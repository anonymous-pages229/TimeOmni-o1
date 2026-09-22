r"""LLM-as-judge for the **explanation quality** of the understanding branch.

Purpose: acceptance test for the answer-first arm -- its selling point is "interpretability at
equal accuracy", and merely reporting "accuracy did not drop" says nothing about whether the
explanations are worth anything. **Arm-agnostic**: whichever infer directory you pass gets
judged (CoT and answer-first are alternative candidates; pick one by accuracy, then judge its
explanations with this module; the explanations of the two arms are not compared with each other).

    python chronos_llm/eval/explanation_judge.py outputs/eval/<run>/infer \
        --out outputs/eval/<run>/explanation_judge --per_dataset 100

Reuses the client / retry / concurrency / resume logic of `text_alignment_eval` (the same
OpenAI-compatible API). The judge API is an external endpoint, so run this where it is reachable.

Three design decisions, none of them obvious:

1. **Sampling is stratified by correct/incorrect answer, not random**. A random sample of 100
   contains ~80 correct answers, but **explanation quality on the incorrect samples is the direct
   signal of "post-hoc rationalisation"** -- if the answer is already wrong yet the explanation is
   still well-structured with sufficient criteria, the explanation is decoupled from the actual
   decision and is just a story made up for a predetermined answer. Hence by default half of
   each stratum is sampled, and the final report splits the two groups to compare mean `support`.

2. **Further stratified by sub-task within a dataset**. VeriTime has 7 sub-tasks; random
   sampling would give CTU (only 144 test rows, and the most anomalous one) just a dozen rows.
   Sub-tasks get equal allocations.

3. **"Citing concrete numeric values" is a penalty, not a bonus** -- specific to this project.
   The model **never sees any numeric text** (the series enters only via soft-prompt injection),
   so any concrete sample value appearing in an explanation is necessarily hallucinated (this is
   exactly how the CTU enumeration degeneration was exposed). A generic judge template would
   reward "well-grounded citation of the data", which here would **score in the wrong direction**,
   so the prompt explicitly asks to flag fabricated numbers and reports a separate
   `hallucinated_numbers` metric.

**Absolute scores must be read against a null hypothesis**: the judge is generous with text that
"sounds plausible", so a support score of 4.72/5 on its own is unanchored. The shuffled-pairing
control is in `scripts/utils/shuffled_explanation_judge_control.py` (swap explanations to test
support, swap reference reasoning to test consistency); it reuses `build_records`/`run_judge_batch`
from this module, so it uses the exact same sample and prompt as the main table.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from chronos_llm.eval.text_alignment_eval import (  # noqa: E402
    DEFAULT_JUDGE_MODEL, _retry_exceptions, _with_retry, build_openai_client,
)
# "agg6" denotes the six-source MMTR understanding pool.
from chronos_llm.scripts.utils.summarize_agg6_eval import extract  # noqa: E402
# Domain assignment shares one implementation with the by_domain aggregation of Table 1 --
# the two must agree, otherwise "explanation quality" and "accuracy" would be reported under
# different domain definitions.
from chronos_llm.scripts.utils.summarize_agg6_by_domain import domain_of  # noqa: E402

JUDGE_SYSTEM = (
    "You are a strict evaluator of explanations produced by a time-series analysis model. "
    "The model sees the time series ONLY through a learned encoder — it has NO access to the "
    "raw numeric values as text. Therefore any concrete numeric value it cites (e.g. "
    "'-0.2859 kWh', 'a peak of 3.2 mV') is necessarily fabricated, and must be flagged. "
    "Reply with a single JSON object and nothing else."
)

JUDGE_USER_TEMPLATE = """Question posed to the model:
{question}

Correct answer: {gt_answer}
Model's answer: {pred_answer}

Model's explanation:
{explanation}

Reference reasoning written by a domain expert (may use different wording):
{gt_reasoning}

Rate the model's EXPLANATION on three axes and reply with JSON:
{{"support": <1-5>, "consistency": <1-5>, "hallucinated_numbers": <0 or 1>, "reason": "<one sentence>"}}

- "support": does the explanation actually justify THE ANSWER THE MODEL GAVE? Judge internal
  coherence only — do NOT reward or punish based on whether that answer is correct.
  1 = unrelated or self-contradictory, 5 = directly and specifically justifies it.
- "consistency": do the explanation's criteria match the reference reasoning's criteria?
  Ignore wording, ordering, and any numeric values. 1 = unrelated criteria, 5 = same criteria.
- "hallucinated_numbers": 1 if the explanation states concrete numeric readings/statistics of
  the series (which the model cannot observe), 0 otherwise. Generic qualitative statements
  ("a sustained high plateau", "frequent drops") are NOT hallucinations.
"""


def split_answer_explanation(text):
    """Split a generation into (answer line, explanation). Both arm formats are covered:

    - answer-first: ``Answer: X\\n\\n<explanation>``  => the explanation comes **after** the answer
    - CoT:          ``<reasoning>\\nAnswer: X``        => the explanation comes **before** the answer

    So we cut at "the `Answer:` line" and join the two sides into the explanation -- no need to
    know which arm produced it.

    The boundary must be the **first line-initial** marker, the same convention as the two
    extractors (`_ANSWER_LINE_RE`): the VeriTime explanation template contains a line
    "Step 6 Summarizing the thinking process to output the answer:"; cutting at the "last"
    occurrence would turn the answer into an empty string and move most of the explanation into
    the "before the answer" part. Only when no line-initial marker exists do we fall back to the
    last occurrence.
    """
    if not text:
        return "", ""
    anchored = list(re.finditer(r"(?m)^[ \t]*[Aa]nswer:[ \t]*", text))
    if anchored:
        hit = anchored[0]
    else:
        loose = list(re.finditer(r"[Aa]nswer:\s*", text))
        if not loose:
            return "", text.strip()      # no answer marker: treat the whole text as the explanation
        hit = loose[-1]
    head = text[: hit.start()]
    tail = text[hit.end():]
    nl = tail.find("\n")
    answer_line = tail if nl < 0 else tail[:nl]
    rest = "" if nl < 0 else tail[nl:]
    return answer_line.strip(), (head.strip() + "\n" + rest.strip()).strip()


def domain_budget(n, k=2.0, floor=40, cap=250):
    """How many rows to sample from one domain: **proportional to sqrt(n)**, then clamped to
    [floor, cap] (and to n when fewer than n rows exist).

    Why sqrt(n) rather than equal or proportional allocation (domains with more samples should
    get somewhat more draws):
    - **Equal** (the old 100 rows per dataset) gives Physiology (50,792) and Energy (160) the same
      sample size, wasting estimation precision on the large domain while oversampling the small one;
    - **Proportional to n** lets Physiology take 72%, with the other 11 domains together under 30%,
      which amounts to evaluating only ECG-QA.
    sqrt(n) is the standard compromise between the two (standard error ~ 1/sqrt(sample size), so a
    sqrt(n) allocation makes the relative precision of the domains roughly comparable).
    `floor=40` guarantees that even the smallest domain can be split into ~20 correct / ~20 incorrect;
    `cap=250` keeps Physiology from eating the budget. With k=2.0 all 12 domains total about 1,070 rows.
    """
    import math
    return max(1, min(n, cap, max(floor, round(k * math.sqrt(n)))))


def stratified_sample(rows, n_total, seed=0):
    """Within a group: allocate equally across sub-tasks first, then within each sub-task split
    half **correct / half incorrect**.

    A short side is topped up from the other side (and the returned records keep the `correct`
    flag, so the summary splits the groups by the actual ratio).
    Deterministic: fixed seed + sorting by id before sampling, so runs are comparable.
    """
    by_task = defaultdict(list)
    for r in rows:
        by_task[r.get("task") or "—"].append(r)
    tasks = sorted(by_task)
    per_task = max(1, n_total // max(len(tasks), 1))
    rng = random.Random(seed)
    out = []
    for t in tasks:
        rs = sorted(by_task[t], key=lambda r: str(r.get("id", "")))
        ok = [r for r in rs if extract(r["ground_truth"]) == extract(r["generated_text"])]
        bad = [r for r in rs if extract(r["ground_truth"]) != extract(r["generated_text"])]
        half = per_task // 2
        take_ok = rng.sample(ok, min(half, len(ok))) if ok else []
        take_bad = rng.sample(bad, min(per_task - len(take_ok), len(bad))) if bad else []
        # Top up from the other side when one side is short
        if len(take_ok) + len(take_bad) < per_task:
            spare = [r for r in ok if r not in take_ok] + [r for r in bad if r not in take_bad]
            need = per_task - len(take_ok) - len(take_bad)
            take_ok += rng.sample(spare, min(need, len(spare))) if spare else []
        for r in take_ok + take_bad:
            out.append((r, extract(r["ground_truth"]) == extract(r["generated_text"])))
    return out


def build_messages(rec):
    user = JUDGE_USER_TEMPLATE.format(
        question=(rec["question"] or "")[:1200],
        gt_answer=rec["gt_answer"],
        pred_answer=rec["pred_answer"] or "(no answer produced)",
        explanation=(rec["explanation"] or "(empty)")[:4000],
        gt_reasoning=(rec["gt_reasoning"] or "(none provided)")[:3000],
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_response(raw):
    """Extract the JSON; return None on failure (the caller records an error instead of silently
    counting it as 0 and dragging the mean down)."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return {
            "support": float(d["support"]),
            "consistency": float(d["consistency"]),
            "hallucinated_numbers": int(d.get("hallucinated_numbers", 0)),
            "reason": str(d.get("reason", ""))[:300],
        }
    except (KeyError, TypeError, ValueError):
        return None


def judge_one(rec, client, model=DEFAULT_JUDGE_MODEL, max_attempts=3):
    def _call():
        return client.chat.completions.create(model=model, messages=build_messages(rec))
    resp = _with_retry(_call, max_attempts=max_attempts, retry_exceptions=_retry_exceptions())
    raw = resp.choices[0].message.content
    parsed = parse_response(raw)
    return {"uid": rec["uid"], **(parsed or {}), "raw": None if parsed else raw,
            "error": None if parsed else "unparsable"}


def summarize(scored, records, group_key="domain"):
    """-> markdown lines. Split by **domain (default) or dataset** x (correct/incorrect); the
    support-score gap between correct and incorrect is exactly the "post-hoc rationalisation"
    criterion."""
    by_uid = {r["uid"]: r for r in records}
    groups = defaultdict(lambda: defaultdict(list))
    for s in scored:
        if s.get("error"):
            continue
        r = by_uid.get(s["uid"])
        if not r:
            continue
        groups[r[group_key]]["correct" if r["correct"] else "wrong"].append(s)

    head = "discipline" if group_key == "domain" else "dataset"
    lines = [f"| {head} | group | n | support(1-5) | consistency(1-5) | hallucinated numbers % |", "|---|---|---|---|---|---|"]
    for ds in sorted(groups):
        for g in ("correct", "wrong"):
            ss = groups[ds][g]
            if not ss:
                continue
            n = len(ss)
            lines.append(
                f"| {ds} | {g} | {n} "
                f"| {sum(x['support'] for x in ss) / n:.2f} "
                f"| {sum(x['consistency'] for x in ss) / n:.2f} "
                f"| {100.0 * sum(x['hallucinated_numbers'] for x in ss) / n:.1f} |")
    # Total + post-hoc rationalisation criterion
    allv = [s for g in groups.values() for ss in g.values() for s in ss]
    ok = [s for ds in groups for s in groups[ds]["correct"]]
    bad = [s for ds in groups for s in groups[ds]["wrong"]]
    if allv:
        lines.append(f"| **all** | — | {len(allv)} "
                     f"| {sum(x['support'] for x in allv) / len(allv):.2f} "
                     f"| {sum(x['consistency'] for x in allv) / len(allv):.2f} "
                     f"| {100.0 * sum(x['hallucinated_numbers'] for x in allv) / len(allv):.1f} |")
    if ok and bad:
        d = sum(x["support"] for x in ok) / len(ok) - sum(x["support"] for x in bad) / len(bad)
        lines += ["", f"**Post-hoc rationalisation criterion**: support gap between the correct and the wrong group = **{d:+.2f}**.",
                  "A gap near 0 => the model produces equally self-consistent explanations for **wrong** answers, i.e. the explanation is decoupled from the actual decision;",
                  "a clearly positive gap => the explanation does track decision quality."]
    return lines



def make_record(r, ds, dom, correct):
    """One infer-jsonl row -> judge record. The three-part `uid` (dataset::id::sub-task) is unique
    across the whole pool; resume and the pairing of the shuffled control both key on it."""
    ans, expl = split_answer_explanation(r.get("generated_text"))
    return {
        "uid": f"{ds}::{r.get('id', '')}::{r.get('task', '')}",
        "dataset": ds, "domain": dom, "task": r.get("task", ""), "correct": correct,
        "question": r.get("input_text", ""), "gt_answer": r.get("ground_truth", ""),
        "pred_answer": ans, "explanation": expl,
        "gt_reasoning": r.get("gt_reasoning", ""),
    }


def load_rows(infer_dir):
    """Read every jsonl under the infer directory -> [(row, dataset_name, domain), ...] (no sampling)."""
    all_rows = []
    for fn in sorted(os.listdir(infer_dir)):
        if not fn.endswith(".jsonl"):
            continue
        ds = fn.replace("_test.jsonl", "").replace(".jsonl", "")
        for line in open(os.path.join(infer_dir, fn), encoding="utf-8"):
            r = json.loads(line)
            all_rows.append((r, ds, domain_of(r)))
    return all_rows


def build_records(infer_dir, by="domain", per_dataset=100, budget_k=2.0, floor=40, cap=250,
                  seed=0, skip_empty_explanation=False, verbose=True):
    """infer directory -> list of sampled judge records.

    **The main table and the shuffled null-hypothesis control
    (`scripts/utils/shuffled_explanation_judge_control.py`) share this one implementation**:
    same parameters + same seed => the same uids are drawn row for row, which is what makes the
    two tables comparable side by side. The sampling scheme itself is not repeated here; see the
    docstrings of `domain_budget` / `stratified_sample`.
    """
    all_rows = load_rows(infer_dir)

    if skip_empty_explanation:
        dropped = defaultdict(int)
        kept = []
        for r, ds, dom in all_rows:
            if split_answer_explanation(r.get("generated_text"))[1].strip():
                kept.append((r, ds, dom))
            else:
                dropped[(dom, ds)] += 1
        if dropped and verbose:
            print(f"[judge] --skip_empty_explanation: dropped rows without an explanation "
                  f"{sum(dropped.values())}/{len(all_rows)}")
            for (dom, ds), n in sorted(dropped.items(), key=lambda kv: -kv[1]):
                print(f"    {dom:26s} {ds:14s} {n}")
        all_rows = kept

    records = []
    if by == "dataset":
        buckets = defaultdict(list)
        for r, ds, dom in all_rows:
            buckets[ds].append((r, dom))
        for ds in sorted(buckets):
            rows = [r for r, _ in buckets[ds]]
            dom_of_row = {id(r): d for r, d in buckets[ds]}
            for r, correct in stratified_sample(rows, per_dataset, seed=seed):
                records.append(make_record(r, ds, dom_of_row[id(r)], correct))
    else:
        buckets = defaultdict(list)
        for r, ds, dom in all_rows:
            buckets[dom].append((r, ds))
        if verbose:
            print(f"[judge] sampling by domain (k={budget_k}*sqrt(n), clamped to [{floor}, {cap}]):")
        for dom in sorted(buckets, key=lambda d: -len(buckets[d])):
            rows = [r for r, _ in buckets[dom]]
            ds_of_row = {id(r): ds for r, ds in buckets[dom]}
            n_take = domain_budget(len(rows), budget_k, floor, cap)
            if verbose:
                print(f"    {dom:26s} total {len(rows):6d} -> sample {n_take:4d}")
            for r, correct in stratified_sample(rows, n_take, seed=seed):
                records.append(make_record(r, ds_of_row[id(r)], dom, correct))

    n_ok = sum(1 for r in records if r["correct"])
    if verbose:
        print(f"[judge] sampled {len(records)} rows (correct {n_ok} / wrong {len(records) - n_ok})"
              f", covering {len({r['domain'] for r in records})} domains, "
              f"{len({r['dataset'] for r in records})} datasets, "
              f"{len({r['task'] for r in records})} sub-tasks")
        empty = sum(1 for r in records if not r["explanation"].strip())
        if empty:
            print(f"[judge] WARNING: {empty} sampled rows have an empty explanation -- if the share is "
                  f"high, this checkpoint is not generating explanations at all; inspect the "
                  f"generations before scoring", file=sys.stderr)
    return records


def run_judge_batch(records, checkpoint_path, client, model=DEFAULT_JUDGE_MODEL, max_workers=3):
    """Concurrent scoring + resume by uid; returns **all** scores in the checkpoint (including
    those from earlier runs).

    The shuffled null-hypothesis control (`scripts/utils/shuffled_explanation_judge_control.py`)
    reuses it, so the null hypothesis and the main table go through the same call / retry / parse
    path -- any difference can only come from the one field that was shuffled.
    """
    done = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"[judge] resuming: {len(done)} rows already done")

    todo = [r for r in records if r["uid"] not in done]
    with open(checkpoint_path, "a", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(judge_one, r, client, model): r for r in todo}
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    s = fut.result()
                except Exception as e:  # noqa: BLE001
                    s = {"uid": futs[fut]["uid"], "error": str(e)[:200]}
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                fout.flush()
                if i % 50 == 0:
                    print(f"[judge] {i}/{len(todo)}")
    with open(checkpoint_path, encoding="utf-8") as f:
        scored = [json.loads(l) for l in f]
    n_err = sum(1 for s in scored if s.get("error"))
    if n_err:
        print(f"[judge] WARNING: {n_err}/{len(scored)} rows failed to score (error recorded, excluded from means)", file=sys.stderr)
    return scored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infer_dir")
    ap.add_argument("--out", required=True, help="output directory (score JSONL + markdown summary)")
    ap.add_argument("--by", choices=("domain", "dataset"), default="domain",
                    help="grouping axis for sampling and summary. domain (default) = SciTS discipline "
                         "convention, same axis as Table 1; dataset = the old per-dataset scheme "
                         "(a fixed --per_dataset rows per group)")
    ap.add_argument("--per_dataset", type=int, default=100, help="only used with --by dataset")
    ap.add_argument("--budget_k", type=float, default=2.0,
                    help="only with --by domain: sample k*sqrt(n) rows per domain (k=2 gives ~1,070 over 12 domains)")
    ap.add_argument("--floor", type=int, default=40, help="only with --by domain: minimum rows per domain")
    ap.add_argument("--cap", type=int, default=250, help="only with --by domain: maximum rows per domain")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max_workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_empty_explanation", action="store_true",
                    help="drop rows whose generation contains no explanation at all **before** sampling. "
                         "ST-Bench is trained/evaluated under the direct-answer convention (its official "
                         "test set has 0%% think) => that dataset has 100%% no explanation: mixing it into "
                         "the sample makes the judge give '(empty)' the lowest support and systematically "
                         "drags down its domain (Urbanism, 96%% ST-Bench) -- that is not poor explanation "
                         "quality, there is no explanation by design. After dropping, the domain budget is "
                         "recomputed from **the number of rows that have an explanation**.")
    ap.add_argument("--dry_run", action="store_true", help="only sample, do not call the API (inspect the sample distribution)")
    args = ap.parse_args(argv)

    records = build_records(
        args.infer_dir, by=args.by, per_dataset=args.per_dataset,
        budget_k=args.budget_k, floor=args.floor, cap=args.cap, seed=args.seed,
        skip_empty_explanation=args.skip_empty_explanation)
    if args.dry_run:
        return 0

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "scores.jsonl")
    scored = run_judge_batch(records, ckpt, build_openai_client(),
                             model=args.model, max_workers=args.max_workers)

    how = (f"{args.per_dataset} rows per dataset" if args.by == "dataset" else
           f"{args.budget_k}*sqrt(n) rows per domain (clamped to [{args.floor}, {args.cap}])")
    if args.skip_empty_explanation:
        how += ", **rows without an explanation dropped before sampling** (ST-Bench uses the direct-answer convention)"
    md = "\n".join(["# Explanation quality LLM-judge", "",
                    f"infer_dir=`{args.infer_dir}`  judge={args.model}  "
                    f"{how}, further stratified by sub-task and correct/wrong within each domain", ""]
                   + summarize(scored, records, group_key=args.by))
    out_md = os.path.join(args.out, "explanation_judge.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[judge] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
