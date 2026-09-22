r"""Understanding side: **mismatched-pairing null-hypothesis control for the explanation-quality judge**.

Background: the support (e.g. 4.72/5) and consistency (e.g. 4.24/5) scores reported by
`eval/explanation_judge.py` are **absolute values without a reference point** -- an LLM judge is
generous towards text that merely "sounds plausible", so without a null hypothesis one cannot answer
"is 4.72 high?". The three forecasting-side LLM-judge metrics each already have a shuffled control
(`shuffled_text_alignment_control.py` / `shuffled_visual_faithfulness_control.py` /
`shuffle_band_null_control.py`); the understanding side was missing one and this script fills the gap
(the more a metric relies on subjective judge scoring, the more it needs a null reference to interpret
absolute scores correctly).

**Two shuffles, each breaking exactly one pairing relation** (prompt unchanged, same `judge_one`):

- **Group expl (null for support)**: replace `explanation` with the explanation of **another** sample;
  question / gold answer / model answer / reference reasoning stay the sample's own. This breaks the
  "explanation <-> the model's own answer" pair, which is precisely what support is defined on.
  **Support should drop significantly**; if it does not, the judge is not reading the pairing at all and
  the 4.72 in the main table is not trustworthy either.
- **Group ref (null for consistency)**: replace `gt_reasoning` (expert reference reasoning) with another
  sample's; everything else unchanged. This breaks the "explanation <-> reference reasoning" pair.
  **Consistency should drop significantly, while support by definition should not move**.

Each group carries an **implementation self-check** that exposes bugs instead of producing a
plausible-looking but wrong table:

1. Group expl pairs by **cyclic shift** inside a scope => the multiset of explanations is unchanged =>
   **the hallucinated-number rate should in theory equal the main table exactly** (the judge decides
   that column from the explanation alone). A large deviation = pairing / sampling misalignment.
2. Group ref leaves explanation and answer untouched => **support should match the main table
   sample by sample**. Its gap to the main table is exactly this judge's **test-retest noise floor**
   (temperature>0 + small context perturbations); any later "support differs by 0.0x" must first be
   compared against it.

Pairing is a cyclic shift **within the same sub-task** (`--scope task`, default): same question type,
just a different sample -- the strictest null hypothesis. Random cross-dataset pairing (an ECG
explanation under a network-traffic question) is trivially rejected by the judge, which makes the null
too low and artificially inflates the relative advantage of the main table.

Sampling is **not re-implemented**: we call `explanation_judge.build_records()` directly with the same
parameters and seed => the same uids as the main table, sample by sample; at start-up the script checks
the uid set against the main table's `scores.jsonl` and fails hard on mismatch (so two tables produced
under drifting protocols cannot be compared side by side).

    # same parameters as the main table
    python chronos_llm/scripts/utils/shuffled_explanation_judge_control.py \
        outputs/eval/<run>/infer \
        --real_scores outputs/eval/<run>/explanation_judge_by_domain/scores.jsonl \
        --out outputs/eval/<run>/explanation_judge_shuffled \
        --skip_empty_explanation

Note: this script needs outbound network access to the judge API.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from chronos_llm.eval.explanation_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL, build_openai_client, build_records, run_judge_batch,
)

SCOPE_KEYS = {"task": ("dataset", "task"), "dataset": ("dataset",), "domain": ("domain",),
              "global": ()}


def build_random_pairs(records, seed=0):
    """-> ``{uid: partner_uid}``: a **uniformly random derangement over the whole pool**, ignoring scope.

    Why it is needed: the name `--scope global` is misleading -- records are sorted by uid and
    cyclically shifted, and uids start with the dataset name => neighbouring uids almost always belong
    to the same dataset; empirically **99.2% of pairs land in the same dataset**, i.e. the hardest
    pairing rather than a random one (after fixing the same bug on the forecasting side, the deltas grew
    4-14x).

    Uses **Sattolo's algorithm** to generate a single-cycle permutation: guarantees **no fixed points**
    (no self-pairing) while still being a **permutation** => the multiset of explanations is conserved,
    so the hallucination-rate self-check of group expl remains valid. (The two forecasting-side scripts
    resample each record independently, which is not a permutation -- they have no self-check that
    depends on multiset conservation, so both designs are valid on their own; do not mix them.)
    """
    uids = sorted(r["uid"] for r in records)
    n = len(uids)
    if n < 2:
        raise ValueError("fewer than 2 samples, cannot build shuffled pairs")
    order = list(uids)
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):          # Sattolo: j strictly < i => single cycle, no fixed points
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    pairs = {order[i]: order[(i + 1) % n] for i in range(n)}
    assert all(u != v for u, v in pairs.items()), "self-pairing found, null hypothesis invalid"
    assert sorted(pairs.values()) == uids, "not a permutation, explanation multiset not conserved"
    return pairs, 0


def build_shuffled_pairs(records, scope="task", shift=1):
    """-> ``{uid: partner_uid}``: sort uids within each scope and cyclically shift by ``shift``.

    A scope containing **only 1** sample cannot be paired within itself (it would self-pair = null
    hypothesis invalid); such samples are collected into an orphan pool that is cyclically shifted on
    its own; only when the pool itself has a single sample do we fall back to any other global sample.
    The returned mapping is guaranteed **self-pairing free**, and within every full scope it is a
    **permutation** (=> explanation multiset conserved, which is the premise of the hallucination-rate
    self-check of group expl).
    """
    keys = SCOPE_KEYS[scope]
    buckets = defaultdict(list)
    for r in records:
        buckets[tuple(r[k] for k in keys)].append(r["uid"])

    pairs, orphans = {}, []
    for key in sorted(buckets):
        uids = sorted(buckets[key])
        if len(uids) < 2:
            orphans.extend(uids)
            continue
        s = shift % len(uids) or 1
        for i, u in enumerate(uids):
            pairs[u] = uids[(i + s) % len(uids)]

    if orphans:
        orphans = sorted(orphans)
        if len(orphans) >= 2:
            for i, u in enumerate(orphans):
                pairs[u] = orphans[(i + 1) % len(orphans)]
        else:
            u = orphans[0]
            others = sorted(r["uid"] for r in records if r["uid"] != u)
            if not others:
                raise ValueError("fewer than 2 samples, cannot build shuffled pairs")
            pairs[u] = others[0]

    assert all(u != v for u, v in pairs.items()), "self-pairing found, null hypothesis invalid"
    assert len(pairs) == len(records), (len(pairs), len(records))
    return pairs, len(orphans)


def apply_shuffle(records, pairs, field):
    """Replace ``field`` of every record with the same field of its paired sample; keep everything else.

    Additionally writes ``_from_uid`` recording the true owner of the swapped content, which the
    per-sample table uses for paired comparison.
    """
    by_uid = {r["uid"]: r for r in records}
    out = []
    for r in records:
        partner = by_uid[pairs[r["uid"]]]
        rec = dict(r)
        rec[field] = partner[field]
        rec["_from_uid"] = partner["uid"]
        out.append(rec)
    return out


def score_index(scored):
    """Score list -> ``{uid: score_dict}``, skipping error rows (not silently counted as 0)."""
    return {s["uid"]: s for s in scored if not s.get("error") and "support" in s}


def agg(scores):
    """A group of scores -> mean over the three axes (hallucination column in percent)."""
    if not scores:
        return {"n": 0, "support": float("nan"), "consistency": float("nan"),
                "halluc": float("nan")}
    n = len(scores)
    return {
        "n": n,
        "support": sum(s["support"] for s in scores) / n,
        "consistency": sum(s["consistency"] for s in scores) / n,
        "halluc": 100.0 * sum(s["hallucinated_numbers"] for s in scores) / n,
    }


def paired_delta(real_idx, shuf_idx, pairs, field_owner_is_partner=True):
    """**Paired** support difference of the same shuffled content under "true pairing vs mismatched pairing".

    In group expl the record with uid=j is judged on the explanation of uid=i (i=pairs[j] is the
    owner), so it must be compared against ``real[i]``, not ``real[j]`` -- otherwise "the explanation
    changed owner" gets confused with "the sample changed".
    """
    deltas, win, tie, loss = [], 0, 0, 0
    for uid, s in shuf_idx.items():
        owner = pairs[uid] if field_owner_is_partner else uid
        r = real_idx.get(owner)
        if r is None:
            continue
        d = s["support"] - r["support"]
        deltas.append(d)
        win += d > 0
        tie += d == 0
        loss += d < 0
    mean = sum(deltas) / len(deltas) if deltas else float("nan")
    return {"n": len(deltas), "mean_delta": mean, "win": win, "tie": tie, "loss": loss}


def by_domain_table(records, real_idx, expl_idx, ref_idx):
    """-> markdown rows: one row per discipline, real / both shuffles side by side."""
    dom_of = {r["uid"]: r["domain"] or "Unknown" for r in records}
    groups = defaultdict(lambda: defaultdict(list))
    for name, idx in (("real", real_idx), ("expl", expl_idx), ("ref", ref_idx)):
        for uid, s in idx.items():
            groups[dom_of.get(uid, "Unknown")][name].append(s)

    lines = ["| discipline | n | support real | support shuf-expl | **delta (lower=better)** | consistency real "
             "| consistency shuf-ref | **delta (lower=better)** | halluc% real/shuf-expl |", "|---|---|---|---|---|---|---|---|---|"]
    for dom in sorted(groups):
        a, b, c = (agg(groups[dom][k]) for k in ("real", "expl", "ref"))
        lines.append(
            f"| {dom} | {a['n']} | {a['support']:.2f} | {b['support']:.2f} "
            f"| **{b['support'] - a['support']:+.2f}** | {a['consistency']:.2f} "
            f"| {c['consistency']:.2f} | **{c['consistency'] - a['consistency']:+.2f}** "
            f"| {a['halluc']:.1f} / {b['halluc']:.1f} |")
    A, B, C = (agg([s for g in groups.values() for s in g[k]]) for k in ("real", "expl", "ref"))
    lines.append(
        f"| **All** | **{A['n']}** | **{A['support']:.2f}** | **{B['support']:.2f}** "
        f"| **{B['support'] - A['support']:+.2f}** | **{A['consistency']:.2f}** "
        f"| **{C['consistency']:.2f}** | **{C['consistency'] - A['consistency']:+.2f}** "
        f"| **{A['halluc']:.1f} / {B['halluc']:.1f}** |")
    return lines, A, B, C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infer_dir")
    ap.add_argument("--real_scores", required=True,
                    help="scores.jsonl of the main table (side-by-side comparison + uid-set consistency check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", choices=tuple(SCOPE_KEYS), default="task",
                    help="pairing scope: task (default, same dataset & sub-task, strictest) / dataset / domain / global")
    ap.add_argument("--shift", type=int, default=1)
    ap.add_argument("--pairing", choices=["shift", "random"], default="shift",
                    help="shift = cyclic shift within scope (legacy protocol; empirically 99.2% same-dataset, "
                         "not a random null); random = uniform derangement over the whole pool "
                         "(Sattolo, still a permutation => self-checks stay valid)")
    ap.add_argument("--by", choices=("domain", "dataset"), default="domain")
    ap.add_argument("--per_dataset", type=int, default=100)
    ap.add_argument("--budget_k", type=float, default=2.0)
    ap.add_argument("--floor", type=int, default=40)
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_empty_explanation", action="store_true")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max_workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0: judge only the first N records per group (sorted by uid; saves API cost; "
                         "the official table uses the full set)")
    ap.add_argument("--dry_run", action="store_true", help="only build the pairs, do not call the API")
    args = ap.parse_args(argv)

    records = build_records(
        args.infer_dir, by=args.by, per_dataset=args.per_dataset, budget_k=args.budget_k,
        floor=args.floor, cap=args.cap, seed=args.seed,
        skip_empty_explanation=args.skip_empty_explanation)

    real_scored = [json.loads(l) for l in open(args.real_scores, encoding="utf-8")]
    real_idx = score_index(real_scored)
    uids, real_uids = {r["uid"] for r in records}, {s["uid"] for s in real_scored}
    if uids != real_uids:
        raise SystemExit(
            f"[control] sample set differs from the main table: this run {len(uids)} / main table {len(real_uids)}, "
            f"only here {len(uids - real_uids)}, only in main table {len(real_uids - uids)}. "
            f"Side-by-side comparison requires identical sampling -- align --by/--budget_k/--floor/--cap/--seed/"
            f"--skip_empty_explanation and rerun.")
    print(f"[control] uid set identical to the main table ({len(uids)} records)")

    if args.pairing == "random":
        pairs, n_orphan = build_random_pairs(records, seed=args.seed + 1)
    else:
        pairs, n_orphan = build_shuffled_pairs(records, scope=args.scope, shift=args.shift)
    dom_of = {r["uid"]: r["domain"] for r in records}
    ds_of = {r["uid"]: r["dataset"] for r in records}
    same_dom = sum(1 for u, v in pairs.items() if dom_of[u] == dom_of[v])
    same_ds = sum(1 for u, v in pairs.items() if ds_of[u] == ds_of[v])
    # The pairing protocol must be spelled out verbatim in the artifacts: the shift protocol is
    # empirically 99.2% same-dataset and not a random null at all; writing only scope/shift would
    # mislead a later reader into assuming random pairing.
    pairing_desc = (f"pairing=random (uniform derangement over the pool, Sattolo) seed={args.seed + 1}"
                    if args.pairing == "random"
                    else f"pairing=shift (cyclic shift within scope={args.scope}, shift={args.shift})")
    print(f"[control] {pairing_desc}: {len(pairs)} pairs, no self-pairing, {n_orphan} orphans, "
          f"same dataset {same_ds}/{len(pairs)} ({100 * same_ds / len(pairs):.1f}%), "
          f"same domain {same_dom}/{len(pairs)}")

    shuf_expl = apply_shuffle(records, pairs, "explanation")
    shuf_ref = apply_shuffle(records, pairs, "gt_reasoning")
    if args.limit > 0:
        keep = {r["uid"] for r in sorted(records, key=lambda x: x["uid"])[:args.limit]}
        shuf_expl = [r for r in shuf_expl if r["uid"] in keep]
        shuf_ref = [r for r in shuf_ref if r["uid"] in keep]
        print(f"[control] --limit {args.limit}: judging only {len(shuf_expl)} records per group")
    if args.dry_run:
        for r in shuf_expl[:2]:
            print(f"\n  uid={r['uid']}\n  explanation taken from {r['_from_uid']}\n  "
                  f"first 120 chars of explanation: {r['explanation'][:120]!r}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    client = build_openai_client()
    print(f"\n[control] group expl (support null, swapped explanation) n={len(shuf_expl)}")
    expl_idx = score_index(run_judge_batch(
        shuf_expl, os.path.join(args.out, "scores_shuffle_expl.jsonl"), client,
        model=args.model, max_workers=args.max_workers))
    print(f"\n[control] group ref (consistency null, swapped reference reasoning) n={len(shuf_ref)}")
    ref_idx = score_index(run_judge_batch(
        shuf_ref, os.path.join(args.out, "scores_shuffle_ref.jsonl"), client,
        model=args.model, max_workers=args.max_workers))

    # Compare only on uids scored in all three groups (same denominator, so a failed judge call in one
    # group cannot give the three columns different bases)
    common = set(real_idx) & set(expl_idx) & set(ref_idx)
    real_c = {u: real_idx[u] for u in common}
    expl_c = {u: expl_idx[u] for u in common}
    ref_c = {u: ref_idx[u] for u in common}
    lines, A, B, C = by_domain_table(records, real_c, expl_c, ref_c)

    pd_expl = paired_delta(real_idx, expl_c, pairs)
    halluc_gap = B["halluc"] - A["halluc"]
    supp_noise = C["support"] - A["support"]
    ref_halluc_same = sum(1 for u in common
                          if ref_c[u]["hallucinated_numbers"] == real_c[u]["hallucinated_numbers"])

    md = "\n".join([
        "# Explanation-quality judge: mismatched-pairing null-hypothesis control", "",
        f"infer_dir=`{args.infer_dir}`  judge={args.model}  **{pairing_desc}**  "
        f"same-dataset pairing rate **{100 * same_ds / len(pairs):.1f}%**  "
        f"{len(common)} records comparable across all three groups (sampling identical to the main table)", "",
        "Each shuffle breaks exactly one pairing relation, prompt unchanged: **group \"shuffled explanation\"** "
        "replaces the explanation with another sample's (breaks explanation <-> the model's own answer, "
        "targeting the definition of support); **group \"shuffled reference\"** replaces the expert reference "
        "reasoning with another sample's (breaks explanation <-> reference reasoning, targeting the definition "
        "of consistency).", "",
    ] + lines + [
        "", "## Self-checks (a bug shows up here instead of producing a plausible-looking wrong table)", "",
        f"1. **Hallucinated-number rate conservation**: the pairing of group \"shuffled explanation\" is a "
        f"**permutation** (both cyclic shift and derangement are) => the multiset of explanations is unchanged "
        f"=> this column should in theory be identical. Observed {A['halluc']:.1f}% -> {B['halluc']:.1f}% "
        f"(delta={halluc_gap:+.1f} pp); the deviation comes from the judge's own randomness."
        + ("**Note that under a random derangement this check becomes looser**: the multiset is conserved, "
           "but the context the judge sees is not -- an ECG explanation placed under a network-traffic question "
           "shifts the judgement (shift protocol deviates by +1.4pp, random derangement by +5.1pp). It still "
           "catches gross pairing misalignment, but is no longer a precision instrument."
           if args.pairing == "random" else ""),
        f"2. **Support test-retest noise floor**: group \"shuffled reference\" leaves explanation and answer "
        f"untouched => support should by definition not change. Observed {A['support']:.2f} -> {C['support']:.2f} "
        f"(delta={supp_noise:+.2f}) -- **this is the noise floor of this judge; any support difference smaller "
        f"than it is not a real difference**.",
        f"3. **Per-record agreement of the hallucination column** (group \"shuffled reference\" vs main table, "
        f"the same explanation judged twice): "
        f"{100.0 * ref_halluc_same / max(len(common), 1):.1f}%.", "",
        "## Paired test (same explanation: true pairing vs mismatched pairing)", "",
        f"n={pd_expl['n']}, paired support difference **{pd_expl['mean_delta']:+.2f}**, "
        f"mismatched higher {pd_expl['win']} / tie {pd_expl['tie']} / mismatched lower {pd_expl['loss']}.",
        "(The same explanation is judged once under each pairing and compared pairwise, which removes the "
        "per-explanation quality variation.)",
    ])
    out_md = os.path.join(args.out, "shuffled_explanation_judge.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("\n" + md)

    csv_path = os.path.join(args.out, "per_sample.csv")
    by_uid = {r["uid"]: r for r in records}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "uid", "domain", "dataset", "task", "correct", "paired_with",
            "real_support", "shuf_expl_support", "shuf_ref_support",
            "real_consistency", "shuf_expl_consistency", "shuf_ref_consistency",
            "real_halluc", "shuf_expl_halluc", "shuf_ref_halluc"])
        w.writeheader()
        for uid in sorted(common):
            r = by_uid[uid]
            w.writerow({
                "uid": uid, "domain": r["domain"] or "Unknown", "dataset": r["dataset"],
                "task": r["task"], "correct": r["correct"], "paired_with": pairs[uid],
                "real_support": real_c[uid]["support"],
                "shuf_expl_support": expl_c[uid]["support"],
                "shuf_ref_support": ref_c[uid]["support"],
                "real_consistency": real_c[uid]["consistency"],
                "shuf_expl_consistency": expl_c[uid]["consistency"],
                "shuf_ref_consistency": ref_c[uid]["consistency"],
                "real_halluc": real_c[uid]["hallucinated_numbers"],
                "shuf_expl_halluc": expl_c[uid]["hallucinated_numbers"],
                "shuf_ref_halluc": ref_c[uid]["hallucinated_numbers"],
            })
    print(f"\n[control] wrote {out_md} and {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
