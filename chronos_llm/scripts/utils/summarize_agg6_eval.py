"""Evaluation summary for the six-source MMTR understanding pool ("agg6") -- **each dataset is
scored with the native metric and split convention of its own paper**, and placed side by side
with the baseline / SOTA numbers transcribed from those papers.

Consumes the output directory of `infer_understanding.py` (one jsonl per test set, same file
name, containing `generated_text`/`ground_truth`/`task`/`scene`) and emits a markdown
comparison table.

    python chronos_llm/scripts/utils/summarize_agg6_eval.py outputs/eval/agg6_epoch1/infer

**Comparability notes (which numbers can be compared against the papers and which cannot)**:

- **OpenTSLM (three subsets)**: dispatched through the official-protocol scorer in
  `eval/eval_opentslm_answer.py`. ECG accuracy matches the paper's Table 2 protocol and can be
  compared directly (our setting is stricter: the official prompt lists per-template candidate
  answers, ours is open-ended). ECG F1 is computed globally and **cannot** be compared (the
  official one is per-template; our HF mirror lacks `template_id`). **HAR/Sleep Acc/F1 cannot be
  compared directly**: the scoring protocol replicates the official `allowed_labels`/
  `_canonicalize_label`, but **the candidate space of the task input differs** -- the official
  loader lists all classes in the prompt (HAR 8 classes / Sleep 6 classes), whereas the CSV
  `prompt` field we pass through is the two-way prompt used when the CoT annotations were
  generated (chance 50% vs 12.5%/16.7%); the official train/eval never uses that field, so those
  numbers are reference only.
- **ST-Bench**: MCQ accuracy on T1/T2/T3. Our test counts (207/1194/1592) **match exactly** the
  ST-Test of the paper's Table 7, so they are directly comparable. The T4 forecasting subtask is
  not part of the understanding branch (it goes through the forecasting branch).
- **VeriTime**: accuracy on the three Scenario subtasks and the four Knowledge datasets. Our test
  is **the full officially released test file** (`Dataset/Scenario/scenario_shuffled_test.jsonl`
  461 rows + `Dataset/Knowledge/knowledge_test.jsonl` 694 rows = 1,155, matching ours row by
  row; question text is verbatim and the candidate set is kept inside the question). Residual
  difference: the per-family counts printed in the paper's Table 9 (CTU 270 / ECG 780 / EMG 450 /
  RCW 320) are larger than the released files (144/200/150/200), so the paper's numbers were
  measured on its **paper-version** test, which is not the same set as the release; comparisons
  must state "official released test, not the paper's printed counts".
- **Time-RA**: two-level weighted F1 -- **Label F1** (binary: Normal Sequence vs any anomaly) and
  **Action F1** (fine-grained type). The paper's Table 2 protocol is a **text-only LLM with 14-shot
  all-class exemplars + CoT**; ours is an encoder route, zero-shot (classes listed in the prompt),
  so **the protocols differ and the numbers are only an order-of-magnitude reference**. We also
  removed the 62.8% mislabeled samples of the official Uni test, so the denominators differ too.
- **HiTSR**: L2/L3 accuracy. The paper's Table 1 is an **OOD main table** (L2 Local samples 500
  from BEDTime, L2 Global 120 from MMTS-Bench, L3 100 from MCQ2; sample indices unreleased), while
  we evaluate **HiTSR's own test split** (in-domain), so **these are not the same questions and
  cannot be compared directly**; they only show our level on the in-distribution data.
- **TelecomTS**: we expand `QnA.{network,anomalies}` into QA pairs and score exact-match answer
  accuracy; the paper's Tables 2/3/4/6 report **per-subtask** F1/Acc (detection F1, duration F1,
  11-class root-cause Acc, QA split into Trend/Traffic/Mobility/Congestion/Location), so **the
  granularity differs and there is no direct comparison**; the numbers serve only as our own
  internal comparison.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

# Share the single definition of the "line-initial Answer: marker" with the official-protocol
# extractor (the two must agree, otherwise the same generations would yield different answers
# on the two evaluation paths).
from chronos_llm.eval.eval_opentslm_answer import first_answer_line  # noqa: E402


# --------------------------- generic answer extraction ---------------------------
# Without an `Answer:` marker, anything longer than this is judged "cannot be an answer; it is an
# unfinished reasoning trace". Calibration: across all 19 subtasks the longest ground_truth is
# 133 characters (and 100% of them carry the marker), while unfinished reasoning has a median of
# several thousand characters -- an order of magnitude apart, so 300 leaves ample margin on both sides.
_BARE_ANSWER_MAX_CHARS = 300
def extract(text):
    """Take the content after `Answer:` and normalise it; strip the reasoning segment first (the
    answer comes after `</think>`).

    **Two "no answer was written" checks**, both required:

    1. `<think>` was opened but never closed => generation was truncated and there is no answer
       segment at all.
    2. **There is no `Answer:` marker at all** => the answer segment was likewise never reached.
       This check was added because check 1 is structurally blind for the **CoT arm**: the CoT
       inference prefix stops at the **opening** `<think>`, that token lives in the **prefix**, not
       in `generated_text`, so "opened but not closed" can never fire and an unfinished reasoning
       trace would be taken wholesale as an "answer". For accuracy this does not matter (wrong
       either way), but weighted F1 (the reported metric for Time-RA) would mint a unique
       pseudo-class for every such sample and inflate the denominator -- 0.62% of time_ra_uni
       samples fell into this case for one CoT checkpoint.

    Check 2 **must carry a length criterion**: a **bare answer** without `Answer:` (e.g.
    "Sudden Spike Anomaly") is legitimate input and must be kept; only "no marker **and**
    implausibly long for an answer" is treated as empty. The threshold is
    `_BARE_ANSWER_MAX_CHARS` -- measured across all 19 subtasks the longest ground_truth is 133
    characters (all with the marker), whereas unfinished reasoning has a median of several
    thousand characters, an order of magnitude apart, so 300 leaves margin on both sides. (A first
    version omitted the length criterion and was caught by the existing "bare answer" test case.)
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        return ""
    line = first_answer_line(text)
    parts = re.split(r"[Aa]nswer:\s*", text)
    if line is not None:
        # **First line-anchored marker + only that line**: in answer-first mode the supervised
        # segment is `Answer: X\n\n<explanation>` -- the answer comes **first**, the explanation
        # after. Taking the whole segment would treat the explanation as the answer; taking the
        # **last** `Answer:` (the original logic, written for "reasoning first, answer last" CoT)
        # gets fooled by mid-sentence markers inside the explanation -- the VeriTime explanation
        # template's "Step 6 Summarizing the thinking process to output the answer:" is exactly
        # what drove anomaly detection / scenario attribution / inferential calculation down to
        # 2.78/1.14/0.95 (the generations were in fact fully correct).
        s = line
    elif len(parts) > 1:
        s = parts[-1].split("\n")[0]      # no line-initial marker: fall back to the original logic (keeps historical numbers unchanged)
    elif len(text) > _BARE_ANSWER_MAX_CHARS:
        return ""
    else:
        s = text          # bare answer: still goes through the same normalisation below (strip special tokens / trailing dots / lowercase)
    s = re.sub(r"<\|.*?\|>|<eos>$", "", s).strip()
    return re.sub(r"[.\s]+$", "", s).strip().lower()


def accuracy(rows):
    if not rows:
        return float("nan")
    return 100.0 * sum(extract(r["ground_truth"]) == extract(r["generated_text"]) for r in rows) / len(rows)


def weighted_f1(rows, keyfn):
    """Support-weighted F1 (the Time-RA paper's metric). ``keyfn`` maps a sample to a class label."""
    tp, fp, fn, sup = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        g, p = keyfn(extract(r["ground_truth"])), keyfn(extract(r["generated_text"]))
        sup[g] += 1
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1
    total = sum(sup.values())
    if not total:
        return float("nan")
    acc = 0.0
    for c, n in sup.items():
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc += f1 * n
    return acc / total


def load(infer_dir, name):
    p = os.path.join(infer_dir, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def by_task(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("task")].append(r)
    return d


# --------------------- baselines (transcribed from the respective papers) ---------------------
# Each entry = (description of the comparison target, value). **Only protocol-matched targets go
# here**; targets with protocol doubts are explained in the NOTE and excluded from the
# "beats SOTA" judgement.
BASELINES = {
    "ecg":        ("OpenTSLM best Flamingo/Llama3.2-3B", 46.25),
    # har/sleep were moved out of the comparable list after checking the official loader source:
    # the official HARCoTQADataset/SleepEDFCoTQADataset build the prompt at runtime and list
    # **all classes** as candidates (HAR 8 classes / Sleep 6 classes, chance 12.5%/16.7%); the
    # paper's Table 2 was measured in that setting. The CSV `prompt` field passed through as our
    # input_text is the **two-way** prompt from the CoT annotation stage (chance 50%) and is never
    # used by the official train/eval => different candidate space, neither Acc nor F1 is directly
    # comparable, so they live in REFERENCE_ONLY.
    # (ECG has no such issue and the direction is reversed: the official prompt carries a
    # per-template candidate list, ours is open-ended; our setting is stricter, Acc is comparable.)
    "st_t1":      ("STReasoner-8B (SOTA)", 95.65),
    "st_t2":      ("STReasoner-8B (SOTA)", 75.71),
    "st_t3":      ("STReasoner-8B (SOTA)", 87.12),
    "vt_anomaly": ("VeriTime-4B SFT+RL (SOTA)", 91.11),
    "vt_attr":    ("VeriTime-4B SFT+RL (SOTA)", 87.50),
    "vt_infer":   ("VeriTime-4B SFT+RL (SOTA)", 77.14),
    "vt_ctu":     ("VeriTime-4B (SOTA)", 67.50),
    "vt_ecg":     ("VeriTime-4B (SOTA)", 30.30),
    "vt_emg":     ("VeriTime-3B (SOTA)", 64.96),
    "vt_rcw":     ("VeriTime-3B (SOTA)", 64.89),
}
# Targets with a different protocol -- order-of-magnitude reference only
REFERENCE_ONLY = {
    "har":            ("OpenTSLM best SoftPrompt/Llama3.2-1B (all-class listing setting, not our two-way prompt)", 71.48),
    "sleep":          ("OpenTSLM best SoftPrompt/Llama3.2-1B (all-class listing setting, not our two-way prompt)", 81.08),
    "ra_uni_label":   ("Time-RA uni best Label-F1: Qwen2.5-3B zero-shot", 0.9000),
    "ra_uni_action":  ("Time-RA uni best Action-F1: Llama-3-8B SFT", 0.1511),
    "ra_multi_label": ("Time-RA multi best Label-F1: Qwen2.5-7B SFT", 0.8544),
    "ra_multi_action": ("Time-RA multi best Action-F1: Phi-4-mini SFT", 0.4372),
    "hitsr_l2":       ("LLaTiSA L2-Local (OOD sample, different questions)", 75.6),
    "hitsr_l3":       ("LLaTiSA L3 (OOD sample, different questions)", 67.0),
}


def fmt(v, nd=2):
    return "—" if v != v else f"{v:.{nd}f}"


def delta(ours, base, nd=2):
    if ours != ours:
        return "—"
    d = ours - base
    return f"{'+' if d >= 0 else ''}{d:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("infer_dir")
    ap.add_argument("--out", default=None, help="markdown output path (default: print only)")
    args = ap.parse_args(argv)
    D = args.infer_dir
    L = []

    def w(s=""):
        L.append(s)

    # --- OpenTSLM three subsets: reuse the official-protocol scorer ---
    from chronos_llm.eval.eval_opentslm_answer import detect_protocol, evaluate as ots_eval
    w("## 1. OpenTSLM subsets (official-protocol scoring; **only ECG is directly comparable**, HAR/Sleep use a different setting †)")
    w()
    w("| Subset | N | Ours Acc | Paper best Acc | Δ | Ours F1 | Paper best F1 |")
    w("|---|---|---|---|---|---|---|")
    for key, fname, f1_base in [("ecg", "test.jsonl", 40.25),
                                ("har", "har_test.jsonl", 65.44),
                                ("sleep", "sleep_test.jsonl", 69.88)]:
        rr = load(D, fname)
        if not rr:
            continue
        m = ots_eval(rr, protocol=detect_protocol(rr, fname))
        acc = 100.0 * m["accuracy"]                 # evaluate() returns a fraction in 0~1
        f1 = 100.0 * m["macro_f1_global"]
        if key == "ecg":
            who, base = BASELINES[key]
            w(f"| ECG | {m['n']} | **{fmt(acc)}** | {fmt(base)} | **{delta(acc, base)}** | "
              f"{fmt(f1)} | not comparable |")
        else:
            who, base = REFERENCE_ONLY[key]
            w(f"| {key.upper()} † | {m['n']} | **{fmt(acc)}** | {fmt(base)} † | not comparable † | "
              f"{fmt(f1)} | {fmt(f1_base)} † |")
    w()
    w("> **ECG is comparable and our setting is stricter**: accuracy follows the official protocol "
      "verbatim; moreover the official loader splices the template's candidate answer list into the "
      "prompt (`get_possible_answers_for_template`), while our input_text only has the clinical "
      "context + question with **no candidates** => we obtain this score in the harder open-ended "
      "setting. ECG F1 is computed globally and **cannot** be placed next to the paper's (the "
      "official one is per-template; our HF mirror lacks `template_id`).")
    w(">")
    w("> † **HAR/Sleep are not directly comparable (the task input's candidate space differs)**: the "
      "official `HARCoTQADataset`/`SleepEDFCoTQADataset` build the prompt at runtime and list **all "
      "classes** as candidates (HAR 8 classes, Sleep 6 classes, chance 12.5%/16.7%); the paper's "
      "Table 2 numbers were measured in that setting. Our input_text passes through the CSV `prompt` "
      "field -- the two-way prompt from the **CoT annotation stage** (chance 50%), which the official "
      "train/eval never uses. Different candidate spaces => Acc/F1 are reference only and excluded "
      "from win/loss counts.")
    w()

    # --- ST-Bench ---
    rows = load(D, "stbench_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 2. ST-Bench / STReasoner (ST-Test; counts match the paper's Table 7 exactly => directly comparable)")
        w()
        w("| Subtask | N | Ours Acc | STReasoner-8B (SOTA) | Δ | Qwen3-8B+SFT | ChatTS-8B | GPT-5.2(Text) |")
        w("|---|---|---|---|---|---|---|---|")
        for tk, label, key, sft, chatts, gpt in [
                ("stbench_etiological", "T1 etiological", "st_t1", 82.13, 56.52, 83.09),
                ("stbench_entity", "T2 entity", "st_t2", 62.65, 19.51, 38.78),
                ("stbench_correlation", "T3 correlation/causality", "st_t3", 79.84, 41.08, 58.79)]:
            a = accuracy(t.get(tk, []))
            who, base = BASELINES[key]
            w(f"| {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {fmt(base)} | **{delta(a, base)}** | "
              f"{fmt(sft)} | {fmt(chatts)} | {fmt(gpt)} |")
        w()

    # --- VeriTime ---
    rows = load(D, "veritime_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 3. VeriTime / TSRBench (full official released test, verbatim questions; the paper's Table 9 counts are from its paper-version test => must be stated)")
        w()
        w("| Family | Subtask | N | Ours Acc | VeriTime SOTA | Δ | ChatTS(SFT) | GPT-4o-mini |")
        w("|---|---|---|---|---|---|---|---|")
        for tk, fam, label, key, chatts, gpt in [
                ("veritime_Anomaly_detection", "Scenario", "Anomaly detection", "vt_anomaly", 89.44, 82.12),
                ("veritime_Scenario_attribution", "Scenario", "Scenario attribution", "vt_attr", 80.68, 61.36),
                ("veritime_Inferential_calculation", "Scenario", "Inferential calculation", "vt_infer", 72.38, 65.71)]:
            a = accuracy(t.get(tk, []))
            who, base = BASELINES[key]
            w(f"| {fam} | {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {fmt(base)} | "
              f"**{delta(a, base)}** | {fmt(chatts)} | {fmt(gpt)} |")
        for tk, label, key, cls in [("veritime_CTU", "CTU", "vt_ctu", 67.20),
                                    ("veritime_ECG", "ECG", "vt_ecg", 28.39),
                                    ("veritime_EMG", "EMG", "vt_emg", 73.33),
                                    ("veritime_RCW", "RCW", "vt_rcw", 62.28)]:
            a = accuracy(t.get(tk, []))
            who, base = BASELINES[key]
            w(f"| Knowledge | {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {fmt(base)} | "
              f"**{delta(a, base)}** | classical best {fmt(cls)} | — |")
        w()
        w("> The Knowledge family's \"classical best\" is the best per-task fully supervised classical "
          "model in the paper's Table 2. ECG at 15-30% across the board (4 arrhythmia classes) is a "
          "well-known near-chance task.")
        w()

    # --- Time-RA ---
    rows = load(D, "time_ra_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 4. Time-RA / RATs40K (different protocol, order-of-magnitude reference only)")
        w()
        w("| Subset | N | Ours Label-F1 | Paper best Label-F1 | Ours Action-F1 | Paper best Action-F1 |")
        w("|---|---|---|---|---|---|")
        norm = lambda s: "normal" if "normal" in s else "anomaly"
        for tk, label, lk, ak in [("time_ra_uni", "Uni", "ra_uni_label", "ra_uni_action"),
                                  ("time_ra_multi", "Multi", "ra_multi_label", "ra_multi_action")]:
            rr = t.get(tk, [])
            lf = weighted_f1(rr, norm)
            af = weighted_f1(rr, lambda s: s)
            w(f"| {label} | {len(rr)} | **{fmt(lf, 4)}** | {REFERENCE_ONLY[lk][1]:.4f} | "
              f"**{fmt(af, 4)}** | {REFERENCE_ONLY[ak][1]:.4f} |")
        w()
        w("> The paper's Table 2 protocol is a **text-only LLM + 14-shot all-class exemplars + CoT**; "
          "ours is an encoder route, zero-shot. **The Label-F1 ceiling of ~0.90 is an artefact of the "
          "83.7% anomaly skew (predicting anomaly everywhere already scores high).**")
        w(">")
        w("> **The large lead in the Action-F1 column must be discounted and cannot be read as a clean "
          "SOTA improvement**: we removed the 62.8% mislabeled samples of the official Uni test "
          "(univariate series labeled with multivariate-only types), **whereas the paper's 0.1511 was "
          "computed on the full test including those unanswerable samples** -- this alone explains a "
          "large part of the gap (62.8% of the denominator is unanswerable for any model). To obtain a "
          "comparable number, rerun the conversion with `--keep_mismatched` and re-evaluate on the "
          "**original full test**.")
        w()

    # --- HiTSR ---
    rows = load(D, "hitsr_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 5. HiTSR / LLaTiSA (we evaluate the in-domain test split; the paper's Table 1 is an OOD sample => different questions)")
        w()
        w("| Level | N | Ours Acc | LLaTiSA(OOD) | ChatTS(OOD) | GPT-4o Text(OOD) |")
        w("|---|---|---|---|---|---|")
        for tk, label, key, chatts, gpt in [("hitsr_l2", "L2", "hitsr_l2", 57.0, 47.6),
                                            ("hitsr_l3", "L3", "hitsr_l3", 59.0, 43.0)]:
            a = accuracy(t.get(tk, []))
            w(f"| {label} | {len(t.get(tk, []))} | **{fmt(a)}** | {REFERENCE_ONLY[key][1]:.1f} | "
              f"{fmt(chatts, 1)} | {fmt(gpt, 1)} |")
        w()
        w("> **These columns are not evidence that \"we win\"** -- the paper's columns are OOD-sampled "
          "questions (BEDTime/MMTS-Bench/MCQ2, sample indices unreleased), while we evaluate HiTSR's "
          "own test. They only show our level on the in-distribution data.")
        w()

    # --- TelecomTS ---
    rows = load(D, "telecomts_test.jsonl")
    if rows:
        t = by_task(rows)
        w("## 6. TelecomTS (different granularity, not directly comparable; internal comparison only)")
        w()
        w("| Subtask | N | Ours Acc | # distinct GT answers |")
        w("|---|---|---|---|")
        for tk, label in [("telecomts_network", "network QA"), ("telecomts_anomalies", "anomalies QA")]:
            rr = t.get(tk, [])
            n_gt = len({extract(r["ground_truth"]) for r in rr})
            flag = "  **degenerate**" if n_gt <= 1 else ""
            w(f"| {label} | {len(rr)} | **{fmt(accuracy(rr))}**{flag} | {n_gt} |")
        w()
        w("> **The accuracy in the anomalies row is meaningless**: the test GT is **all \"No\"** "
          "(1 distinct answer), so answering no everywhere scores 100%. Root cause: "
          "the TelecomTS conversion takes the **last 5%** of rows as test, but the TelecomTS parquet "
          "row order is not random -- anomalous scenarios cluster at the front and the trailing 5% "
          "happen to be all normal windows. **The training data itself is fine** (the 32,585 train "
          "anomalies rows have 317 distinct answers, 89.5% no); only this one metric is affected. The "
          "fix is an evenly spaced row split (still row-based, no temporal leakage), but **train/test "
          "must be re-split together** -- swapping only the test would overlap with the train set of "
          "already-trained models and leak, so it is deferred to the next training round.")
        w()
        w("> The paper's Tables 2/3/4/6 report per-subtask detection F1 / duration F1 / root-cause Acc "
          "(11 classes) / QA split into Trend·Traffic·Mobility·Congestion·Location; we expand "
          "`QnA.{network,anomalies}` wholesale into QA and score exact match => different granularity. "
          "Reference magnitudes: the paper's Toto+Qwen3-4B has detection F1 0.487, root-cause Acc "
          "0.826, QA Trend 0.670 / Traffic 0.988 / Location 0.400.")
        w()

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n-> wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
