r"""Per-epoch trend table for the six-source MMTR understanding pool ("agg6"): compare the
independent evaluation results of several epochs side by side under one native protocol.

`summarize_agg6_eval.py` answers "how does one checkpoint compare with each baseline"; this script
answers **"which epoch is best, and which subtasks degrade as training goes on"** -- the basis for
picking the final checkpoint and for diagnosing "small datasets get swamped" (qualitatively,
VeriTime was seen to degrade with epochs; the per-epoch numbers here pin that down).

The metric protocol is **reused item by item** from `summarize_agg6_eval` rather than
re-implemented: the three OpenTSLM subsets go through the official protocol script
(`eval_opentslm_answer.evaluate`), ST-Bench/VeriTime/HiTSR/TelecomTS use exact-match accuracy,
Time-RA uses support-weighted F1 (the paper's protocol). **Protocol drift would flip the
conclusions of a trend table**, hence no copy-pasted implementation.

Usage:
    python chronos_llm/scripts/utils/compare_agg6_epochs.py \
        epoch1=outputs/eval/agg6_epoch1/infer epoch2=outputs/eval/agg6_epoch2/infer \
        --out outputs/eval/agg6_epoch_trend.md

Missing epochs (not evaluated yet) are skipped automatically without affecting the existing
columns -- look at each one as it finishes, no need to wait for all.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from chronos_llm.scripts.utils.summarize_agg6_eval import (  # noqa: E402
    BASELINES, accuracy, by_task, fmt, load, weighted_f1,
)

# (display name, file name, subtask key or None=whole file, metric kind, baseline key or None)
# The order is the table row order: clustered by dataset so "which dataset degrades as a whole"
# is easy to see.
ROWS = [
    ("ECG (OpenTSLM)",      "test.jsonl",           None,                            "ots",  "ecg"),
    ("HAR (OpenTSLM)",      "har_test.jsonl",       None,                            "ots",  "har"),
    ("Sleep (OpenTSLM)",    "sleep_test.jsonl",     None,                            "ots",  "sleep"),
    ("ST-Bench T1 etiology",    "stbench_test.jsonl",   "stbench_etiological",           "acc",  "st_t1"),
    ("ST-Bench T2 entity",    "stbench_test.jsonl",   "stbench_entity",                "acc",  "st_t2"),
    ("ST-Bench T3 correlation",    "stbench_test.jsonl",   "stbench_correlation",           "acc",  "st_t3"),
    ("VeriTime anomaly detection",   "veritime_test.jsonl",  "veritime_Anomaly_detection",    "acc",  "vt_anomaly"),
    ("VeriTime scenario attribution",   "veritime_test.jsonl",  "veritime_Scenario_attribution", "acc",  "vt_attr"),
    ("VeriTime inferential calculation",   "veritime_test.jsonl",  "veritime_Inferential_calculation", "acc", "vt_infer"),
    ("VeriTime CTU",        "veritime_test.jsonl",  "veritime_CTU",                  "acc",  "vt_ctu"),
    ("VeriTime ECG",        "veritime_test.jsonl",  "veritime_ECG",                  "acc",  "vt_ecg"),
    ("VeriTime EMG",        "veritime_test.jsonl",  "veritime_EMG",                  "acc",  "vt_emg"),
    ("VeriTime RCW",        "veritime_test.jsonl",  "veritime_RCW",                  "acc",  "vt_rcw"),
    # HiTSR tasks are **lower-case** hitsr_l2/l3 (as written by the conversion script); upper case
    # would silently match nothing and the whole row would disappear
    ("HiTSR L2",            "hitsr_test.jsonl",     "hitsr_l2",                      "acc",  None),
    ("HiTSR L3",            "hitsr_test.jsonl",     "hitsr_l3",                      "acc",  None),
    ("TelecomTS network",   "telecomts_test.jsonl", "telecomts_network",             "acc",  None),
    ("TelecomTS anomalies", "telecomts_test.jsonl", "telecomts_anomalies",           "acc",  None),
    # Time-RA is **split into uni/multi** as in the paper (a merged number would land somewhere in
    # between and match neither)
    ("Time-RA uni Label-F1",    "time_ra_test.jsonl", "time_ra_uni",   "raL", None),
    ("Time-RA uni Action-F1",   "time_ra_test.jsonl", "time_ra_uni",   "raA", None),
    ("Time-RA multi Label-F1",  "time_ra_test.jsonl", "time_ra_multi", "raL", None),
    ("Time-RA multi Action-F1", "time_ra_test.jsonl", "time_ra_multi", "raA", None),
]

NAN = float("nan")


def _load_overlaid(infer_dir, fname):
    """Support **directory overlay** `dirA+dirB`: for a file present in several directories the
    **later** directory wins; the remaining files still come from the earlier ones.

    The use case is "re-evaluate one dataset and overwrite it in place" without redoing the whole
    round. Example: the CoT arm's VeriTime outputs were truncated at `MAX_NEW_TOKENS=2048` (CTU
    had up to 72.2% missing answers); after re-evaluating VeriTime alone with 4096 into a second
    directory, passing `ep3=<full_infer_dir>+<veritime_rerun_infer_dir>` makes that one file use
    the re-run while the other 7 datasets stay as they were.
    """
    for d in reversed(infer_dir.split("+")):
        rows = load(d, fname)
        if rows:
            return rows
    return []


def _metric(infer_dir, fname, task, kind, cache):
    """Compute one cell. cache stores loaded files by (dir,fname) so the same jsonl is not read
    repeatedly (ECG has 40k rows)."""
    key = (infer_dir, fname)
    if key not in cache:
        cache[key] = _load_overlaid(infer_dir, fname)
    rows = cache[key]
    if not rows:
        return NAN, 0
    if task is not None:
        rows = by_task(rows).get(task, [])
        if not rows:
            return NAN, 0
    if kind == "ots":
        from chronos_llm.eval.eval_opentslm_answer import detect_protocol, evaluate as ots_eval
        m = ots_eval(rows, protocol=detect_protocol(rows, fname))
        return 100.0 * m["accuracy"], m["n"]
    if kind == "acc":
        return accuracy(rows), len(rows)
    if kind == "raL":   # Label-F1: any anomaly collapses to "anomaly" (binary)
        return 100.0 * weighted_f1(rows, lambda s: "normal" if "normal" in s else "anomaly"), len(rows)
    if kind == "raA":   # Action-F1: fine-grained raw answers
        return 100.0 * weighted_f1(rows, lambda s: s), len(rows)
    raise ValueError(f"unknown metric kind {kind}")


def summary_table(specs, values, counts):
    """Side-by-side comparison under three aggregate criteria -- **do not pick the final
    checkpoint from just one of them**.

    Typically no single epoch is best everywhere (large datasets peak early while small ones are
    still improving), so "the best epoch" is necessarily a weighted judgement; each criterion has
    its own bias and they must be read together:

    - **Equal-weight mean**: the 21 subtasks weigh the same. Biased towards small test sets
      (VeriTime CTU with 144 rows counts as much as OpenTSLM-ECG with 41,093), so single-sample
      jitter is amplified.
    - **Count-weighted mean**: weighted by test-set size. More stable, but dominated by ECG alone
      (58% of the samples), effectively "the ECG curve".
    - **Number of subtasks where it is best**: rank only, ignoring magnitude; immune to metric
      scale differences, low resolution when there are many ties.

    Only finite cells participate; when an epoch lacks a row, that row is excluded from its mean
    (the denominator shrinks accordingly).
    """
    labels = [lab for lab, _ in specs]
    rows = [(f"{k} ({counts.get(k, 0)})", v, counts.get(k, 0)) for k, v in values.items()]
    out = ["## Aggregate criteria side by side (for checkpoint selection; read all three together)", "",
           "| Criterion | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]

    def _row(name, fn):
        vals = [fn(i) for i in range(len(labels))]
        finite = [(i, v) for i, v in enumerate(vals) if v == v]
        best = max(finite, key=lambda iv: iv[1])[0] if finite else -1
        cells = [(f"**{v:.2f}**" if i == best else f"{v:.2f}") if v == v else "—"
                 for i, v in enumerate(vals)]
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    def _mean(i, weighted):
        num = den = 0.0
        for _, vs, n in rows:
            v = vs[i]
            if v != v:
                continue
            w = n if weighted else 1.0
            num += v * w
            den += w
        return num / den if den else NAN

    _row("Equal-weight mean", lambda i: _mean(i, False))
    _row("Count-weighted mean", lambda i: _mean(i, True))
    _row("Number of subtasks where best", lambda i: float(sum(
        1 for _, vs, _ in rows
        if [x for x in vs if x == x] and vs[i] == max(x for x in vs if x == x))))
    out.append("")
    return out


def build_table(specs):
    """specs = [(label, infer_dir), ...] -> (markdown lines, per-row value dict, per-row count dict)."""
    cache = {}
    labels = [lab for lab, _ in specs]
    out, values, counts = [], {}, {}

    head = "| Subtask | N | " + " | ".join(labels) + " | Best | Trend |"
    out.append(head)
    out.append("|" + "---|" * (len(labels) + 4))

    for name, fname, task, kind, base_key in ROWS:
        vals, n_show = [], 0
        for _, d in specs:
            v, n = _metric(d, fname, task, kind, cache)
            vals.append(v)
            n_show = max(n_show, n)
        values[name] = vals
        counts[name] = n_show
        finite = [(i, v) for i, v in enumerate(vals) if v == v]
        if not finite:
            # All-nan has two possible causes: the file does not exist at all (that epoch is not
            # evaluated yet, normal); or **the file exists but the task name is wrong** -- the
            # latter makes the whole row vanish silently and can only be caught by eye (this
            # happened once with an upper-case hitsr_L2 that matched nothing). So warn here
            # explicitly and list the task names actually present in the file.
            if task is not None:
                for _, d in specs:
                    rr = cache.get((d, fname)) or []
                    if rr:
                        real = sorted({r.get("task") for r in rr})
                        print(f"[warn] '{name}' is entirely empty: {fname} has no task={task!r}, "
                              f"actual tasks are {real} -- the task name in ROWS may be wrong", file=sys.stderr)
                        break
            continue
        best_i = max(finite, key=lambda iv: iv[1])[0]
        cells = []
        for i, v in enumerate(vals):
            s = fmt(v)
            cells.append(f"**{s}**" if i == best_i and len(finite) > 1 else s)
        # Trend = last - first (only when both exist); reveals subtasks that degrade with training
        if len(finite) > 1:
            d = finite[-1][1] - finite[0][1]
            trend = f"{'+' if d >= 0 else ''}{d:.2f}"
            if d <= -2.0:
                trend += " ⚠️degraded"
        else:
            trend = "—"
        best_label = labels[best_i] if len(finite) > 1 else "—"
        out.append(f"| {name} | {n_show} | " + " | ".join(cells) + f" | {best_label} | {trend} |")
    return out, values, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+",
                    help="of the form epoch1=outputs/eval/agg6_epoch1/infer; directories can be "
                         "overlaid with `+` (for a file present in several, the later one wins; "
                         "used to overwrite a single re-evaluated dataset)")
    ap.add_argument("--out", default=None, help="markdown output path (default: print only)")
    args = ap.parse_args(argv)

    specs = []
    for s in args.specs:
        if "=" not in s:
            ap.error(f"argument must have the form label=dir: {s}")
        lab, d = s.split("=", 1)
        # With the overlay syntax, continue as long as **at least one** layer exists (the re-run
        # directory may not be produced yet, in which case fall back to the original one)
        parts = [p for p in d.split("+") if os.path.isdir(p)]
        if not parts:
            print(f"[skip] {lab}: directory does not exist {d} (epoch not evaluated yet?)", file=sys.stderr)
            continue
        if len(parts) < len(d.split("+")):
            missing = [p for p in d.split("+") if not os.path.isdir(p)]
            print(f"[warn] {lab}: overlay layers missing {missing}, using only {parts}", file=sys.stderr)
        specs.append((lab, "+".join(parts)))
    if not specs:
        ap.error("no usable evaluation directory")

    lines = ["# Per-epoch trend on the six-source MMTR understanding pool (native protocol per dataset)", ""]
    lines.append("The metric protocol is identical to `summarize_agg6_eval.py` (same functions). The **Best** column gives the")
    lines.append("epoch with the highest value; the **Trend** column = last − first, values ≤ −2.0 are flagged ⚠️degraded.")
    lines.append("")
    tbl, values, counts = build_table(specs)
    lines += tbl
    lines.append("")
    if len(specs) > 1:
        lines += summary_table(specs, values, counts)

    # Degradation list: direct evidence of small datasets being swamped by large ones (turns the
    # qualitative diagnosis into per-epoch numbers)
    degraded = []
    for name, vals in values.items():
        fin = [v for v in vals if v == v]
        if len(fin) > 1 and fin[-1] - fin[0] <= -2.0:
            degraded.append((name, fin[0], fin[-1], fin[-1] - fin[0]))
    if degraded:
        lines.append("## Subtasks that degrade with training (last − first ≤ −2.0)")
        lines.append("")
        lines.append("| Subtask | First epoch | Last epoch | Drop |")
        lines.append("|---|---|---|---|")
        for n, a, b, d in sorted(degraded, key=lambda x: x[3]):
            lines.append(f"| {n} | {fmt(a)} | {fmt(b)} | **{d:.2f}** |")
        lines.append("")

    # Wins against baselines (only the subset that can be compared directly against the tables)
    lines.append("## Wins per epoch against directly comparable baselines")
    lines.append("")
    lines.append("| epoch | wins/comparable | winning subtasks |")
    lines.append("|---|---|---|")
    for i, (lab, _) in enumerate(specs):
        win, tot, names = 0, 0, []
        for name, fname, task, kind, base_key in ROWS:
            if not base_key or base_key not in BASELINES:
                continue
            v = values.get(name, [NAN] * len(specs))[i]
            if v != v:
                continue
            tot += 1
            if v > BASELINES[base_key][1]:
                win += 1
                names.append(name)
        lines.append(f"| {lab} | {win}/{tot} | {', '.join(names) if names else '—'} |")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[compare] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
