#!/usr/bin/env python3
"""Backbone ablation, stage two (domain-focused SFT): per-epoch nine-discipline quick-screen curve
for the three arms (9B LLM / 4B LLM / TimesFM TSFM).

    python chronos_llm/scripts/utils/curve_dom12_backbone.py

Same convention as the released checkpoint's selection point: the equal-weight mean over the 9
disciplines produced by `summarize_agg6_by_domain.py --exclude-domains Physiology` (cross-checked:
the ep18 checkpoints of the 9B / 4B arms reproduce 80.93 / 79.16 exactly); Physiology is kept as a
separate sentinel column.
The three arms are aligned by **epoch, not by step**: with dynamic batching the TimesFM arm needs
34,060 steps in total against 35,020 for the two Chronos-2 arms, so their checkpoint step numbers
never line up. Only quick screens that are fully merged (5 jsonl files, no per-rank shards) are
printed.
"""
import os
import subprocess
import sys

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EVAL = f"{R}/outputs/eval"
SUMMARIZE = f"{R}/chronos_llm/scripts/utils/summarize_agg6_by_domain.py"
# Prefixes of the quick-screen output directories; "dom12" denotes the 12-discipline domain pool
ARMS = {"9B": "v9dom12k", "4B": "bbdom12q4b", "TFM": "bbdom12tfm"}


def steps(prefix):
    return sorted(int(d.split("step")[1]) for d in os.listdir(EVAL)
                  if d.startswith(f"{prefix}_vt_step") and d.split("step")[1].isdigit())


def done(d):
    fs = os.listdir(d) if os.path.isdir(d) else []
    return sum(f.endswith(".jsonl") and "rank" not in f for f in fs) == 5 and not any("rank" in f for f in fs)


def row(out, head, n):
    """First n score columns of the table row starting with `head` (columns 0-2 are the discipline
    name, its description and n, so the scores start at column 3)."""
    for line in out.splitlines():
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].strip("*").startswith(head):
            return cells[3:3 + n]
    raise ValueError(f"row {head} not found")


def main():
    arm_steps = {a: steps(p) for a, p in ARMS.items()}
    print("| ep | 9B 9-dom | 4B 9-dom | TFM 9-dom | TFM-9B | Physiology 9B / 4B / TFM |")
    print("|---|---|---|---|---|---|")
    for ep in range(1, 21):
        runs = []
        for a, p in ARMS.items():
            if ep <= len(arm_steps[a]):
                d = f"{EVAL}/{p}_vt_step{arm_steps[a][ep - 1]}/infer"
                if done(d):
                    runs.append((a, d))
        if not any(a == "TFM" for a, _ in runs):
            continue
        args = [f"{a}={d}" for a, d in runs]
        env = dict(os.environ, PYTHONPATH=R)
        o9 = subprocess.run([sys.executable, SUMMARIZE, "--exclude-domains", "Physiology", *args],
                            capture_output=True, text=True, cwd=R, env=env, check=True).stdout
        o12 = subprocess.run([sys.executable, SUMMARIZE, *args],
                             capture_output=True, text=True, cwd=R, env=env, check=True).stdout
        labels = [a for a, _ in runs]
        v = dict(zip(labels, row(o9, "Avg", len(labels))))
        ph = dict(zip(labels, row(o12, "Physiology", len(labels))))
        diff = f"{float(v['TFM']) - float(v['9B']):+.2f}" if "9B" in v else ""
        print(f"| {ep} | {v.get('9B', '')} | {v.get('4B', '')} | {v['TFM']} | {diff} | "
              f"{ph.get('9B', '')} / {ph.get('4B', '')} / {ph['TFM']} |")


if __name__ == "__main__":
    main()
