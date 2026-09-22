r"""Compare the **hit rate of the model-generated `[Magnitude: X]` tag** under different magnitude conventions.

Why it is needed: the ground truth in `eval_conclusion_accuracy.py` is hard-coded to the **old
convention** (this row's future peak / the q90 of the future peaks over the whole dataset). After the
tag was changed to the per-sample "ROI peak / own history peak", the two runs have different
"ground truth" definitions and the hit rate must be computed **for each run under the convention of
its own training data** -- only that answers the core hypothesis the new convention is meant to test:
**once the tag is self-contained within the sample, does the model infer it correctly more often**.

The ground truth is taken directly from the conclusion prefix (`[Magnitude: X]`) of the respective
parquet -- i.e. the very supervision signal the model saw in training, not recomputed, to avoid
convention drift. The prediction is the same tag inside `gen_text` of `forecast_preds.jsonl`. The
jsonl row order == the test split row order (infer merges in the original row order).

  python chronos_llm/scripts/utils/eval_magnitude_hit.py \
      new=<run>/mid_eval/epoch_19/forecast_preds.jsonl:<new parquet> \
      old=<run>/mid_eval/epoch_19/forecast_preds.jsonl:<old parquet>
"""
import argparse
import json
import re
from collections import Counter

import pandas as pd

_TAG = re.compile(r"\[Magnitude:\s*([A-Z_]+)\]")
ORDER = ["TINY", "WELL_BELOW", "MOD_BELOW", "TYPICAL", "ABOVE"]


def load(jsonl_path, parquet_path):
    df = pd.read_parquet(parquet_path, columns=["conclusion", "split", "dataset_name"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    gt = test["conclusion"].astype(str).str.extract(_TAG.pattern)[0]
    gen = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            m = _TAG.search(row.get("gen_text") or "")
            gen.append(m.group(1) if m else None)
    if len(gen) != len(test):
        raise SystemExit(f"row count mismatch: jsonl {len(gen)} vs test split {len(test)}")
    return gt, pd.Series(gen), test["dataset_name"]


def report(tag, gt, gen, ds):
    ok = gt.notna()
    n = int(ok.sum())
    hit = int((gt[ok] == gen[ok]).sum())
    miss_tag = int(gen[ok].isna().sum())            # the model wrote no tag
    # adjacent band (off by one level) counts as "near"
    idx = {b: i for i, b in enumerate(ORDER)}
    near = sum(1 for a, b in zip(gt[ok], gen[ok])
               if b in idx and a in idx and abs(idx[a] - idx[b]) == 1)
    print(f"\n### {tag} ({n} decidable samples)")
    print(f"- **exact hit {100.0 * hit / n:.1f}%** ({hit}/{n}); adjacent band {100.0 * near / n:.1f}%;"
          f" no tag written {100.0 * miss_tag / n:.1f}%")
    print(f"- ground-truth distribution {dict(Counter(gt[ok]))}")
    print(f"- generated distribution {dict(Counter(x for x in gen[ok] if x))}")
    # per-band recall, to see whether it collapses onto the majority band
    print("- per-band recall: ", end="")
    for b in ORDER:
        sel = ok & (gt == b)
        if int(sel.sum()) == 0:
            continue
        print(f"{b} {100.0 * int((gen[sel] == b).sum()) / int(sel.sum()):.0f}%({int(sel.sum())}) ", end="")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=preds.jsonl:parquet")
    args = ap.parse_args()
    for spec in args.runs:
        name, _, rest = spec.partition("=")
        jsonl, _, parquet = rest.rpartition(":")
        gt, gen, ds = load(jsonl, parquet)
        report(name, gt, gen, ds)


if __name__ == "__main__":
    main()
