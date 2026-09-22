r"""Split the magnitude tag of the conclusion into a **checkable field** and compute a hit rate on two
axes, each with a shuffled null hypothesis.

Why it is needed: the two existing 1-5 point LLM judges on the forecasting side barely separate the
shuffled control from the real pairing (shape 3.331 vs 3.249, image-text 3.256 vs 3.082) -- the
problem is not how the pairs are formed but the **metric itself**: on a 5-point scale, "3 = right
overall direction but wrong details" is a very lenient middle grade, two random curves agree on the
overall direction about half the time, so the judge collapses around 3. Switching the criterion to
**discrete fields judged right/wrong one by one** removes the lenient middle grade and the random
baseline drops to that field's marginal collision rate.

This script only does the cheapest cell: the five-band `[Magnitude: X]` tag -- it **needs no
extraction model** (a regex reads it straight from gen_text) and the band rule is hard-coded
(`make_structured_conclusion.py`: ratio = peak / the p90 of the future peaks over all rows of the
dataset, band edges 0.10/0.40/0.70/1.30).

Two axes:

* **Support / faithfulness axis (no ground truth)**: the band written in the model's text vs the band
  recomputed with the same rule from **its own predicted median curve**. The reference side is pure
  arithmetic, no judge involved => no lenient middle grade to collapse into. This is exactly the
  question the multimodal judge in `forecast_visual_faithfulness.py` tries to ask.
* **Accuracy / consistency axis (against ground truth)**: the band written in the text vs the
  reference band in the parquet conclusion (= the existing `eval_magnitude_hit.py` protocol, computed
  here as well for comparison).

The shuffled null hypothesis **needs no rerun and no API calls**: once the field is discrete, the
expected hit rate of a random mismatch = sum_{i!=j} 1[w_i == p_j] / (n(n-1)), which can be computed
exactly (not a sampling estimate).

  python chronos_llm/scripts/utils/magnitude_faithfulness.py \
      --pred_dir <run>/mid_eval/epoch_19 \
      --parquet data/forecast/mmtr_forecast_corpus.parquet
"""
import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

# Verbatim copy of make_structured_conclusion.py (the band rule must come from the same source, otherwise the protocols drift)
BAND_ORDER = ["tiny", "well_below", "mod_below", "typical", "above"]
BAND_EDGES = [0.10, 0.40, 0.70, 1.30, float("inf")]
BAND_TAG = {"tiny": "TINY", "well_below": "WELL_BELOW", "mod_below": "MOD_BELOW",
            "typical": "TYPICAL", "above": "ABOVE"}
TAG_RE = re.compile(r"\[Magnitude:\s*([A-Z_]+)\]")


def band_of(ratio):
    for name, hi in zip(BAND_ORDER, BAND_EDGES):
        if ratio < hi:
            return name
    return BAND_ORDER[-1]


def _target_series(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        return np.asarray(v[0], dtype=float)
    return np.asarray(v, dtype=float)


def _peak(v):
    a = _target_series(v)
    a = a[np.isfinite(a)]
    return float(np.nanmax(a)) if a.size else np.nan


def dataset_typical(df):
    """Per-dataset "typical peak" = the p90 of the future peaks over all rows of that dataset (same source as the annotation script)."""
    peaks = df["future_values"].map(_peak)
    return peaks.groupby(df["dataset_name"]).quantile(0.90).to_dict()


def shuffled_hit_rate(written, predicted):
    """Expected hit rate of a random mismatch, exact closed form: sum_{i!=j} 1[w_i == p_j] / (n(n-1)).

    Equals (sum_b n_w[b]*n_p[b] - number of self-pair hits) / (n(n-1))."""
    n = len(written)
    if n < 2:
        return float("nan")
    cw, cp = Counter(written), Counter(predicted)
    total = sum(cw[b] * cp[b] for b in set(cw) | set(cp))
    self_hit = sum(1 for a, b in zip(written, predicted) if a == b)
    return (total - self_hit) / (n * (n - 1))


def report(tag, written, other, label_other):
    keep = [(w, o) for w, o in zip(written, other) if w is not None and o is not None]
    n = len(keep)
    if n == 0:
        print(f"\n### {tag}: no decidable samples")
        return
    w = [a for a, _ in keep]
    o = [b for _, b in keep]
    hit = sum(1 for a, b in keep if a == b) / n
    null = shuffled_hit_rate(w, o)
    idx = {BAND_TAG[b]: i for i, b in enumerate(BAND_ORDER)}
    near = sum(1 for a, b in keep
                if a in idx and b in idx and abs(idx[a] - idx[b]) <= 1) / n
    major = max(Counter(o).values()) / n
    print(f"\n### {tag} ({n} decidable samples)")
    print(f"- **real-pair hit rate {100 * hit:.1f}%** (within +-1 band {100 * near:.1f}%)")
    print(f"- **random-mismatch null {100 * null:.1f}%** (exact closed form, not sampled)"
          f"; always answering the majority band {100 * major:.1f}%")
    print(f"- **gap {100 * (hit - null):+.1f} percentage points**")
    print(f"- distribution of the written band {dict(Counter(w))}")
    print(f"- distribution of {label_other} {dict(Counter(o))}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="directory containing forecast_preds.npz/.jsonl")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--dedup_ids", default=None,
                    help="optional: keep only the test ids of this parquet (switch to the de-duplicated test protocol)")
    args = ap.parse_args(argv)

    npz = np.load(f"{args.pred_dir}/forecast_preds.npz", allow_pickle=True)
    med = npz["pred_quantiles"][:, int(np.argmin(np.abs(npz["quantile_levels"] - 0.5))), :]
    valid = npz["valid_mask"]
    ids = [str(x) for x in npz["ids"]]
    ds_names = [str(x) for x in npz["dataset_names"]]

    written = []
    with open(f"{args.pred_dir}/forecast_preds.jsonl") as f:
        for line in f:
            m = TAG_RE.search(json.loads(line).get("gen_text") or "")
            written.append(m.group(1) if m else None)
    if len(written) != len(ids):
        raise SystemExit(f"row count mismatch: jsonl {len(written)} vs npz {len(ids)}")

    df = pd.read_parquet(args.parquet)
    typical = dataset_typical(df)
    test = df[df["split"] == "test"].reset_index(drop=True)
    gt_tag = test["conclusion"].astype(str).str.extract(TAG_RE.pattern)[0].tolist()
    if len(gt_tag) != len(ids):
        print(f"WARNING: parquet test {len(gt_tag)} rows vs npz {len(ids)} rows -- aligning the ground-truth axis by id")
        by_id = dict(zip(test["id"].astype(str), gt_tag))
        gt_tag = [by_id.get(i) for i in ids]

    # The model's own predicted curve -> the band recomputed with the same rule (ground truth untouched)
    pred_band = []
    for i in range(len(ids)):
        v = med[i][valid[i]]
        v = v[np.isfinite(v)]
        t = typical.get(ds_names[i], np.nan)
        if v.size == 0 or not np.isfinite(t) or t <= 1e-9:
            pred_band.append(None)
            continue
        peak = float(v.max())
        pred_band.append(BAND_TAG[band_of(peak / t)] if peak > 1e-9 else BAND_TAG["tiny"])

    keep_ids = None
    if args.dedup_ids:
        d2 = pd.read_parquet(args.dedup_ids, columns=["id", "split"])
        keep_ids = set(d2[d2["split"] == "test"]["id"].astype(str))
        sel = [k for k, i in enumerate(ids) if i in keep_ids]
        print(f"dedup protocol: {len(ids)} -> {len(sel)} rows")
        written = [written[k] for k in sel]
        pred_band = [pred_band[k] for k in sel]
        gt_tag = [gt_tag[k] for k in sel]

    print(f"\n## Magnitude-field hit rate on two axes ({args.pred_dir})")
    report("Support/faithfulness axis: written band vs band recomputed from the model's own predicted curve (no ground truth)",
           written, pred_band, "the band recomputed from the model's own curve")
    report("Accuracy/consistency axis: written band vs ground-truth band",
           written, gt_tag, "the ground-truth band")


if __name__ == "__main__":
    main()
