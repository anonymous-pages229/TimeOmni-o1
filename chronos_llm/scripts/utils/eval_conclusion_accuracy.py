"""Programmatic accuracy evaluation of the generated reasoning/conclusion in the FULL setting
(the paper's "reasoning grounding" metric).

For the forecast_preds.jsonl of infer_forecast (with gen_text/median_pred/roi), check the textual
claims sample by sample:
1. **ROI localisation**: the ROI interval parsed from gen_text (absolute row indices,
   rl.reward.extract_roi) vs the true ROI (the jsonl `roi` relative interval + the parquet past_len
   restores absolute coordinates) -> mean IoU, IoU=1 exact rate, not-mentioned rate.
2. **Magnitude-band accuracy** (data ground-truth convention, same band edges as the corpus magnitude tags):
   true ratio = the day's future peak / the p90 of all future peaks of that dataset, 5 bands;
   gen_text keywords are mapped to a band -> band accuracy, direction bias (negative = underestimate),
   rate of missing magnitude sentence. Only samples whose ground truth is computable (typ>0) count.
3. **Band-match score of the text vs the ground-truth conclusion** (rl.reward.magnitude_match, the
   same convention as the GRPO reward).
4. **Self-consistency (faithfulness)**: the band of the model's own median forecast peak vs the band
   its conclusion claims -- tests whether the fed-back forecast follows its own textual claim (only
   counted when the generation has a magnitude sentence).

jsonl row order == test split row order (infer merges in original row order; an assert on equal
row counts is the safety net).

Usage:
  python eval_conclusion_accuracy.py NAME=preds.jsonl [NAME2=...] --parquet corpus.parquet \
      [--out-csv OUT] [--per-source]
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

from chronos_llm.rl.reward import extract_roi, magnitude_match, roi_iou

# Band edges and keywords aligned with the corpus magnitude bands
_BAND_EDGES = [0.10, 0.40, 0.70, 1.30, float("inf")]
_BAND_NAMES = ["tiny", "well_below", "mod_below", "typical", "above"]
_KEYWORDS = [
    ("tiny", ["small fraction", "tiny fraction", "small sliver", "near-flat"]),
    ("well_below", ["well below", "far under", "strongly suppressed"]),
    ("mod_below", ["moderately below", "somewhat under"]),
    ("typical", ["close to the series' typical", "roughly its usual peak", "near the series' typical"]),
    ("above", ["well above", "distinctly beyond", "well past", "surges"]),
]
_BAND_IDX = {b: i for i, b in enumerate(_BAND_NAMES)}


def band_of(ratio):
    for hi, name in zip(_BAND_EDGES, _BAND_NAMES):
        if ratio < hi:
            return name
    return _BAND_NAMES[-1]


def band_in_text(text):
    t = str(text).lower()
    for name, kws in _KEYWORDS:
        if any(k in t for k in kws):
            return name
    return None


def _peak(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.nanmax(a)) if a.size else np.nan


def eval_run(jsonl, test, typ_map):
    rows = [json.loads(l) for l in open(jsonl)]
    assert len(rows) == len(test), f"{jsonl}: row count mismatch {len(rows)} vs {len(test)}"
    recs = []
    for r, (_, t) in zip(rows, test.iterrows()):
        gen = r.get("gen_text") or ""
        ds = str(t["dataset_name"])
        past = int(t["past_len"])
        # 1) ROI: ground truth = relative roi + past_len restored to absolute (same frame as the absolute row indices in the conclusion text)
        rel = r.get("roi")
        gt_roi = (past + int(rel[0]), past + int(rel[1])) if rel else None
        gen_roi = extract_roi(gen)
        iou = roi_iou(gen_roi, gt_roi) if gt_roi else np.nan
        # 2) magnitude band from the data ground truth
        typ = typ_map.get(ds, np.nan)
        ratio = _peak(t["future_values"]) / typ if (np.isfinite(typ) and typ > 1e-9) else np.nan
        true_b = band_of(ratio) if np.isfinite(ratio) else None
        gen_b = band_in_text(gen)
        # 4) self-consistency: band of the median forecast peak vs the claimed band
        pred_b = None
        if r.get("median_pred") and np.isfinite(typ) and typ > 1e-9:
            pk = _peak(r["median_pred"])
            pred_b = band_of(pk / typ) if np.isfinite(pk) else None
        recs.append(dict(
            ds=ds, iou=iou, roi_exact=(iou == 1.0), roi_missing=(gen_roi is None),
            true_b=true_b, gen_b=gen_b, pred_b=pred_b,
            mag_match_text=magnitude_match(gen, r.get("gt_conclusion", "")),
        ))
    return recs


def summarize(recs, label):
    n = len(recs)
    iou = np.array([r["iou"] for r in recs], dtype=float)
    out = {
        "n": n,
        "roi_iou_mean": np.nanmean(iou),
        "roi_exact_rate": np.mean([r["roi_exact"] for r in recs]),
        "roi_missing_rate": np.mean([r["roi_missing"] for r in recs]),
        "mag_match_text": np.mean([r["mag_match_text"] for r in recs]),
    }
    scored = [r for r in recs if r["true_b"] is not None]
    stated = [r for r in scored if r["gen_b"] is not None]
    out["mag_n_scored"] = len(scored)
    out["mag_stated_rate"] = len(stated) / len(scored) if scored else np.nan
    out["mag_acc(stated)"] = (np.mean([r["gen_b"] == r["true_b"] for r in stated])
                              if stated else np.nan)
    # Lenient convention: no magnitude mentioned -> treated as the typical band (consistent with the reward's "not mentioned = band 0")
    out["mag_acc(none=typical)"] = (np.mean([(r["gen_b"] or "typical") == r["true_b"] for r in scored])
                                    if scored else np.nan)
    out["mag_dir_bias"] = (np.mean([_BAND_IDX[r["gen_b"]] - _BAND_IDX[r["true_b"]] for r in stated])
                           if stated else np.nan)
    faith = [r for r in stated if r["pred_b"] is not None]
    out["faithfulness"] = (np.mean([r["pred_b"] == r["gen_b"] for r in faith]) if faith else np.nan)
    out["faith_n"] = len(faith)
    print(f"\n===== {label} (n={n}) =====")
    print(f"  ROI:  mean IoU={out['roi_iou_mean']:.3f}  exact-match rate={out['roi_exact_rate']:.3f}"
          f"  not-mentioned rate={out['roi_missing_rate']:.3f}")
    print(f"  MAG:  scorable samples={out['mag_n_scored']}  magnitude-sentence rate={out['mag_stated_rate']:.3f}"
          f"  band acc (stated)={out['mag_acc(stated)']:.3f}"
          f"  band acc (default=typical)={out['mag_acc(none=typical)']:.3f}"
          f"  direction bias={out['mag_dir_bias']:+.3f} (negative=underestimate)")
    print(f"  text band-match score (vs true conclusion, GRPO convention)={out['mag_match_text']:.3f}")
    print(f"  faithfulness (forecast-peak band == claimed band, n={out['faith_n']})={out['faithfulness']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=preds.jsonl")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--per-source", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    test = df[df["split"] == "test"].reset_index(drop=True)
    pk_all = df["future_values"].map(_peak)
    typ_map = pk_all.groupby(df["dataset_name"]).quantile(0.90).to_dict()

    all_rows = []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        recs = eval_run(path, test, typ_map)
        s = summarize(recs, name)
        all_rows.append({"run": name, "source": "ALL", **{k: v for k, v in s.items()}})
        if args.per_source:
            for ds in sorted(set(r["ds"] for r in recs)):
                sub = [r for r in recs if r["ds"] == ds]
                s = summarize(sub, f"{name} / {ds}")
                all_rows.append({"run": name, "source": ds, **{k: v for k, v in s.items()}})

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        keys = list(all_rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"\nwrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
