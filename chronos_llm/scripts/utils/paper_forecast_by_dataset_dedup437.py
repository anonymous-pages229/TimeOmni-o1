r"""Forecasting-side **per-dataset (10 datasets)** metric table on the 437-row de-leaked test protocol -- for the paper
appendix "Full per-dataset results".

Same filtering/metric functions as `rescore_forecast_dedup.py` (which aggregates over the 5 domains), only at
`dataset_name` granularity. Within each dataset the metrics are computed per sample and then averaged (the same
convention as domain/Overall); a domain score pools all samples of that domain and is **not** the mean of its
dataset scores; Overall = equal-weight mean over the 5 domains.

By default it reads four npz files (all produced on the original 444-row test split; the script filters them to the
437 de-duplicated rows using the test ids of the de-leaked parquet):
  - Ours (gen)  = mid-training evaluation of the released TimeOmni-o1 forecasting checkpoint at its selected epoch
    (the Table 2 axis)
  - Ours (tf)   = the teacher-forced counterpart of the same run (already filtered)
  - Chronos-2 finetuned = mid-training evaluation of the chronos_only arm at its selected epoch (its per-domain CRPS
    matches the Table 2 row: Solar 0.378 / Load 0.037 / Traffic 0.138 / Finance 0.025 / Climate 0.065 / Overall 0.129)
  - Chronos-2 zero-shot = `outputs/baseline_forecast_zeroshot_corpus_v2/zeroshot_preds.npz`
    (this npz has no `dataset_names`; the id->dataset_name map from the parquet fills it in)
The run directories come from the HEADLINE_RUN_DIR / CHRONOS_FT_RUN_DIR environment variables, or pass
label=npz pairs explicitly.

    python chronos_llm/scripts/utils/paper_forecast_by_dataset_dedup437.py \
        [--out-csv outputs/eval/forecast_dedup437_by_dataset.csv]

Pure numpy/pyarrow; no model is loaded.
"""
import argparse
import csv
import os
import sys

import numpy as np
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from chronos_llm.eval.metrics import forecast_metrics  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "chronos_llm/scripts/utils"))
from paper_forecast_by_domain import DOMAIN_SOURCES  # noqa: E402

DEDUP = os.environ.get("FORECAST_PARQUET", "data/forecast/mmtr_forecast_corpus.parquet")
HEADLINE_RUN = os.environ.get("HEADLINE_RUN_DIR", os.path.join(REPO, "outputs/timeomni_o1_forecast_run"))
CHRONOS_FT_RUN = os.environ.get("CHRONOS_FT_RUN_DIR", os.path.join(REPO, "outputs/chronos_only_forecast_run"))
DEFAULT_RUNS = [
    ("Ours (gen, mid_eval ep19)", os.path.join(HEADLINE_RUN, "mid_eval/epoch_19/forecast_preds.npz")),
    ("Ours (tf, mid_eval ep19)", os.path.join(REPO, "outputs/eval/_dedup437/headline_tf.npz")),
    ("Chronos-2 finetuned (mid_eval ep25)", os.path.join(CHRONOS_FT_RUN, "mid_eval/epoch_25/forecast_preds.npz")),
    ("Chronos-2 zero-shot", os.path.join(REPO, "outputs/baseline_forecast_zeroshot_corpus_v2/zeroshot_preds.npz")),
]
# Display order: grouped by domain (Solar / Load / Traffic / Finance / Climate)
ORDER = [s for dom in ("Solar", "Load", "Traffic", "Finance", "Climate") for s in DOMAIN_SOURCES[dom]]
SRC2DOM = {s: d for d, ss in DOMAIN_SOURCES.items() for s in ss}


def load_meta(parquet):
    t = pq.read_table(parquet, columns=["id", "split", "dataset_name"]).to_pandas()
    keep = set(t[t["split"] == "test"]["id"].astype(str))
    id2ds = dict(zip(t["id"].astype(str), t["dataset_name"]))
    return keep, id2ds


def per_dataset(npz_path, keep, id2ds):
    d = np.load(npz_path, allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    sel = np.array([i in keep for i in ids])
    pred, gt = d["pred_quantiles"][sel], d["gt"][sel]
    valid = d["valid_mask"][sel].astype(bool)
    roi = valid & (d["roi_mask"][sel] > 0.5)
    lv = d["quantile_levels"]
    ds = np.array([id2ds[i] for i in ids[sel]])
    rows = []
    for s in ORDER:
        q = ds == s
        mf = forecast_metrics(pred[q], gt[q], valid[q], lv)
        mr = forecast_metrics(pred[q], gt[q], roi[q], lv)
        rows.append(dict(dataset=s, domain=SRC2DOM[s], n=int(q.sum()),
                         CRPS_full=mf["CRPS"], CRPS_roi=mr["CRPS"],
                         PCC_full=mf["PCC"], PCC_roi=mr["PCC"],
                         MAPE_full=mf["MAPE"], MAPE_roi=mr["MAPE"]))
    dom = {}
    for k, srcs in DOMAIN_SOURCES.items():
        q = np.isin(ds, srcs)
        dom[k] = (forecast_metrics(pred[q], gt[q], valid[q], lv)["CRPS"],
                  forecast_metrics(pred[q], gt[q], roi[q], lv)["CRPS"])
    overall = (float(np.mean([v[0] for v in dom.values()])), float(np.mean([v[1] for v in dom.values()])))
    return rows, dom, overall, int(sel.sum()), int((~sel).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-parquet", default=DEDUP)
    ap.add_argument("--out-csv", default=os.path.join(REPO, "outputs/eval/forecast_dedup437_by_dataset.csv"))
    ap.add_argument("runs", nargs="*", help="optional: label=npz path pairs; without them the four defaults in the script are used")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs] if args.runs else DEFAULT_RUNS
    keep, id2ds = load_meta(args.dedup_parquet)
    out = []
    for label, path in runs:
        rows, dom, overall, n, dropped = per_dataset(path, keep, id2ds)
        print(f"\n== {label}  (n={n}, dropped={dropped})  Overall(5-domain equal-weight) full/ROI = "
              f"{overall[0]:.4f} / {overall[1]:.4f}")
        print("   domain CRPS full/ROI: " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in dom.items()))
        print(f"   {'dataset':38s} {'dom':8s} n   CRPS_full CRPS_roi PCC_full PCC_roi")
        for r in rows:
            print(f"   {r['dataset']:38s} {r['domain']:8s} {r['n']:3d} {r['CRPS_full']:.4f}   {r['CRPS_roi']:.4f}  "
                  f"{r['PCC_full']:.3f}   {r['PCC_roi']:.3f}")
            out.append(dict(run=label, **r))
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
