"""Paper main table: split the forecast npz of several runs by dataset_name (data source) into
sub-test-sets and compute the metrics for each.

Same grouping logic as analyze_per_dataset.py (reuses its resolve_ds_names/group_metrics: prefer
the dataset_names stored in the npz, fall back to positional alignment with the parquet for old
outputs); the difference is that any number of runs is supported and the output is a long CSV +
a per-source pivot table (CRPS/PCC x full/roi) from which the paper's forecasting table can be
filled directly.

The metric convention is strictly the same as the overall evaluation: forecast_metrics computes
per-sample values and averages over samples, and CRPS is the within-sample normalised WQL, so
re-aggregating any subset == evaluating that subset on its own.

Usage:
  python paper_forecast_by_source.py NAME=path/to/preds.npz [NAME2=...] \
      [--parquet corpus.parquet] [--out-csv OUT] [--ref NAME]
--ref names the reference run (e.g. PO); in the pivot table the other runs are marked with * where
they beat it (lower CRPS / higher PCC).
"""
import argparse
import csv
import os

import numpy as np

from chronos_llm.eval.metrics import forecast_metrics
from chronos_llm.scripts.utils.analyze_per_dataset import group_metrics, resolve_ds_names


def load_run(spec, parquet):
    name, path = spec.split("=", 1)
    z = np.load(path, allow_pickle=True)
    ds = resolve_ds_names(z, parquet)
    per = group_metrics(z, ds)
    # Overall row (ALL): same convention as the overall numbers in forecast_metrics.csv
    valid = z["valid_mask"].astype(bool)
    roi = valid & (z["roi_mask"] > 0.5)
    per["__ALL__"] = {
        "n": len(ds),
        "full": forecast_metrics(z["pred_quantiles"], z["gt"], valid, z["quantile_levels"]),
        "roi": forecast_metrics(z["pred_quantiles"], z["gt"], roi, z["quantile_levels"]),
    }
    return name, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=npz_path")
    ap.add_argument("--parquet", default=None, help="positional-alignment fallback when the npz has no dataset_names")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--ref", default=None, help="name of the reference run; other runs are marked with * relative to it")
    args = ap.parse_args()

    runs = dict(load_run(s, args.parquet) for s in args.runs)
    names = [s.split("=", 1)[0] for s in args.runs]
    sources = sorted({ds for per in runs.values() for ds in per if ds != "__ALL__"},
                     key=lambda d: -runs[names[0]][d]["n"])

    for metric, lower_better in (("CRPS", True), ("PCC", False)):
        for region in ("full", "roi"):
            print(f"\n########## {metric} / {region} ({'lower is better' if lower_better else 'higher is better'}"
                  f"{', *=better than ' + args.ref if args.ref else ''}) ##########")
            hdr = f"{'source':40} {'n':>4} | " + " | ".join(f"{n:>16}" for n in names)
            print(hdr)
            print("-" * len(hdr))
            for ds in sources + ["__ALL__"]:
                cells = []
                refv = runs[args.ref][ds][region][metric] if args.ref and args.ref in runs else None
                for n in names:
                    v = runs[n][ds][region][metric]
                    star = " "
                    if refv is not None and n != args.ref:
                        star = "*" if ((v < refv) if lower_better else (v > refv)) else " "
                    cells.append(f"{v:15.4f}{star}")
                label = "ALL" if ds == "__ALL__" else ds
                print(f"{label:40} {runs[names[0]][ds]['n']:>4} | " + " | ".join(cells))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "source", "region", "n", "MAPE", "PCC", "CRPS"])
            for n in names:
                for ds in sources + ["__ALL__"]:
                    for region in ("full", "roi"):
                        m = runs[n][ds][region]
                        w.writerow([n, "ALL" if ds == "__ALL__" else ds, region,
                                    runs[n][ds]["n"],
                                    f"{m['MAPE']:.4f}", f"{m['PCC']:.6f}", f"{m['CRPS']:.6f}"])
        print(f"\nwritten -> {args.out_csv}")


if __name__ == "__main__":
    main()
