"""Paper main table: group the 10 data sources of a forecast npz into 5 domains and report the
metrics of each domain directly.

Key difference from paper_forecast_by_source.py: a domain row is **not** the weighted average of
per-source means; instead all raw samples of the domain's sources are pooled and forecast_metrics
is called once on the pooled samples (per-sample computation, then nanmean). The two protocols
give close but not algebraically identical numbers -- CRPS/PCC treat certain samples (e.g.
all-zero ground truth) as nan, which nanmean drops; "per-source means first" weights by the raw n
of each source, "direct pooling" by the valid samples after nan removal. Domain rows follow this
script (pooling), consistent with the main table's footnote "CRPS = normalised WQL, computed per
sample then averaged => a domain row = the direct mean over all samples of that domain".

DOMAIN_SOURCES is the 10->5 grouping fixed for the paper; changing the grouping only requires
editing this dict.

Usage:
  python paper_forecast_by_domain.py NAME=path/to/preds.npz [NAME2=...] \
      [--parquet corpus.parquet] [--out-csv OUT]
"""
import argparse
import csv
import os

import numpy as np

from chronos_llm.eval.metrics import forecast_metrics
from chronos_llm.scripts.utils.analyze_per_dataset import resolve_ds_names

DOMAIN_SOURCES = {
    "Solar": ["CGTSF/MSPG", "fidelts/Canada_photovoltaics_plants",
              "fidelts/Germany_Renewable_Power_Grid"],
    "Load": ["fnf/load", "fidelts/California_ISO"],
    "Traffic": ["CGTSF/PTF", "fnf/traffic"],
    "Finance": ["finnews/MTBench_finance", "fnf/bitcoin"],
    "Climate": ["timemmd/Climate"],
}


def load_run(spec, parquet):
    name, path = spec.split("=", 1)
    z = np.load(path, allow_pickle=True)
    ds = resolve_ds_names(z, parquet)
    valid = z["valid_mask"].astype(bool)
    roi = valid & (z["roi_mask"] > 0.5)
    out = {}
    for dom, srcs in DOMAIN_SOURCES.items():
        idx = np.where(np.isin(ds, srcs))[0]
        mf = forecast_metrics(z["pred_quantiles"][idx], z["gt"][idx], valid[idx], z["quantile_levels"])
        mr = forecast_metrics(z["pred_quantiles"][idx], z["gt"][idx], roi[idx], z["quantile_levels"])
        out[dom] = {"n": len(idx), "full": mf, "roi": mr}
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=npz_path")
    ap.add_argument("--parquet", default=None, help="positional-alignment fallback when the npz has no dataset_names")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    runs = dict(load_run(s, args.parquet) for s in args.runs)
    names = [s.split("=", 1)[0] for s in args.runs]

    hdr = f"{'domain':10} {'n':>4} | " + " | ".join(f"{n:>26}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for dom in DOMAIN_SOURCES:
        cells = [f"full={runs[n][dom]['full']['CRPS']:.4f}/roi={runs[n][dom]['roi']['CRPS']:.4f}"
                 for n in names]
        print(f"{dom:10} {runs[names[0]][dom]['n']:>4} | " + " | ".join(f"{c:>26}" for c in cells))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "domain", "n", "CRPS_full", "CRPS_roi", "PCC_full", "PCC_roi"])
            for n in names:
                for dom in DOMAIN_SOURCES:
                    r = runs[n][dom]
                    w.writerow([n, dom, r["n"],
                                f"{r['full']['CRPS']:.4f}", f"{r['roi']['CRPS']:.4f}",
                                f"{r['full']['PCC']:.4f}", f"{r['roi']['PCC']:.4f}"])
        print(f"\nwrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
