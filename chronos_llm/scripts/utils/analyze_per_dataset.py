"""Split forecasting predictions by dataset_name and compare the per-dataset metrics of the model
(best epoch) against a baseline.

Usage:
  python analyze_per_dataset.py <model_npz> <baseline_npz> [test_parquet]

Group labels preferentially come from the npz's own ``dataset_names`` (infer_forecast writes one per
row, same convention for the generate and teacher-forced modes) -- then no test_parquet is needed and
the model/baseline npz files need not share a row order. Only when that key is missing (older
artefacts) does the script fall back to the old "align by position against test_parquet" path, in
which case test_parquet is mandatory and positional alignment is the only mapping that holds, because
the ids are not unique.
"""
import argparse
import functools

import numpy as np

from chronos_llm.eval.metrics import forecast_metrics


@functools.lru_cache(maxsize=4)
def test_dataset_order(parquet):
    """Fallback path: ids are NOT unique inside the test split (500 rows carry only 324 distinct ids,
    111 of which appear in more than one dataset), so alignment must be **positional**: row i of the
    npz corresponds to row i of the test split after reset_index. That the positional alignment holds
    strictly was double-checked both via the id order and by comparing gt against future_values
    numerically.
    parquet path -> (dataset_name array, id list) is deterministic, so the result is cached to keep
    batch scripts from re-reading the file (~0.4s per read of the 13MB parquet, which noticeably slows
    down a scan over hundreds of epochs)."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["id", "dataset_name", "split"])
    df = t.to_pandas()
    df = df[df["split"] == "test"].reset_index(drop=True)
    return df["dataset_name"].astype(str).to_numpy(), df["id"].astype(str).tolist()


def resolve_ds_names(npz, parquet):
    """Per-row dataset label: prefer the npz's own dataset_names, otherwise fall back to positional
    alignment against the parquet."""
    if "dataset_names" in npz.files:
        return np.array([str(x) for x in npz["dataset_names"]])
    if parquet is None:
        raise SystemExit("npz has no dataset_names (old artefact); pass test_parquet to fall back to "
                         "positional alignment")
    ds_names, ref_ids = test_dataset_order(parquet)
    ids = [str(x) for x in npz["ids"]]
    assert ids == ref_ids, "npz id order differs from the test split -- positional alignment is invalid!"
    return ds_names


def group_metrics(npz, ds_names):
    pred, gt = npz["pred_quantiles"], npz["gt"]
    roi_mask, valid, levels = npz["roi_mask"], npz["valid_mask"].astype(bool), npz["quantile_levels"]
    roi = valid & (roi_mask > 0.5)
    out = {}
    for ds in sorted(set(ds_names)):
        idx = np.where(ds_names == ds)[0]
        full = forecast_metrics(pred[idx], gt[idx], valid[idx], levels)
        r = forecast_metrics(pred[idx], gt[idx], roi[idx], levels)
        out[ds] = {"n": len(idx), "full": full, "roi": r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_npz")
    ap.add_argument("baseline_npz")
    ap.add_argument("test_parquet", nargs="?", default=None,
                    help="only needed when the npz carries no dataset_names of its own (old "
                         "artefacts); used for the positional-alignment fallback")
    args = ap.parse_args()

    mz = np.load(args.model_npz, allow_pickle=True)
    bz = np.load(args.baseline_npz, allow_pickle=True)
    M = group_metrics(mz, resolve_ds_names(mz, args.test_parquet))
    B = group_metrics(bz, resolve_ds_names(bz, args.test_parquet))

    print("=" * 130)
    print("Per-dataset comparison: model(best epoch) vs baseline."
          " Δ=model-baseline; lower is better for MAPE/CRPS (Δ<0 good), higher is better for PCC"
          " (Δ>0 good). ★=model wins")
    print("=" * 130)

    for region in ("full", "roi"):
        print(f"\n########## region = {region} ##########")
        hdr = (f"{'dataset':38} {'n':>3} | "
               f"{'MAPE(m/b/Δ)':>26} | {'PCC(m/b/Δ)':>26} | {'CRPS(m/b/Δ)':>28}")
        print(hdr)
        print("-" * len(hdr))
        # sorted by descending n
        for ds in sorted(M.keys(), key=lambda d: -M[d]["n"]):
            m, b = M[ds][region], B[ds][region]
            def fmt(mk, bk, lower):
                dv = mk - bk
                better = (dv < 0) if lower else (dv > 0)
                star = "★" if better else " "
                return f"{mk:7.2f}/{bk:7.2f}/{dv:+6.2f}{star}"
            def fmtf(mk, bk, lower):
                dv = mk - bk
                better = (dv < 0) if lower else (dv > 0)
                star = "★" if better else " "
                return f"{mk:.3f}/{bk:.3f}/{dv:+.3f}{star}"
            print(f"{ds:38} {M[ds]['n']:>3} | "
                  f"{fmt(m['MAPE'], b['MAPE'], True):>26} | "
                  f"{fmtf(m['PCC'], b['PCC'], False):>26} | "
                  f"{fmtf(m['CRPS'], b['CRPS'], True):>28}")


if __name__ == "__main__":
    main()
