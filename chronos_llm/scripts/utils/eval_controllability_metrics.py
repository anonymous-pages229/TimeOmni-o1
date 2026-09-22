"""Controllability metrics: after sweeping the magnitude band in the edited conclusion, does the
forecast follow the stated band?

Input: the forecast_preds_tf.npz of each variant produced by controllability_infer.sh (identical
row order, all in test-row order) + ctrl_orig.parquet (typical reference and dataset_name).

Per sample, ratio_hat(v) = peak of variant v's forecast median within ROI∩valid (valid if empty) /
the typical peak of that dataset (p90 of the future peaks over the whole parquet, same convention
as the corpus magnitude tags -- the ctrl parquet only contains test rows, so the typical reference is
estimated from those rows directly; the 5 band variants share the same reference => comparisons
between bands are unaffected).

Metrics (ALL + per dataset):
- **monotonicity**: per-sample Spearman rho between the 5 band ordinals and ratio_hat -- mean,
  fraction with rho>0, fraction strictly ordered;
- **direction-follow rate**: fraction of samples with ratio_hat(above) > ratio_hat(well_below)
  (the edit direction is followed);
- **response magnitude**: mean ratio_hat per band (offset relative to the orig reference);
- **perturbation relative to orig**: mean |ratio_hat(v) - ratio_hat(orig)| (how much the edit
  actually changed the forecast).

Usage: python eval_controllability_metrics.py --root outputs/eval/controllability \
        --parquet .../ctrl_orig.parquet [--out-csv OUT]
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BANDS = ["tiny", "well_below", "mod_below", "typical", "above"]


def _peak(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        v = v[0]
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.nanmax(a)) if a.size else np.nan


def pred_peaks(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    q = z["pred_quantiles"]                       # (N, Q, H)
    med = q[:, q.shape[1] // 2, :]                # median row (levels are symmetric and include 0.5)
    valid = z["valid_mask"].astype(bool)
    roi = valid & (z["roi_mask"] > 0.5)
    peaks = np.full(len(med), np.nan)
    for i in range(len(med)):
        m = roi[i] if roi[i].any() else valid[i]
        if m.any():
            peaks[i] = np.nanmax(med[i][m])
    return peaks


def summarize(mat, orig, mask, label):
    """mat: (5, N) ratio_hat per band; orig: (N,) reference; mask: samples entering the statistics."""
    idx = np.where(mask)[0]
    rhos, strict = [], 0
    for i in idx:
        col = mat[:, i]
        if np.isfinite(col).all():
            r = spearmanr(np.arange(len(BANDS)), col).statistic
            if np.isfinite(r):
                rhos.append(r)
                strict += int(np.all(np.diff(col) > 0))
    rhos = np.array(rhos)
    ab = mat[BANDS.index("above"), idx]
    wb = mat[BANDS.index("well_below"), idx]
    ok = np.isfinite(ab) & np.isfinite(wb)
    dir_rate = float(np.mean(ab[ok] > wb[ok])) if ok.any() else np.nan
    dev = np.nanmean([np.nanmean(np.abs(mat[b, idx] - orig[idx])) for b in range(len(BANDS))])
    row = {
        "group": label, "n": len(idx),
        "spearman_mean": float(np.mean(rhos)) if rhos.size else np.nan,
        "spearman_pos_rate": float(np.mean(rhos > 0)) if rhos.size else np.nan,
        "strict_mono_rate": strict / len(rhos) if rhos.size else np.nan,
        "dir_follow_rate(above>well_below)": dir_rate,
        "mean_abs_dev_vs_orig": float(dev),
    }
    for b, name in enumerate(BANDS):
        row[f"ratio_{name}"] = float(np.nanmean(mat[b, idx]))
    row["ratio_orig"] = float(np.nanmean(orig[idx]))
    print(f"[{label}] n={row['n']} Spearman={row['spearman_mean']:.3f} "
          f"(rho>0 {row['spearman_pos_rate']:.2f}, strictly monotone {row['strict_mono_rate']:.2f}) "
          f"direction-follow rate={row['dir_follow_rate(above>well_below)']:.2f} "
          f"band means tiny..above = " + ", ".join(f"{row[f'ratio_{n}']:.2f}" for n in BANDS)
          + f" (orig {row['ratio_orig']:.2f})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    test = pd.read_parquet(args.parquet)
    ds_names = test["dataset_name"].astype(str).to_numpy()
    pk = test["future_values"].map(_peak)
    typ_map = pk.groupby(test["dataset_name"]).quantile(0.90).to_dict()
    typ = np.array([typ_map.get(d, np.nan) for d in ds_names])
    typ_ok = np.isfinite(typ) & (typ > 1e-9)

    orig = pred_peaks(os.path.join(args.root, "orig", "forecast_preds_tf.npz")) / np.where(typ_ok, typ, np.nan)
    mat = np.stack([
        pred_peaks(os.path.join(args.root, b, "forecast_preds_tf.npz")) / np.where(typ_ok, typ, np.nan)
        for b in BANDS
    ])  # (5, N)

    rows = [summarize(mat, orig, typ_ok, "ALL")]
    for d in sorted(set(ds_names)):
        rows.append(summarize(mat, orig, typ_ok & (ds_names == d), d))

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"Wrote -> {args.out_csv}")


if __name__ == "__main__":
    main()
