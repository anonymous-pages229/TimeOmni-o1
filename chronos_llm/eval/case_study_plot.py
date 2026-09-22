"""Case-study plot: draw a single-sample comparison from **saved real prediction npz files**
(baseline zero-shot vs. ChronosLLM).

Loads no model -- pure numpy/pandas/matplotlib reading npz + parquet (history/text), seconds on CPU.
Samples are aligned with the two npz files row by row by **row position** (double-checked with the
GT values); the id is for display only (ids are not unique within the parquet).

Usage:
    python chronos_llm/eval/case_study_plot.py --idx 306 --out docs/case_study_id115.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronos_llm.eval.metrics import sample_pcc, sample_wql, _median_index

BASE = "outputs/baseline_forecast_zeroshot/zeroshot_preds.npz"
LLM = "outputs/eval/forecast/forecast_preds.npz"
PARQUET = "data/forecast/mmtr_forecast_corpus.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=306)
    ap.add_argument("--out", default="docs/case_study.png")
    args = ap.parse_args()

    base = np.load(BASE, allow_pickle=True)
    llm = np.load(LLM, allow_pickle=True)
    assert list(base["ids"]) == list(llm["ids"]), "the two npz files have different row orders"
    levels = base["quantile_levels"].astype(float)
    qi = _median_index(levels)
    lo_i = int(np.argmin(np.abs(levels - 0.1)))
    hi_i = int(np.argmin(np.abs(levels - 0.9)))

    n = args.idx
    gt = base["gt"][n]
    valid = base["valid_mask"][n]
    roi = (base["roi_mask"][n] > 0) & valid
    H = int(valid.sum())                       # valid horizon length
    t = np.arange(H)

    base_med = base["pred_quantiles"][n, qi, :H]
    llm_med = llm["pred_quantiles"][n, qi, :H]
    llm_lo = llm["pred_quantiles"][n, lo_i, :H]
    llm_hi = llm["pred_quantiles"][n, hi_i, :H]
    gt_v = gt[:H]
    roi_v = roi[:H]

    # per-sample ROI metrics (actually computed)
    pcc_b = sample_pcc(gt_v, base_med, roi_v)
    pcc_l = sample_pcc(gt_v, llm_med, roi_v)
    wql_b = sample_wql(gt_v, base["pred_quantiles"][n][:, :H], levels, roi_v)
    wql_l = sample_wql(gt_v, llm["pred_quantiles"][n][:, :H], levels, roi_v)

    # history (left part of the plot) from the parquet, taken by row position with an id check
    df = pd.read_parquet(PARQUET)
    dft = df[df["split"] == "test"].reset_index(drop=True)
    row = dft.iloc[n]
    assert str(row["id"]) == str(base["ids"][n]), "row alignment id mismatch"
    hist = np.asarray(row["history_values"], dtype=float).ravel()
    past_len = len(hist)
    roi_idx = np.where(roi_v)[0]
    roi_a, roi_b = (int(roi_idx[0]), int(roi_idx[-1] + 1)) if roi_idx.size else (0, 0)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 5.2))
    th = np.arange(past_len)
    tf = past_len + t                          # absolute time axis of the future segment

    ax.plot(th, hist, color="#444", lw=1.4, label="History (observed)")
    ax.axvline(past_len - 0.5, color="#999", ls=":", lw=1)
    # ROI shading
    if roi_b > roi_a:
        ax.axvspan(past_len + roi_a - 0.5, past_len + roi_b - 0.5,
                   color="#ffe9a8", alpha=0.6, label="ROI (region of interest)")
    # ground truth
    ax.plot(tf, gt_v, color="#111", lw=2.2, label="Ground truth (future)")
    # baseline
    ax.plot(tf, base_med, color="#e8743b", lw=1.8, ls="--",
            label=f"Chronos-2 zero-shot (ROI PCC={pcc_b:.2f})")
    # ChronosLLM + quantile band
    ax.fill_between(tf, llm_lo, llm_hi, color="#1f77b4", alpha=0.18,
                    label="ChronosLLM 10-90% quantile band")
    ax.plot(tf, llm_med, color="#1f77b4", lw=1.9, ls="--",
            label=f"ChronosLLM (ROI PCC={pcc_l:.2f})")

    ax.set_title(
        f"Case study  id={base['ids'][n]}  ({row['dataset_name']})\n"
        f"ROI trend corr (PCC): {pcc_b:.2f} -> {pcc_l:.2f}   |   "
        f"ROI quantile loss (WQL): {wql_b:.2f} -> {wql_l:.2f}",
        fontsize=11)
    ax.set_xlabel("time index"); ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Saved figure -> {args.out}")
    print(f"id={base['ids'][n]} dataset={row['dataset_name']} ROI=[{roi_a+past_len},{roi_b+past_len}) "
          f"roi_shape={row.get('roi_shape')}")
    print(f"ROI PCC  base={pcc_b:.3f}  llm={pcc_l:.3f}")
    print(f"ROI WQL  base={wql_b:.3f}  llm={wql_l:.3f}")


if __name__ == "__main__":
    main()
