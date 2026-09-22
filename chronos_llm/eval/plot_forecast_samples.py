"""Sample plots: history + future GT + future prediction (ChronosLLM) + future zero-shot prediction,
**x-axis = real timestamps**.

Reads the infer npz (``pred_quantiles/gt/roi_mask/valid_mask/ids``) + the companion jsonl with the
same stem (model-generated reasoning ``gen_text``, same row order; ``gen_text`` embedded in old npz
files is still supported as a fallback) + an optional zero-shot baseline npz (overlaid zero-shot
prediction line) + the parquet (history values/timestamps/event, reasoning and conclusion text).
Samples are aligned with the npz by **row position** (double-checked with the GT values, since ids
are not unique within the parquet); N samples are drawn and each is written as a PNG, and every
sample's "model-generated reasoning + GT event/reasoning/conclusion" is dumped into one markdown
file. Pure numpy/pandas/matplotlib, CPU.

Usage:
    python chronos_llm/eval/plot_forecast_samples.py \
        --pred_npz outputs/.../mid_eval/epoch_X/forecast_preds.npz \
        --zeroshot_npz outputs/baseline_forecast_zeroshot/zeroshot_preds.npz \
        --n 8 --seed 0 --band --out_dir outputs/forecast_plots
    # or give explicit row numbers: --rows 306,305,291
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chronos_llm.eval.metrics import sample_pcc, sample_wql, _median_index

PARQUET = "data/forecast/mmtr_forecast_corpus.parquet"


def _q_index(levels, q):
    return int(np.argmin(np.abs(np.asarray(levels, float) - q)))


def _align_parquet_row(dft, npz_ids, gt_npz, valid, n):
    """Take the parquet row by position and double-check the alignment with id + GT values (parquet
    ids are not unique, so the id alone is unreliable)."""
    row = dft.iloc[n]
    H = int(valid.sum())
    fut = np.asarray(row["future_values"], float).ravel()
    id_ok = str(row["id"]) == str(npz_ids[n])
    gt_ok = len(fut) >= H and np.allclose(fut[:H], gt_npz[:H], atol=1e-3, equal_nan=True)
    return row, H, (id_ok and gt_ok)


def plot_one(ax, row, H, npz, zs, n, qi, lo_i, hi_i, draw_band):
    """Plot one sample on ax; returns (pcc_pred, pcc_zs, wql_pred, wql_zs, roi_idx)."""
    levels = npz["quantile_levels"].astype(float)
    gt = npz["gt"][n][:H]
    roi = (npz["roi_mask"][n][:H] > 0)
    med = npz["pred_quantiles"][n, qi, :H]
    lo = npz["pred_quantiles"][n, lo_i, :H]
    hi = npz["pred_quantiles"][n, hi_i, :H]

    hist = np.asarray(row["history_values"], float).ravel()
    hts = pd.to_datetime(np.asarray(row["history_timestamps"]).ravel())
    fts = pd.to_datetime(np.asarray(row["future_timestamps"]).ravel())[:H]
    # align history with its timestamps (take the tail in case the lengths differ)
    hts = hts[-len(hist):] if len(hts) >= len(hist) else hts
    hist = hist[-len(hts):]

    ax.plot(hts, hist, color="#444", lw=1.3, label="History")
    if len(fts):
        ax.axvline(fts[0], color="#999", ls=":", lw=1)
    roi_idx = np.where(roi)[0]
    if roi_idx.size:
        ax.axvspan(fts[roi_idx[0]], fts[min(roi_idx[-1], H - 1)],
                   color="#ffe9a8", alpha=0.55, label="ROI")
    ax.plot(fts, gt, color="#111", lw=2.0, label="Future GT")
    pcc_zs = wql_zs = float("nan")
    if zs is not None:
        zmed = zs["pred_quantiles"][n, qi, :H]
        ax.plot(fts, zmed, color="#e8743b", lw=1.7, ls="--", label="Zero-shot pred")
        pcc_zs = sample_pcc(gt, zmed, roi)
        wql_zs = sample_wql(gt, zs["pred_quantiles"][n][:, :H], levels, roi)
    if draw_band:
        ax.fill_between(fts, lo, hi, color="#1f77b4", alpha=0.16, label="Pred 10-90%")
    ax.plot(fts, med, color="#1f77b4", lw=1.8, ls="--", label="ChronosLLM pred")
    pcc_pred = sample_pcc(gt, med, roi)
    wql_pred = sample_wql(gt, npz["pred_quantiles"][n][:, :H], levels, roi)
    return pcc_pred, pcc_zs, wql_pred, wql_zs, roi_idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_npz", required=True,
                    help="ChronosLLM infer npz (the reasoning is read from the .jsonl with the same "
                         "stem; keep both in the same directory)")
    ap.add_argument("--zeroshot_npz", default=None,
                    help="zero-shot baseline npz (overlaid comparison line, optional)")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--rows", default=None,
                    help="comma-separated npz row numbers (takes precedence over --n random sampling)")
    ap.add_argument("--n", type=int, default=8, help="number of randomly sampled rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", action="store_true", help="draw the 10-90% quantile band")
    ap.add_argument("--out_dir", default="outputs/forecast_plots")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    npz = np.load(args.pred_npz, allow_pickle=True)
    zs = np.load(args.zeroshot_npz, allow_pickle=True) if args.zeroshot_npz else None
    if zs is not None:
        assert zs["pred_quantiles"].shape[0] == npz["pred_quantiles"].shape[0], \
            "zero-shot and pred row counts differ; cannot align by row"
    levels = npz["quantile_levels"].astype(float)
    qi, lo_i, hi_i = _median_index(levels), _q_index(levels, 0.1), _q_index(levels, 0.9)
    ids = npz["ids"]
    # Model-generated reasoning comes from the companion jsonl (same stem and row order as the npz);
    # gen_text embedded in old npz files is the fallback.
    jsonl_path = (args.pred_npz[:-4] + ".jsonl") if args.pred_npz.endswith(".npz") \
        else args.pred_npz + ".jsonl"
    gen_text = None
    if os.path.exists(jsonl_path):
        gen_text = [json.loads(l).get("gen_text", "")
                    for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    elif "gen_text" in npz.files:
        gen_text = list(npz["gen_text"])
    has_text = gen_text is not None

    df = pd.read_parquet(args.parquet)
    dft = df[df["split"] == args.split].reset_index(drop=True)
    N = len(ids)
    assert len(dft) == N, f"parquet split={args.split} has {len(dft)} rows != npz {N}"

    if args.rows:
        rows = [int(x) for x in args.rows.split(",") if x.strip() != ""]
    else:
        rng = np.random.default_rng(args.seed)
        rows = sorted(rng.choice(N, size=min(args.n, N), replace=False).tolist())

    md = [f"# Forecast sample visualization ({len(rows)} samples)\n",
          f"- pred_npz: `{args.pred_npz}`",
          f"- zeroshot_npz: `{args.zeroshot_npz}`",
          f"- model-generated reasoning: {'loaded (jsonl gen_text)' if has_text else '**gen_text not found** (no companion jsonl and the old npz has none either; rerun infer to produce it)'}\n"]

    for n in rows:
        row, H, ok = _align_parquet_row(dft, ids, npz["gt"][n], npz["valid_mask"][n], n)
        if not ok:
            print(f"[warn] row {n} failed the alignment check (id/GT mismatch), skipped")
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        pcc_p, pcc_z, wql_p, wql_z, roi_idx = plot_one(
            ax, row, H, npz, zs, n, qi, lo_i, hi_i, args.band)
        zspart = f"  |  zero-shot PCC={pcc_z:.2f} WQL={wql_z:.2f}" if zs is not None else ""
        ax.set_title(f"row={n}  id={ids[n]}  ({row['dataset_name']})\n"
                     f"ROI: ChronosLLM PCC={pcc_p:.2f} WQL={wql_p:.2f}{zspart}", fontsize=10.5)
        ax.set_xlabel("timestamp"); ax.set_ylabel("value")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.25)
        fig.autofmt_xdate(); fig.tight_layout()
        png = os.path.join(args.out_dir, f"sample_row{n}_id{ids[n]}.png")
        fig.savefig(png, dpi=130); plt.close(fig)

        md.append(f"\n## row={n}  id={ids[n]}  {row['dataset_name']}")
        md.append(f"![](sample_row{n}_id{ids[n]}.png)\n")
        md.append(f"- ROI ChronosLLM: PCC={pcc_p:.3f}  WQL={wql_p:.3f}"
                  + (f"  |  zero-shot: PCC={pcc_z:.3f}  WQL={wql_z:.3f}" if zs is not None else ""))
        md.append(f"\n**GT event (EVENT)**: {str(row.get('event','')).strip()}")
        md.append(f"\n**GT reasoning (REASONING)**: {str(row.get('reasoning','')).strip()}")
        md.append(f"\n**GT conclusion (CONCLUSION)**: {str(row.get('conclusion','')).strip()}")
        if has_text:
            md.append(f"\n**Model-generated (generate)**:\n```\n{str(gen_text[n]).strip()}\n```")
        print(f"Saved {png}  (ROI PCC pred={pcc_p:.2f}" +
              (f" / zs={pcc_z:.2f}" if zs is not None else "") + ")")

    md_path = os.path.join(args.out_dir, "samples.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved visualization summary -> {md_path}")


if __name__ == "__main__":
    main()
