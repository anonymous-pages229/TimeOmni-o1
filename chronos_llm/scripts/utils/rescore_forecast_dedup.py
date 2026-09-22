r"""Re-score existing forecast evaluation artefacts (npz files over the original 444-row test
split) **against the deduplicated test whitelist**, without re-running GPU inference.

Background: the test split of the original training corpus contained **7 rows duplicated from
train** (6x CGTSF/PTF + 1x fnf/traffic, test indices 63/78/90/91/92/93/426). The deduplicated
corpus removes these 7 rows from **test** (444->437) while **train (4,630 rows) is unchanged
character for character** (prompt/conclusion/reasoning/background/event of the shared rows are
identical in both parquets) => **already-trained models are completely unaffected; only the
evaluation convention changes**.

The npz written by `infer_forecast.py` carries `ids`, so every historical artefact can be filtered
by the whitelist after the fact and re-scored on the same axis as the deduplicated corpus -- no
inference has to be re-run.

    python chronos_llm/scripts/utils/rescore_forecast_dedup.py \
        headline=outputs/eval/forecast/<run_a>/forecast_preds.npz \
        other=outputs/eval/<run_b>/forecast_preds.npz \
        [--save-filtered-npz outputs/eval/_dedup437]   # save the filtered npz for the by_domain script

**Primary convention = equal-weight arithmetic mean over the 5 domains** (this is the Overall
reported in the main tables); the pooled number is shown only as a secondary column. Note that the 7
duplicated rows **all fall in the Traffic domain** (6x CGTSF/PTF + 1x fnf/traffic) => dedup changes
only the Traffic column, the other four domains are element-wise unchanged; under the equal-weight
convention the effect is further diluted by the 1/5 weight.

Filtered numbers can only be compared with numbers that were filtered the same way. Also note that
**the independent-evaluation axis != the mid-training-evaluation axis**: numbers from the two are not
on the same axis to begin with, so do not mix them when switching axes.
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

DEDUP = ("data/forecast/"
         "mmtr_forecast_corpus.parquet")


def keep_ids(parquet):
    t = pq.read_table(parquet, columns=["id", "split"]).to_pandas()
    return set(t[t["split"] == "test"]["id"].astype(str))


# TimesFM-3.0 emits 9 quantiles (0.1...0.9), Chronos-2 emits 21 (0.01...0.99). CRPS here is the
# **normalised WQL** (sum 2|(y-q)(1{y<=q}-alpha)| / (K * sum|y|)): K sits in the denominator and the
# pinball weight min(alpha, 1-alpha) of the extreme quantiles (alpha=0.01/0.99) is tiny by
# construction => **the CRPS of 21 quantiles is systematically lower than that of 9**, so absolute
# scores of two backbones cannot be put side by side without aligning them first.
# Within one arm (FULL vs. its own TSFM-only lower bound, i.e. the coupling gain) this does not
# matter -- both sides share the same quantile grid.
#
# Alignment: **interpolate the 9 TimesFM quantiles onto the 21 Chronos ones**, rather than cutting
# both down to the 9 they share. Cutting would move every number off the published convention (the
# headline goes 0.1162 -> 0.1316 and every main table has to be recomputed); interpolating only
# touches the TimesFM rows and keeps them on the same axis as the main tables.
# The interpolation is linear in the quantile function; the 4 target levels outside [0.1, 0.898]
# (0.01/0.05/0.949/0.988) are linearly extrapolated with the slope of the two endpoint quantiles --
# the quantile function is increasing and the endpoint slope is positive, so the result stays
# monotone and no quantile crossing is introduced.
# The target grid must be the levels **actually stored** in the checkpoint, not the round
# 0.01/0.05/.../0.99: training runs in bf16, so the levels tensor is cast to bf16 and back
# (0.01 -> 0.010009766, 0.3 -> 0.30078125, 0.99 -> 0.98828125; up to 0.0017 off). With a round
# target grid the TimesFM side would sit on round levels and the Chronos side on bf16 levels, and
# the pinball weights alpha would not be identical on both sides.
# The 9 bf16 TimesFM levels are bit-identical to entries 2/4/6/8/10/12/14/16/18 of this grid => the
# interpolation anchors align exactly and only the 12 added levels are interpolated.
CHRONOS_21 = (0.010009766, 0.050048828, 0.100097656, 0.150390625, 0.200195312, 0.25,
              0.30078125, 0.349609375, 0.400390625, 0.44921875, 0.5, 0.55078125,
              0.6015625, 0.6484375, 0.69921875, 0.75, 0.80078125, 0.8515625,
              0.8984375, 0.94921875, 0.98828125)


def _interp_quantiles(pred, src_levels, dst_levels=CHRONOS_21, tol=0.01):
    """(N, Q_src, H) quantile predictions -> (N, len(dst), H). Returned unchanged if already on the target grid."""
    src = np.asarray(src_levels, dtype=float)
    dst = np.asarray(dst_levels, dtype=float)
    # Already 21 levels: leave untouched. tol=0.01 tolerates the two ways the same grid gets stored --
    # training artefacts carry bf16-quantised levels (0.99 -> 0.98828125) while a zero-shot run on CPU
    # runs in float32 and stores the exact 0.99. The two differ by <=0.0017, are the same grid at
    # different precision, and must not trigger a spurious resampling (which would drift Chronos-2
    # zero-shot from 0.1727 to 0.1730). With K identical they are comparable; the effect of that
    # <=0.0017 level difference on CRPS is <=0.0003.
    if len(src) == len(dst) and np.all(np.abs(src - dst) <= tol):
        return pred, src
    n, _, h = pred.shape
    flat = np.asarray(pred, dtype=np.float64).transpose(0, 2, 1).reshape(-1, len(src))
    out = np.empty((flat.shape[0], len(dst)), dtype=np.float64)
    for j, q in enumerate(dst):
        if q <= src[0]:                       # extrapolate on the left
            slope = (flat[:, 1] - flat[:, 0]) / (src[1] - src[0])
            out[:, j] = flat[:, 0] + slope * (q - src[0])
        elif q >= src[-1]:                    # extrapolate on the right
            slope = (flat[:, -1] - flat[:, -2]) / (src[-1] - src[-2])
            out[:, j] = flat[:, -1] + slope * (q - src[-1])
        else:
            i = int(np.searchsorted(src, q) - 1)
            w = (q - src[i]) / (src[i + 1] - src[i])
            out[:, j] = flat[:, i] * (1.0 - w) + flat[:, i + 1] * w
    return out.reshape(n, h, len(dst)).transpose(0, 2, 1).astype(pred.dtype), dst


def rescore(npz_path, keep, save_dir=None, label="", interp21=False):
    d = np.load(npz_path, allow_pickle=True)
    ids = np.array([str(x) for x in d["ids"]])
    sel = np.array([i in keep for i in ids])
    dropped = (~sel).sum()
    pred, gt = d["pred_quantiles"][sel], d["gt"][sel]
    valid = d["valid_mask"][sel].astype(bool)
    roi = valid & (d["roi_mask"][sel] > 0.5)
    levels = d["quantile_levels"]
    if interp21:
        pred, levels = _interp_quantiles(pred, levels)
    full_m = forecast_metrics(pred, gt, valid, levels)
    roi_m = forecast_metrics(pred, gt, roi, levels)
    # Primary convention: equal weight over the 5 domains (the Overall definition of the main tables)
    ds = np.array([str(x) for x in d["dataset_names"]])[sel]
    per_dom, dom_f, dom_r = {}, [], []
    for dom, srcs in DOMAIN_SOURCES.items():
        idx = np.where(np.isin(ds, srcs))[0]
        if len(idx) == 0:
            continue
        f = forecast_metrics(pred[idx], gt[idx], valid[idx], levels)
        r = forecast_metrics(pred[idx], gt[idx], roi[idx], levels)
        per_dom[dom] = {"n": len(idx), "full": f["CRPS"], "roi": r["CRPS"]}
        dom_f.append(f["CRPS"]); dom_r.append(r["CRPS"])
    full_m["EQ5"] = float(np.mean(dom_f)) if dom_f else float("nan")
    roi_m["EQ5"] = float(np.mean(dom_r)) if dom_r else float("nan")
    full_m["per_dom"] = per_dom
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{label or os.path.basename(os.path.dirname(npz_path))}.npz")
        np.savez(out, pred_quantiles=pred, gt=gt, roi_mask=d["roi_mask"][sel],
                 valid_mask=d["valid_mask"][sel], quantile_levels=levels, ids=ids[sel],
                 dataset_names=np.array([str(x) for x in d["dataset_names"]])[sel])
        print(f"    filtered npz -> {out}")
    return full_m, roi_m, int(sel.sum()), int(dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=npz path")
    ap.add_argument("--dedup-parquet", default=DEDUP)
    ap.add_argument("--interp21", action="store_true",
                    help="linearly interpolate 9-quantile predictions onto the 21 Chronos levels before "
                         "computing CRPS -- required when comparing absolute scores across backbones "
                         "(TimesFM emits 9, Chronos 21, and K sits in the WQL denominator)")
    ap.add_argument("--save-filtered-npz", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    keep = keep_ids(args.dedup_parquet)
    print(f"[rescore] dedup test whitelist: {len(keep)} rows ({os.path.basename(args.dedup_parquet)})\n")
    rows = []
    print("Primary convention = equal weight over the 5 domains (the Overall definition of the main tables); pooled columns are for reference only\n")
    print(f"{'run':30s} | {'n':>4s} | {'eq full':>8s} | {'eq roi':>8s} | {'pool full':>8s} | "
          f"{'pool roi':>8s} | per-domain full (Solar/Load/Traffic/Finance/Climate)")
    print("-" * 132)
    for spec in args.runs:
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"{label:34s} | WARNING: not found {path}")
            continue
        f, r, n, dr = rescore(path, keep, args.save_filtered_npz, label, interp21=args.interp21)
        pd_str = " ".join(f"{f['per_dom'][d]['full']:.4f}" for d in DOMAIN_SOURCES if d in f["per_dom"])
        print(f"{label:30s} | {n:4d} | {f['EQ5']:8.4f} | {r['EQ5']:8.4f} | {f['CRPS']:8.4f} | "
              f"{r['CRPS']:8.4f} | {pd_str}")
        row = {"run": label, "n": n, "dropped": dr,
               "EQ5_full": f["EQ5"], "EQ5_roi": r["EQ5"],
               "pooled_full": f["CRPS"], "pooled_roi": r["CRPS"],
               "full_PCC": f["PCC"], "roi_PCC": r["PCC"]}
        for dom, v in f["per_dom"].items():
            row[f"{dom}_full"], row[f"{dom}_roi"], row[f"{dom}_n"] = v["full"], v["roi"], v["n"]
        rows.append(row)
    if args.out_csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
