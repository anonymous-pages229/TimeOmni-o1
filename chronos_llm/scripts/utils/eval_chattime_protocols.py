"""Settle the metric protocol once and for all: take ChatTime's own forecasts on the CGTSF test split (output of
run_chattime_cgtsf.py), compute MAE under every candidate normalisation protocol, and compare with its own two
rows in the paper's Table 5 (ChatTime / ChatTime-).

Logic: same model, same data, same official inference code => the only unknown is the MAE normalisation
protocol. The protocol that **simultaneously** reproduces the Table 5 numbers for all three datasets x two
variants (with/without text) (within ±10-15%, allowing for sub-sampling error and sampling randomness) is the
paper's true evaluation protocol. Once found, the TimeOmni-o1 / Chronos-2 zero-shot numbers under the same
protocol are emitted as well (restricted to the same sub-sample ChatTime ran on, for a head-to-head
comparison; numbers on the full test split are given too).

Usage (CPU):
  python chronos_llm/scripts/utils/eval_chattime_protocols.py \
      --chattime_jsonl outputs/eval/cgtsf/chattime/chattime_preds.jsonl \
      --out outputs/eval/cgtsf/chattime/protocol_verdict.json
"""
import argparse
import json

import numpy as np
import pandas as pd

from chronos_llm.scripts.utils.cgtsf_protocol_probe import (
    DATASETS, EVAL_PARQUET, NPZ, TABLE5_REF, candidate_maes, station_train_stats,
)


def load_chattime(jsonl_path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    return pd.DataFrame(rows)


def model_rows_from_npz(path: str) -> dict:
    """npz -> {id: (median_pred, gt, valid)}."""
    npz = np.load(path, allow_pickle=True)
    qi = int(np.argmin(np.abs(np.asarray(npz["quantile_levels"]) - 0.5)))
    med = npz["pred_quantiles"][:, qi, :]
    gt = npz["gt"]
    valid = np.asarray(npz["valid_mask"]) & np.isfinite(gt)
    return {str(i): (med[k], gt[k], valid[k]) for k, i in enumerate(npz["ids"])}


def eval_subset(ids, preds, gts, hists, ds, stats) -> dict:
    """Compute the MAE under every candidate protocol for a subset of rows (within one dataset). preds/gts have variable length -> padded to a common length."""
    fl = max(len(g) for g in gts)
    P = np.full((len(ids), fl), np.nan)
    G = np.full((len(ids), fl), np.nan)
    for k, (p, g) in enumerate(zip(preds, gts)):
        P[k, :len(p)] = p
        G[k, :len(g)] = g
    valid = np.isfinite(G) & np.isfinite(P)
    idxs = [int(i.split("__")[1]) for i in ids]
    st_std = np.array([stats[(ds, i)]["std"] for i in idxs])
    st_rng = np.array([stats[(ds, i)]["range"] for i in idxs])
    g_std = np.full(len(ids), stats[(ds, None)]["std"])
    # Failed rows (all-NaN forecast) are counted before being honestly excluded from the mean
    n_fail = int((~valid.any(axis=1)).sum())
    keep = valid.any(axis=1)
    out = candidate_maes(P[keep], G[keep], [hists[k] for k in np.where(keep)[0]],
                         st_std[keep], st_rng[keep], g_std[keep], valid[keep])
    out["n"] = int(keep.sum())
    out["n_fail"] = n_fail
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chattime_jsonl", required=True)
    ap.add_argument("--eval_parquet", default=EVAL_PARQUET)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = station_train_stats()
    pq = pd.read_parquet(args.eval_parquet)
    id_to_hist = {r["id"]: np.asarray(r["history_values"], dtype=np.float64)
                  for _, r in pq.iterrows()}

    ct = load_chattime(args.chattime_jsonl)
    report = {}
    for ds in DATASETS:
        sub = ct[ct["dataset"] == f"CGTSF/{ds}"]
        ids = sub["id"].tolist()
        hists = [id_to_hist[i] for i in ids]
        gts = [np.asarray(g, dtype=np.float64) for g in sub["gt"]]
        entry = {"table5_ref": TABLE5_REF[ds], "n_subsample": len(ids)}
        for variant, col in [("chattime_text", "pred_text"), ("chattime_plain", "pred_plain")]:
            preds = [np.asarray(p, dtype=np.float64) for p in sub[col]]
            entry[variant] = eval_subset(ids, preds, gts, hists, ds, stats)
        # Our two models: restricted to the same sub-sample (head-to-head), npz aligned by id
        for model, path in NPZ.items():
            rows = model_rows_from_npz(path)
            got = [(i, rows[i]) for i in ids if i in rows]
            if not got:
                continue
            m_ids = [i for i, _ in got]
            m_pred = [r[0][r[2]] for _, r in got]   # valid positions only
            m_gt = [r[1][r[2]] for _, r in got]
            m_hist = [id_to_hist[i] for i in m_ids]
            entry[f"{model}_same_subset"] = eval_subset(m_ids, m_pred, m_gt, m_hist, ds, stats)
        report[ds] = entry

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Quick verdict: relative deviation of each protocol over ChatTime's two variants x 3 datasets
    print("\n===== protocol verdict (smaller |log(measured/Table5)| = better match) =====")
    for proto in ["raw", "minmax_window", "z_global", "z_station", "z_window", "minmax_station"]:
        devs = []
        for ds in DATASETS:
            for variant, ref_key in [("chattime_text", "ChatTime"), ("chattime_plain", "ChatTime-")]:
                got = report[ds][variant][proto]
                ref = TABLE5_REF[ds][ref_key]
                if got and np.isfinite(got) and got > 0:
                    devs.append(abs(np.log(got / ref)))
        print(f"  {proto:15s} mean |log deviation|={np.mean(devs):.3f}  (geometric-mean deviation x{np.exp(np.mean(devs)):.2f})")


if __name__ == "__main__":
    main()
