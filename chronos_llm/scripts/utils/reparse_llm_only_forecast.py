"""Offline re-parsing of LLM-only forecasting evaluation outputs (repairs existing results affected
by a batch-level horizon parsing bug).

Background: `_generate_forecast_llm_only` used to call `parse_forecast_values` with one and the same
`horizon` for the whole batch (= the max FL within the batch after the collator's right NaN padding)
-- in mixed batches, samples with a short fl were required to write more values than the task itself
asked for and were systematically misjudged as "parse failures" (most of the failures in an
independent evaluation of a fine-tuned checkpoint were such pseudo-failures; the zero-shot success
rate was depressed the same way). The generated text itself is unaffected (the n=fl in the prompt
instruction was always correct per sample), so it can be re-parsed offline from gen_text in the jsonl
without rerunning GPU inference.

For every evaluation directory (must contain forecast_preds.jsonl + forecast_preds.npz):
  1. per row, take the true horizon = position of the last True in the npz valid_mask row + 1
     (equivalent to the trimmed length of gt_series in the jsonl; genuine NaN gaps inside the row stay
     within the length);
  2. `parse_forecast_values(gen_text, true h)` rebuilds the pred_quantiles row (the point forecast fills
     all 21 quantiles; on failure the whole row stays NaN -- the protocol itself is unchanged, only the
     horizon convention is fixed);
  3. overwrite the npz and the jsonl's median_pred field in place and print the success counts before/after.
Afterwards rerun `eval_forecast.py --pred <npz>` to refresh forecast_metrics.csv (this script does not
do it; pure CPU, seconds).

Usage:
  PYTHONPATH=. python chronos_llm/scripts/utils/reparse_llm_only_forecast.py \
      outputs/eval/llm_only_ft_forecast_ep20 [outputs/eval/llm_only_zeroshot_forecast ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from chronos_llm.data.ts_text import parse_forecast_values  # noqa: E402


def reparse_dir(d: str) -> None:
    npz_path = os.path.join(d, "forecast_preds.npz")
    jsonl_path = os.path.join(d, "forecast_preds.jsonl")
    with np.load(npz_path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    rows = [json.loads(l) for l in open(jsonl_path)]
    P, V, ids = data["pred_quantiles"], data["valid_mask"], data["ids"]
    assert len(rows) == P.shape[0], f"{d}: jsonl row count {len(rows)} != npz {P.shape[0]}"
    for i, r in enumerate(rows):
        assert str(r["id"]) == str(ids[i]), f"{d} row {i} id misaligned: {r['id']} vs {ids[i]}"
    Q = P.shape[1]
    # success = all finite on the valid positions (same sr definition as paper_forecast_success_rate.py)
    before = sum(int(V[i].any() and np.isfinite(P[i][:, V[i]]).all()) for i in range(P.shape[0]))
    ok = 0
    for i, r in enumerate(rows):
        nz = np.nonzero(V[i])[0]
        if nz.size == 0:
            continue
        h = int(nz.max()) + 1
        P[i] = np.nan
        vals = parse_forecast_values(r.get("gen_text") or "", h)
        if vals is not None:
            P[i, :, :h] = np.broadcast_to(vals, (Q, h))
            ok += 1
            r["median_pred"] = [float(v) for v in vals]
        else:
            r["median_pred"] = [float("nan")] * h
    data["pred_quantiles"] = P
    np.savez_compressed(npz_path, **data)
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{d}: successfully parsed {before} -> {ok} / {len(rows)}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        reparse_dir(d)
