"""Counterfactual data augmentation: produce the cfaug variant of the training parquet.

Diagnosis: in the training data the "magnitude instruction" can almost always be inferred back out of
the history => chronos learns the history->future shortcut and ignores the text fed back from the
LLM. Even after the feedback gate was fixed (tanh(gate) ~ 0.72) and scheduled sampling had squeezed
the gen-minus-tf gap down to ~0.017, the gap did not close. This script cuts the shortcut directly:
**the same history x a counterfactual magnitude instruction x a future rescaled to match that
instruction** -- the history no longer determines the future, so the magnitude can only be read out
of the text.

Self-consistency (the core of the design, i.e. why this is not just "rewrite the magnitude sentence
of the conclusion"):
  - Lesson from the earlier magnitude-injection variant: a
    magnitude sentence in the text that is not backed by evidence visible in the prompt makes the text
    CE teach hallucination (the model emits a confident band with no evidence for it). A
    counterfactual row's magnitude contradicts both the event and the history, so rewriting only the
    conclusion would reproduce that pathology at scale.
  - The original reasoning/conclusion text is full of qualitative magnitude wording bound to the
    original future ("suppressed" / "near-zero" / "peaking well above ten units"); rewriting it by
    regex is unreliable, and leftover old magnitude wording teaches the model to ignore the text --
    worse than no augmentation at all.
  - Solution: the counterfactual instruction becomes an **advisory section in the user prompt**
    ("## External magnitude advisory (established fact)"), and reasoning/conclusion are replaced by
    templates that cite that advisory (zero leftover old magnitude wording; the ROI span sentence is
    kept -- rescaling changes neither shape nor position, so the span statement stays true). This
    gives:
      (i) a text CE that is fully self-consistent and learnable (advisory -> restatement), with no
          need to mask labels or touch train.py / the model code;
      (ii) scheduled-sampling batches are safe by construction: self-generation starts from a prefix
           that contains the advisory, so the generated text restates it; and even when it does not,
           under feedback_scope=all the advisory is itself part of the hidden states fed back;
      (iii) the magnitude wording reuses the evaluation's 5-band taxonomy verbatim
            (causal_faithfulness.BAND_CLAUSES), so the training supervision and the faithfulness
            evaluation share one convention.
  - All original rows are kept as they are (augmentation only appends rows); not a single split=test
    row is modified.

Bands / rescaling: ratio = this row's future peak / the p90 of the future peaks over all rows of the
same dataset (same convention as the corpus magnitude tags); the target band is drawn deterministically by
hashing the id after excluding the current band from the 5, the target ratio is jittered uniformly
inside the band, and factor = target ratio / current ratio (clipped to [0.02, 40]; out-of-range
rotates to the next band, and a row whose every band is out of range is skipped).
Only the 7 periodic datasets with a clear notion of magnitude are augmented (photovoltaics / load /
traffic); the PCC~0 ceiling-bound ones are excluded (MTBench / bitcoin / Climate -- the established
conclusion of the magnitude-injection variant: on random-walk-like series a magnitude sentence is just noise).

Usage:
  python chronos_llm/scripts/utils/make_cfaug_parquet.py \
      --in  data/forecast/mmtr_forecast_corpus.parquet \
      --out data/forecast/mmtr_forecast_corpus_cfaug.parquet [--frac 0.5]
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

# The 5-band magnitude wording: verbatim identical to chronos_llm/eval/causal_faithfulness.BAND_CLAUSES
# (which in turn was cross-checked against the keyword families of make_controllability_parquets and
# of the RL reward). It is not imported from there: the eval module's import chain pulls in
# scipy/torch, while this script must run in a minimal environment; a test asserts that the two dicts
# are verbatim equal, guarding against drift.
BAND_CLAUSES = {
    "tiny": "the peak reaches only a small fraction of the series' typical high level",
    "well_below": "the peak runs well below the series' typical highs",
    "mod_below": "the peak sits moderately below the series' typical highs",
    "typical": "the peak sits close to the series' typical high level",
    "above": "the peak runs well above the series' typical highs",
}
BAND_ORDER = ["tiny", "well_below", "mod_below", "typical", "above"]
# Band edges match the corpus magnitude bands; the target ratio is jittered uniformly inside an "inner
# interval" of the band (staying away from the edges).
BAND_EDGES = [0.10, 0.40, 0.70, 1.30, float("inf")]
BAND_TARGET_RANGE = {
    "tiny": (0.03, 0.08),
    "well_below": (0.15, 0.35),
    "mod_below": (0.45, 0.65),
    "typical": (0.80, 1.20),
    "above": (1.40, 2.00),
}
# Periodic datasets with a clear notion of magnitude (they have a "typical peak"); finnews/MTBench_finance,
# fnf/bitcoin and timemmd/Climate are excluded (PCC~0 ceiling-bound, where a magnitude instruction is
# just noise -- the same exclusion as in the magnitude-injection variant).
ELIGIBLE_DATASETS = [
    "CGTSF/MSPG", "fidelts/Canada_photovoltaics_plants", "fidelts/Germany_Renewable_Power_Grid",
    "CGTSF/PTF", "fidelts/California_ISO", "fnf/load", "fnf/traffic",
]

ADVISORY_SECTION = (
    "## External magnitude advisory (established fact)\n"
    "An external planning advisory for the forecast window indicates that {clause}. Treat this "
    "advisory as an established fact that supersedes whatever peak magnitude the event or the "
    "recent history would otherwise imply; the temporal shape of the future window still follows "
    "from the event and history as usual."
)
REASONING_TMPL = (
    "The ROI covers indices [{rs}, {re}) of the overall series, lining up with the event's target "
    "window as usual. The history window establishes the series' habitual daily shape and typical "
    "peak level, and the event describes the conditions of the day, but the external magnitude "
    "advisory supersedes the peak magnitude they would imply. The future window therefore keeps "
    "its expected temporal shape while its amplitude is set so that {clause}."
)
CONCLUSION_TMPL = (
    "In the span [{rs}, {re}), the profile keeps its expected temporal shape at the "
    "advisory-set amplitude. Overall, {clause}."
)


def _hash_u(key: str) -> float:
    """Deterministic uniform number in [0,1) (hash of the key; no global RNG, so it is reproducible
    and independent of row order)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10**8 / 10**8


def _target_series(v):
    v = list(v) if not isinstance(v, (list, np.ndarray)) else v
    if len(v) and np.ndim(v[0]) > 0:
        return np.asarray(v[0], dtype=float)
    return np.asarray(v, dtype=float)


def _future_peak(v):
    a = _target_series(v)
    a = a[np.isfinite(a)]
    return float(np.nanmax(a)) if a.size else np.nan


def band_of(ratio: float) -> str:
    for name, hi in zip(BAND_ORDER, BAND_EDGES):
        if ratio < hi:
            return name
    return BAND_ORDER[-1]


def pick_counterfactual(ratio: float, rid: str, min_factor=0.02, max_factor=40.0):
    """Deterministically pick a counterfactual band: exclude the current one, try the rest in a
    hash-determined rotation, move on to the next band whenever the factor falls out of range, and
    return None if every band is out of range. Returns (band, target_ratio, factor)."""
    cur = band_of(ratio)
    cands = [b for b in BAND_ORDER if b != cur]
    start = int(_hash_u(rid + "#band") * len(cands))
    for k in range(len(cands)):
        band = cands[(start + k) % len(cands)]
        lo, hi = BAND_TARGET_RANGE[band]
        tgt = lo + (hi - lo) * _hash_u(rid + "#jit#" + band)
        factor = tgt / ratio
        if min_factor <= factor <= max_factor:
            return band, tgt, factor
    return None


def make_cf_row(row: pd.Series, typical: float, min_factor=0.02, max_factor=40.0):
    """Turn one original row into a counterfactual row (dict); return None when it cannot be augmented
    (invalid ratio / every factor out of range / missing span).
    Only a single-channel future is supported (the current forecasting corpus is entirely 1D; a
    multi-channel corpus would first need target-row selection logic)."""
    fv = row["future_values"]
    if len(fv) and np.ndim(fv[0]) > 0:
        return None  # multi-channel: not augmented (avoids rescaling covariate rows by mistake)
    if not np.isfinite(typical) or typical <= 1e-9:
        return None
    peak = _future_peak(fv)
    if not np.isfinite(peak) or peak <= 1e-9:
        return None
    ratio = peak / typical
    picked = pick_counterfactual(ratio, str(row["id"]), min_factor, max_factor)
    if picked is None:
        return None
    band, tgt, factor = picked

    def _safe_int(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return int(f) if np.isfinite(f) else None

    rs, re = _safe_int(row.get("roi_start_idx")), _safe_int(row.get("roi_end_idx"))
    if rs is None or re is None or re <= rs:
        pl, fl = _safe_int(row.get("past_len")), _safe_int(row.get("future_len"))
        if pl is None or fl is None:
            return None
        # No ROI: the span degenerates to the whole future window (absolute coordinates, as
        # elsewhere in the corpus).
        rs, re = pl, pl + fl

    clause = BAND_CLAUSES[band]
    new = row.to_dict()
    new["id"] = f"{row['id']}#cf_{band}"
    scaled = np.asarray(fv, dtype=np.float32) * np.float32(factor)
    new["future_values"] = scaled.tolist()  # NaN*f=NaN, so the padding semantics are preserved
    prompt = str(row.get("prompt") or "")
    sec = ADVISORY_SECTION.format(clause=clause)
    marker = "\n## Task"
    # Insert before ## Task (matching the existing section style); append at the end if absent.
    if marker in prompt:
        i = prompt.index(marker)
        new["prompt"] = prompt[:i] + "\n" + sec + prompt[i:]
    else:
        new["prompt"] = prompt.rstrip() + "\n\n" + sec
    pp = row.get("plain_prompt")
    if pp is not None and str(pp).strip():
        new["plain_prompt"] = (str(pp).rstrip() + " An external planning advisory for the "
                               f"forecast window, treated as established fact, indicates that {clause}.")
    new["reasoning"] = REASONING_TMPL.format(rs=rs, re=re, clause=clause)
    new["conclusion"] = CONCLUSION_TMPL.format(rs=rs, re=re, clause=clause)
    # Provenance columns (empty on the original rows): they never reach the model (the dataset does
    # not read unknown columns) and exist only for tests / diagnostics.
    new["cf_band"], new["cf_scale"], new["cf_src_id"] = band, float(factor), str(row["id"])
    return new


def augment(df: pd.DataFrame, frac=0.5, salt="cfaug1", min_factor=0.02, max_factor=40.0):
    """Return (augmented df, stats dict). The original rows (including every test row) are kept as
    they are; only counterfactual rows are appended."""
    typ_map = (df["future_values"].map(_future_peak)
               .groupby(df["dataset_name"]).quantile(0.90).to_dict())
    cf_rows, n_skip = [], 0
    for _, row in df.iterrows():
        if row.get("split") != "train" or str(row["dataset_name"]) not in ELIGIBLE_DATASETS:
            continue
        if _hash_u(f"{salt}#{row['id']}#sel") >= frac:
            continue
        cf = make_cf_row(row, typ_map.get(row["dataset_name"], np.nan), min_factor, max_factor)
        if cf is None:
            n_skip += 1
            continue
        cf_rows.append(cf)
    out = pd.concat([df, pd.DataFrame(cf_rows)], ignore_index=True) if cf_rows else df.copy()
    stats = {
        "n_orig": len(df), "n_cf": len(cf_rows), "n_skip": n_skip,
        "by_band": pd.Series([r["cf_band"] for r in cf_rows]).value_counts().to_dict(),
        "by_ds": pd.Series([r["dataset_name"] for r in cf_rows]).value_counts().to_dict(),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--frac", type=float, default=0.5,
                    help="fraction of the eligible train rows that get augmented")
    ap.add_argument("--salt", default="cfaug1",
                    help="hash salt for row/band selection (a different salt = a different "
                         "augmentation set)")
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    out, stats = augment(df, frac=args.frac, salt=args.salt)
    out.to_parquet(args.dst)
    print(f"augmentation done: {stats['n_orig']} original rows + {stats['n_cf']} counterfactual rows"
          f" ({stats['n_skip']} skipped) -> {args.dst}")
    print("by band:", stats["by_band"])
    print("by dataset:", stats["by_ds"])


if __name__ == "__main__":
    main()
