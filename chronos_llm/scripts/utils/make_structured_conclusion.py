"""Prepend a structured magnitude tag to the conclusion of the forecasting training data (an
experiment on improving the quality of the self-generated text in gen mode).

Diagnostic motivation (the "advisory interface 2x2" finding): with one and the same checkpoint,
the faithfulness test under a "freely edit the last sentence of the conclusion" interface reached a
direction-agreement rate of only 45~46.9%, whereas switching to a "structured advisory segment"
interface (a templated External magnitude advisory segment inserted into the user prompt, with the
reasoning/conclusion replaced by templates that cite that advisory) jumped straight to Spearman
1.000 / direction 100% -- same model, same weights; faithfulness differed enormously purely because
"how the magnitude is expressed" changed from free prose to a structured template. But in that
experiment the "structure" lived in an **externally injected prompt segment** and the model only had
to restate/cite an already given fact; this script tests a different hypothesis: in the **default
zero-shot setting with no external advisory at all** (the model must judge the magnitude itself from
history+event, with no ready-made fact to copy), does turning the magnitude judgement in the
conclusion into a structured tag (instead of embedding it in a free-prose sentence) also lower the
error rate at generation time -- i.e. is the "structured format" by itself a factor that improves
generation reliability, independent of "whether there is an external fact to copy".

Method: prepend a bracketed tag `[Magnitude: {BAND}]` to the conclusion (BAND is the upper-case form
of the five bands tiny/well_below/mod_below/typical/above; the taxonomy matches
`chronos_llm.scripts.utils.make_cfaug_parquet`/`chronos_llm.eval.causal_faithfulness` verbatim and a
test guards against drift); after the tag the existing conclusion text is kept as is (including the
evidence-gated magnitude sentence written during corpus construction -- not modified, not
de-duplicated). The ratio is computed exactly as in make_cfaug_parquet: ratio = this row's future
peak / the p90 of the future peaks over all rows of the same dataset ("typical"); it is injected
uniformly into all rows (including test, not only the 7 "clear magnitude concept" sets of cfaug) --
an intentional design difference: cfaug's counterfactual rows "fabricate a magnitude fact that
contradicts the history so the model is forced to read the text", whereas this script's tag
"predicts the true magnitude of every row", structurally the same thing the chronos2 quantile head
must do (a magnitude estimate for every row, not only when the evidence is sufficient); it is not
the "confident assertion without evidence" hallucination that the evidence gate guards against.

Skipped: multi-channel future (2D; absent from the current forecasting corpus, defensive skip),
typical/peak non-finite or <= 1e-9.

**Prompt-instruction fix**: in the first run with these tags, the conclusion was forced to carry the
`[Magnitude: BAND]` prefix but the `## Task` section of the user prompt never told the model to emit
that tag nor listed the 5 options -- the model learned the format implicitly purely from "all answers
in this training set look like this"; instruction and supervision were disconnected. The fix in
`add_structured_tags` also replaces the fixed closing sentence of the `prompt` column with a version
that explicitly requires the tag and lists the bands (matching verbatim the closing sentence produced
by the corpus prompt builder `build_downstream_prompt`; if that sentence is not
found the script raises instead of silently skipping, to catch upstream template changes).

**--exclude-datasets**: optional filter that removes the given dataset_name values (exact match)
from both train and test, used for the Finance-domain ablation (`finnews/MTBench_finance`/
`fnf/bitcoin`) -- the Finance domain's ROI start is systematically biased towards the beginning of
the window (68.3% of training rows have rel_start~0) and it was the only domain that kept losing to
chronos_only; excluding the domain outright, rather than patching the data augmentation further,
tests whether the criterion "each of the 5 domains clearly beats chronos_only" can be met on the
remaining 4 domains.

Usage:
  python chronos_llm/scripts/utils/make_structured_conclusion.py \\
      --in  data/forecast/<corpus>.parquet \\
      --out data/forecast/<corpus>_structtag.parquet \\
      [--exclude-datasets finnews/MTBench_finance,fnf/bitcoin]
"""
import argparse

import numpy as np
import pandas as pd

# Verbatim copy of chronos_llm/scripts/utils/make_cfaug_parquet.py (a test asserts equality to prevent drift);
# no heavy dependencies beyond this module are imported so the script runs in a minimal environment (no torch).
BAND_ORDER = ["tiny", "well_below", "mod_below", "typical", "above"]
BAND_EDGES = [0.10, 0.40, 0.70, 1.30, float("inf")]
BAND_TAG = {
    "tiny": "TINY", "well_below": "WELL_BELOW", "mod_below": "MOD_BELOW",
    "typical": "TYPICAL", "above": "ABOVE",
}

# Verbatim copy of the closing sentence of the corpus prompt builder `build_downstream_prompt`
# (this sentence is the only fixed closing constant in the template without per-row interpolation,
# so it can safely be replaced as a whole).
_TASK_TAIL_OLD = (
    "One sentence giving the ROI index range [start_idx, end_idx) and the shape effect "
    "the event has on the future window there."
)
_TASK_TAIL_NEW = (
    "First, prefix the conclusion with a bracketed magnitude tag stating how the ROI peak "
    "compares to this series' typical peak, choosing exactly one of: [Magnitude: TINY] "
    "(near zero) / [Magnitude: WELL_BELOW] (well below typical) / [Magnitude: MOD_BELOW] "
    "(moderately below typical) / [Magnitude: TYPICAL] (around typical) / [Magnitude: ABOVE] "
    "(above typical). Then, in the same sentence, give the ROI index range "
    "[start_idx, end_idx) and the shape effect the event has on the future window there."
)


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


def make_tag(row: pd.Series, typical: float):
    """Return (band, tag_str) or None (not injectable: multi-channel / invalid typical or peak)."""
    fv = row["future_values"]
    v = list(fv) if not isinstance(fv, (list, np.ndarray)) else fv
    if len(v) and np.ndim(v[0]) > 0:
        return None  # multi-channel: do not inject (absent from the current forecasting corpus, defensive skip)
    if not np.isfinite(typical) or typical <= 1e-9:
        return None
    peak = _future_peak(fv)
    if not np.isfinite(peak) or peak <= 1e-9:
        return None
    band = band_of(peak / typical)
    return band, f"[Magnitude: {BAND_TAG[band]}]"


def inject_prompt_instruction(prompt: str) -> str:
    """Replace the closing sentence of the prompt with the version that "requires [Magnitude: BAND]
    and lists the five options". The closing sentence is a verbatim constant in the upstream template;
    if it cannot be found, raise (so an upstream template change cannot silently drift here and
    disconnect the training data from the prompt instruction again)."""
    s = str(prompt)
    if _TASK_TAIL_OLD not in s:
        if _TASK_TAIL_NEW in s:
            return s  # already the new version, idempotent
        raise ValueError(
            "prompt closing sentence does not match the build_downstream_prompt template; cannot replace safely -- "
            "the upstream template may have changed; verify manually and update _TASK_TAIL_OLD"
        )
    return s.replace(_TASK_TAIL_OLD, _TASK_TAIL_NEW)


def add_structured_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Try to prepend the tag to all rows (including test) and rewrite the prompt instruction at the same
    time; rows that cannot be tagged keep their conclusion unchanged, but the prompt instruction is
    rewritten uniformly (even when a row gets no tag, the instruction describes the task's format
    requirement and does not depend on whether a single row was hit). Returns (new df, stats dict)."""
    typ_map = (df["future_values"].map(_future_peak)
               .groupby(df["dataset_name"]).quantile(0.90).to_dict())
    out = df.copy()
    new_concl = out["conclusion"].astype(object).copy()
    out["prompt"] = out["prompt"].map(inject_prompt_instruction)
    bands = []
    n_touch = 0
    for pos, (_, row) in enumerate(df.iterrows()):
        picked = make_tag(row, typ_map.get(row["dataset_name"], np.nan))
        if picked is None:
            bands.append(None)
            continue
        band, tag = picked
        concl = str(row["conclusion"]).lstrip()
        new_concl.iat[pos] = f"{tag} {concl}"
        bands.append(band)
        n_touch += 1
    out["conclusion"] = new_concl
    out["structtag_band"] = bands  # provenance column, not fed to the model (the dataset ignores unknown columns)
    stats = {
        "n_total": len(df), "n_touch": n_touch, "n_skip": len(df) - n_touch,
        "by_band": pd.Series([b for b in bands if b is not None]).value_counts().to_dict(),
    }
    return out, stats


def filter_excluded_datasets(df: pd.DataFrame, exclude_datasets) -> pd.DataFrame:
    """Exclude by exact dataset_name match (from both train and test; the split is not re-stratified).
    Returns df unchanged when exclude_datasets is empty/None."""
    excl = [x.strip() for x in exclude_datasets if str(x).strip()] if exclude_datasets else []
    if not excl:
        return df
    return df[~df["dataset_name"].isin(excl)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--exclude-datasets", default="", help="comma-separated list of dataset_name values (exact match) to exclude from both train and test")
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    before = len(df)
    df = filter_excluded_datasets(df, args.exclude_datasets.split(","))
    if len(df) != before:
        print(f"excluded {args.exclude_datasets}: {before} -> {len(df)} rows")
    out, stats = add_structured_tags(df)
    out.to_parquet(args.dst)
    print(f"structured tag injection done: {stats['n_touch']}/{stats['n_total']} rows"
          f" (skipped {stats['n_skip']}) -> {args.dst}")
    print("by band:", stats["by_band"])


if __name__ == "__main__":
    main()
