"""Controllability experiment: generate the "magnitude-band sweep" edited-variant parquets.

For every conclusion in the test split: strip the existing "Overall, ..." magnitude sentence (if
any), then uniformly append the generic magnitude sentence of the requested band (the first wording
of each corpus magnitude band, not tied to an evidence cue -- the intervention must be
decoupled from the event text so that "the forecast follows the instruction, not the event" can be
attributed). The reasoning is left untouched (the injection never touched it, so the edited
conditioning text stays self-consistent). 5 band variants + the original reference = 6 parquets
(test rows only; train rows are dropped).

Afterwards run infer_forecast --mode teacher_forced on each variant to feed back the forecasts, and
eval_controllability_metrics.py quantifies: is the forecast peak ratio monotone in the claimed band
(Spearman), and the above-vs-below direction agreement rate.

Usage: python make_controllability_parquets.py --in corpus.parquet --out-dir OUT
"""
import argparse
import os
import re

import pandas as pd

from chronos_llm.scripts.utils.make_structured_conclusion import BAND_TAG

# band -> generic magnitude sentence (aligned with the corpus magnitude bands / rl.reward keywords)
BANDS = {
    "tiny": "the peak reaches only a small fraction of the series' typical high level",
    "well_below": "the peak runs well below the series' typical highs",
    "mod_below": "the peak sits moderately below the series' typical highs",
    "typical": "the peak sits close to the series' typical high level",
    "above": "the peak runs well above the series' typical highs",
}
_OVERALL_RE = re.compile(r"\s*Overall,.*$", re.S)
_MAGTAG_RE = re.compile(r"^\[Magnitude:\s*[A-Z_]+\]\s*")


def edit_conclusion(concl, clause, band=None):
    """When band is not None, also replace the leading structured magnitude tag
    (make_structured_conclusion format) -- otherwise the prefix tag still shows the original band and
    contradicts the edited final sentence, making the constructed counterfactual impure (diagnosed on
    a first causal-faithfulness run on structured-tag data: with `[Magnitude: TAG]` in front of the
    conclusion and only the last sentence edited, Spearman fell from its normal level to 0.06 -- the
    evaluation had not caught up with the new data format, the model was not actually unfaithful)."""
    s = str(concl)
    prefix = ""
    m = _MAGTAG_RE.match(s)
    if m:
        s = s[m.end():]
        prefix = f"[Magnitude: {BAND_TAG[band]}] " if band is not None else m.group(0)
    base = _OVERALL_RE.sub("", s).rstrip()
    if base and base[-1] not in ".!?":
        base += "."
    return f"{prefix}{base} Overall, {clause}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    test = df[df["split"] == "test"].reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)

    ref = os.path.join(args.out_dir, "ctrl_orig.parquet")
    test.to_parquet(ref)
    print(f"orig (unedited reference, {len(test)} test rows) -> {ref}")

    n_had = int(test["conclusion"].astype(str).str.contains("Overall, ").sum())
    print(f"test rows that already contain a magnitude sentence: {n_had}/{len(test)}")

    for band, clause in BANDS.items():
        v = test.copy()
        v["conclusion"] = [edit_conclusion(c, clause, band=band) for c in v["conclusion"]]
        p = os.path.join(args.out_dir, f"ctrl_{band}.parquet")
        v.to_parquet(p)
        print(f"{band:>10} -> {p}")
    # Print one example row for a visual sanity check
    i = 0
    print("\nExample (original vs 'above' edit):")
    print("  original ...", str(test['conclusion'].iat[i])[-140:])
    print("  edited   ...", edit_conclusion(test['conclusion'].iat[i], BANDS['above'], band='above')[-140:])


if __name__ == "__main__":
    main()
