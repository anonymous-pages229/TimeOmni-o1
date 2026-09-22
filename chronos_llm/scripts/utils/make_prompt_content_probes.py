"""Build two "text-content" counterfactual evaluation sets for the FULL arm (checkpoints that consume the
`prompt` column and generate `<think>...</think>` + conclusion). Only the text is changed (the three
columns `background`/`event`/`prompt`); history_values/future_values/roi_* are untouched -- the forecast
is still scored against the real values, and the only variable is whether the text fed to the LLM is
correct / present at all.

**Important pitfall (an earlier version of this script produced invalid results because of it)**: the
user text that `ForecastParquetDataset._user_text()` (`chronos_llm/data/forecast_dataset.py`) actually
feeds to the model was assembled as ``background column + "\\n\\nEvent: " + event column + "\\n\\n" +
prompt column`` -- **`background`/`event` are read as two separate columns, independent of the `prompt`
string**. The `prompt` column also embeds a verbatim copy of "## Dataset background/## Event
(established fact)" (so that its own template is self-contained), but that copy is only a repetition;
what the model actually sees is decided by the two separate columns plus the concatenation order. The
earlier version only edited the copy embedded in the `prompt` string and did not update the
`background`/`event` columns themselves -- so the model still read the **real, unmodified** background
and event from the first two segments assembled by `_user_text()`, and the edit inside `prompt` was
merely a later repetition with no effect (the damage/information had already been read). **All three
columns (`background`/`event`/`prompt`) are now changed together, and a self-check that replicates the
`_user_text()` assembly verifies that the old content really disappears from "the text actually fed to
the model", not just from the single `prompt` column.**

Two modes:
  shuffle: replace only the event (the `event` column + the copy embedded in `prompt`) with the event of
    another row of the same dataset group (cyclic shift by 1 within the group); background/series
    metadata stay as they are -- the single variable is "is the text content related to the real
    situation of that day"; intended for **normal generation** evaluation (the model reasons over the
    wrong event and feeds it back).
  noinfo: replace background + event (the `background`/`event` columns + both copies embedded in
    `prompt`) with fixed placeholder sentences; series metadata stay as they are -- for use with
    ``--fb_mask_generated`` (skip generation, feed back only the prompt prefix) or with normal
    generation, to measure "what level the model degrades to without any content signal".

``--splits`` (default ``test``, comma-separated) controls which splits are rewritten: by default only the
test split changes, producing an **inference-time counterfactual probe** (wrong/empty text temporarily
fed to an already-trained checkpoint); pass ``train,test`` to rewrite the training set as well, producing
a dataset that **can be fed to train_forecast.sh for training from scratch** ("train in its own setting"
-- the model never sees genuinely related text, and the train/test text distributions are consistent).
With several splits, shifting/replacement is grouped by ``(dataset_name, split)`` and the donor never
crosses splits (a train row's donor is another row of the same split); with splits={"test"} the
behaviour is element-wise identical to the old version.

Usage:
  python chronos_llm/scripts/utils/make_prompt_content_probes.py --mode shuffle \\
      --in data/forecast/mmtr_forecast_corpus.parquet --out data/forecast/mmtr_forecast_corpus_eventshuffled.parquet
  python chronos_llm/scripts/utils/make_prompt_content_probes.py --mode noinfo \\
      --in data/forecast/mmtr_forecast_corpus.parquet --out data/forecast/mmtr_forecast_corpus_noinfo.parquet
  # rewrite train + test (for training from scratch):
  python chronos_llm/scripts/utils/make_prompt_content_probes.py --mode shuffle --splits train,test \\
      --in data/forecast/mmtr_forecast_corpus.parquet --out data/forecast/mmtr_forecast_corpus_eventshuffled_traintest.parquet
"""
import argparse

import pandas as pd

NOINFO_BACKGROUND = "No background information is available for this series."
NOINFO_EVENT = "No event information is available for this time range."


def effective_user_text(background: str, event: str, prompt: str) -> str:
    """Replicate the assembly logic of ForecastParquetDataset._user_text() (without the forecast_prompt_only
    branch, which the FULL-arm checkpoints do not use) -- a self-check that the script's edits really affect
    "the text actually fed to the model", instead of changing one column that the assembly order then bypasses."""
    parts = [background, "Event: " + event, prompt]
    return "\n\n".join(p for p in parts if p)


def _anchor_replace_prompt(prompt: str, old_background: str, old_event: str,
                           new_background: str, new_event: str) -> str:
    """Replace the background/event text embedded in the prompt string with the new text (this is the copy
    repeated inside the prompt's own template; kept consistent with the edits to the background/event columns.
    It is not the path that makes this script effective, but it avoids the two copies contradicting each other);
    before replacing, assert that both anchors occur exactly once in this prompt."""
    assert prompt.count(old_background) == 1, "background anchor is missing or not unique; the template may have drifted"
    assert prompt.count(old_event) == 1, "event anchor is missing or not unique; the template may have drifted"
    out = prompt.replace(old_background, new_background, 1)
    out = out.replace(old_event, new_event, 1)
    return out


def shuffle_event(df: pd.DataFrame, splits: set) -> tuple[pd.DataFrame, dict]:
    """Swap only the event (event column + the copy embedded in prompt) by a cyclic shift of 1 within the group;
    background/metadata untouched. Grouped by (dataset_name, split), donors never cross splits; with
    splits={"test"} the result is element-wise identical to the old version."""
    out = df.copy()
    new_prompt = out["prompt"].astype(object).copy()
    new_event = out["event"].astype(object).copy()
    src_id_col = pd.Series([None] * len(out), index=out.index, dtype=object)
    mask = out["split"].isin(splits)
    n_touched = 0
    for _, group in out[mask].groupby(["dataset_name", "split"]):
        idx = group.index.tolist()
        m = len(idx)
        if m < 2:
            continue  # a single row in the group cannot be shifted (defensive skip)
        for pos, i in enumerate(idx):
            donor = idx[(pos + 1) % m]  # cyclic shift by 1; donor != i always holds (m>=2)
            donor_event = out.at[donor, "event"]
            new_prompt.at[i] = _anchor_replace_prompt(
                out.at[i, "prompt"], out.at[i, "background"], out.at[i, "event"],
                out.at[i, "background"], donor_event)
            new_event.at[i] = donor_event
            src_id_col.at[i] = out.at[donor, "id"]
            n_touched += 1
    out["prompt"] = new_prompt
    out["event"] = new_event
    out["event_shuffled_from_id"] = src_id_col  # provenance column, not fed to the model
    return out, {"n_masked_rows": int(mask.sum()), "n_touched": n_touched}


def noinfo(df: pd.DataFrame, splits: set) -> tuple[pd.DataFrame, dict]:
    """Replace background + event (both columns + both copies embedded in prompt) with fixed placeholder
    sentences; series metadata untouched."""
    out = df.copy()
    new_prompt = out["prompt"].astype(object).copy()
    new_background = out["background"].astype(object).copy()
    new_event = out["event"].astype(object).copy()
    mask = out["split"].isin(splits)
    n_touched = 0
    for i in out[mask].index:
        new_prompt.at[i] = _anchor_replace_prompt(
            out.at[i, "prompt"], out.at[i, "background"], out.at[i, "event"],
            NOINFO_BACKGROUND, NOINFO_EVENT)
        new_background.at[i] = NOINFO_BACKGROUND
        new_event.at[i] = NOINFO_EVENT
        n_touched += 1
    out["prompt"] = new_prompt
    out["background"] = new_background
    out["event"] = new_event
    return out, {"n_masked_rows": int(mask.sum()), "n_touched": n_touched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["shuffle", "noinfo"])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--splits", default="test",
                     help="Comma-separated splits to rewrite (default test = the inference-time probe behaviour; "
                          "train,test = rewrite the training set too, producing a dataset for training from scratch)")
    args = ap.parse_args()
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    df = pd.read_parquet(args.src)
    fn = shuffle_event if args.mode == "shuffle" else noinfo
    out, stats = fn(df, splits)

    # Defensive check 1: rows marked as changed really have a different prompt string.
    # The tolerance adapts to the split size: at train scale (thousands of rows) some sources (e.g. the
    # "weekend conditions" of CGTSF/MSPG, the "light snow" of Canada_photovoltaics_plants) have 5%~26% verbatim
    # duplicate event texts (reused weather templates), so a cyclic shift by 1 hits a donor with identical text
    # far more often than in test-only mode (444 rows, where 2% suffices); the 2%/6% thresholds are only a
    # coarse "most rows really changed" filter, not a precise metric.
    touched_rows = out[out["split"].isin(splits)]
    n_changed = sum(1 for i in touched_rows.index if out.at[i, "prompt"] != df.at[i, "prompt"])
    n_coincident = stats["n_touched"] - n_changed
    tol_ratio = 0.02 if splits == {"test"} else 0.06
    assert n_coincident <= max(5, stats["n_touched"] * tol_ratio), (
        f"Claimed to change {stats['n_touched']} rows, but only {n_changed} prompt strings actually changed "
        f"({n_coincident} rows coincidentally identical, above the {tol_ratio:.0%} tolerance) -- check the replacement logic first."
    )
    if n_coincident:
        print(f"Note: {n_coincident} rows have donor text identical to the original by coincidence (the donor row index does differ; not a bug).")

    # Defensive check 2 (the core of the fix): replicate the _user_text() assembly and confirm that the old
    # content really disappears from "the text actually fed to the model" (new_eff) -- not merely from one
    # column that the assembly order bypasses (which was exactly the earlier bug: only the copy embedded in the
    # prompt string was changed while the background/event columns stayed intact and still leaked at the front
    # of the assembled text).
    n_leaked = 0
    for i in touched_rows.index:
        new_eff = effective_user_text(out.at[i, "background"], out.at[i, "event"], out.at[i, "prompt"])
        old_background, old_event = df.at[i, "background"], df.at[i, "event"]
        if args.mode == "noinfo":
            leaked = (old_background and old_background in new_eff) or (old_event and old_event in new_eff)
        else:  # shuffle: the new assembled text must no longer contain the row's own original event (background unchanged, not checked)
            leaked = out.at[i, "event_shuffled_from_id"] is not None and old_event and old_event in new_eff
        n_leaked += int(bool(leaked))
    assert n_leaked <= max(5, stats["n_touched"] * tol_ratio), (
        f"Self-check failed: the old content is still present in the assembled model-input text of {n_leaked} rows -- "
        f"the edit is bypassed by the background/event/prompt assembly order and is not really in effect."
    )

    out.to_parquet(args.dst)
    print(f"[{args.mode}] splits={sorted(splits)} {stats['n_touched']}/{stats['n_masked_rows']} rows rewritten "
          f"(background/event columns + the copy embedded in prompt changed together, assembled-text self-check passed) -> {args.dst}")


if __name__ == "__main__":
    main()
