#!/usr/bin/env python3
"""Summary tables for the backbone ablation (TSFM swapped to TimesFM-3.0 / LLM swapped to a
smaller Qwen3.5-4B).

The question is not "which backbone is stronger" but **whether the coupling gain survives a
backbone swap**: every arm ships its own lower bound (TSFM-only on the forecasting side), so on
top of the absolute scores every table also reports `delta = lower bound - FULL`
(positive = attaching the LLM helps).

Usage (pure CPU, no accelerator needed)::

    python chronos_llm/scripts/utils/summarize_backbone_ablation.py --side forecast
    python chronos_llm/scripts/utils/summarize_backbone_ablation.py --side forecast --curve
    python chronos_llm/scripts/utils/summarize_backbone_ablation.py --side understanding

Run locations: the ablation runs are scanned under BACKBONE_F_DIR / BACKBONE_U_DIR (defaults
outputs/forecast_backbone_abl, outputs/understanding_backbone_abl); the pre-existing A0 runs are
given as comma-separated directories in A0_FULL_RUNS and A0_CHRONOS_ONLY_RUNS.

**The A0 reference numbers are not hard-coded**: they are recomputed on the fly from the
mid-training-evaluation npz of the corresponding runs through exactly the same rescoring path as
the ablation arms (`A0_RUNS`) -- hard-coded numbers are the easiest way to end up with a silent
mismatch whenever the convention changes (444->437 rows, pooled->equal weight, gen->teacher-forced).

Three convention choices, identical to the forecasting protocol of the paper's main tables:
  (1) the 437-row deduplicated test whitelist; (2) equal-weight arithmetic mean over the 5 domains
  (no pooling); (3) the mid-training-evaluation axis (the Overall of the main forecasting table is
  measured on that axis).

**Both the gen and the tf readings are printed**: tf (teacher-forced) skips autoregressive
generation and therefore removes the text-divergence noise introduced by the LLM's linear-attention
kernel being re-JIT-compiled for every new batch shape (that noise was measured to land almost
entirely on the Solar and Climate domains). So when a per-domain gain flips sign, only the tf
column can tell whether the feedback path itself or generation jitter is responsible.

Decision thresholds (the measured noise floors): on the forecasting side an equal-weight |delta|
must exceed 0.0217 to count as a real difference; on the understanding side ECG accuracy needs
|delta| > 1.5 points.
WARNING: a mid-training epoch cannot be used to rank arms -- the released forecasting checkpoint
itself has the shape "degrades in the middle, then recovers sharply at ep19", so the selection
point is pinned to **ep19** (the epoch used for the released checkpoint in the main table) and an
8-epoch sliding mean is printed next to it as a denoised auxiliary column.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Output roots of the ablation runs: train them with OUTPUT_DIR pointing here (see the README),
# so this scan does not pick up unrelated runs. Every run directory name carries its arm tag
# (`_timesfm3`, `_qwen3-5-4b`, `_chronosonly`), which is what `arm_of` keys on.
F_DIR = os.environ.get("BACKBONE_F_DIR", os.path.join(_REPO, "outputs/forecast_backbone_abl"))
U_DIR = os.environ.get("BACKBONE_U_DIR", os.path.join(_REPO, "outputs/understanding_backbone_abl"))

# A0 = the unmodified backbone pair (Chronos-2 + Qwen3.5-9B, i.e. the released forecasting recipe)
# and its Chronos-2-only lower bound. They are not retrained for the ablation; their numbers are
# recomputed from the mid-training evaluation of the existing runs, whose directories are passed as
# comma-separated lists. In the paper A0_FULL has two trainings of the very same recipe -- the gap
# between their ep19 scores is what defines the run-to-run noise floor.
A0_RUNS = {
    "A0_FULL": [x for x in os.environ.get("A0_FULL_RUNS", "").split(",") if x],
    "chronos_only": [x for x in os.environ.get("A0_CHRONOS_ONLY_RUNS", "").split(",") if x],
}
DOMS = ["Solar", "Load", "Traffic", "Finance", "Climate"]
# Which lower bound belongs to which FULL arm: A2 only swaps the LLM, and with no LLM attached the
# model is LLM-independent => it simply reuses A0's chronos_only run.
LOWER_OF = {"A0_FULL": "chronos_only", "A2_q4b": "chronos_only", "A1_tfm": "tfm_only"}
NOISE = 0.0217   # forecasting-side decision threshold (measured run-to-run noise floor)
A0_ECG = 52.64   # understanding side: mean of A0's two replicas, 53.33 / 51.95 (independent full 4000-row eval)
E_DIR = os.path.join(_REPO, "outputs/eval/backbone_abl")


def arm_of(dirname: str) -> str:
    if "chronosonly" in dirname:
        return "tfm_only" if "timesfm3" in dirname else "chronos_only"
    if "timesfm3" in dirname:
        return "A1_tfm"
    if "qwen3-5-4b" in dirname:
        return "A2_q4b"
    return "?"


def seed_of(dirname: str) -> str:
    m = re.search(r"_s(\d+)_", dirname)
    return m.group(1) if m else "-"


def _collect_runs(complete_only: bool = False) -> dict[str, list[str]]:
    """arm -> list of run directories (the ablation arms plus the pre-existing A0 runs).

    `complete_only`: keep only runs that reached the full 30 epochs. A disk-full incident killed
    three arms at ep20-22; **their ep19 data is valid** but they have no ep30 => without the filter
    the ep19 and ep30 tables would be averaged over different replica sets (4/3 vs 2/2) and the two
    selection points would no longer describe the same batch of runs. Turn it on for tables that
    report the paper convention.
    """
    runs: dict[str, list[str]] = {}
    for d in sorted(glob.glob(os.path.join(F_DIR, "*"))):
        if not os.path.isdir(d):
            continue
        if complete_only and not os.path.exists(
                os.path.join(d, "mid_eval", "epoch_30", "forecast_preds.npz")):
            continue
        runs.setdefault(arm_of(os.path.basename(d)), []).append(d)
    for k, v in A0_RUNS.items():
        have = runs.setdefault(k, [])
        have.extend(x for x in v if os.path.isdir(x) and x not in have)
    return runs


def _score(run: str, epoch: int, tf: bool, interp21: bool = False, _cache={}):
    """One epoch of one run -> (equal-weight full CRPS, {domain: full CRPS}); None if missing."""
    import chronos_llm.scripts.utils.rescore_forecast_dedup as R
    key = (run, epoch, tf, interp21)
    if key in _cache:
        return _cache[key]
    npz = os.path.join(run, "mid_eval", f"epoch_{epoch}",
                       f"forecast_preds{'_tf' if tf else ''}.npz")
    if not os.path.exists(npz):
        _cache[key] = None
        return None
    if not hasattr(_score, "_keep"):
        _score._keep = R.keep_ids(R.DEDUP)
    f, _, _, _ = R.rescore(npz, _score._keep, None, "", interp21=interp21)
    out = (f["EQ5"], {d: f["per_dom"][d]["full"] for d in DOMS if d in f["per_dom"]})
    _cache[key] = out
    return out


# The lower-bound arms (no LLM attached) have no tf reading at all: tf means "feed the ground-truth
# reasoning/conclusion and take its hidden states as the feedback signal", and without a feedback
# path this is literally the same forward pass as gen. Measured: `tfm_only`'s forecast_metrics.csv
# and its _tf.csv are **element-wise identical**, which confirms it. For the older A0 chronos_only
# run the two do differ (CRPS 0.1862 vs 0.1787, even n_valid_pcc differs) and the sample order of
# its tf npz does not line up with the test split (a legacy-code artefact) => gen is always used as
# the lower bound and is shared by both readings.
_NO_TF_ARMS = ("chronos_only", "tfm_only")


def _arm_scores(runs, arm, epoch, tf, interp21=False):
    """Scores of every replica of one arm at a given epoch (replicas that never got there are skipped)."""
    if arm in _NO_TF_ARMS:
        tf = False
    return [s for s in (_score(r, epoch, tf, interp21) for r in runs.get(arm, [])) if s]


def _mean_dom(scores):
    eq = st.mean(s[0] for s in scores)
    dom = {d: st.mean(s[1][d] for s in scores if d in s[1]) for d in DOMS}
    return eq, dom


def forecast_side(epoch: int, window: int, curve: bool, complete_only: bool = False,
                  interp21: bool = False, per_seed: bool = False) -> None:
    runs = _collect_runs(complete_only)

    if curve:
        print(f"\n### Full curve (437-row dedup / mid-training eval / gen / equal-weight 5-domain CRPS, full region)\n")
        cols = [(a, r) for a in ("A1_tfm", "A2_q4b", "tfm_only") for r in runs.get(a, [])]
        hdr = [f"{a}_s{seed_of(os.path.basename(r))}" for a, r in cols]
        print("| ep | " + " | ".join(hdr) + " |")
        print("|" + "---|" * (len(hdr) + 1))
        for e in range(1, 31):
            row = [(f"{s[0]:.4f}" if (s := _score(r, e, False)) else "·") for _, r in cols]
            if any(c != "·" for c in row):
                print(f"| {e} | " + " | ".join(row) + " |")

    for tf in (False, True):
        tag = "tf (teacher-forced, no generation noise)" if tf else "gen (autoregressive generation, primary reading)"
        print(f"\n## Forecasting side - {tag} - epoch {epoch} (437-row dedup / mid-training eval / equal-weight 5-domain CRPS, full region)\n")
        print("| arm | replicas | " + " | ".join(DOMS) + " | **equal-weight** |")
        print("|" + "---|" * (len(DOMS) + 3))
        agg = {}
        for arm in ("A0_FULL", "chronos_only", "A1_tfm", "tfm_only", "A2_q4b"):
            sc = _arm_scores(runs, arm, epoch, tf, interp21)
            if not sc:
                continue
            eq, dom = _mean_dom(sc)
            agg[arm] = (eq, dom)
            cells = " | ".join(f"{dom[d]:.4f}" for d in DOMS)
            print(f"| {arm} | {len(sc)} | {cells} | **{eq:.4f}** |")

        if per_seed:
            print("\n**Per-replica detail** (the table to read when picking the best replica)\n")
            print("| arm | seed | " + " | ".join(DOMS) + " | equal-weight |")
            print("|" + "---|" * (len(DOMS) + 3))
            for arm in ("A0_FULL", "chronos_only", "A1_tfm", "tfm_only", "A2_q4b"):
                for r in runs.get(arm, []):
                    sc = _score(r, epoch, False if arm in _NO_TF_ARMS else tf, interp21)
                    if not sc:
                        continue
                    eq, dom = sc
                    print(f"| {arm} | {seed_of(os.path.basename(r))} | "
                          + " | ".join(f"{dom[d]:.4f}" for d in DOMS) + f" | {eq:.4f} |")

        print(f"\n### Coupling gain delta = own TSFM-only lower bound - FULL (positive = attaching the LLM helps)\n")
        print("| arm | lower bound | " + " | ".join(f"Δ{d}" for d in DOMS) + " | **Δ equal-weight** | all domains positive? |")
        print("|" + "---|" * (len(DOMS) + 4))
        for arm in ("A0_FULL", "A1_tfm", "A2_q4b"):
            low_arm = LOWER_OF[arm]
            if arm not in agg or low_arm not in agg:
                continue
            eq, dom = agg[arm]
            leq, ldom = agg[low_arm]
            dd = {d: ldom[d] - dom[d] for d in DOMS}
            cells = " | ".join(f"{dd[d]:+.4f}" for d in DOMS)
            n_pos = sum(v > 0 for v in dd.values())
            print(f"| {arm} | {low_arm} | {cells} | **{leq - eq:+.4f}** | {n_pos}/5 |")
        print(f"\n> Decision threshold |delta equal-weight| > {NOISE} (measured run-to-run noise floor); "
              f"the stricter per-domain criterion additionally requires all 5 domains to beat the lower bound **individually**.")

    if window:
        lo, hi = epoch - window + 1, epoch
        print(f"\n### Denoising aid: sliding mean over epochs {lo}-{hi} (gen, equal weight) -- single-epoch jitter is far larger than the decision threshold, hence this extra column\n")
        print("| arm | valid epochs | sliding mean | ep%d single point |" % epoch)
        print("|---|---|---|---|")
        for arm in ("A0_FULL", "chronos_only", "A1_tfm", "tfm_only", "A2_q4b"):
            vals = [s[0] for e in range(lo, hi + 1) for s in _arm_scores(runs, arm, e, False, interp21)]
            pt = _arm_scores(runs, arm, epoch, False, interp21)
            if vals:
                print(f"| {arm} | {len(vals)} | {st.mean(vals):.4f} | "
                      f"{st.mean(s[0] for s in pt):.4f} |" if pt else
                      f"| {arm} | {len(vals)} | {st.mean(vals):.4f} | · |")


def _ecg_acc(jsonl: str) -> tuple[int, float] | None:
    from chronos_llm.eval.eval_opentslm_answer import evaluate as ecg_eval
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    if not rows:
        return None
    return len(rows), ecg_eval(rows, "ecg")["accuracy"] * 100


def understanding_eval() -> None:
    """Independent full evaluation (4000 rows) + time-series mismatch probe -- the **decisive**
    reading on the understanding side.

    The drop = orig - shuffled is a direct measure of "the model really reads the time series";
    what is compared across arms is the **drop**, not the absolute score.
    WARNING: the reference has to be the A0 run trained with the *same* recipe (cold start,
    60k samples x 4 epochs), not the earlier answer-first checkpoint whose -16.2-point drop comes
    from a different recipe and sits 12 points higher in absolute accuracy (an earlier version of
    the probe list had that one attached by mistake).
    """
    arms: dict[str, dict[str, tuple[int, float]]] = {}
    for d in sorted(glob.glob(os.path.join(E_DIR, "u_*"))):
        name = os.path.basename(d)[2:]
        kind = "probe" if name.endswith("-probe") else "orig"
        arm = name[:-6] if kind == "probe" else name
        for f in glob.glob(os.path.join(d, "infer", "*.jsonl")):
            r = _ecg_acc(f)
            if r:
                arms.setdefault(arm, {})[kind] = r

    print("\n## Understanding side, decisive reading: independent full evaluation (ECG-QA, 4000 rows) + time-series mismatch probe\n")
    print("| arm | orig | mismatch probe | **drop** |")
    print("|---|---|---|---|")
    for arm in sorted(arms):
        o, p = arms[arm].get("orig"), arms[arm].get("probe")
        so = f"{o[1]:.2f}" if o else "running"
        sp = f"{p[1]:.2f}" if p else "running"
        sd = f"**{p[1] - o[1]:+.2f}**" if (o and p) else "—"
        print(f"| {arm} | {so} | {sp} | {sd} |")
    print("\n> Decision threshold: accuracy |delta| > 1.5 points (1 sigma ~ 0.77 at n=4000). What is "
          "compared across arms is the **drop** -- drops of the same magnitude mean that "
          "'the model really reads the time series' does not depend on the particular TSFM.")


def best_epoch(interp21: bool = True, emax: int = 30, complete_only: bool = True) -> None:
    """Scan every epoch of every run and report each run's **own best epoch** (gen, smallest
    equal-weight CRPS).

    When comparing backbones it is unfair to pin everything to the ep19 selection point: that point
    was picked for the released checkpoint's configuration and is not necessarily optimal for the
    others. The fair protocol is to let every configuration pick its own best epoch.
    """
    # By default only runs that completed 30 epochs are considered: the three runs killed by the
    # disk-full incident stop at ep20-22, so their "best" is a minimum over a smaller candidate set
    # and is not comparable with a complete run (different min ranges => different selection bias).
    runs = _collect_runs(complete_only)
    print(f"\n## Best epoch per run (gen / 437-row dedup / mid-training eval / equal-weight 5-domain CRPS"
          f"{' / quantiles interpolated to 21 levels' if interp21 else ''})\n")
    print("| arm | seed | best ep | " + " | ".join(DOMS) + " | **Overall** | ep19 reference |")
    print("|" + "---|" * (len(DOMS) + 5))
    best_of_arm: dict[str, tuple] = {}
    for arm in ("A0_FULL", "chronos_only", "A1_tfm", "tfm_only", "A2_q4b"):
        for r in runs.get(arm, []):
            cur = None
            for e in range(1, emax + 1):
                sc = _score(r, e, False, interp21)
                if sc and (cur is None or sc[0] < cur[1]):
                    cur = (e, sc[0], sc[1])
            if cur is None:
                continue
            e19 = _score(r, 19, False, interp21)
            sd = f"{e19[0]:.4f}" if e19 else "·"
            print(f"| {arm} | {seed_of(os.path.basename(r))} | {cur[0]} | "
                  + " | ".join(f"{cur[2][d]:.4f}" for d in DOMS)
                  + f" | **{cur[1]:.4f}** | {sd} |")
            if arm not in best_of_arm or cur[1] < best_of_arm[arm][1]:
                best_of_arm[arm] = cur
    print("\n### Best replica x best epoch of every configuration\n")
    print("| configuration | best ep | " + " | ".join(DOMS) + " | **Overall** |")
    print("|" + "---|" * (len(DOMS) + 3))
    for arm, (e, eq, dom) in sorted(best_of_arm.items()):
        print(f"| {arm} | {e} | " + " | ".join(f"{dom[d]:.4f}" for d in DOMS) + f" | **{eq:.4f}** |")


def understanding_side() -> None:
    from chronos_llm.eval.eval_opentslm_answer import evaluate as ecg_eval

    def load(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    print("\n## Understanding side (ECG-QA; the capped mid-training subset only shows the trend, the decisive numbers come from the independent 4000-row evaluation)\n")
    print("| arm | seed | epoch | n | accuracy |")
    print("|---|---|---|---|---|")
    for d in sorted(glob.glob(os.path.join(U_DIR, "*"))):
        base = os.path.basename(d)
        arm, seed = arm_of(base), seed_of(base)
        for ep in sorted(glob.glob(os.path.join(d, "mid_eval", "epoch_*")),
                         key=lambda p: int(p.rsplit("_", 1)[1])):
            e = int(ep.rsplit("_", 1)[1])
            for f in glob.glob(os.path.join(ep, "understanding", "*ecgqa*.jsonl")):
                rows = load(f)
                if not rows:
                    continue
                acc = ecg_eval(rows, "ecg")["accuracy"]
                # evaluate() returns a 0-1 fraction while the A0 reference is a percentage;
                # multiply by 100 here so the two can never be compared at the wrong scale
                print(f"| {arm} | {seed} | {e} | {len(rows)} | {acc * 100:.2f} |")
    print(f"\n> A0 reference: mean of the two ECG-QA replicas **{A0_ECG:.2f}** (53.33 / 51.95, independent full 4000-row evaluation); decision threshold 1.5 points.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["forecast", "understanding"], required=True)
    ap.add_argument("--epoch", type=int, default=19, help="selection point (default 19 = the epoch used for the released forecasting checkpoint in the main table)")
    ap.add_argument("--window", type=int, default=8, help="width of the sliding mean, 0 = off")
    ap.add_argument("--curve", action="store_true", help="also print the full per-epoch curve")
    ap.add_argument("--complete-only", action="store_true",
                    help="only count runs that completed 30 epochs -- mandatory when ep19 and ep30 are shown side by side, otherwise the two selection points cover different replica sets")
    ap.add_argument("--interp21", action="store_true",
                    help="interpolate TimesFM's 9 quantiles onto Chronos's 21 before computing CRPS -- mandatory when comparing absolute scores across backbones")
    ap.add_argument("--per-seed", action="store_true", help="also print the per-replica detail table")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="with --best-epoch, also include runs that never reached 30 epochs (excluded by default, see the note inside the function)")
    ap.add_argument("--best-epoch", action="store_true",
                    help="scan all epochs and report each run's own best epoch instead of pinning to the ep19 selection point")
    ap.add_argument("--eval", action="store_true",
                    help="understanding side: read the independent full evaluation + mismatch probe artefacts (the decisive reading) instead of the mid-training trend")
    a = ap.parse_args()
    if a.side == "forecast" and a.best_epoch:
        best_epoch(a.interp21, complete_only=not a.include_incomplete)
    elif a.side == "forecast":
        forecast_side(a.epoch, a.window, a.curve, a.complete_only,
                      a.interp21, a.per_seed)
    elif a.eval:
        understanding_eval()
    else:
        understanding_side()
