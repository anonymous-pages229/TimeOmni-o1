"""GRPO reward components (to close the gap between free generation and teacher forcing in the
forecast branch).

Background: magnitude injection raised the teacher-forced ceiling of the FULL arm, but the
**generated** conclusions are error-prone in ROI localisation (IoU only ~0.57) and in the magnitude
band, so the gen-vs-tf gap actually widened. GRPO optimises generation quality directly through a
reward.

Reward design (more direct than "ROI + shape LLM scoring"):
- **main reward = -forecast CRPS**: the forecast obtained after feeding the generated conclusion back
  into chronos vs the ground truth. Aligned with the final objective, and shape correctness is
  implied by CRPS (wrong shape -> wrong forecast -> high CRPS), so no LLM shape scoring is needed.
- **auxiliary reward = ROI IoU**: dense shaping that corrects the generated ROI localisation (the
  generated ROI only reaches ~0.57 IoU, lots of headroom).
- **auxiliary reward = magnitude-band match**: generated magnitude band vs the true band (the core
  signal of the magnitude-annotated data).

This module holds only **pure text parsing + arithmetic scoring** (testable on CPU); the CRPS main
reward is computed in the training loop from the model outputs (reusing eval.metrics.forecast_metrics).
"""
import re

# magnitude-band keywords -> ordered band (for the magnitude-match score). Wording aligned with the magnitude bands of the forecasting corpus.
_MAG_BANDS = [
    (-2, ["small fraction", "tiny fraction", "small sliver", "near-flat", "near-zero"]),
    (-1, ["well below", "far under", "strongly suppressed", "moderately below", "somewhat under"]),
    (0, []),                                    # typical / not mentioned
    (1, ["well above", "climbs above", "distinctly beyond", "surges", "spikes far", "above the"]),
]


def extract_roi(text):
    """Parse the half-open ROI interval (start, end) from generated text. Supports '[a, b)' and 'rows a through b'."""
    text = str(text)
    m = re.search(r"\[(\d+),\s*(\d+)\)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"rows?\s+(\d+)\s+through\s+(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)) + 1   # 'through' is a closed interval -> half-open
    return None


def roi_iou(a, b):
    """IoU of two half-open intervals; 0 if either is None."""
    if a is None or b is None:
        return 0.0
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    ov = max(0, hi - lo)
    un = (a[1] - a[0]) + (b[1] - b[0]) - ov
    return ov / un if un > 0 else 0.0


def magnitude_band(text):
    """Map text to a magnitude band {-2,-1,0,1}; no keyword hit -> 0 (typical / not mentioned). The most extreme hit wins."""
    t = str(text).lower()
    best = 0
    for band, kws in _MAG_BANDS:
        if any(k in t for k in kws):
            if abs(band) > abs(best):
                best = band
    return best


def magnitude_match(gen_text, gt_text):
    """Match score in [0,1] between the generated and the true magnitude band: same band=1, one band off=0.5, opposite direction=0."""
    g, t = magnitude_band(gen_text), magnitude_band(gt_text)
    d = abs(g - t)
    if g * t < 0:            # opposite direction (says low when it is high) -- worst case
        return 0.0
    return max(0.0, 1.0 - 0.5 * d)


def text_reward(gen_text, gt_roi, gt_text, w_roi=1.0, w_mag=1.0):
    """Dense shaping reward from the text alone (excludes the CRPS main term, computed in the
    training loop). gt_roi=(a,b) true ROI; gt_text=true conclusion (with magnitude). Returns
    (total, breakdown)."""
    iou = roi_iou(extract_roi(gen_text), gt_roi)
    mag = magnitude_match(gen_text, gt_text)
    total = w_roi * iou + w_mag * mag
    return total, {"roi_iou": iou, "mag_match": mag}
