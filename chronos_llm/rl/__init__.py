"""GRPO RL components: close the gap between generated and teacher-forced forecasting (see the reward.py docs)."""
from .reward import (
    extract_roi,
    magnitude_band,
    magnitude_match,
    roi_iou,
    text_reward,
)

__all__ = ["extract_roi", "roi_iou", "magnitude_band", "magnitude_match", "text_reward"]
