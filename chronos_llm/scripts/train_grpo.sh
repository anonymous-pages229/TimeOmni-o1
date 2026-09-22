#!/bin/bash
# GRPO over the forecasting reasoning (reported as a negative result in the paper's analysis):
# starting from a forecasting SFT checkpoint, the LLM LoRA is updated with rewards computed from
# the forecaster's error under the sampled instruction. Single GPU; hyper-parameters via env vars.
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INIT_CKPT=${INIT_CKPT:?set INIT_CKPT to a forecasting SFT checkpoint directory}
PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
EPOCHS=${EPOCHS:-1}
BS=${BS:-8}; G=${G:-6}; TEMP=${TEMP:-1.0}; LR=${LR:-1e-6}
W_ROI=${W_ROI:-0.3}; W_MAG=${W_MAG:-0.3}; MNT=${MAX_NEW_TOKENS:-256}
MAX_STEPS=${MAX_STEPS:-0}
SAVE_EVERY=${SAVE_EVERY:-50}
RUN_TS=${RUN_TS:-$(date +%Y%m%d-%H%M%S)}
RUN_NAME="grpo_g${G}_lr${LR}_wroi${W_ROI}_wmag${W_MAG}_bs${BS}_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_forecast_run}/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

python -m chronos_llm.rl.train_grpo \
  --init_ckpt "$INIT_CKPT" --parquet "$PARQUET" --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" --batch_size "$BS" --group_size "$G" --temperature "$TEMP" \
  --lr "$LR" --w_roi "$W_ROI" --w_mag "$W_MAG" --max_new_tokens "$MNT" --max_steps "$MAX_STEPS" \
  --save_every_steps "$SAVE_EVERY" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
