#!/bin/bash
# Joint (mixed) training of both task types in one model: understanding jsonl batches and
# forecasting parquet batches are interleaved within every epoch (the interleaving ratio follows
# the actual batch counts). Per-epoch repeats let the two branches see different numbers of passes.
#
# Usage:
#   bash chronos_llm/scripts/train_chronos_llm.sh
#
# Paper recipe for the unified checkpoint (Table "one checkpoint, both tasks"; 8 GPUs, 10 epochs,
# checkpoint of epoch 4), starting from the GRPO understanding checkpoint whose feedback pathway
# has never been trained, hence the gate warm start:
#   INIT_FROM_CHECKPOINT=<GRPO checkpoint> UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr_dom12.txt \
#   PARQUET=data/forecast/mmtr_forecast_corpus.parquet U_REPEATS=1 F_REPEATS=9 LR_LORA=3e-5 \
#   GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_START_RATIO=1.0 SS_END_RATIO=0.7 \
#   bash chronos_llm/scripts/train_chronos_llm.sh
# Do NOT pass GATE_INIT / ZERO_CROSS_OUT_PROJ when the starting checkpoint has already trained the
# forecasting pathway (they would overwrite the learned gates / output projections).
set -e -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARQUET=${PARQUET:-data/forecast/mmtr_forecast_corpus.parquet}
DATA_TAG="_$(basename "$PARQUET" .parquet)"
UNDERSTANDING_LIST=${UNDERSTANDING_LIST:-chronos_llm/configs/understanding_train_mmtr.txt}
read_manifest UNDERSTANDING_JSONL "$UNDERSTANDING_LIST"
EVAL_LIST=${EVAL_LIST:-chronos_llm/configs/understanding_test_mmtr.txt}
read_manifest EVAL_JSONL "$EVAL_LIST"
EVAL_BASE_DIR=${EVAL_BASE_DIR:-.}

EPOCHS=${EPOCHS:-10}
U_REPEATS=${U_REPEATS:-1}; F_REPEATS=${F_REPEATS:-3}   # passes per epoch of each branch
LR=${LR:-1e-5}; LR_LORA=${LR_LORA:-1e-4}; LR_QFORMER=${LR_QFORMER:-1e-4}; LR_GATE=${LR_GATE:-1e-3}
UNDERSTANDING_BS=${UNDERSTANDING_BS:-70}; FORECAST_BS=${FORECAST_BS:-40}
U_TOKEN_BUDGET=${U_TOKEN_BUDGET:-20000}; F_TOKEN_BUDGET=${F_TOKEN_BUDGET:-40000}   # per-rank padded-token budgets
PATCH_BUDGET=${PATCH_BUDGET:-0}; MAX_DYN_BS=${MAX_DYN_BS:-150}; ENCODE_MAX_ROWS=${ENCODE_MAX_ROWS:-58}
HISTORY_STATS_TOKEN=${HISTORY_STATS_TOKEN:-1}; SHORT_MIN_PATCHES=${SHORT_MIN_PATCHES:-4}
BS_LADDER=${BS_LADDER:-"1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150"}
SAMPLE_CHUNKS=${SAMPLE_CHUNKS:-15}; OVERVIEW_CHUNK=${OVERVIEW_CHUNK:-1}
SS_START_RATIO=${SS_START_RATIO:-1.0}; SS_END_RATIO=${SS_END_RATIO:-1.0}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0}
RUN_TS=${RUN_TS:-$(date +%Y%m%d-%H%M%S)}

EXTRA=(); TAG=""
[ -n "${INIT_FROM_CHECKPOINT:-}" ] && { EXTRA+=(--init_from_checkpoint "$INIT_FROM_CHECKPOINT"); TAG="${TAG}_initckpt"; }
[ "${INIT_MERGE_REOPEN:-0}" = "1" ]  && { EXTRA+=(--init_merge_reopen); TAG="${TAG}merge"; }
[ -n "${GATE_INIT:-}" ]              && { EXTRA+=(--gate_init "$GATE_INIT"); TAG="${TAG}_g${GATE_INIT}"; }
[ -n "${ZERO_CROSS_OUT_PROJ:-}" ]    && { EXTRA+=(--zero_cross_out_proj); TAG="${TAG}_zop"; }
[ "${NO_REASONING:-0}" = "1" ]       && { EXTRA+=(--understanding_no_reasoning); TAG="${TAG}_nocot"; }
[ "${ANSWER_FIRST:-0}" = "1" ]       && { EXTRA+=(--understanding_answer_first); TAG="${TAG}_af"; }
[ "${FREEZE_TS:-0}" = "1" ]          && { EXTRA+=(--freeze_ts); TAG="${TAG}_freezets"; }
[ "$HISTORY_STATS_TOKEN" = "1" ]     && EXTRA+=(--history_stats_token)
if [ "$SS_END_RATIO" != "1.0" ] || [ "$SS_START_RATIO" != "1.0" ]; then
  EXTRA+=(--ss_start_ratio "$SS_START_RATIO" --ss_end_ratio "$SS_END_RATIO" --ss_max_new_tokens "${SS_MAX_NEW_TOKENS:-288}")
  TAG="${TAG}_ss${SS_START_RATIO}-${SS_END_RATIO}"
fi
[ -n "${SEED:-}" ] && { EXTRA+=(--seed "$SEED"); TAG="${TAG}_s${SEED}"; }
if [ -n "${MAX_STEPS:-}" ]; then STEP_ARG=(--max_steps "$MAX_STEPS"); else STEP_ARG=(--num_train_epochs "$EPOCHS"); fi

RUN_NAME="ep${EPOCHS}_ux${U_REPEATS}fx${F_REPEATS}_lr${LR}_lora${LR_LORA}_qf${LR_QFORMER}_gate${LR_GATE}_utb${U_TOKEN_BUDGET}_ftb${F_TOKEN_BUDGET}${TAG}${BACKBONE_TAG}${DATA_TAG}_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/joint_run}/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"; archive_script "$OUTPUT_DIR" "${BASH_SOURCE[0]}"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d-%H%M%S)_node${NODE_RANK}.log"
echo "[train_chronos_llm] OUTPUT_DIR=$OUTPUT_DIR"

torchrun --nnodes=$NODE_COUNT --node_rank=$NODE_RANK --nproc_per_node=$PROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=${MASTER_PORT:-29500} chronos_llm/train.py \
  --chronos_ckpt "$CHRONOS" --llm_path "$LLM" \
  --tsfm_backbone "$TSFM_BACKBONE" \
  --forecast_parquet "$PARQUET" \
  --understanding_jsonl "${UNDERSTANDING_JSONL[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --bf16 --gradient_checkpointing --deepspeed chronos_llm/configs/ds_zero2.json \
  --learning_rate $LR --lr_lora $LR_LORA --lr_qformer $LR_QFORMER --lr_gate $LR_GATE \
  "${STEP_ARG[@]}" --warmup_ratio 0.03 \
  --understanding_epoch_repeats $U_REPEATS --forecast_epoch_repeats $F_REPEATS \
  --understanding_bs $UNDERSTANDING_BS --forecast_bs $FORECAST_BS \
  --understanding_token_budget $U_TOKEN_BUDGET --forecast_token_budget $F_TOKEN_BUDGET \
  --patch_budget $PATCH_BUDGET --max_dynamic_bs $MAX_DYN_BS --dynamic_bs_ladder "$BS_LADDER" \
  --encode_max_rows $ENCODE_MAX_ROWS --short_min_patches $SHORT_MIN_PATCHES \
  --encode_sample_chunks $SAMPLE_CHUNKS --encode_overview_chunk $OVERVIEW_CHUNK \
  --understanding_max_context 240000 --forecast_max_context 8192 \
  --max_user_tokens ${MAX_USER_TOKENS:-1500} --max_tokens ${MAX_TOKENS:-4096} \
  --text_loss_weight 1.0 --pred_loss_weight 1.0 --roi_loss_weight 1.0 \
  --eval_every_epochs ${EVAL_EVERY_EPOCHS:-1} \
  --eval_understanding_jsonl "${EVAL_JSONL[@]}" --eval_understanding_base_dir "$EVAL_BASE_DIR" \
  --eval_understanding_limit ${EVAL_U_LIMIT:-0} --eval_understanding_bs ${EVAL_U_BS:-400} \
  --eval_understanding_max_new_tokens ${EVAL_U_MAX_NEW_TOKENS:-2048} \
  --eval_forecast_bs ${EVAL_F_BS:-64} \
  "${EXTRA[@]}" \
  --logging_steps ${LOGGING_STEPS:-10} --save_total_limit $SAVE_TOTAL_LIMIT --save_only_model \
  --dataloader_num_workers ${NUM_WORKERS:-8} \
  2>&1 | tee -a "$LOG_FILE"
