#!/bin/bash
# Controllability experiment (GPU side): for each of the 6 edited parquet variants (orig + 5
# magnitude levels) run one teacher-forced feedback forecast with the released forecasting
# checkpoint; the npz lands in outputs/eval/controllability/<variant>/forecast_preds_tf.npz.
# Afterwards eval_controllability_metrics.py (CPU) computes direction agreement / magnitude
# monotonicity.
#   bash chronos_llm/scripts/utils/controllability_infer.sh
set -e -o pipefail
REPO=.
cd "$REPO"; export PYTHONPATH="$REPO:$PYTHONPATH"; export PYTHONUNBUFFERED=1

MODEL_PATH=${MODEL_PATH:-checkpoints/timeomni-o1}
PQ_DIR=${PQ_DIR:-$REPO/outputs/eval/controllability/parquets}
GATE_SCALE=${GATE_SCALE:-1.0}   # when != 1 the output root automatically gets a _gs<alpha> suffix (diagnosing a nearly closed feedback gate)
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/eval/controllability$( [ "$GATE_SCALE" != "1.0" ] && echo "_gs${GATE_SCALE%.*}" )}
VARIANTS=${VARIANTS:-orig tiny well_below mod_below typical above}

PROC_PER_NODE=${PROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$PROC_PER_NODE" -ge 1 ] || PROC_PER_NODE=1

for v in $VARIANTS; do
  out="$OUT_ROOT/$v"
  mkdir -p "$out"
  if [ -f "$out/forecast_preds_tf.npz" ]; then
    echo "[ctrl] $v already exists, skipping"; continue
  fi
  echo "[ctrl] ===== variant=$v ====="
  torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$PROC_PER_NODE \
    --master_addr=127.0.0.1 --master_port=$(( (RANDOM % 20000) + 20000 )) \
    chronos_llm/eval/infer_forecast.py \
    --model_path "$MODEL_PATH" \
    --parquet "$PQ_DIR/ctrl_$v.parquet" \
    --split test --limit 500 \
    --batch_size ${BATCH_SIZE:-32} \
    --num_workers 0 \
    --gate_scale "$GATE_SCALE" \
    --mode teacher_forced \
    --output "$out/forecast_preds_tf.npz" \
    2>&1 | tee "$out/infer_$(date +%Y%m%d-%H%M%S).log"
done
echo "[ctrl] all variants done"

python chronos_llm/scripts/utils/eval_controllability_metrics.py \
  --root "$OUT_ROOT" \
  --parquet "$PQ_DIR/ctrl_orig.parquet" \
  --out-csv "$OUT_ROOT/controllability_metrics.csv" \
  2>&1 | tee "$OUT_ROOT/metrics_$(date +%Y%m%d-%H%M%S).log"
