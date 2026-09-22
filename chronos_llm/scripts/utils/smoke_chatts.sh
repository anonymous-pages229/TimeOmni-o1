#!/bin/bash
# Chat-TS GPU smoke test: 3 sets x 5 samples x two prompt styles (inst/plain) -- output coherence tells whether the token semantics are aligned
set -ex
REPO=.
SMOKE_LIST=$REPO/chronos_llm/configs/understanding_smoke3_jsonl.txt
for style in inst plain; do
  BASELINE=chatts_2503 LIMIT=5 JSONL_LIST=$SMOKE_LIST PROMPT_STYLE=$style \
    OUT_DIR=$REPO/outputs/eval/baseline_chatts_2503/smoke_$style \
    bash $REPO/chronos_llm/scripts/utils/run_baseline_understanding_generic.sh
done
echo SMOKE_DONE
