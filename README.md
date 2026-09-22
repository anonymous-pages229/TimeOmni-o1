# TimeOmni-o1 and MMTR

Anonymous code and data release for the paper *TimeOmni-o1: multimodal time series reasoning by
coupling a time-series foundation model with an LLM* (under review).

**TimeOmni-o1** couples a pretrained time-series foundation model (TSFM; Chronos-2) with an LLM
(Qwen3.5-9B) in both directions and serves the two task types of multimodal time series reasoning
with one model:

* **series -> LLM**: the TSFM encodes the (multivariate, arbitrarily long) series; a
  sliding-window Q-Former compresses the patch embeddings into a short soft prompt that is spliced
  into the LLM's context next to the text; the LLM writes a reasoning trace and an answer
  (**understanding**), or a trace that ends in a *forecast instruction* (**forecasting**);
* **LLM -> TSFM**: a feedback Q-Former compresses the LLM's last-layer hidden states and injects
  them through gated cross-attention into every encoder block of the TSFM, which then produces the
  21-quantile forecast. The whole loop is differentiable end to end.

**MMTR** (Multimodal Time series Reasoning) is the reasoning-trace-annotated dataset the model is
trained on: an understanding part (497,956 train / 70,630 test samples, 12 disciplines, six public
reasoning-annotated corpora unified under one schema) and an event-conditioned forecasting part
(4,630 train / 437 test samples, five domains, ten subsets of five public multimodal forecasting
sources, annotated by us with reasoning chains and forecast instructions).

This repository contains the model, SFT / GRPO / joint training, inference, evaluation (including
the reasoning-trace judges, the counterfactual faithfulness test and the external benchmarks), the
external baselines, the backbone and architecture ablations, unit tests, and the MMTR test sets.
Model weights are not included in this anonymous version.

---

## Repository layout

```
chronos_llm/                 this work (import name kept from development)
  models/                    ChronosLLM (PreTrainedModel), sliding-window / feedback Q-Formers, Chronos-2 with gated cross-attention
  data/                      jsonl / parquet datasets, chat template + supervision masking, multichannel collator, dynamic batching sampler
  train.py, trainer.py       HF-Trainer-based training (DeepSpeed ZeRO-2, grouped learning rates, scheduled sampling, mid-training eval)
  rl/                        GRPO for understanding (stage 2), GRPO over forecasting instructions, rewards
  eval/                      inference (torchrun, multi-node), metrics, LLM-as-judge trace evaluation, causal faithfulness, external baselines
  scripts/                   entry points: train_*.sh, eval_*.sh (see "Training" / "Evaluation")
  scripts/utils/             per-discipline / per-domain aggregation, ablation data and probes,
                             external-benchmark converters, backbone-ablation summaries
  configs/                   DeepSpeed config and data manifests
  tests/                     CPU unit tests (real Chronos-2 + tiny LLM stand-in); run_all_cpu.sh
  train_headcls.py           TSFM-only baseline (Chronos-2 + per-subtask classification heads)
data/                        MMTR test sets, the full forecasting corpus, CiK / ST-Bench T4 conversions (see data/README.md)
docs/                        DATASET.md (dataset card), EVALUATION.md (metrics and protocols)
src/chronos/                 unmodified upstream Chronos-2 code (Apache-2.0)
src/timesfm3/                unmodified upstream TimesFM-3.0 code (Apache-2.0), used by the backbone ablation
```

## Setup

```bash
conda create -n timeomni python=3.12 && conda activate timeomni
pip install -r requirements.txt
pip install -e .                      # installs `chronos` (upstream) and `chronos_llm`

# backbones (any local path works; the scripts read CHRONOS2_PATH / LLM_PATH)
hf download amazon/chronos-2 --local-dir checkpoints/chronos-2
hf download Qwen/Qwen3.5-9B --local-dir checkpoints/Qwen3.5-9B
# only for the backbone ablation (TIMESFM3_PATH / LLM_PATH)
hf download google/timesfm-3.0-pytorch --local-dir checkpoints/TimesFM3.0
hf download Qwen/Qwen3.5-4B --local-dir checkpoints/Qwen3.5-4B
```

Notes on the environment: `transformers>=5`; Chronos-2 is loaded through
`chronos_llm.models.cross_attn_chronos.load_chronos2_with_cross_attn` (direct construction +
`load_state_dict`), which is also what wraps every encoder block with gated cross-attention.
Qwen3.5's hybrid linear attention needs `flash-linear-attention`; its kernels are specialized per
batch size, which is why the training scripts quantize dynamic batch sizes to a ladder.

## Data

See `data/README.md` for what is shipped and `docs/DATASET.md` for the dataset card.

* **Understanding test sets** are in `data/understanding/` (full official test splits; ECG-QA
  subsampled to 250 recordings for size). Training splits are rebuilt from the public releases into
  the same jsonl schema; the expected locations are listed in
  `chronos_llm/configs/understanding_train_mmtr.txt`. Discipline-rebalanced and GRPO pools are
  derived with `make_domain12_pool.py` and `make_grpo_domain_pools_v10.py`.
* **Forecasting corpus** (train + test, annotations included): `data/forecast/mmtr_forecast_corpus.parquet`.

## Model

`chronos_llm/models/chronos_llm_model.py` defines `ChronosLLMConfig` / `ChronosLLM`:

| component | file | notes |
|---|---|---|
| TSFM encoder with feedback | `chronos_llm/models/cross_attn_chronos.py` | Chronos-2 (d=768, context 8192, 21 quantiles) with a gated cross-attention module `tanh(g) * CrossAttn(h, Z)` added to every encoder block; `cross_states=None` reproduces the original model exactly. The released recipe starts the gate open (`gate_init=1.0`) and zero-initialises the cross-attention output projection, so training starts from the identity. |
| sliding-window Q-Former | `chronos_llm/models/qformer.py::SlidingWindowQFormer` | non-overlapping windows over the patch sequence, `k` learned queries per window plus `G` global queries with sinusoidal time encoding; the soft-token budget is log-linear in the number of patches x channels (`--sw_token_min/max/pc_ref`), capped at 200 tokens. Internal width 768 with an output projection to the LLM width. |
| statistics token | `--history_stats_token` | per-window location / scale / range statistics fed back as a token (instance normalisation removes them from the encoder). |
| feedback Q-Former | `chronos_llm/models/qformer.py::QFormer` | `M=16` queries over the LLM's last-layer hidden states -> `(M, 768)` cross-states for the TSFM. |
| LLM adaptation | `add_lora()` | LoRA on the LLM only (r=8, alpha=32); TSFM, both Q-Formers and gates are fully trained (`modules_to_save`). |
| chat template / masking | `chronos_llm/data/chat_utils.py` | the supervised span is located by a longest-common-prefix trick against the chat template; the inference prefix is token-identical to the training context. Understanding: `<think>trace</think>` + answer (or direct answer when the trace is empty); forecasting: `<think>trace</think>` + forecast instruction. |
| losses | `chronos_llm/models/chronos_llm_model.py::forward` | text CE + masked pinball loss over the horizon + pinball loss restricted to the event ROI (weights `--text/pred/roi_loss_weight`). |

Multivariate inputs: the collator folds all channels of all samples into `(sum C, L)` rows with
group ids; the TSFM attends across channels within a sample only; each sample yields exactly one
soft prompt. Long histories (up to 240k steps for understanding) are encoded chunk-wise with
uniform chunk sampling (`--encode_sample_chunks`) plus one down-sampled overview chunk.

## Training

All entry points are in `chronos_llm/scripts/`; every hyper-parameter is an environment variable
with the paper's value as default (see the header of each script). Multi-GPU / multi-node via
`torchrun` + DeepSpeed ZeRO-2; four learning-rate groups (TSFM 1e-5, LoRA 1e-4, Q-Formers 1e-4,
gates 1e-3, no weight decay on gates).

```bash
# Stage 1 - understanding SFT on MMTR with reasoning supervision (8 GPUs, 10 epochs)
UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr.txt \
EVAL_U_MAX_NEW_TOKENS=2048 bash chronos_llm/scripts/train_understanding.sh
#   no-reasoning control arm: NO_REASONING=1 ...        LLM-only backbone baseline: LLM_ONLY=1 PROC_PER_NODE=2 U_TOKEN_BUDGET=80000 ...

# Stage 2 - discipline-rebalanced SFT of the LLM LoRA only, from the stage-1 checkpoint (20 epochs)
python chronos_llm/scripts/utils/make_domain12_pool.py
UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr_dom12.txt FREEZE_TS=1 EPOCHS=20 \
SAVE_TOTAL_LIMIT=0 INIT_FROM_CHECKPOINT=outputs/understanding_run/<stage1>/checkpoint-<N> \
bash chronos_llm/scripts/train_understanding.sh

# Stage 3 - GRPO on the LLM LoRA (answer-correctness reward, G=8, T=1.0, 8 GPUs, 1000 steps; step 300 reported)
python chronos_llm/scripts/utils/make_grpo_domain_pools_v10.py
INIT_CKPT=outputs/understanding_run/<stage2>/checkpoint-<N> bash chronos_llm/scripts/train_grpo_understanding.sh

# Forecasting SFT on the MMTR forecasting corpus (2 GPUs, 30 epochs; epoch 19 reported)
GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_END_RATIO=0.7 PROC_PER_NODE=2 bash chronos_llm/scripts/train_forecast.sh
#   ablations: CHRONOS_ONLY=1 | FORECAST_PROMPT_ONLY=1 | FEEDBACK_SCOPE=conclusion | CHRONOS_RANDOM_INIT=1 | LLM_ONLY=1

# One checkpoint for both tasks: joint training from the stage-3 checkpoint (8 GPUs, 10 epochs; epoch 4 reported)
INIT_FROM_CHECKPOINT=outputs/grpo_understanding_run/<run>/checkpoint-300 \
UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr_dom12.txt U_REPEATS=1 F_REPEATS=9 LR_LORA=3e-5 \
GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_END_RATIO=0.7 bash chronos_llm/scripts/train_chronos_llm.sh
```

Scheduled sampling (`SS_START_RATIO -> SS_END_RATIO`) anneals the text fed back into the TSFM
from the annotated trace to the model's own generation during forecasting training, narrowing the
train/inference gap of the feedback pathway. Checkpoints are PEFT adapters (LoRA + TSFM +
Q-Formers) plus `config.json`; `ChronosLLM.from_pretrained(path, merge=False)` reloads them.
Mid-training evaluation on the test manifests runs every epoch (`EVAL_EVERY_EPOCHS`).

Architecture ablations are switches of the same scripts: `HISTORY_COMPRESSOR=pool`,
`FEEDBACK_COMPRESSOR=pool`, `HISTORY_ENCODE_LAYER=k`, `FEEDBACK_LLM_LAYER=k`,
`CROSS_ATTN_LAYERS=last6`, `SW_GLOBAL_QUERIES=0`, `SW_TOKEN_MAX=...`.

### Backbone ablation

The same scripts swap either backbone; the Q-Formers, the gated cross-attention feedback, the
losses and the recipe stay exactly as above, and the backbone is recorded in the checkpoint config
and in the run name.

* **TSFM -> TimesFM-3.0**: `TSFM_BACKBONE=timesfm3` (weights from `TIMESFM3_PATH`, default
  `checkpoints/TimesFM3.0`). `chronos_llm/models/timesfm_backbone.py` wraps TimesFM-3.0 into the
  Chronos-2 interface (patch embeddings for the sliding-window Q-Former, quantile head for the
  forecast) and `chronos_llm/models/cross_attn_timesfm.py` adds the same gated cross-attention to
  every TimesFM transformer layer, again starting from the identity (zero-initialised output projection).
* **LLM -> Qwen3.5-4B**: `LLM_PATH=checkpoints/Qwen3.5-4B`.

```bash
# forecasting: each FULL arm is paired with its own TSFM-only lower bound; two seeds per arm
for SEED in 42 43; do
  COMMON="OUTPUT_DIR=outputs/forecast_backbone_abl SEED=$SEED GATE_INIT=1.0 ZERO_CROSS_OUT_PROJ=1 SS_END_RATIO=0.7 PROC_PER_NODE=2"
  env $COMMON TSFM_BACKBONE=timesfm3                bash chronos_llm/scripts/train_forecast.sh   # TimesFM-3.0 + 9B
  env $COMMON TSFM_BACKBONE=timesfm3 CHRONOS_ONLY=1 bash chronos_llm/scripts/train_forecast.sh   # TimesFM-3.0 alone
  env $COMMON LLM_PATH=checkpoints/Qwen3.5-4B       bash chronos_llm/scripts/train_forecast.sh   # Chronos-2 + 4B
done
python chronos_llm/eval/baseline_forecast_zeroshot.py --tsfm_backbone timesfm3 \
  --output outputs/eval/zeroshot_timesfm3/preds.npz --output_csv outputs/eval/zeroshot_timesfm3/metrics.csv
# summary table (437-row test, equal weight over the 5 domains, each run at its own best epoch);
# --interp21 interpolates TimesFM's 9 quantiles onto the 21 Chronos-2 levels so CRPS is comparable
A0_FULL_RUNS=<released Chronos-2 + 9B runs> A0_CHRONOS_ONLY_RUNS=<Chronos-2-only run> \
  python chronos_llm/scripts/utils/summarize_backbone_ablation.py --side forecast --interp21 --best-epoch

# understanding: rerun the three stages above with the switch on stage 1 and stage 2
# (stage 3 reads the backbone from the stage-2 checkpoint), then evaluate as usual
TSFM_BACKBONE=timesfm3 UNDERSTANDING_LIST=chronos_llm/configs/understanding_train_mmtr.txt \
  EVAL_U_MAX_NEW_TOKENS=2048 bash chronos_llm/scripts/train_understanding.sh
```

CRPS is a normalised weighted quantile loss with the number of quantiles in the denominator, so
absolute scores of a 9-quantile and a 21-quantile model are only comparable after the alignment
done by `--interp21` (also available in `scripts/utils/rescore_forecast_dedup.py`); comparisons
within one arm (FULL vs. its own lower bound) are unaffected.

## Inference and evaluation

```bash
# understanding: sharded generation + scoring (per-file CSV, native per-subtask protocols, per-discipline table)
bash chronos_llm/scripts/eval_understanding.sh <checkpoint_dir>
# forecasting: generate + teacher-forced passes, CRPS / PCC over the horizon and the ROI, per-domain table
bash chronos_llm/scripts/eval_forecast.sh <checkpoint_dir>
```

`docs/EVALUATION.md` describes the metrics and protocols in detail. Further evaluation scripts:

| what | script |
|---|---|
| reasoning-trace quality, understanding (LLM judge: consistency / supportiveness, correct-vs-wrong stratified) | `chronos_llm/eval/explanation_judge.py` |
| reasoning-trace quality, forecasting (ROI overlap, shape-consistency judge) | `chronos_llm/eval/text_alignment_eval.py` |
| counterfactual faithfulness of the forecast to the instruction (five magnitude bands, Spearman / direction agreement) | `scripts/utils/make_controllability_parquets.py`, `scripts/utils/controllability_infer.sh`, `scripts/utils/eval_controllability_metrics.py`, `chronos_llm/eval/causal_faithfulness.py` |
| magnitude-tag hit rates (text vs. own forecast, text vs. ground truth, shuffled null) | `scripts/utils/magnitude_faithfulness.py`, `scripts/utils/eval_conclusion_accuracy.py` |
| paired significance tests between two forecast runs | `scripts/utils/significance_test.py` |
| reasoning-content ablations (event text shuffled / removed), prompt-only variants | `scripts/utils/make_prompt_content_probes.py`, `build_plain_prompt.py`, `make_shuffled_event_plain_prompt.py` |

### External benchmarks

* **SciTS** (understanding, retrained on the official split): `UNDERSTANDING_LIST=chronos_llm/configs/scits_train.txt
  EVAL_LIST=chronos_llm/configs/scits_test.txt EVAL_BASE_DIR=data/raw/scits/Release_v1 SAMPLE_CHUNKS=8 EVAL_U_MAX_NEW_TOKENS=200
  bash chronos_llm/scripts/train_understanding.sh`; aggregation with `scripts/utils/paper_understanding_domains.py`.
* **CiK** (forecasting, zero-shot, official RCRPS): `build_cik_eval_parquet.py` -> `eval_cik.sh` -> `eval_cik_rcrps.py`
  (the converted test split is shipped as `data/external/cik_eval.parquet`; the RCRPS step imports the official benchmark code).
* **ST-Bench T4** (forecasting, fine-tuned, official MAE): `build_stbench_forecast_parquet.py` (shipped as
  `data/external/stbench_t4_forecast.parquet`) -> `train_forecast.sh` with `FORECAST_BS=32 F_TOKEN_BUDGET=0`.

### Baselines

`chronos_llm/eval/baseline_*.py` (+ wrappers in `scripts/utils/`) run the external models under
the same test protocols: ChatTime-1-7B, Chat-TS-8B, OpenTSLM, Time-MQA, TimeOmni-VL-15B,
TimeOmni-1-4B, TimeReasoner, DoubleCast, UniTS (fine-tuned), TimeOmni (fine-tuned on MMTR),
Chronos-2 zero-shot, and the two backbone-alone rows (LLM with the series serialized as text:
`LLM_ONLY=1` / `LLM_ONLY_ZEROSHOT=1`; TSFM with classification heads: `train_headcls.sh`).
Each baseline script expects the third-party code / weights under `third_party/` and
`checkpoints/`; see its docstring.

## Tests

```bash
bash chronos_llm/tests/run_all_cpu.sh            # ~60 CPU tests: real Chronos-2 / TimesFM-3.0 + tiny LLM stand-in
bash chronos_llm/tests/run_train_save_check.sh   # train -> save -> reload equivalence
bash chronos_llm/scripts/smoke_test.sh           # GPU: six steps with the real 9B LLM
```

The tests cover the gate-zero / zero-output-projection identity, batched-vs-single generation
equivalence with left padding, inference-prefix == training-context alignment, multichannel
folding, bucketed encoding equivalence, dynamic batching, resume, metrics against hand-computed
values, the TimesFM-3.0 adapter and its cross-attention identity, and the evaluation pipeline end to end.

## License

Code in `src/chronos/` is the upstream Chronos-2 release (Apache-2.0, see `LICENSE` and
`src/chronos/NOTICE`) and code in `src/timesfm3/` is the upstream TimesFM-3.0 release (Apache-2.0,
see `src/timesfm3/LICENSE`); the rest of the code is released under the same Apache-2.0 license.
No model weights are shipped; the TimesFM-3.0 weights come with their own license on the Hugging Face hub.
Dataset files under `data/` are redistributed for review purposes only; each source keeps its own
license (see `data/README.md`).
