# Evaluation protocols

## Understanding

Inference: `chronos_llm/eval/infer_understanding.py` (via `scripts/eval_understanding.sh`) writes one
jsonl per test file with `generated_text`, `ground_truth`, `task`, `scene`. Greedy decoding,
`max_new_tokens=2048` for MMTR (VeriTime traces are long), the inference prefix opens a
`<think>` segment except for sources whose test protocol is direct answering (ST-Bench).

Scoring follows each source's **native protocol** (`scripts/utils/summarize_agg6_eval.py`):

| source | metric |
|---|---|
| OpenTSLM ECG-QA / HAR / Sleep | the official OpenTSLM evaluation (`eval/eval_opentslm_answer.py`: answer extraction after `Answer:`, label canonicalisation, accuracy / F1) |
| ST-Bench, HiTSR, VeriTime | exact-match accuracy of the extracted answer (letter boundaries are enforced for single-letter MCQ answers) |
| TelecomTS | exact match against the reference answer |
| Time-RA | support-weighted F1 over the anomaly types (the source paper's metric) |

Subtask scores aggregate into disciplines by sample count
(`scripts/utils/summarize_agg6_by_domain.py`, `domain_of` assigns every sample to one discipline);
the overall score is the arithmetic mean of the 12 discipline scores. A single evaluation pass is
not bit-reproducible (batch-composition-dependent linear-attention kernels change about half of
the generated traces part-way); measured on one checkpoint this moves the overall average by about
0.5 and small disciplines by up to 5 points, so differences below one point are read as ties.

Baselines are scored by exactly the same scripts on their generated text
(`scripts/utils/score_agg6_baseline.sh`, `split_agg6_for_baselines.py`,
`merge_agg6_baseline_parts.py`); single-channel baselines read the first channel.

## Forecasting

Inference: `chronos_llm/eval/infer_forecast.py` (via `scripts/eval_forecast.sh`) on the parquet
test split. `--mode generate` lets the model write its own trace and instruction (the reported
setting); `--mode teacher_forced` feeds back the annotated instruction instead (an upper bound
that isolates the feedback pathway from generation quality). Outputs: an `npz` with the 21
quantile forecasts, ground truth, validity and ROI masks, and a jsonl with the generated text.

Metrics (`chronos_llm/eval/metrics.py`, per sample then averaged):

* **CRPS** = normalised weighted quantile loss over the 21 levels,
  `sum_q 2|(y - q_hat)(1{y <= q_hat} - alpha)| / (K * sum |y|)`, consistent with the TSFM's pinball
  training loss; samples with an all-zero target are excluded (nan).
* **PCC** of the median forecast with the target.
* **MAPE** is computed but not used as a headline metric (pathological on near-zero series).
* each metric over the **full horizon** and over the **event ROI**.

Reporting: per-domain scores pool the samples of a domain (`scripts/utils/paper_forecast_by_domain.py`,
ten subsets -> five domains) and the overall score is the **equal-weight mean of the five domains**.
Baselines that fail to produce a parsable forecast on some samples are weighted by their parse
success rate (`scripts/utils/paper_forecast_success_rate.py`). Paired significance between two
runs: `scripts/utils/significance_test.py` (paired t, Wilcoxon, paired bootstrap).
Run-to-run training noise for this corpus is about 0.01 CRPS; the evaluation itself is stable to
about 0.001.

## Reasoning-trace quality

One framework across both tasks (LLM judge = GPT-5.4 through an OpenAI-compatible endpoint,
`OPENAI_API_KEY` / `--api_base_url`):

* **Understanding** (`eval/explanation_judge.py`): samples are stratified by discipline and by
  correct / wrong answer; the judge scores *consistency* with the annotated trace and
  *supportiveness* of the model's own answer (1–5). Faithfulness is behavioural: the
  correct-minus-wrong supportiveness gap (post-hoc rationalisation would defend both equally).
  `scripts/utils/shuffled_explanation_judge_control.py` gives the mismatched-pair null.
* **Forecasting** (`eval/text_alignment_eval.py`): relaxed ROI overlap and strict IoU between the
  generated and annotated ROI (parse failures counted in the denominator), plus a shape-consistency
  judge between the generated and annotated trace; `shuffled_text_alignment_control.py` is the null.
  `eval/forecast_visual_faithfulness.py` additionally asks a multimodal judge whether the plotted
  forecast matches the generated trace.
* **Counterfactual faithfulness** (forecasting): `scripts/utils/make_controllability_parquets.py`
  rewrites the magnitude statement of every test instruction into each of five bands,
  `controllability_infer.sh` teacher-forces each variant through the feedback pathway, and
  `eval_controllability_metrics.py` reports the per-sample Spearman correlation between the stated
  band and the resulting forecast peak and the above-vs-below direction agreement (chance: 0 / 50%).
  `eval/causal_faithfulness.py` bundles the same test with the gate-zero identity check.
* **Magnitude tags** (`scripts/utils/magnitude_faithfulness.py`): the `[Magnitude: BAND]` written
  by the model is compared with the band implied by its own median forecast (supportiveness) and
  with the annotated band (consistency), with a shuffled null.

## External benchmarks

* **SciTS**: the architecture is retrained on the official training split (23 datasets, 21 usable)
  and scored with the suite's protocol (F1 for understanding tasks, accuracy for MCQ, discipline =
  simple mean over its tasks; `scripts/utils/paper_understanding_domains.py`,
  `paper_scits_baselines.py`).
* **CiK**: zero-shot; the official RCRPS implementation is imported from the benchmark code and
  fed the 21 quantiles as samples (`scripts/utils/eval_cik_rcrps.py`).
* **ST-Bench T4**: fine-tuned on the 650 ST-CoT forecasting rows (ground truth recovered from
  ST-SFT by exact input match), scored with the official MAE on the 280 ST-Test rows
  (`scripts/utils/build_stbench_forecast_parquet.py`, `docs` in its docstring).
