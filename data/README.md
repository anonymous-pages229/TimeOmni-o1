# Data shipped with this repository

```
data/
  forecast/mmtr_forecast_corpus.parquet   MMTR forecasting part: 4,630 train + 437 test samples (full corpus, 14 MB)
  understanding/<source>/<name>_test.jsonl MMTR understanding test sets (unified jsonl schema)
  understanding/<source>/signals/*.npy     the waveforms referenced by the jsonl rows (`ori_path` relative to the repo root;
                                           OpenTSLM keeps its three subsets in ptbxl_signals/, har_signals/, sleep_signals/)
  understanding/MANIFEST.json              row / file counts and the sampling rule of every shipped file
  external/cik_eval.parquet                CiK official test split (355 instances) in our parquet schema
  external/stbench_t4_forecast.parquet     ST-Bench T4 forecasting split (650 train / 280 test) in our parquet schema
```

## Understanding test sets

All eight files are the official test splits used in the paper, with one exception: **ECG-QA is
subsampled** (250 of the 3,207 PTB-XL recordings, chosen uniformly at random with seed 0, and all
QA rows of a kept recording: 2,927 of 41,093 rows) to keep the repository under 300 MB — the
full ECG waveforms alone are 770 MB. Every other file is complete (HAR 8,199, Sleep 928, ST-Bench
2,993, VeriTime 1,155, TelecomTS 8,000, HiTSR 5,472, Time-RA 2,790). The full ECG-QA test set can be
rebuilt from the public PTB-XL release and the OpenTSLM ECG-QA-CoT mirror.

Row schema (one JSON object per line):

| key | meaning |
|---|---|
| `id`, `dataset_name`, `task`, `scene` | identifiers; `task` selects the scoring protocol, `scene` the source domain label |
| `input_text` | the question / instruction shown to the model (list with one string) |
| `think` | the annotated reasoning trace (empty for ST-Bench, whose official protocol is direct answering) |
| `gt_text` | the ground-truth answer, always prefixed `Answer: ` |
| `input_ts.original.ori_path` | waveform file, relative to the repository root (e.g. `data/understanding/opentslm/ptbxl_signals/16952.npy`); `channel` = number of channels, `ori_length` = length |

The training splits are not shipped (they are 1.6 GB of waveforms); they are rebuilt from the
public releases of the eight sources into the same jsonl schema as the shipped test files, and
`chronos_llm/configs/understanding_train_mmtr.txt` lists where they are expected.

Source licences: OpenTSLM data (PTB-XL CC-BY 4.0, UCI-HAR, Sleep-EDF), ST-Bench (Apache-2.0),
TelecomTS (MIT), Time-RA (Apache-2.0, gated download on the Hugging Face hub), VeriTime and
HiTSR (released without a licence file). The copies here are for review only; please obtain
each dataset from its original release for any other use.

## Forecasting corpus

`mmtr_forecast_corpus.parquet` is the complete event-conditioned forecasting part of MMTR
(annotations included), 23 columns:

| column | meaning |
|---|---|
| `id`, `dataset_name`, `batch`, `annotator` | `dataset_name` = `<source>/<subset>` (10 subsets); `annotator` = the LLM that wrote the trace |
| `freq`, `past_len`, `future_len`, `total_len` | sampling frequency and window geometry |
| `history_values`, `history_timestamps` | the history window (model input) |
| `future_values`, `future_timestamps` | the forecast target (never shown to the model) |
| `background`, `event` | dataset background and the event description (an established fact available at forecast time) |
| `prompt` | the full user prompt (background + event + series metadata + task instruction) |
| `plain_prompt` | the same facts without the reasoning instruction (used by the prompt-only ablation) |
| `reasoning` | the annotated reasoning chain (2–4 sentences: event time -> ROI, event + history -> shape) |
| `conclusion` | the forecast instruction: `[Magnitude: BAND]` + ROI index range + shape effect + relative-magnitude sentence |
| `structtag_band` | the magnitude band of the conclusion (TINY / WELL_BELOW / MOD_BELOW / TYPICAL / ABOVE) |
| `roi_start_idx`, `roi_end_idx`, `roi_shape` | the event region of interest, half-open `[start, end)` in absolute indices, and its shape phrase |
| `split` | `train` (4,630) / `test` (437) |

Domains (equal-weight averaging in the paper): Solar = CGTSF/MSPG, fidelts/Canada_photovoltaics_plants,
fidelts/Germany_Renewable_Power_Grid; Load = fnf/load, fidelts/California_ISO; Traffic = CGTSF/PTF,
fnf/traffic; Finance = finnews/MTBench_finance, fnf/bitcoin; Climate = timemmd/Climate. Seven test
rows whose series also appeared in the training split (an artefact of the upstream window sampling)
were removed from the test split (444 -> 437); the training split is unchanged. See
`docs/DATASET.md` for the dataset card.
