# MMTR dataset card

MMTR supervises multimodal time series reasoning with an annotated reasoning trace for both task
types. Totals: **502,586 train / 71,074 test** samples.

## Understanding part (497,956 train / 70,630 test)

Six public reasoning-annotated corpora unified under one schema (series, question, trace, answer):

| source | subtasks (`task` field) | answer format | train | test | channels x length |
|---|---|---|---|---|---|
| OpenTSLM ECG-QA-CoT | `ecg_qa_cot` | open QA (`Answer: yes/no/<attribute>`) | 159,313 | 41,093 | 12 x 5000 |
| OpenTSLM HAR-CoT | `har_cot` | classification | 68,384 | 8,199 | 3 x 128 |
| OpenTSLM Sleep-CoT | `sleep_cot` | classification | 7,428 | 928 | 1 x 1500 |
| ST-Bench T1/T2/T3 | `stbench_{etiological,entity,correlation}` | MCQ (A–D) | 23,305 | 2,993 | 3–10 nodes x 48–360 |
| VeriTime / TSRBench | `veritime_{Scenario_attribution,Inferential_calculation,Anomaly_detection,ECG,EMG,RCW,CTU}` | MCQ / numeric / classification | 3,200 | 1,155 | 1 x 256–720 |
| TelecomTS | `telecomts_{network,anomalies}` | open QA | 154,185 | 8,000 | 16 x 128 |
| HiTSR L2 / L3 | `hitsr_{l2,l3}` | MCQ (A–D) | 49,147 | 5,472 | 1 x ~160 |
| Time-RA / RATs40K | `time_ra_{uni,multi}` | anomaly-type classification (15 classes) | 32,994 | 2,790 | 1–10 x 128 |

Every sample is assigned to one of 12 disciplines; the paper's overall score is the arithmetic
mean of the 12 discipline scores.

| discipline | test | train | contributing sources |
|---|---:|---:|---|
| Physiology | 50,792 | 241,804 | ECG-QA, HAR, Time-RA, VeriTime |
| Telecommunications | 8,000 | 154,185 | TelecomTS |
| Cross-domain / Synthetic | 5,488 | 49,324 | HiTSR L2/L3, Time-RA `Artificial` |
| Urbanism | 3,029 | 23,587 | ST-Bench, VeriTime, Time-RA `Traffic_Flow` |
| Information Technology | 937 | 11,894 | Time-RA, VeriTime (K8s / Redis / Oracle / web services) |
| Neuroscience | 928 | 7,428 | Sleep |
| Earth Science | 492 | 4,895 | Time-RA, VeriTime |
| Meteorology | 219 | 993 | Time-RA `Ocean-Tropical Atmosphere`, VeriTime `Weather` |
| Economics | 201 | 1,405 | VeriTime (finance / retail / advertising / media ...), Time-RA |
| Bioacoustics | 200 | 127 | VeriTime RCW |
| Manufacturing | 167 | 2,034 | Time-RA `Machine-System`, VeriTime |
| Energy | 160 | 198 | VeriTime CTU |

Sleep is filed under Neuroscience and HAR under Physiology, following the SciTS taxonomy.
The train/test discipline mix is not identical (Telecommunications is 31% of training but 11% of
test; Bioacoustics has fewer training than test samples), which motivates the discipline-rebalanced
SFT pool used in stage 2 (`make_domain12_pool.py`: nine small/medium disciplines in full, the three
smallest up-sampled to 1,000 rows, Physiology / Telecommunications / Cross-domain down-sampled to
20k / 8k / 8k; 90,243 rows per epoch).

## Forecasting part (4,630 train / 437 test)

| domain | subset | frequency | history | horizon | train | test |
|---|---|---|---:|---:|---:|---:|
| Solar | CGTSF/MSPG | 15 min | 480 | 96 | 613 | 59 |
| Solar | fidelts/Canada_photovoltaics_plants | 1 h | 96 | 24 | 650 | 48 |
| Solar | fidelts/Germany_Renewable_Power_Grid | 15 min | 284–292 | 92–96 | 565 | 56 |
| Load | fnf/load | 30 min | 48 | 48 | 736 | 75 |
| Load | fidelts/California_ISO | 5 min | 576 | 288 | 464 | 46 |
| Traffic | CGTSF/PTF | 1 h | 120 | 24 | 532 | 47 |
| Traffic | fnf/traffic | 1 h | 120 | 48 | 365 | 35 |
| Finance | finnews/MTBench_finance | intraday | 276–403 | 7–81 | 405 | 41 |
| Finance | fnf/bitcoin | 1 day | 60 | 14 | 173 | 17 |
| Climate | timemmd/Climate | 1 week | 36 | 12 | 127 | 13 |

The two photovoltaic subsets with a `#dupK` suffix in `id` contain up-sampled cloudy-day windows
(training split only). Annotators: Claude Sonnet (2,500 windows) and GPT-5.4 (3,000 windows before
subset filtering). Median lengths: event 52 words, reasoning 81 words, conclusion 23 words; the
ROI covers a median 24% of the horizon.

Each sample provides the history, the background, the event (an established fact available at
forecast time), the annotated reasoning chain and forecast instruction (ROI + shape + relative
magnitude, with a `[Magnitude: BAND]` tag), and the explicit ROI used for ROI-weighted training and
ROI metrics. Column-level documentation is in `data/README.md`.
