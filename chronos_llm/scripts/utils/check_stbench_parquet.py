"""One-off check: verify that the parquet produced by build_stbench_forecast_parquet.py can be loaded/rendered
correctly by ForecastParquetDataset (train supervised rendering + test inference rendering), scanning every
row to catch lazy skips. Loads a tokenizer.
"""
import os
import sys

sys.path.insert(0, ".")
from transformers import AutoTokenizer

from chronos_llm.data.forecast_dataset import ForecastParquetDataset

LLM = os.environ.get("LLM_PATH", "checkpoints/Qwen3.5-9B")
PARQUET = "data/forecast/mmtr_forecast_corpus.parquet"

tok = AutoTokenizer.from_pretrained(LLM)

print("=== train split (inference_mode=False) ===")
ds = ForecastParquetDataset(PARQUET, tok, split="train", max_user_tokens=1500, max_tokens=4096)
print("len:", len(ds))
item = ds[0]
print("keys:", list(item.keys()))
print("history shape:", item["history"].shape, "future shape:", item["future"].shape, "roi shape:", item["roi"].shape)
print("input_ids len:", len(item["input_ids"]), " labels len:", len(item["labels"]))
n_supervised = sum(1 for l in item["labels"] if l != -100)
print("supervised label count:", n_supervised)

print("\n=== test split (inference_mode=True) ===")
ds_test = ForecastParquetDataset(
    PARQUET, tok, split="test", inference_mode=True, emit_meta=True,
    max_user_tokens=1500, max_tokens=4096,
)
print("len:", len(ds_test))
item_t = ds_test[0]
print("keys:", list(item_t.keys()))
print("history shape:", item_t["history"].shape, "future shape:", item_t["future"].shape)
print("meta:", {k: (v if k != "input_text" else v[:80]) for k, v in item_t["meta"].items()})

print("\n=== full rendering pass over train (catch lazy skips / exceptions) ===")
for i in range(len(ds)):
    ds[i]
print("train _skip set size:", len(ds._skip))

print("\n=== full rendering pass over test ===")
for i in range(len(ds_test)):
    ds_test[i]
print("test _skip set size:", len(ds_test._skip))
print("\nOK - all passed")
