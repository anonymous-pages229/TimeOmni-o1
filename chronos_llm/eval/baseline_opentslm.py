"""OpenTSLM (arXiv 2510.02410, MIT-licensed, fully open source) zero-shot understanding baseline.

## Model choice

OpenTSLM releases a **matrix of checkpoints split by curriculum stage / task** (tsqa/har/sleep/ecg
x sp/flamingo x llama-1B/3B/gemma) with no single "general-purpose" model. Our 23 test sets are
cross-domain content QA => we take the **TSQA stage** (the general time-series QA of curriculum
stage 1, closest to "general understanding"); the backbone is its largest, llama-3.2-3B; both the
SoftPrompt and Flamingo variants were downloaded and the better-behaved one was kept after a GPU
smoke comparison (the paper promotes the Flamingo variant; its demo script defaults to SP only to
save memory).

## Protocol

Reuses the official `opentslm` package (pip install -e; the model code is not re-implemented):
`OpenTSLM.load_pretrained(repo_or_path)` + the official batch dict format
(pre_prompt/time_series/time_series_text/post_prompt) + the official collate (pad to a multiple of
patch_size) + `model.generate(batch)`.
- Series are z-scored (its TSQADataset convention); `time_series_text` uses its verbatim template
  "This is the time series, it has mean {m:.4f} and std {s:.4f}." (feeding back the true
  statistics -- exactly the discriminative signal validated in our short-window detection, and the
  format matches its training distribution).
- post_prompt uses its TSQA training template "Predict the {task} Answer:" (chosen after a smoke
  comparison against the generic "Answer:").
- Series are uniformly downsampled to `--max_points` (default 512, same convention as
  ChatTime / Time-MQA).

## Usage (GPU node, dedicated venv per the setup notes)

  python chronos_llm/eval/baseline_opentslm.py \\
    --model_path checkpoints/OpenTSLM-llama3b-tsqa-flamingo \\
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \\
    --base_dir data/raw/scits/Release_v1 \\
    --out_dir outputs/eval/baseline_opentslm/understanding [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch


def load_ts_first_channel(path, max_len=0):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".csv":
            import pandas as pd
            arr = pd.read_csv(path, header=None).apply(
                lambda c: pd.to_numeric(c, errors="coerce")).values
        elif ext in (".wav", ".flac", ".mp3", ".m4a"):
            import soundfile as sf
            arr, _ = sf.read(path, dtype="float32", always_2d=True)
        else:
            raise ValueError(f"unsupported ext {ext}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"load {path} failed: {e}")
        return np.zeros(16, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.T
    if arr.ndim == 1:
        arr = arr[None, :]
    for row in arr:
        if np.isfinite(row).any():
            s = row
            break
    else:
        return np.zeros(16, dtype=np.float32)
    if max_len and len(s) > max_len:
        idx = np.linspace(0, len(s) - 1, max_len).round().astype(int)
        s = s[idx]
    return np.nan_to_num(s, nan=0.0)


def build_batch_element(question: str, series: np.ndarray, task: str, post_style: str):
    m, s = float(series.mean()), float(series.std())
    z = (series - m) / (s + 1e-8)
    if post_style == "task":
        post = f"Predict the {task} Answer:"
    else:
        post = "Answer:"
    return {
        "pre_prompt": question.strip(),
        "time_series": [z.tolist()],
        "time_series_text": [
            f"This is the time series, it has mean {m:.4f} and std {s:.4f}."],
        "post_prompt": post,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="local checkpoint directory (containing model_checkpoint.pt)")
    ap.add_argument("--base_llm", default="checkpoints/Llama-3.2-3B-nopad",
                     help="local path of the backbone LLM (the unsloth mirror: same weights as meta-llama, not gated; "
                          "the -nopad variant removes the preset tokenizer pad_token to match the vocabulary layout "
                          "the official checkpoint was trained with -- the unsloth mirror sets pad_token by default, "
                          "which makes the resized vocabulary one row short and load_state_dict fail with an "
                          "embed_tokens/lm_head size mismatch)")
    ap.add_argument("--variant", default="flamingo", choices=["flamingo", "sp"])
    ap.add_argument("--jsonl_list", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_points", type=int, default=512)
    ap.add_argument("--post_style", default="task", choices=["task", "plain"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="MIMII")
    ap.add_argument("--file_shard_idx", type=int, default=0)
    ap.add_argument("--file_num_shards", type=int, default=1)
    args = ap.parse_args()

    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate)
    from opentslm.model_config import PATCH_SIZE

    # We do not use the official OpenTSLM.load_pretrained (it hard-wires a HF hub download of the
    # checkpoint and maps the backbone to the gated meta-llama/Llama-3.2-3B -- guaranteed to fail
    # on an offline compute node): assemble locally in an equivalent way following its internal
    # logic (Flamingo variant + cross_attn_every_n_layers=1 are the fixed parameters of its
    # load_pretrained), with the backbone taken from the local unsloth/Llama-3.2-3B mirror (same
    # weights, not gated; passed via --base_llm).
    print(f"[opentslm] loading {args.model_path} (base={args.base_llm}) ...")
    t0 = time.time()
    if args.variant == "flamingo":
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
        model = OpenTSLMFlamingo(device="cuda", llm_id=args.base_llm,
                                 cross_attn_every_n_layers=1, gradient_checkpointing=False)
    else:
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
        model = OpenTSLMSP(llm_id=args.base_llm, device="cuda")
    ckpt = args.model_path
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "model_checkpoint.pt")
    model.load_from_file(ckpt)
    # In the checkpoint the vision_encoder is stored in fp32 while the lang_encoder (+ flamingo
    # cross-attn layers) is bf16 -- mixing them fails on dtype at the cross-attn LayerNorm
    # ("expected BFloat16 but found Float"). Cast everything to bf16 (matches the native precision
    # of the dominant lang_encoder; the lower precision of the vision_encoder has no material effect
    # on pure inference).
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"[opentslm] loaded in {time.time()-t0:.1f}s")

    files = [l.strip() for l in open(args.jsonl_list)
             if l.strip() and not l.strip().startswith("#")]
    files = [f for f in files if not any(e in f for e in args.exclude.split(",") if e)]
    if args.file_num_shards > 1:
        files = files[args.file_shard_idx:: args.file_num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[opentslm] shard {args.file_shard_idx}/{args.file_num_shards}: {len(files)} files")

    for jf in files:
        rows = [json.loads(l) for l in open(jf)]
        if args.limit:
            rows = rows[: args.limit]
        name = os.path.basename(jf)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name} already exists")
            continue
        print(f"[opentslm] {name}: {len(rows)} rows")
        results = []
        n_fail = 0
        for ridx, r in enumerate(rows):
            t_row = time.time()
            ori = ((r.get("input_ts") or {}).get("original") or {}).get("ori_path") or ""
            path = ori if os.path.isabs(ori) else os.path.join(args.base_dir, ori)
            q = (r.get("input_text") or [""])[0]
            gen_text = ""
            try:
                s = load_ts_first_channel(path, max_len=args.max_points)
                el = build_batch_element(q, s, str(r.get("task") or "QA"), args.post_style)
                batch = extend_time_series_to_match_patch_size_and_aggregate(
                    [el], patch_size=PATCH_SIZE)
                # The official util hard-codes time_series as fp32; the model was cast to bf16
                # above, so cast here to match, otherwise the first conv/norm layer of the
                # vision_encoder fails on dtype.
                for item in batch:
                    if isinstance(item.get("time_series"), torch.Tensor):
                        item["time_series"] = item["time_series"].to(torch.bfloat16)
                with torch.no_grad():
                    preds = model.generate(batch, max_new_tokens=args.max_new_tokens)
                gen_text = (preds[0] if preds else "").strip()
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  [warn] {name} row {ridx} failed: {type(e).__name__}: {e}")

            results.append({
                "id": r.get("id"), "uid": r.get("uid"),
                "dataset_name": r.get("dataset_name"), "task": r.get("task"),
                "scene": r.get("scene"), "input_text": r.get("input_text"),
                "generated_text": gen_text,
                "ground_truth": r.get("gt_text"), "gt_result": r.get("gt_result"),
            })
            if (ridx + 1) % 20 == 0 or ridx == 0:
                print(f"  {ridx+1}/{len(rows)} (last row {time.time()-t_row:.1f}s, failures so far {n_fail})")

        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_path} (fail {n_fail}/{len(rows)})")


if __name__ == "__main__":
    main()
