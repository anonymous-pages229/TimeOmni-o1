"""Scheduled-sampling training: continue training from the magnitude SFT checkpoint, feeding the
model's **own generated** conclusion back into chronos to obtain the pred/roi losses, and **train
only chronos + feedback_qformer** (LLM frozen) -- so that chronos adapts to the generation
distribution and the gen-tf gap shrinks. Goal: turn the teacher-forced advantage of the magnitude
SFT checkpoint into an advantage under free generation (surpassing the prompt-only arm).

Usage (scripts/train_ss.sh):
  python -m chronos_llm.rl.train_ss --init_ckpt <mag_best> --parquet <corpus_mag> --output_dir ...
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.models.chronos_llm_model import ChronosLLM


def _save_ckpt(model, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.save_pretrained(path)


def _freeze_ss(model, scope="feedback_crossattn"):
    """Freezing strategies:
    - ``feedback_crossattn`` (default, stable): train only the **feedback path** -- feedback_qformer +
      the gated cross_attn injected into chronos, **leaving the Chronos-2 backbone untouched** (the
      pretrained weights are sensitive to generated feedback; training everything makes the gradient
      explode).
    - ``chronos_feedback``: train the whole chronos + feedback (more adaptive but unstable).
    - ``no_llm`` (companion of stage2 / tf mode): freeze only the LLM, train chronos + both
      Q-formers -- the only difference from regular training is that the LLM stops updating, so the
      gen distribution no longer drifts and chronos's convergence under tf transfers ~1:1 to gen.
    The LLM (including LoRA) is frozen; the first two scopes also freeze history_qformer."""
    n_train = 0
    for n, p in model.named_parameters():
        if scope == "feedback_crossattn":
            keep = ("feedback_qformer" in n) or (".cross_attn." in n)
        elif scope == "no_llm":
            keep = ("chronos" in n) or ("feedback_qformer" in n) or ("history_qformer" in n)
        else:
            keep = ("chronos" in n or "feedback_qformer" in n) and ("history_qformer" not in n)
        p.requires_grad_(keep)
        if keep:
            n_train += p.numel()
    return n_train


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_user_tokens", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every_steps", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--train_scope", default="feedback_crossattn",
                    choices=["feedback_crossattn", "chronos_feedback", "no_llm"])
    ap.add_argument("--mode", default="ss", choices=["ss", "tf"],
                    help="ss=feed back the self-generated conclusion (shrinks the gap); tf=feed back the teacher "
                         "text (stage2: LLM frozen, keep converging chronos; no generation, fast and stable, the "
                         "dataset automatically uses the training rendering)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    print(f"loading magnitude SFT checkpoint {args.init_ckpt} ...")
    model = ChronosLLM.from_pretrained(args.init_ckpt, merge=False, is_trainable=True).to(args.device)
    tok = model.tokenizer
    n_train = _freeze_ss(model, args.train_scope)
    print(f"SS train_scope={args.train_scope}: trainable parameters {n_train/1e6:.2f}M")

    ds = ForecastParquetDataset(args.parquet, tok, split="train",
                                inference_mode=(args.mode == "ss"),
                                max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=ChronosLLMCollator(tokenizer=tok))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            future = batch.get("future")
            if future is None or future.shape[0] == 0:
                continue
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if args.mode == "tf":
                out = model.tf_forecast_loss(batch)
            else:
                out = model.ss_forecast_loss(batch, max_new_tokens=args.max_new_tokens)
            opt.zero_grad()
            out["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                pl = float(out["pred_loss"]); rl = out["roi_loss"]
                print(f"ep{epoch} step{step} | loss={float(out['loss']):.4f} pred={pl:.4f} "
                      f"roi={float(rl):.4f} grad_norm={float(gn):.3f}" if rl is not None else
                      f"ep{epoch} step{step} | loss={float(out['loss']):.4f} pred={pl:.4f} grad_norm={float(gn):.3f}",
                      flush=True)
            if step % args.save_every_steps == 0:
                _save_ckpt(model, os.path.join(args.output_dir, f"checkpoint-{step}"))
                print(f"saved checkpoint-{step}", flush=True)
            if args.max_steps and step >= args.max_steps:
                _save_ckpt(model, os.path.join(args.output_dir, "final"))
                print("reached max_steps, stopping", flush=True); return
    _save_ckpt(model, os.path.join(args.output_dir, "final"))
    print("SS training finished")


if __name__ == "__main__":
    main()
