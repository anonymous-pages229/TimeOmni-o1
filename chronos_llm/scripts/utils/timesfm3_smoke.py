"""TimesFM-3.0 loading smoke test: check that the weights really made it into the model and that
the forward / decode paths produce sane numbers.

Why "did the weights actually load" deserves its own check: under transformers 5.x, Chronos-2's
``from_pretrained`` **silently** returns a model whose weights were never loaded, which is why this
project builds that backbone explicitly and calls ``load_state_dict`` instead. TimesFM3 goes
through huggingface_hub's ``PyTorchModelHubMixin`` (a different implementation), but the same trap
is worth nailing down before building anything on top of it -- this script compares every raw
tensor in the safetensors file against ``model.state_dict()`` and raises on any mismatch.

Usage (CPU only is enough)::

    python chronos_llm/scripts/utils/timesfm3_smoke.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

DEFAULT_CKPT = "checkpoints/TimesFM3.0"


def check_weights_loaded(model: torch.nn.Module, ckpt_dir: str) -> None:
    """Compare the raw safetensors file with the loaded model tensor by tensor, guarding against a
    silent "weights were never loaded" failure."""
    from safetensors.torch import load_file

    raw = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    sd = model.state_dict()

    missing = [k for k in raw if k not in sd]
    extra = [k for k in sd if k not in raw]
    if missing:
        raise RuntimeError(f"{len(missing)} tensors of the checkpoint never reached the model, e.g. {missing[:5]}")

    n_checked, n_bad = 0, []
    for k, v in raw.items():
        got = sd[k]
        if got.shape != v.shape:
            n_bad.append(f"{k}: shape {tuple(got.shape)} != {tuple(v.shape)}")
            continue
        if not torch.equal(got.to(v.dtype).cpu(), v.cpu()):
            n_bad.append(f"{k}: values differ (max|delta|={(got.cpu().float() - v.cpu().float()).abs().max():.3e})")
        n_checked += 1
    if n_bad:
        raise RuntimeError(f"{len(n_bad)} tensors disagree with the checkpoint:\n  " + "\n  ".join(n_bad[:10]))

    print(f"[weights] OK: {n_checked} tensors match value by value; {len(extra)} tensors exist only in the model (not in the checkpoint)"
          + (f", e.g. {extra[:3]}" if extra else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()

    from timesfm3.torch.model import TimesFM3Torch

    print(f"[load] {args.ckpt}")
    model = TimesFM3Torch.from_pretrained(args.ckpt)
    model.eval()

    n_param = sum(p.numel() for p in model.parameters())
    print(f"[load] params {n_param / 1e6:.1f}M | d_model={model.transformer_config.transformer.model_dims} "
          f"| layers={len(model.transformer_stack.layers)} | quantiles={model.quantiles} "
          f"| input_patch={model.input_patch_len} output_patch={model.output_patch_len}")

    check_weights_loaded(model, args.ckpt)

    # ---- decode: extrapolate a sine wave and check the numbers are plausible
    #      (this is the official inference path, wrapped in @no_grad) ----
    torch.manual_seed(0)
    ctx_len, horizon = 512, 64
    t = np.arange(ctx_len + horizon, dtype=np.float32)
    wave = np.sin(2 * np.pi * t / 64.0) * 3.0 + 10.0
    ctx = torch.from_numpy(wave[None, None, :ctx_len])          # (b=1, u=1, ctx)
    truth = wave[ctx_len:]

    out = model.decode(target=ctx, horizon=horizon)
    print(f"[decode] logits shape={tuple(out.shape)}  (expected (b, v, horizon, num_quantiles))")
    assert out.shape == (1, 1, horizon, len(model.quantiles)), out.shape
    assert torch.isfinite(out).all(), "decode output contains NaN/Inf"

    median = out[0, 0, :, len(model.quantiles) // 2].float().numpy()
    mae = float(np.abs(median - truth).mean())
    print(f"[decode] sine extrapolation MAE={mae:.4f} (amplitude 3.0, offset 10.0; guessing at random scores about 1.9)")
    assert mae < 1.0, f"sine extrapolation MAE={mae:.3f} is far too large, weights or forward pass are probably wrong"

    # ---- forward: obtain transformer_output (the KV source consumed by the history Q-former) ----
    n_ctx_patches = ctx_len // model.input_patch_len
    n_hor_patches = horizon // model.input_patch_len
    n_patches = n_ctx_patches + n_hor_patches
    values = torch.zeros(1, 1, n_patches, model.input_patch_len)
    values[:, :, :n_ctx_patches] = ctx.view(1, 1, n_ctx_patches, model.input_patch_len)
    masks = torch.zeros_like(values, dtype=torch.bool)          # False = valid (opposite of chronos)
    masks[:, :, n_ctx_patches:] = True                          # horizon patches carry no observation
    patch_is_target = torch.zeros(1, 1, n_patches, dtype=torch.bool)
    patch_is_target[:, :, n_ctx_patches:] = True

    res = model.forward(
        {"values": values, "masks": masks, "patch_is_target": patch_is_target},
        return_aux_outputs=True,
    )
    hid = res["__call__:transformer_output"]
    print(f"[forward] transformer_output shape={tuple(hid.shape)}  (b, v, n_patch, d_model)")
    assert hid.shape[-1] == model.transformer_config.transformer.model_dims
    assert torch.isfinite(hid).all(), "transformer_output contains NaN/Inf"
    print(f"[forward] hidden stats mean={hid.mean():.4f} std={hid.std():.4f}")

    # ---- multivariate: can variate attention handle C>1 ----
    ctx2 = torch.cat([ctx, ctx * 0.5 + 1.0], dim=1)             # (1, 2, ctx)
    out2 = model.decode(target=ctx2, horizon=horizon)
    print(f"[multivariate] C=2 decode shape={tuple(out2.shape)}")
    assert out2.shape == (1, 2, horizon, len(model.quantiles))
    assert torch.isfinite(out2).all()

    # ---- gradients: the training path must back-propagate (decode is @no_grad, forward is not) ----
    model.train()
    res_g = model.forward({"values": values.requires_grad_(False), "masks": masks,
                           "patch_is_target": patch_is_target})
    loss = res_g["logits"].float().square().mean()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_all = sum(1 for _ in model.parameters())
    print(f"[grad] forward back-propagates: {n_grad}/{n_all} parameters received a finite gradient")
    assert n_grad > n_all * 0.8, "most parameters received no gradient, the training path is broken"

    print("\nAll TimesFM-3.0 smoke checks passed")


if __name__ == "__main__":
    main()
