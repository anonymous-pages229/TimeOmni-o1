"""Diagnostic: when back-propagating with `cross_states` attached, which gradients actually reach
the TimesFM backbone, and which reach `cross_states` itself.

Hypothesis: the backbone getting no gradient in the fully wired model is specific to the
**feedback** path -- the back-propagation cases of the adapter-layer unit tests never enable
cross_states (one case runs without feedback, another runs under no_grad), so that path has never
been verified by a backward pass.
"""
import os, sys, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_REPO, os.path.join(_REPO, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
from chronos_llm.models.timesfm_backbone import load_timesfm3_backbone

CKPT = "checkpoints/TimesFM3.0"
torch.manual_seed(0)
bb = load_timesfm3_backbone(CKPT)
bb.train()
d = bb.config.d_model
R, L, fl = 4, 192, 64
nop = 1

def report(tag, cs=None, nan_pad=False):
    bb.zero_grad(set_to_none=True)
    ctx = torch.randn(R, L) * 3 + 10
    if nan_pad:
        ctx[1, :96] = float('nan')     # in the full-model tests the context is left-padded with NaN
    future = torch.randn(R, fl) * 3 + 10
    kw = {}
    if cs is not None:
        kw = {"cross_states": cs, "cross_states_mask": torch.ones(R, cs.shape[1])}
    res = bb.forecast_losses(context=ctx, future_target=future, num_output_patches=nop,
                             group_ids=torch.arange(R), **kw)
    pl = res["pred_loss"]
    pl.backward()
    body = [(n, p) for n, p in bb.named_parameters() if ".cross_attn." not in ("." + n)]
    xatt = [(n, p) for n, p in bb.named_parameters() if ".cross_attn." in ("." + n)]
    nz_body = sum(1 for _, p in body if p.grad is not None and float(p.grad.abs().sum()) > 0)
    nz_x = sum(1 for _, p in xatt if p.grad is not None and float(p.grad.abs().sum()) > 0)
    prb = bb.tfm.pre_transformer_resblock.hidden_layer.weight
    g = prb.grad
    gs = 'None' if g is None else f'{float(g.abs().sum()):.3e}'
    csg = 'n/a' if cs is None else ('None' if cs.grad is None else f'{float(cs.grad.abs().sum()):.3e}')
    print(f"[{tag}] loss={float(pl):.4f} backbone nonzero grads {nz_body}/{len(body)} | "
          f"cross_attn nonzero {nz_x}/{len(xatt)} | resblock.hidden |g|={gs} | cross_states.grad={csg}")

report("no feedback")
report("no feedback + NaN pad", nan_pad=True)
for gate in (0.0, 0.3):
    with torch.no_grad():
        for n, p in bb.named_parameters():
            if n.endswith(".cross_attn.gate"):
                p.fill_(gate)
    cs = torch.randn(R, 12, d, requires_grad=True)
    report(f"feedback gate={gate}", cs=cs)
    cs2 = torch.randn(R, 12, d, requires_grad=True)
    report(f"feedback gate={gate} + NaN pad", cs=cs2, nan_pad=True)
