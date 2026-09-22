#!/usr/bin/env python3
"""Quantify the cost of the TimesFM adapter's "bucket by channel count" strategy -- the explanation
for full understanding-side training being 2.8x slower.

Background: `TimesFM3Backbone._run_tfm` groups the rows of a batch by the number of variates per
group and runs one forward per bucket (otherwise SDPA returns NaN on fully masked rows, and padded
rows would pollute variate attention). On the forecasting side the channel count is uniform, so
there is about one bucket; the understanding-side pool (`agg6`, the six-source MMTR understanding
mix) however contains 7 distinct channel counts (1/3/5/9/10/12/16), so with random batching every
step runs 4-7 forwards.

For the same total number of rows this script compares:
  A) a single channel count (1 bucket)        -- the forecasting-side shape
  B) the real agg6 channel distribution       -- the understanding-side shape
Both do exactly the same amount of arithmetic, so the difference is **the cost of bucketing
itself** (kernel launches plus poor GPU utilisation at small batch sizes).
Chronos-2 is benchmarked next to it as a reference (it folds rows through group_ids and needs no
bucketing at all).

    python chronos_llm/scripts/utils/bench_timesfm_bucketing.py
"""
import os, sys, time
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from chronos_llm.models.timesfm_backbone import load_timesfm3_backbone   # noqa: E402
from chronos_llm.models.cross_attn_chronos import load_chronos2_with_cross_attn  # noqa: E402

TFM = "checkpoints/TimesFM3.0"
CHR = "checkpoints/chronos-2"
# measured agg6 distribution (share of samples): 1ch 18.1% / 3ch 14.9% / 5ch 1.6% / 9ch 0.2% / 10ch 2.2% / 12ch 32.0% / 16ch 31.0%
AGG6 = [(1, .181), (3, .149), (5, .016), (9, .002), (10, .022), (12, .320), (16, .310)]
L = 2048          # time steps per channel (the order of magnitude seen on the understanding side after encode_sample_chunks)
N_SAMPLES = 24    # samples per batch


def make_groups(spec, n_samples):
    """spec=[(channel count, share)] -> group_ids (one group per sample, C rows inside a group)."""
    gids, g = [], 0
    for c, frac in spec:
        for _ in range(max(1, round(n_samples * frac)) if frac < 1 else n_samples):
            gids += [g] * c
            g += 1
    return torch.tensor(gids, dtype=torch.long)


def bench(model, gids, dev, dtype, tag, iters=8):
    R = gids.numel()
    ctx = torch.randn(R, L, device=dev, dtype=torch.float32)
    loc = torch.zeros(R, device=dev); scale = torch.ones(R, device=dev)
    gids = gids.to(dev)
    n_buckets = len(torch.unique(torch.bincount(gids)[torch.bincount(gids) > 0]).tolist())
    with torch.no_grad():
        for i in range(iters + 2):
            if i == 2:
                torch.cuda.synchronize(); t0 = time.time()
            model(ctx, num_output_patches=1, group_ids=gids)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    print(f"  {tag:38s} rows={R:4d} groups={int(gids.max())+1:3d} buckets={n_buckets} -> {dt*1000:8.1f} ms/call")
    return dt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"device={dev} dtype={dtype}\n")

    tfm = load_timesfm3_backbone(TFM, dtype=dtype).to(dev).eval()
    print("== TimesFM-3.0 ==")
    # A) single channel count: the total row count is tuned close to B so that the arithmetic is comparable
    g_single = make_groups([(12, 1.0)], N_SAMPLES)
    a = bench(tfm, g_single, dev, dtype, "A) single channel count (12ch, 1 bucket)")
    g_mixed = make_groups(AGG6, N_SAMPLES)
    b = bench(tfm, g_mixed, dev, dtype, "B) real agg6 distribution (many buckets)")
    print(f"  -> per-row cost: A {a/ g_single.numel()*1e3:.3f} ms/row   B {b/ g_mixed.numel()*1e3:.3f} ms/row"
          f"   bucketing overhead ~ {(b/g_mixed.numel())/(a/g_single.numel()):.2f}x\n")

    del tfm; torch.cuda.empty_cache()
    chr_ = load_chronos2_with_cross_attn(CHR, dtype=dtype).to(dev).eval()
    print("== Chronos-2 (reference, no bucketing needed) ==")
    for tag, g in (("A) single channel count (12ch)", g_single), ("B) real agg6 distribution", g_mixed)):
        R = g.numel()
        ctx = torch.randn(R, L, device=dev, dtype=torch.float32)
        with torch.no_grad():
            for i in range(10):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                chr_(ctx, num_output_patches=1, group_ids=g.to(dev))
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 8
        print(f"  {tag:38s} rows={R:4d} -> {dt*1000:8.1f} ms/call  ({dt/R*1e3:.3f} ms/row)")


if __name__ == "__main__":
    main()
