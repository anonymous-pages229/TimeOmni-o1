"""Paired significance test: per-sample CRPS/PCC differences between two independent evaluation
npz files (same test split, same samples).

- CRPS is already computed per sample and then averaged (metrics.sample_wql) => it naturally
  supports a paired test.
- Three tests cross-check each other: paired t (mean difference), Wilcoxon signed-rank (robust to
  skew), paired bootstrap 10k (resample samples; reports the 95% CI of the difference and P(A<B)).
- Usage: python significance_test.py <npz_A> <npz_B> [label_A] [label_B]
"""
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from chronos_llm.eval.metrics import sample_pcc, sample_wql


def per_sample(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    q, gt, roi, vm, lv = (d["pred_quantiles"], d["gt"], d["roi_mask"],
                          d["valid_mask"], d["quantile_levels"])
    med = q[:, np.abs(lv - 0.5).argmin(), :]
    n = len(gt)
    crps_f = np.array([sample_wql(gt[i], q[i], lv, vm[i]) for i in range(n)])
    crps_r = np.array([sample_wql(gt[i], q[i], lv, vm[i] & (roi[i] > 0)) for i in range(n)])
    pcc_f = np.array([sample_pcc(gt[i], med[i], vm[i]) for i in range(n)])
    return d["ids"], {"full CRPS": crps_f, "roi CRPS": crps_r, "full PCC": pcc_f}


def paired_tests(a, b, name, la, lb, higher_better=False):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = a - b
    t_p = stats.ttest_rel(a, b).pvalue
    w_p = stats.wilcoxon(a, b, zero_method="wilcox").pvalue if np.any(diff != 0) else 1.0
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(10000, len(a)))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])
    p_better = float((boot < 0).mean()) if not higher_better else float((boot > 0).mean())
    win = float((a < b).mean()) if not higher_better else float((a > b).mean())
    print(f"\n== {name} (n={len(a)}) ==")
    print(f"  {la}: mean={a.mean():.5f}   {lb}: mean={b.mean():.5f}   diff={a.mean()-b.mean():+.5f}")
    print(f"  per-sample win rate ({la} better): {win:.1%}")
    print(f"  paired t-test  p={t_p:.4f}")
    print(f"  Wilcoxon      p={w_p:.4f}")
    print(f"  bootstrap 95% CI of diff: [{ci[0]:+.5f}, {ci[1]:+.5f}]   P({la} better)={p_better:.1%}")


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    ids_a, ma = per_sample(pa)
    ids_b, mb = per_sample(pb)
    assert list(ids_a) == list(ids_b), "the two npz files are not sample-aligned"
    print(f"paired samples: {len(ids_a)}   A={la}({pa})   B={lb}({pb})")
    for k in ma:
        paired_tests(ma[k], mb[k], k, la, lb, higher_better=("PCC" in k))


if __name__ == "__main__":
    main()
