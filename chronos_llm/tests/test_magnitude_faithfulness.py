"""Unit tests for the magnitude-field faithfulness script: band edges + closed-form shuffled null
hypothesis == brute-force enumeration.

The closed form sum_{i!=j} 1[w_i == p_j] / (n(n-1)) is the core of that script (it makes the
"shuffled control" need neither a judge rerun nor sampling, so it has no sampling noise); if it were
wrong it would not raise but silently give a plausible-looking wrong number -- so it must be checked
case by case against the brute-force double loop. Pure numpy, no model loaded.
"""
import random
import sys

sys.path.insert(0, ".")

from chronos_llm.scripts.utils.magnitude_faithfulness import (  # noqa: E402
    BAND_ORDER, band_of, shuffled_hit_rate,
)


def brute_force_null(written, predicted):
    n = len(written)
    hits = sum(1 for i in range(n) for j in range(n) if i != j and written[i] == predicted[j])
    return hits / (n * (n - 1))


def test_band_edges():
    # band edges 0.10/0.40/0.70/1.30, closed on the left, open on the right
    assert band_of(0.0) == "tiny"
    assert band_of(0.0999) == "tiny"
    assert band_of(0.10) == "well_below"
    assert band_of(0.3999) == "well_below"
    assert band_of(0.40) == "mod_below"
    assert band_of(0.70) == "typical"
    assert band_of(1.2999) == "typical"
    assert band_of(1.30) == "above"
    assert band_of(1e6) == "above"
    print("  band_of edges OK")


def test_null_matches_brute_force():
    rng = random.Random(0)
    for trial in range(200):
        n = rng.randint(2, 12)
        w = [rng.choice(BAND_ORDER) for _ in range(n)]
        # in half of the trials predicted is correlated with written (many self-matches), in the
        # other half fully independent
        if trial % 2 == 0:
            p = [w[i] if rng.random() < 0.6 else rng.choice(BAND_ORDER) for i in range(n)]
        else:
            p = [rng.choice(BAND_ORDER) for _ in range(n)]
        got, want = shuffled_hit_rate(w, p), brute_force_null(w, p)
        assert abs(got - want) < 1e-12, f"n={n} got={got} want={want}\nw={w}\np={p}"
    print("  closed-form null hypothesis == brute-force enumeration (200 random cases) OK")


def test_null_excludes_self_pairing():
    """When every self-pair is a hit, the mismatched null hypothesis must be clearly below the true
    hit rate of 1.0 -- the diagonal must not be counted."""
    w = ["tiny", "typical", "above", "tiny"]
    assert shuffled_hit_rate(w, list(w)) == brute_force_null(w, list(w))
    assert shuffled_hit_rate(w, list(w)) < 1.0
    # constant column: even mismatched pairs always hit => null hypothesis == 1.0 (the degenerate
    # zero-discrimination case, which must be reported honestly as 1.0)
    c = ["typical"] * 5
    assert abs(shuffled_hit_rate(c, list(c)) - 1.0) < 1e-12
    print("  self-pairing excluded / constant-column degeneracy OK")


def test_degenerate_sizes():
    import math
    assert math.isnan(shuffled_hit_rate([], []))
    assert math.isnan(shuffled_hit_rate(["tiny"], ["tiny"]))
    print("  n<2 returns nan OK")


if __name__ == "__main__":
    test_band_edges()
    test_null_matches_brute_force()
    test_null_excludes_self_pairing()
    test_degenerate_sizes()
    print("ALL OK")
