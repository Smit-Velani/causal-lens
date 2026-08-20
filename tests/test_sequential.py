"""
test_sequential.py — Tests for always-valid inference.

The claim being tested is that peeking breaks fixed-horizon tests and does
not break sequential ones. Both halves need checking: a method that never
rejects controls alpha trivially and is useless.
"""

import numpy as np
import pytest

from sequential import (fixed_horizon_pvalue, always_valid_pvalue,
                        msprt_statistic, confidence_sequence, simulate_peeking)


def test_always_valid_pvalue_in_range():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    p = always_valid_pvalue(a, b)
    assert 0.0 <= p <= 1.0


def test_always_valid_is_more_conservative_than_fixed():
    """
    On the same data the always-valid p-value should never be smaller than
    the fixed-horizon one -- that gap is what pays for anytime-validity.
    """
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(0.1, 1, 400)
        b = rng.normal(0, 1, 400)
        assert always_valid_pvalue(a, b) >= fixed_horizon_pvalue(a, b) - 1e-12


def test_msprt_detects_large_effect():
    """A clearly real effect should push the likelihood ratio well above 1."""
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1, 1000)
    b = rng.normal(0.0, 1, 1000)
    assert msprt_statistic(a, b) > 20


def test_msprt_stays_low_under_null():
    """Under A/A the ratio should not favour the alternative."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)
    assert msprt_statistic(a, b) < 20


def test_confidence_sequence_wider_than_fixed_horizon():
    """
    Anytime-validity costs width. If the confidence sequence were not wider
    than a fixed-horizon interval it would not be providing a uniform
    guarantee.
    """
    from scipy import stats

    rng = np.random.default_rng(4)
    a = rng.normal(0.5, 1, 1000)
    b = rng.normal(0.0, 1, 1000)

    _, lo, hi = confidence_sequence(a, b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    fixed_width = 2 * stats.norm.ppf(0.975) * se

    assert (hi - lo) > fixed_width


def test_peeking_inflates_fixed_horizon_error_rate():
    """
    The core demonstration, at reduced scale so it runs in CI. Twenty looks
    at an A/A test should push the fixed-horizon false-positive rate well
    past its nominal 5% while the sequential rate stays at or below it.
    """
    _, s = simulate_peeking(n_experiments=200, n_max=1000, n_peeks=20,
                            true_effect=0.0, seed=0)

    assert s["fixed_horizon_rate"] > 0.12, "peeking should inflate fixed-horizon FPR"
    assert s["always_valid_rate"] <= 0.05, "always-valid should hold at alpha"


def test_sequential_still_detects_real_effects():
    """
    Controlling false positives is trivial if you never reject. This confirms
    the method has power against a real effect rather than simply being
    silent.
    """
    _, s = simulate_peeking(n_experiments=200, n_max=2000, n_peeks=20,
                            true_effect=0.3, seed=1)
    assert s["always_valid_rate"] > 0.5