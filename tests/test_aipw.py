"""
test_aipw.py — Tests for the doubly robust estimator.

The important test here is the double robustness pattern: breaking one
model should leave the estimate intact, breaking both should not. Without
the "both broken" case the first three prove nothing, which is a mistake
this module actually made in an earlier version.
"""

import numpy as np
import pytest

from data_gen import generate_synthetic_experiment
from aipw import estimate_aipw, double_robustness_check


def test_aipw_recovers_true_effect():
    """With both models correct, AIPW should be close to the true ATE."""
    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)
    r = estimate_aipw(df)

    assert abs(r["ate"] - 5.0) < 0.3
    assert r["ci_low"] <= 5.0 <= r["ci_high"]


def test_aipw_beats_naive_under_confounding():
    """The whole point: adjustment should remove most of the naive bias."""
    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)
    naive = (df.loc[df.treatment == 1, "post_period_metric"].mean()
             - df.loc[df.treatment == 0, "post_period_metric"].mean())
    r = estimate_aipw(df)

    assert abs(r["ate"] - 5.0) < abs(naive - 5.0) / 5


def test_double_robustness_one_correct_model_suffices():
    """
    Breaking either model alone should leave the estimate near the truth,
    because the other one carries it.
    """
    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)
    dr = double_robustness_check(df).set_index("scenario")

    for scenario in ("Both correct", "Outcome model broken",
                     "Propensity model broken"):
        assert abs(dr.loc[scenario, "bias"]) < 0.3, f"{scenario} should be near-unbiased"
        assert dr.loc[scenario, "covers_true"], f"{scenario} CI should cover 5.0"


def test_double_robustness_both_broken_fails():
    """
    The control case. If this passes too, the misspecification is not biting
    and the three rows above are meaningless -- which is exactly what
    happened when the "broken" propensity model was accidentally still
    correctly specified.
    """
    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)
    dr = double_robustness_check(df).set_index("scenario")

    assert abs(dr.loc["Both broken", "bias"]) > 1.0
    assert not dr.loc["Both broken", "covers_true"]


def test_both_broken_collapses_to_naive():
    """
    With the confounder absent from both models there is nothing left to
    adjust with, so AIPW should reduce to the unadjusted difference in means.
    """
    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)
    naive = (df.loc[df.treatment == 1, "post_period_metric"].mean()
             - df.loc[df.treatment == 0, "post_period_metric"].mean())
    dr = double_robustness_check(df).set_index("scenario")

    assert abs(dr.loc["Both broken", "ate"] - naive) < 0.05


def test_propensity_trimming_bounds_weights():
    """Trimming should keep the maximum IPW weight finite and modest."""
    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)
    r = estimate_aipw(df, trim=0.01)

    assert r["propensity_min"] >= 0.01
    assert r["propensity_max"] <= 0.99
    assert r["max_weight"] < 100
    assert r["effective_n"] > 0.5 * r["n"]