"""
test_diagnostics.py — Tests for the assumption diagnostics.

These check that each diagnostic reports correctly on the assumption it
is responsible for, including the case where the assumption holds. A
diagnostic that only ever fails is not a diagnostic.
"""

import numpy as np
import pytest

from data_gen import generate_synthetic_experiment
from cuped import cuped_adjust, check_residual_correlation
from matching import smd_before_after, naive_paired_ci, bootstrap_psm_ci, rosenbaum_bounds
from did import pre_trends_test


def test_cuped_removes_residual_correlation():
    """
    CUPED subtracts the component of Y linearly predictable from X_pre,
    so the adjusted outcome should have essentially zero correlation with
    the covariate — this is the algebraic guarantee, not an approximation.
    """
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0, seed=0)
    y = df.post_period_metric.values
    x_pre = df.pre_period_metric.values

    y_adj, _ = cuped_adjust(y, x_pre)
    corr_before, corr_after, _ = check_residual_correlation(y, y_adj, x_pre)

    assert abs(corr_before) > 0.5, "covariate should be strongly predictive to begin with"
    assert abs(corr_after) < 1e-10, f"residual correlation should be ~0, got {corr_after}"


def test_psm_balances_the_confounder():
    """
    X1 drives treatment assignment, so it should be badly imbalanced before
    matching and comfortably inside the |SMD| < 0.1 threshold afterwards.
    X2 and X3 do not drive assignment and should stay balanced throughout.
    """
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=0)
    balance = smd_before_after(df).set_index("covariate")

    assert abs(balance.loc["X1", "smd_before"]) > 0.2, "X1 should start imbalanced"
    assert abs(balance.loc["X1", "smd_after"]) < 0.1, "X1 should be balanced after matching"
    assert balance.balanced_after.all(), "all covariates should be balanced after matching"


def test_pre_trends_test_detects_violation():
    """With a growing X1 effect, the groups diverge before treatment."""
    df = generate_synthetic_experiment(
        confounding_strength=0.5, seed=0, n_pre_periods=3, x1_trend_per_period=1.0
    )
    res = pre_trends_test(df)

    assert res["p_value"] < 0.05
    assert res["parallel_trends_holds"] is False
    assert res["pre_trend_coef"] > 0.1


def test_pre_trends_test_passes_when_assumption_holds():
    """
    The specificity check, and the more important of the pair: with the
    trend removed the test must return a null result. Note the groups still
    differ by a large CONSTANT level gap here — DiD differences that away by
    design, so a correct test must ignore it and look only at the slope.
    """
    df = generate_synthetic_experiment(
        confounding_strength=0.5, seed=0, n_pre_periods=3, x1_trend_per_period=0.0
    )

    level_gap = (
        df.loc[df.treatment == 1, "pre_period_metric_1"].mean()
        - df.loc[df.treatment == 0, "pre_period_metric_1"].mean()
    )
    assert abs(level_gap) > 0.5, "a level gap should exist for this test to be meaningful"

    res = pre_trends_test(df)
    assert res["p_value"] > 0.05
    assert res["parallel_trends_holds"] is True
    assert abs(res["pre_trend_coef"]) < 0.1


def test_pre_trends_test_requires_multiple_pre_periods():
    """
    With one pre-period there is no trend to test, only a level difference.
    The function should say so rather than returning a misleading number.
    """
    df = generate_synthetic_experiment(seed=0)
    with pytest.raises(ValueError, match="at least 2 pre-periods"):
        pre_trends_test(df)


def test_bootstrap_ci_covers_true_effect():
    """The bootstrap interval should contain the known true ATE of 5.0."""
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=0)
    ate, lo, hi, boots = bootstrap_psm_ci(df, n_boot=60, seed=0)

    assert lo < hi
    assert lo <= 5.0 <= hi, f"CI ({lo:.3f}, {hi:.3f}) misses true ATE 5.0"
    assert len(boots) >= 50, "too many bootstrap replicates failed"


def test_bootstrap_se_exceeds_naive_paired_se():
    """
    The naive paired SE conditions on the matching as if the propensity model
    were known. The bootstrap refits it every replicate, so it must come out
    strictly larger — if it doesn't, the bootstrap isn't re-running the
    pipeline properly.
    """
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=0)
    _, se_naive, _, _ = naive_paired_ci(df)
    _, _, _, boots = bootstrap_psm_ci(df, n_boot=60, seed=0)

    assert boots.std(ddof=1) > se_naive


def test_rosenbaum_gamma_star_exceeds_one():
    """
    A true effect of 5.0 with no hidden bias in the generator should survive
    a substantial hypothetical confounder. Gamma* = 1 would mean even the
    slightest hidden bias overturns it.
    """
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=0)
    bounds, gamma_star = rosenbaum_bounds(df)

    assert gamma_star > 1.5
    assert bounds.loc[bounds.gamma == 1.0, "still_significant"].iloc[0]


def test_rosenbaum_bounds_are_monotone():
    """
    Larger Gamma allows more hidden bias, so the upper p-value bound must be
    non-decreasing. A non-monotone sequence means the bound is miscomputed.
    """
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=0)
    bounds, _ = rosenbaum_bounds(df)

    p = bounds.p_upper_bound.values
    assert all(p[i] <= p[i + 1] + 1e-12 for i in range(len(p) - 1))