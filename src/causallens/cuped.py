"""
cuped.py — CUPED (Controlled-experiment Using Pre-Experiment Data) variance
reduction.

Uses a pre-experiment covariate correlated with the outcome to strip out
predictable noise, tightening the confidence interval around the treatment
effect without changing the effect estimate itself.
"""

import numpy as np
from scipy import stats


def cuped_adjust(y, x_pre):
    """
    theta = Cov(Y, X_pre) / Var(X_pre)
    Y_adjusted = Y - theta * (X_pre - mean(X_pre))
    """
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre)
    y_adjusted = y - theta * (x_pre - np.mean(x_pre))
    return y_adjusted, theta


def estimate_ate_with_ci(y, treatment, alpha=0.05):
    """Diff-in-means ATE with a 95% (or given alpha) confidence interval."""
    y1 = y[treatment == 1]
    y0 = y[treatment == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - alpha / 2)
    return ate, se, ate - z * se, ate + z * se


if __name__ == "__main__":
    from data_gen import generate_synthetic_experiment

    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0)
    y = df["post_period_metric"].values
    x_pre = df["pre_period_metric"].values
    treatment = df["treatment"].values

    ate_before, se_before, lo_before, hi_before = estimate_ate_with_ci(y, treatment)
    y_adj, theta = cuped_adjust(y, x_pre)
    ate_after, se_after, lo_after, hi_after = estimate_ate_with_ci(y_adj, treatment)
    var_reduction = 1 - (y_adj.var() / y.var())

    print(f"theta (CUPED coefficient): {theta:.3f}")
    print()
    print(f"BEFORE CUPED: ATE={ate_before:.3f}  95% CI=({lo_before:.3f}, {hi_before:.3f})  width={hi_before-lo_before:.3f}")
    print(f"AFTER  CUPED: ATE={ate_after:.3f}  95% CI=({lo_after:.3f}, {hi_after:.3f})  width={hi_after-lo_after:.3f}")
    print()
    print(f"Variance reduction: {var_reduction*100:.1f}%")