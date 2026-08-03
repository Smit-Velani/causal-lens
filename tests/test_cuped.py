from data_gen import generate_synthetic_experiment
from cuped import cuped_adjust, estimate_ate_with_ci


def test_cuped_reduces_variance():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.0, seed=42)
    y, x_pre = df.post_period_metric.values, df.pre_period_metric.values
    y_adj, theta = cuped_adjust(y, x_pre)
    var_reduction = 1 - y_adj.var() / y.var()
    assert var_reduction > 0.3, "CUPED should meaningfully reduce variance given a correlated covariate"


def test_cuped_preserves_ate():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.0, seed=42)
    y, x_pre, t = df.post_period_metric.values, df.pre_period_metric.values, df.treatment.values
    ate_before, _, _, _ = estimate_ate_with_ci(y, t)
    y_adj, theta = cuped_adjust(y, x_pre)
    ate_after, _, _, _ = estimate_ate_with_ci(y_adj, t)
    assert abs(ate_after - ate_before) < 0.5, "CUPED should not meaningfully shift the ATE estimate"