from data_gen import generate_synthetic_experiment
from bayesian_ab import bayesian_ab_test


def test_bayesian_posterior_matches_naive():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.0, seed=42)
    y, t = df.post_period_metric.values, df.treatment.values
    naive = y[t == 1].mean() - y[t == 0].mean()
    diff_mean, diff_std, prob_pos, lo, hi = bayesian_ab_test(y, t)
    assert abs(diff_mean - naive) < 1e-6, "posterior mean should match naive diff with a flat prior"
    assert prob_pos > 0.9, "P(treatment>control) should be high given a clear effect"