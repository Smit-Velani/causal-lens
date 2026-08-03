"""
bayesian_ab.py — Bayesian A/B testing via a Normal-Normal conjugate model.

Instead of a p-value, this gives a full posterior over the treatment
effect: P(treatment > control) and a credible interval. With a near-flat
prior and enough data, it converges to the frequentist estimate — that's
expected, and shown explicitly below as a sanity check.
"""

import numpy as np
from scipy import stats


def bayesian_ab_test(y, treatment, prior_mean=0.0, prior_var=1e6):
    y1 = y[treatment == 1]
    y0 = y[treatment == 0]

    def group_posterior(y_group):
        n = len(y_group)
        sample_mean = y_group.mean()
        sample_var = y_group.var(ddof=1)
        prior_precision = 1 / prior_var
        data_precision = n / sample_var
        post_precision = prior_precision + data_precision
        post_var = 1 / post_precision
        post_mean = post_var * (prior_mean * prior_precision + sample_mean * data_precision)
        return post_mean, post_var

    mean1, var1 = group_posterior(y1)
    mean0, var0 = group_posterior(y0)

    diff_mean = mean1 - mean0
    diff_std = np.sqrt(var1 + var0)
    prob_positive = 1 - stats.norm.cdf(0, loc=diff_mean, scale=diff_std)
    ci_low, ci_high = stats.norm.ppf([0.025, 0.975], loc=diff_mean, scale=diff_std)
    return diff_mean, diff_std, prob_positive, ci_low, ci_high


if __name__ == "__main__":
    from data_gen import generate_synthetic_experiment

    # Bayesian A/B runs on the CLEAN (randomized) experiment, like CUPED --
    # it isn't designed to correct for confounding, DiD/PSM handle that
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.0)
    y, treatment = df.post_period_metric.values, df.treatment.values

    diff_mean, diff_std, prob_pos, lo, hi = bayesian_ab_test(y, treatment)
    print(f"Posterior mean diff: {diff_mean:.3f}")
    print(f"Posterior std:       {diff_std:.3f}")
    print(f"P(treatment > control): {prob_pos:.4f}")
    print(f"95% credible interval: ({lo:.3f}, {hi:.3f})")

    naive = y[treatment == 1].mean() - y[treatment == 0].mean()
    print(f"\nSanity check -- naive diff-in-means: {naive:.3f} (should match posterior mean)")