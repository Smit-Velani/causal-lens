"""
data_gen.py — Synthetic ground-truth experiment generator for CausalLens.

Every causal method (CUPED, DiD, PSM, Bayesian A/B, uplift) gets checked
against data where we KNOW the true treatment effect, because we injected
it ourselves. Real-world data never gives you that — this does.
"""

import numpy as np
import pandas as pd


def generate_synthetic_experiment(
    n=10000,
    true_ate=5.0,
    confounding_strength=0.5,
    heterogeneous=True,
    noise_std=6.0,
    seed=42,
):
    """
    confounding_strength: how strongly X1 affects BOTH treatment assignment
    and the outcome (drives the Phase-1 confounding bias demo).

    user_trend: a latent per-user baseline level shared between the pre-
    and post-period metrics — this is what CUPED exploits. In the real
    world this is why "last week's value" predicts "this week's value"
    so well: it's largely the same underlying user behavior.

    heterogeneous: effect size varies by X3, averages to exactly true_ate
    across the population (drives the Phase-6 uplift modeling demo).
    """
    rng = np.random.default_rng(seed)
    user_id = np.arange(n)

    X1 = rng.normal(0, 1, n)           # confounder — drives assignment + baseline
    X2 = rng.normal(0, 1, n)           # pure noise — sanity check
    X3 = rng.normal(0, 1, n)           # effect modifier — changes effect SIZE only
    user_trend = rng.normal(0, 7, n)   # shared pre/post signal — CUPED's target

    pre_period_metric = 20 + 3 * X1 + user_trend + rng.normal(0, 4, n)

    propensity_logit = confounding_strength * X1
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, propensity)

    if heterogeneous:
        true_individual_effect = true_ate * (0.75 + 0.5 * (X3 > 0))
    else:
        true_individual_effect = np.full(n, true_ate)

    baseline = 50 + 4 * X1 + user_trend + rng.normal(0, noise_std, n)
    post_period_metric = baseline + treatment * true_individual_effect

    return pd.DataFrame({
        "user_id": user_id, "X1": X1, "X2": X2, "X3": X3,
        "pre_period_metric": pre_period_metric,
        "treatment": treatment,
        "post_period_metric": post_period_metric,
        "true_individual_effect": true_individual_effect,
    })


if __name__ == "__main__":
    true_ate = 5.0
    df = generate_synthetic_experiment(true_ate=true_ate, confounding_strength=0)

    naive_diff = (
        df.loc[df.treatment == 1, "post_period_metric"].mean()
        - df.loc[df.treatment == 0, "post_period_metric"].mean()
    )
    print(f"True ATE:              {true_ate}")
    print(f"Naive diff-in-means:   {naive_diff:.3f}")
    print(f"Bias (naive - true):   {naive_diff - true_ate:.3f}")