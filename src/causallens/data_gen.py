"""
data_gen.py — Synthetic ground-truth experiment generator for CausalLens.

Every causal method (CUPED, DiD, PSM, Bayesian A/B, uplift) gets checked
against data where we KNOW the true treatment effect, because we injected
it ourselves. Real-world data never gives you that — this does.

The generator can also produce multiple pre-treatment periods, which is
what makes DiD's parallel-trends assumption testable: with a single
pre-period there is no trend to compare, only a level difference, and DiD
differences that away by design.
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
    n_pre_periods=1,
    x1_trend_per_period=1.0,
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

    n_pre_periods: how many pre-treatment periods to emit. With 1 (the
    default) behaviour is unchanged. With >1, extra columns
    pre_period_metric_1 .. pre_period_metric_k are added, ordered earliest
    to latest, and pre_period_metric aliases the latest one so every
    existing caller keeps working.

    x1_trend_per_period: how much X1's coefficient grows per period. This
    is the parallel-trends violation, made explicit and tunable. X1 drives
    treatment assignment, so if X1's effect grows over time the treated
    group drifts upward faster than control even absent any treatment —
    exactly what DiD assumes away. Set to 0 for a clean, parallel world
    where DiD is unbiased.
    """
    rng = np.random.default_rng(seed)
    user_id = np.arange(n)

    X1 = rng.normal(0, 1, n)           # confounder — drives assignment + baseline
    X2 = rng.normal(0, 1, n)           # pure noise — sanity check
    X3 = rng.normal(0, 1, n)           # effect modifier — changes effect SIZE only
    user_trend = rng.normal(0, 7, n)   # shared pre/post signal — CUPED's target

    # Period index: the last pre-period is t=0, earlier ones are negative,
    # and the post-period is t=+1. X1's coefficient at period t is
    # 3 + t * x1_trend_per_period, so with the default the last pre-period
    # sits at 3 and the post-period at 4 — matching the original design.
    pre_offsets = list(range(-(n_pre_periods - 1), 1))

    pre_metrics = {}
    for j, t in enumerate(pre_offsets, start=1):
        coef = 3.0 + t * x1_trend_per_period
        pre_metrics[f"pre_period_metric_{j}"] = (
            20 + coef * X1 + user_trend + rng.normal(0, 4, n)
        )

    # Latest pre-period is the canonical one every other module uses
    pre_period_metric = pre_metrics[f"pre_period_metric_{n_pre_periods}"]

    propensity_logit = confounding_strength * X1
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, propensity)

    if heterogeneous:
        true_individual_effect = true_ate * (0.75 + 0.5 * (X3 > 0))
    else:
        true_individual_effect = np.full(n, true_ate)

    post_coef = 3.0 + 1 * x1_trend_per_period
    baseline = 50 + post_coef * X1 + user_trend + rng.normal(0, noise_std, n)
    post_period_metric = baseline + treatment * true_individual_effect

    out = pd.DataFrame({
        "user_id": user_id, "X1": X1, "X2": X2, "X3": X3,
        "pre_period_metric": pre_period_metric,
        "treatment": treatment,
        "post_period_metric": post_period_metric,
        "true_individual_effect": true_individual_effect,
    })
    for col, vals in pre_metrics.items():
        out[col] = vals
    return out


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

    print()
    df3 = generate_synthetic_experiment(n_pre_periods=3)
    pre_cols = [c for c in df3.columns if c.startswith("pre_period_metric_")]
    print(f"With n_pre_periods=3, emitted: {pre_cols}")
    for c in pre_cols:
        gap = df3.loc[df3.treatment == 1, c].mean() - df3.loc[df3.treatment == 0, c].mean()
        print(f"  {c}: treated - control = {gap:+.3f}")
    print("  Gap widening across pre-periods = parallel trends violated.")