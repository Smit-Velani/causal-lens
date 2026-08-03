"""
did.py — Difference-in-Differences (DiD) estimator.

Handles treatment assignment that's confounded by a baseline-level
difference between groups, by differencing out that level difference using
pre-period data. Works even when a naive post-only comparison is biased —
as long as, absent treatment, both groups would have trended in parallel.
"""

import pandas as pd
import statsmodels.formula.api as smf


def estimate_did(df):
    """
    Reshapes wide (pre/post columns) data into a long panel and fits
    value ~ treated * post. The treated:post interaction coefficient
    is the DiD estimate.
    """
    pre = pd.DataFrame({
        "user_id": df.user_id, "value": df.pre_period_metric,
        "treated": df.treatment, "post": 0,
    })
    post = pd.DataFrame({
        "user_id": df.user_id, "value": df.post_period_metric,
        "treated": df.treatment, "post": 1,
    })
    long_df = pd.concat([pre, post], ignore_index=True)

    model = smf.ols("value ~ treated * post", data=long_df).fit()
    did_estimate = model.params["treated:post"]
    ci_low, ci_high = model.conf_int().loc["treated:post"]
    return did_estimate, ci_low, ci_high


if __name__ == "__main__":
    from data_gen import generate_synthetic_experiment

    for cs in [0.5, 0.0]:
        df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=cs)

        naive = (
            df.loc[df.treatment == 1, "post_period_metric"].mean()
            - df.loc[df.treatment == 0, "post_period_metric"].mean()
        )
        did_est, lo, hi = estimate_did(df)

        print(f"confounding_strength={cs}")
        print(f"  Naive post-only diff: {naive:.3f}  (bias={naive-5.0:.3f})")
        print(f"  DiD estimate:         {did_est:.3f}  (bias={did_est-5.0:.3f})  95% CI=({lo:.3f}, {hi:.3f})")
        print()