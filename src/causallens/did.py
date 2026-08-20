"""
did.py — Difference-in-Differences (DiD) estimator.

Handles treatment assignment that's confounded by a baseline-level
difference between groups, by differencing out that level difference using
pre-period data. Works even when a naive post-only comparison is biased —
as long as, absent treatment, both groups would have trended in parallel.

That last clause is an assumption, not a guarantee, so this module also
ships a placebo test for it: with two or more pre-treatment periods, the
treated-vs-control gap should be flat before treatment. If it isn't,
DiD's estimate is biased and its confidence intervals undercover.
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


def _pre_period_columns(df):
    """Pre-period metric columns, ordered earliest to latest."""
    cols = [c for c in df.columns if c.startswith("pre_period_metric_")]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def pre_trends_test(df, alpha=0.05):
    """
    Placebo test of the parallel-trends assumption.

    Fits value ~ treated * period using ONLY pre-treatment periods, where
    no treatment effect can exist by construction. Any non-zero
    treated:period interaction is therefore a pre-existing divergence
    between the groups — precisely the thing DiD assumes away.

    A significant coefficient means DiD is biased, and the coefficient
    itself estimates roughly how much bias to expect per period.

    Returns a dict with the coefficient, p-value, CI, and a pass/fail flag.
    """
    pre_cols = _pre_period_columns(df)
    if len(pre_cols) < 2:
        raise ValueError(
            "Parallel-trends testing needs at least 2 pre-periods; this "
            f"data has {len(pre_cols)}. Regenerate with "
            "generate_synthetic_experiment(n_pre_periods=3)."
        )

    frames = []
    for i, col in enumerate(pre_cols, start=1):
        frames.append(pd.DataFrame({
            "user_id": df.user_id,
            "value": df[col],
            "treated": df.treatment,
            "period": i,
        }))
    long_df = pd.concat(frames, ignore_index=True)

    model = smf.ols("value ~ treated * period", data=long_df).fit()
    coef = model.params["treated:period"]
    pval = model.pvalues["treated:period"]
    ci_low, ci_high = model.conf_int().loc["treated:period"]

    return {
        "pre_trend_coef": coef,
        "p_value": pval,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "parallel_trends_holds": bool(pval > alpha),
        "n_pre_periods": len(pre_cols),
    }


def pre_period_gaps(df):
    """
    Treated-minus-control gap in each pre-period — the event-study view.
    A flat column means parallel trends holds; a widening one does not.
    """
    rows = []
    for i, col in enumerate(_pre_period_columns(df), start=1):
        gap = (
            df.loc[df.treatment == 1, col].mean()
            - df.loc[df.treatment == 0, col].mean()
        )
        rows.append({"period": i, "treated_minus_control": gap})
    return pd.DataFrame(rows)


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

    print("=" * 60)
    print("PARALLEL-TRENDS PLACEBO TEST")
    print("=" * 60)

    for trend, label in [(1.0, "default generator"), (0.0, "clean, parallel world")]:
        df = generate_synthetic_experiment(
            n_pre_periods=3, x1_trend_per_period=trend, confounding_strength=0.5
        )
        gaps = pre_period_gaps(df)
        res = pre_trends_test(df)
        did_est, _, _ = estimate_did(df)

        print(f"\nx1_trend_per_period={trend}  ({label})")
        print(gaps.to_string(index=False))
        verdict = "HOLDS" if res["parallel_trends_holds"] else "VIOLATED"
        print(f"  pre-trend coef = {res['pre_trend_coef']:+.4f}  "
              f"(p={res['p_value']:.2e})  -> parallel trends {verdict}")
        print(f"  actual DiD bias = {did_est - 5.0:+.4f}")