"""
validate.py — The validation harness: runs every confounding-correction
method across many random seeds and checks bias against the known
synthetic ground truth. This is the file that proves the project works,
rather than just claiming it.

Also runs the per-method diagnostics — PSM covariate balance, the CUPED
residual-correlation check, and a placebo test of DiD's parallel-trends
assumption — so each estimator is verified on its own terms, not just by
whether its final number looks right.
"""

import pandas as pd
from data_gen import generate_synthetic_experiment
from did import estimate_did, pre_trends_test, pre_period_gaps
from matching import estimate_psm, smd_before_after, plot_smd
from cuped import cuped_adjust, check_residual_correlation


def run_confounding_comparison(true_ate=5.0, confounding_strength=0.5, n_seeds=30, n=10000):
    results = []
    for seed in range(n_seeds):
        df = generate_synthetic_experiment(
            n=n, true_ate=true_ate, confounding_strength=confounding_strength, seed=seed
        )
        naive = (
            df.loc[df.treatment == 1, "post_period_metric"].mean()
            - df.loc[df.treatment == 0, "post_period_metric"].mean()
        )
        did_est, did_lo, did_hi = estimate_did(df)
        psm_est, n_matched, n_treated = estimate_psm(df)

        results.append({
            "seed": seed,
            "naive": naive, "naive_bias": naive - true_ate,
            "did": did_est, "did_bias": did_est - true_ate,
            "did_covers_true": did_lo <= true_ate <= did_hi,
            "psm": psm_est, "psm_bias": psm_est - true_ate,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    results = run_confounding_comparison()

    summary = pd.DataFrame({
        "method": ["Naive diff-in-means", "Difference-in-Differences", "Propensity Score Matching"],
        "mean_bias": [results.naive_bias.mean(), results.did_bias.mean(), results.psm_bias.mean()],
        "mean_abs_bias": [
            results.naive_bias.abs().mean(),
            results.did_bias.abs().mean(),
            results.psm_bias.abs().mean(),
        ],
        "std_of_estimates": [results.naive.std(), results.did.std(), results.psm.std()],
    })

    print(f"Across {len(results)} random seeds, confounding_strength=0.5, true ATE=5.0:\n")
    print(summary.to_string(index=False))
    print()
    print(f"DiD 95% CI coverage rate: {results.did_covers_true.mean()*100:.0f}% (should be close to 95%)")

    results.to_csv("reports/validation_results.csv", index=False)
    print("\nSaved raw results to reports/validation_results.csv")

    # ------------------------------------------------------------------
    # Diagnostics — each estimator checked on its own assumptions
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("PSM COVARIATE BALANCE (seed 0, confounding_strength=0.5)")
    print("=" * 60)

    df0 = generate_synthetic_experiment(
        n=10000, true_ate=5.0, confounding_strength=0.5, seed=0
    )
    balance = smd_before_after(df0)
    print(balance.to_string(index=False))
    balance.to_csv("reports/psm_balance.csv", index=False)
    plot_smd(balance, save_path="reports/psm_balance.png")
    print("\nSaved to reports/psm_balance.csv and reports/psm_balance.png")

    n_bal = int(balance.balanced_after.sum())
    print(f"Covariates balanced after matching (|SMD| < 0.1): {n_bal}/{len(balance)}")

    print("\n" + "=" * 60)
    print("CUPED RESIDUAL CORRELATION (seed 0, clean experiment)")
    print("=" * 60)

    df_clean = generate_synthetic_experiment(
        n=10000, true_ate=5.0, confounding_strength=0, seed=0
    )
    y = df_clean.post_period_metric.values
    x_pre = df_clean.pre_period_metric.values
    y_adj, theta = cuped_adjust(y, x_pre)
    corr_before, corr_after, p_after = check_residual_correlation(y, y_adj, x_pre)
    var_reduction = 1 - (y_adj.var() / y.var())

    print(f"theta                = {theta:.4f}")
    print(f"corr(Y, X_pre)       = {corr_before:+.6f}")
    print(f"corr(Y_cuped, X_pre) = {corr_after:+.6f}  (p={p_after:.3f})")
    print(f"Variance reduction   = {var_reduction*100:.1f}%")

    pd.DataFrame([{
        "theta": theta,
        "corr_before": corr_before,
        "corr_after": corr_after,
        "p_after": p_after,
        "variance_reduction": var_reduction,
    }]).to_csv("reports/cuped_diagnostics.csv", index=False)
    print("\nSaved to reports/cuped_diagnostics.csv")

    # ------------------------------------------------------------------
    # Parallel-trends placebo test — the assumption behind the 83% coverage
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("PARALLEL-TRENDS PLACEBO TEST (3 pre-periods)")
    print("=" * 60)

    rows = []
    for trend, label in [(1.0, "default"), (0.0, "parallel world")]:
        df_pt = generate_synthetic_experiment(
            n=10000, true_ate=5.0, confounding_strength=0.5,
            seed=0, n_pre_periods=3, x1_trend_per_period=trend,
        )
        res = pre_trends_test(df_pt)
        did_est, _, _ = estimate_did(df_pt)
        res.update({
            "scenario": label,
            "x1_trend_per_period": trend,
            "actual_did_bias": did_est - 5.0,
        })
        rows.append(res)

        print(f"\n{label} (x1_trend_per_period={trend})")
        print(pre_period_gaps(df_pt).to_string(index=False))
        verdict = "HOLDS" if res["parallel_trends_holds"] else "VIOLATED"
        print(f"  pre-trend coef  = {res['pre_trend_coef']:+.4f} (p={res['p_value']:.2e})")
        print(f"  verdict         = parallel trends {verdict}")
        print(f"  actual DiD bias = {res['actual_did_bias']:+.4f}")

    pd.DataFrame(rows).to_csv("reports/parallel_trends_test.csv", index=False)
    print("\nSaved to reports/parallel_trends_test.csv")