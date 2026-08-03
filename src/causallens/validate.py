"""
validate.py — The validation harness: runs every confounding-correction
method across many random seeds and checks bias against the known
synthetic ground truth. This is the file that proves the project works,
rather than just claiming it.
"""

import pandas as pd
from data_gen import generate_synthetic_experiment
from did import estimate_did
from matching import estimate_psm


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