"""
matching.py — Propensity Score Matching (PSM).

Corrects for confounded (non-randomized) treatment assignment by matching
each treated unit to a control unit with a similar propensity score
(P(treatment=1 | covariates)), then comparing outcomes within matched pairs.

Includes SMD covariate-balance diagnostics — matching is only credible if
the covariates it was supposed to balance actually came out balanced.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def estimate_psm(df, covariates=["X1", "X2", "X3"], caliper_multiplier=0.2,
                 return_matches=False):
    """
    Fits a logistic regression for propensity scores, matches each treated
    unit to its nearest-neighbor control on the logit-propensity scale
    (the standard convention), drops matches beyond a caliper, then reports
    the average outcome difference across matched pairs as the ATE.

    If return_matches=True, also returns the matched index arrays so the
    balance diagnostics can be computed on exactly the same matched sample.
    """
    X = df[covariates].values
    treatment = df.treatment.values
    y = df.post_period_metric.values

    ps_model = LogisticRegression()
    ps_model.fit(X, treatment)
    propensity = np.clip(ps_model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    logit_ps = np.log(propensity / (1 - propensity))

    caliper = caliper_multiplier * logit_ps.std()

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(logit_ps[control_idx].reshape(-1, 1))
    dist, idx = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))

    matched_control_idx = control_idx[idx.flatten()]
    within_caliper = dist.flatten() <= caliper

    matched_treated = treated_idx[within_caliper]
    matched_control = matched_control_idx[within_caliper]

    ate_matched = (y[matched_treated] - y[matched_control]).mean()

    if return_matches:
        matches = {
            "matched_treated": matched_treated,
            "matched_control": matched_control,
            "all_treated": treated_idx,
            "all_control": control_idx,
        }
        return ate_matched, within_caliper.sum(), len(treated_idx), matches

    return ate_matched, within_caliper.sum(), len(treated_idx)


def compute_smd(df, covariates, treated_idx, control_idx):
    """
    Standardized Mean Difference per covariate:

        SMD = (mean_treated - mean_control) / sqrt((var_t + var_c) / 2)

    |SMD| < 0.1 is the conventional threshold for acceptable balance.
    Standardizing makes the imbalance comparable across covariates that
    live on different scales.
    """
    rows = []
    for cov in covariates:
        t = df[cov].values[treated_idx]
        c = df[cov].values[control_idx]
        pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
        smd = (t.mean() - c.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"covariate": cov, "smd": smd})
    return pd.DataFrame(rows)


def smd_before_after(df, covariates=["X1", "X2", "X3"], caliper_multiplier=0.2):
    """
    Covariate balance before vs after matching, one row per covariate.

    'Before' compares all treated against all control units; 'after'
    compares only the matched pairs. A large |SMD| that collapses below
    0.1 after matching is the evidence that PSM actually worked.
    """
    _, _, _, m = estimate_psm(
        df, covariates, caliper_multiplier, return_matches=True
    )
    before = compute_smd(df, covariates, m["all_treated"], m["all_control"])
    after = compute_smd(df, covariates, m["matched_treated"], m["matched_control"])

    out = before.rename(columns={"smd": "smd_before"}).merge(
        after.rename(columns={"smd": "smd_after"}), on="covariate"
    )
    out["abs_reduction"] = out.smd_before.abs() - out.smd_after.abs()
    out["balanced_after"] = out.smd_after.abs() < 0.1
    return out


def plot_smd(balance_df, save_path=None):
    """
    Love plot — the standard way to present PSM balance. One row per
    covariate, |SMD| before and after matching, with the 0.1 threshold
    marked.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 0.6 * len(balance_df) + 1.5))
    y_pos = np.arange(len(balance_df))

    ax.scatter(balance_df.smd_before.abs(), y_pos, label="Before matching", s=60)
    ax.scatter(balance_df.smd_after.abs(), y_pos, label="After matching",
               s=60, marker="D")

    ax.axvline(0.1, ls="--", color="grey", lw=1)
    ax.text(0.1, -0.6, " |SMD| = 0.1 threshold", fontsize=8, color="grey")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df.covariate)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("Covariate balance before vs after PSM")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    from data_gen import generate_synthetic_experiment

    for cs in [0.5, 0.0]:
        df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=cs)

        naive = (
            df.loc[df.treatment == 1, "post_period_metric"].mean()
            - df.loc[df.treatment == 0, "post_period_metric"].mean()
        )
        ate_matched, n_matched, n_treated = estimate_psm(df)

        print(f"confounding_strength={cs}")
        print(f"  Naive diff:  {naive:.3f}  (bias={naive-5.0:.3f})")
        print(f"  PSM ATE:     {ate_matched:.3f}  (bias={ate_matched-5.0:.3f})  matched {n_matched}/{n_treated} treated units")
        print()
        print("  Covariate balance:")
        balance = smd_before_after(df)
        print(balance.to_string(index=False))
        print()

    # Love plot for the confounded case
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5)
    balance = smd_before_after(df)
    plot_smd(balance, save_path="reports/psm_balance.png")
    print("Saved Love plot to reports/psm_balance.png")