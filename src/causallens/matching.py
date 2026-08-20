"""
matching.py — Propensity Score Matching (PSM).

Corrects for confounded (non-randomized) treatment assignment by matching
each treated unit to a control unit with a similar propensity score
(P(treatment=1 | covariates)), then comparing outcomes within matched pairs.

Ships three diagnostics alongside the estimate, because a matched point
estimate on its own says nothing about whether matching worked:

  - SMD covariate balance: did the covariates actually come out balanced?
  - Bootstrap confidence interval: how much sampling uncertainty is there?
  - Rosenbaum sensitivity bounds: how strong would an unmeasured confounder
    have to be to overturn the conclusion?
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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

    X = StandardScaler().fit_transform(X)
    ps_model = LogisticRegression(max_iter=1000)
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


def naive_paired_ci(df, covariates=["X1", "X2", "X3"], caliper_multiplier=0.2,
                    alpha=0.05):
    """
    The interval you get by treating the matched pairs as if they were a
    simple paired sample: SE = sd(pair differences) / sqrt(n_pairs).

    This is what most implementations report, and it is too narrow. It
    conditions on the matching as though the propensity model were known
    rather than estimated, so it ignores a whole layer of uncertainty.
    Included here specifically so it can be compared against the bootstrap
    interval below -- the gap between the two is the cost of pretending
    the propensity model was free.
    """
    _, _, _, m = estimate_psm(df, covariates, caliper_multiplier, return_matches=True)
    y = df.post_period_metric.values
    diffs = y[m["matched_treated"]] - y[m["matched_control"]]

    ate = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    z = stats.norm.ppf(1 - alpha / 2)
    return ate, se, ate - z * se, ate + z * se


def bootstrap_psm_ci(df, covariates=["X1", "X2", "X3"], caliper_multiplier=0.2,
                     n_boot=200, alpha=0.05, seed=0, verbose=False):
    """
    Percentile bootstrap confidence interval for the PSM estimate.

    The whole pipeline is re-run inside each bootstrap replicate: resample
    units with replacement, refit the propensity model on the resampled
    data, rematch, recompute the ATE. Resampling only the matched pairs
    would treat the matching as fixed and understate uncertainty, since
    the propensity scores were estimated from the same data.

    KNOWN CAVEAT -- Abadie & Imbens (2008) showed the standard bootstrap is
    not formally valid for nearest-neighbour matching estimators with a
    fixed number of matches: the matching function is not smooth enough for
    the bootstrap's asymptotic guarantees to hold, and the resulting
    intervals can have incorrect coverage in either direction. This is
    reported as an approximation and a clear improvement over the naive
    paired interval, not as a formally correct one. A formally valid
    alternative is the Abadie-Imbens analytic variance estimator, which is
    not implemented here.

    Returns (ate, ci_low, ci_high, boot_estimates).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    ate, _, _ = estimate_psm(df, covariates, caliper_multiplier)

    boot_estimates = []
    failures = 0
    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[boot_idx].reset_index(drop=True)

        # A resample can end up with one arm empty or too small to match.
        if boot_df.treatment.nunique() < 2:
            failures += 1
            continue
        try:
            est, n_matched, _ = estimate_psm(boot_df, covariates, caliper_multiplier)
            if n_matched > 0 and np.isfinite(est):
                boot_estimates.append(est)
            else:
                failures += 1
        except Exception:
            failures += 1

        if verbose and (b + 1) % 50 == 0:
            print(f"  bootstrap {b+1}/{n_boot}")

    boot_estimates = np.array(boot_estimates)
    if len(boot_estimates) < 20:
        raise RuntimeError(
            f"Only {len(boot_estimates)} of {n_boot} bootstrap replicates "
            f"succeeded -- too few for a usable interval."
        )

    lo = np.percentile(boot_estimates, 100 * alpha / 2)
    hi = np.percentile(boot_estimates, 100 * (1 - alpha / 2))
    return ate, lo, hi, boot_estimates


def rosenbaum_bounds(df, covariates=["X1", "X2", "X3"], caliper_multiplier=0.2,
                     gammas=None, alpha=0.05):
    """
    Rosenbaum sensitivity analysis for unmeasured confounding.

    Matching only balances covariates you measured. The obvious objection
    to any PSM result is "what about a confounder you didn't observe?" --
    and it cannot be answered from the data, because the confounder is by
    definition unobserved. What Rosenbaum bounds do instead is quantify how
    strong such a confounder would have to be before the conclusion breaks.

    Gamma is the odds ratio by which an unmeasured confounder could shift a
    unit's probability of being treated. Gamma = 1 means no hidden bias --
    within a matched pair, either unit was equally likely to be treated.
    Gamma = 2 means one unit could have been twice as likely as its match,
    despite being identical on every measured covariate.

    For each Gamma, the Wilcoxon signed-rank test on pair differences is
    bounded under the most and least favourable assignment patterns
    consistent with that Gamma. Gamma* is the point at which the upper
    bound crosses alpha -- the smallest hidden bias that could explain the
    result away.

    Interpretation: a high Gamma* means the finding is robust; a Gamma*
    near 1 means even slight hidden bias could account for it.

    Returns (bounds_df, gamma_critical).
    """
    if gammas is None:
        gammas = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.1, 2.2, 2.3, 2.4,
                  2.5, 2.75, 3.0, 4.0, 5.0, 7.0, 10.0]

    _, _, _, m = estimate_psm(df, covariates, caliper_multiplier, return_matches=True)
    y = df.post_period_metric.values
    diffs = y[m["matched_treated"]] - y[m["matched_control"]]

    # Wilcoxon signed-rank: drop zero differences, rank by absolute value,
    # sum the ranks belonging to positive differences.
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n < 10:
        raise ValueError(f"Only {n} non-zero pair differences -- too few.")

    ranks = stats.rankdata(np.abs(diffs))
    w_plus = ranks[diffs > 0].sum()

    total_rank = n * (n + 1) / 2
    rank_sq_sum = n * (n + 1) * (2 * n + 1) / 6

    rows = []
    for g in gammas:
        p_hi = g / (1 + g)     # worst case: treatment favoured -> largest null mean
        p_lo = 1 / (1 + g)     # best case

        # Upper bound on the p-value uses the null distribution shifted up,
        # making the observed statistic look least extreme.
        mu_hi = p_hi * total_rank
        sd_hi = np.sqrt(p_hi * (1 - p_hi) * rank_sq_sum)
        p_upper = 1 - stats.norm.cdf((w_plus - mu_hi) / sd_hi)

        mu_lo = p_lo * total_rank
        sd_lo = np.sqrt(p_lo * (1 - p_lo) * rank_sq_sum)
        p_lower = 1 - stats.norm.cdf((w_plus - mu_lo) / sd_lo)

        rows.append({
            "gamma": g,
            "p_lower_bound": p_lower,
            "p_upper_bound": p_upper,
            "still_significant": bool(p_upper < alpha),
        })

    bounds = pd.DataFrame(rows)

    sig = bounds[bounds.still_significant]
    gamma_critical = float(sig.gamma.max()) if len(sig) else 1.0
    return bounds, gamma_critical


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


def plot_bootstrap(boot_estimates, ate, lo, hi, true_ate=None, save_path=None):
    """Bootstrap sampling distribution with the percentile interval marked."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot_estimates, bins=40, alpha=0.75, edgecolor="white")
    ax.axvline(ate, color="black", lw=2, label=f"PSM estimate {ate:.3f}")
    ax.axvline(lo, color="grey", ls="--", lw=1.5, label=f"95% CI ({lo:.3f}, {hi:.3f})")
    ax.axvline(hi, color="grey", ls="--", lw=1.5)
    if true_ate is not None:
        ax.axvline(true_ate, color="crimson", lw=2, ls=":", label=f"True ATE {true_ate}")
    ax.set_xlabel("Bootstrap PSM estimate")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrap distribution ({len(boot_estimates)} replicates)")
    ax.legend(fontsize=8)
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

    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5)
    balance = smd_before_after(df)
    plot_smd(balance, save_path="reports/psm_balance.png")
    print("Saved Love plot to reports/psm_balance.png")

    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 60)

    ate_n, se_n, lo_n, hi_n = naive_paired_ci(df)
    print(f"Naive paired CI:  {ate_n:.3f}  95% CI ({lo_n:.3f}, {hi_n:.3f})  "
          f"width {hi_n-lo_n:.3f}")

    print("Running 200 bootstrap replicates (refits propensity each time)...")
    ate_b, lo_b, hi_b, boots = bootstrap_psm_ci(df, n_boot=200, verbose=True)
    print(f"Bootstrap CI:     {ate_b:.3f}  95% CI ({lo_b:.3f}, {hi_b:.3f})  "
          f"width {hi_b-lo_b:.3f}")
    print(f"Bootstrap SE:     {boots.std(ddof=1):.4f}   (naive SE {se_n:.4f})")
    print(f"Covers true ATE 5.0: {lo_b <= 5.0 <= hi_b}")

    plot_bootstrap(boots, ate_b, lo_b, hi_b, true_ate=5.0,
                   save_path="reports/psm_bootstrap.png")
    print("Saved bootstrap plot to reports/psm_bootstrap.png")

    print("\n" + "=" * 60)
    print("ROSENBAUM SENSITIVITY BOUNDS")
    print("=" * 60)

    bounds, gamma_star = rosenbaum_bounds(df)
    print(bounds.to_string(index=False))
    print()
    print(f"Gamma* = {gamma_star:.2f}")
    print(
        f"An unmeasured confounder would need to shift the odds of treatment "
        f"by a factor of more than {gamma_star:.2f}, for units identical on "
        f"every measured covariate, before this result stops being significant."
    )