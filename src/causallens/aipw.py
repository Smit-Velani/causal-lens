"""
aipw.py — Augmented Inverse Propensity Weighting (Doubly Robust estimation).

PSM throws away data: unmatched controls are discarded, and the estimate
depends entirely on the propensity model being right. AIPW instead uses
every observation and combines two models -- a propensity model for
treatment assignment and an outcome model for the response surface.

The double robustness property: the estimator is consistent if EITHER
model is correctly specified, even when the other is wrong. You get two
chances to be right instead of one. That is a genuinely different
guarantee from matching, which has exactly one.

    ATE = mean[ mu1(X) - mu0(X)
                + T*(Y - mu1(X))/e(X)
                - (1-T)*(Y - mu0(X))/(1-e(X)) ]

The first term is the outcome-regression estimate. The remaining terms
are the augmentation: an IPW-weighted correction built from the outcome
model's residuals. If the outcome model is perfect the residuals vanish
and the propensity model stops mattering; if the propensity model is
perfect the correction term is unbiased regardless of how bad the
outcome model is.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def _fit_propensity(X, treatment, trim=0.01, misspecify=False):
    """
    Propensity model e(X) = P(T=1 | X), with trimming.

    Trimming matters: the AIPW correction divides by e(X) and 1-e(X), so a
    propensity near 0 or 1 produces an enormous weight and a single unit
    can dominate the estimate. Clipping to [trim, 1-trim] bounds that at
    the cost of a small bias -- the standard trade, made explicit.

    misspecify=True fits on X2 and X3 only, omitting X1. Since X1 is the
    sole driver of treatment assignment in this generator, that leaves the
    model with no signal at all -- it can only predict the base rate. This
    is a genuine misspecification rather than a smaller correct model,
    which matters: an earlier version of this function "broke" the model by
    fitting X1 alone, which is in fact the exactly-correct specification
    and made the robustness check vacuous.
    """
    X_fit = X[:, 1:] if misspecify else X
    X_fit = StandardScaler().fit_transform(X_fit)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_fit, treatment)
    e = model.predict_proba(X_fit)[:, 1]
    return np.clip(e, trim, 1 - trim)


def _fit_outcome_models(X, treatment, y, learner="linear", misspecify=False):
    """
    Two outcome models: mu1(X) = E[Y | T=1, X] and mu0(X) = E[Y | T=0, X].

    Each is fit on its own arm only, then used to predict the counterfactual
    for every unit -- so mu1 predicts "what would this control unit have
    scored if treated."

    learner: 'linear' (correctly specified for this generator, whose outcome
    is linear in X1) or 'gbm' (flexible, no functional-form assumption).

    misspecify=True fits on X2 and X3 only, omitting the confounder X1 --
    the same omission used to break the propensity model, so the two
    failure modes are symmetric.
    """
    X_fit = X[:, 1:] if misspecify else X

    make = (lambda: LinearRegression()) if learner == "linear" else (
        lambda: GradientBoostingRegressor(random_state=0, n_estimators=100)
    )

    m1 = make()
    m1.fit(X_fit[treatment == 1], y[treatment == 1])
    m0 = make()
    m0.fit(X_fit[treatment == 0], y[treatment == 0])

    return m1.predict(X_fit), m0.predict(X_fit)


def estimate_aipw(df, covariates=["X1", "X2", "X3"], learner="linear",
                  trim=0.01, alpha=0.05,
                  misspecify_propensity=False, misspecify_outcome=False):
    """
    Doubly robust ATE via AIPW, with an analytic confidence interval.

    Unlike PSM, the influence function here is known in closed form, so the
    standard error comes from the sample variance of the per-unit scores --
    no bootstrap needed, and unlike the matching bootstrap it is formally
    valid rather than an approximation.

    Returns a dict with the estimate, SE, CI, effective sample size, and
    which models were deliberately broken.
    """
    X = df[covariates].values
    t = df.treatment.values
    y = df.post_period_metric.values

    e = _fit_propensity(X, t, trim=trim, misspecify=misspecify_propensity)
    mu1, mu0 = _fit_outcome_models(X, t, y, learner=learner,
                                   misspecify=misspecify_outcome)

    # Per-unit influence-function scores; the ATE is their mean.
    scores = (mu1 - mu0
              + t * (y - mu1) / e
              - (1 - t) * (y - mu0) / (1 - e))

    ate = scores.mean()
    se = scores.std(ddof=1) / np.sqrt(len(scores))
    z = stats.norm.ppf(1 - alpha / 2)

    # Kish effective sample size on the IPW weights -- how much of the data
    # is actually contributing once extreme weights are accounted for.
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    ess = w.sum() ** 2 / (w ** 2).sum()

    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
        "n": len(y),
        "effective_n": ess,
        "max_weight": w.max(),
        "propensity_min": e.min(),
        "propensity_max": e.max(),
        "learner": learner,
        "misspecify_propensity": misspecify_propensity,
        "misspecify_outcome": misspecify_outcome,
    }


def double_robustness_check(df, covariates=["X1", "X2", "X3"], true_ate=5.0,
                            learner="linear"):
    """
    The claim is that AIPW is consistent if EITHER model is right. That is
    testable: break one, then the other, then both, and see which cases
    still recover the truth.

    Expected pattern:
      both correct        -> unbiased
      outcome broken      -> unbiased (propensity rescues it)
      propensity broken   -> unbiased (outcome model rescues it)
      both broken         -> biased (nothing left to rescue it)

    The last row is the important one. Without it, the first three could
    just mean the problem is easy -- and in an earlier version of this
    module they did exactly that, because the "broken" propensity model
    was accidentally still correct.
    """
    cases = [
        ("Both correct", False, False),
        ("Outcome model broken", False, True),
        ("Propensity model broken", True, False),
        ("Both broken", True, True),
    ]

    rows = []
    for label, bad_ps, bad_out in cases:
        r = estimate_aipw(df, covariates, learner=learner,
                          misspecify_propensity=bad_ps,
                          misspecify_outcome=bad_out)
        rows.append({
            "scenario": label,
            "propensity_ok": not bad_ps,
            "outcome_ok": not bad_out,
            "ate": r["ate"],
            "bias": r["ate"] - true_ate,
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "covers_true": r["ci_low"] <= true_ate <= r["ci_high"],
        })
    return pd.DataFrame(rows)


def plot_propensity_overlap(df, covariates=["X1", "X2", "X3"], trim=0.01,
                            save_path=None):
    """
    Overlap (positivity) check: the propensity distributions of treated and
    control units must share support. Where they do not, there is no
    comparable unit in the other arm, and no method -- matching, weighting,
    or regression -- can recover a causal effect there without
    extrapolating beyond the data.
    """
    import matplotlib.pyplot as plt

    X = df[covariates].values
    t = df.treatment.values
    e = _fit_propensity(X, t, trim=trim)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(e[t == 1], bins=40, alpha=0.6, label="Treated", density=True)
    ax.hist(e[t == 0], bins=40, alpha=0.6, label="Control", density=True)
    ax.set_xlabel("Estimated propensity e(X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity overlap between arms")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    from data_gen import generate_synthetic_experiment
    from matching import estimate_psm

    df = generate_synthetic_experiment(n=10000, true_ate=5.0,
                                       confounding_strength=0.5, seed=0)

    naive = (df.loc[df.treatment == 1, "post_period_metric"].mean()
             - df.loc[df.treatment == 0, "post_period_metric"].mean())

    print("=" * 64)
    print("AIPW vs PSM (n=10,000, confounding_strength=0.5, true ATE=5.0)")
    print("=" * 64)

    psm_est, n_matched, n_treated = estimate_psm(df)
    r = estimate_aipw(df)

    print(f"Naive: {naive:.4f}  (bias {naive-5.0:+.4f})")
    print(f"PSM  : {psm_est:.4f}  (bias {psm_est-5.0:+.4f})  "
          f"matched {n_matched}/{n_treated}, no analytic CI")
    print(f"AIPW : {r['ate']:.4f}  (bias {r['ate']-5.0:+.4f})  "
          f"95% CI ({r['ci_low']:.4f}, {r['ci_high']:.4f})  SE {r['se']:.4f}")
    print()
    print(f"AIPW uses all {r['n']:,} units; PSM discards unmatched controls.")
    print(f"Effective sample size after IPW weighting: {r['effective_n']:,.0f} "
          f"({r['effective_n']/r['n']*100:.1f}% of nominal)")
    print(f"Propensity range: [{r['propensity_min']:.4f}, {r['propensity_max']:.4f}], "
          f"max weight {r['max_weight']:.2f}")

    print("\n" + "=" * 64)
    print("DOUBLE ROBUSTNESS CHECK -- linear outcome learner")
    print("=" * 64)
    print("Each model is broken by omitting X1, the only confounder. The claim")
    print("is that ONE correct model suffices; the 'both broken' row is the")
    print("control that proves the first three are not just an easy problem.\n")

    dr = double_robustness_check(df, learner="linear")
    print(dr.to_string(index=False))
    print(f"\nFor reference, the naive bias is {naive-5.0:+.4f} -- 'both broken'")
    print("should collapse back toward it, since neither model can adjust for")
    print("a confounder that is absent from both.")

    print("\n" + "=" * 64)
    print("DOUBLE ROBUSTNESS CHECK -- gradient boosting outcome learner")
    print("=" * 64)
    print("Same test with a flexible outcome model that makes no functional-")
    print("form assumption about how X affects Y.\n")

    dr_gbm = double_robustness_check(df, learner="gbm")
    print(dr_gbm.to_string(index=False))

    plot_propensity_overlap(df, save_path="reports/propensity_overlap.png")
    print("\nSaved overlap plot to reports/propensity_overlap.png")