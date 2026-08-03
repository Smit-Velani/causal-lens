"""
matching.py — Propensity Score Matching (PSM).

Corrects for confounded (non-randomized) treatment assignment by matching
each treated unit to a control unit with a similar propensity score
(P(treatment=1 | covariates)), then comparing outcomes within matched pairs.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def estimate_psm(df, covariates=["X1", "X2", "X3"], caliper_multiplier=0.2):
    """
    Fits a logistic regression for propensity scores, matches each treated
    unit to its nearest-neighbor control on the logit-propensity scale
    (the standard convention), drops matches beyond a caliper, then reports
    the average outcome difference across matched pairs as the ATE.
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
    return ate_matched, within_caliper.sum(), len(treated_idx)


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