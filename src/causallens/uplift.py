"""
uplift.py — Uplift modeling via a manual T-Learner.

Trains two separate models -- one on treated units, one on control units --
then predicts each user's individual treatment effect as the gap between
the two models' predictions. Evaluated against the synthetic ground truth
(true_individual_effect), which real-world data never gives you: this
directly checks whether the model finds WHO benefits more, not just
whether the average effect is right.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


def fit_t_learner(train_df, covariates=["X1", "X2", "X3"]):
    X_train = train_df[covariates].values
    t_train = train_df.treatment.values
    y_train = train_df.post_period_metric.values

    model_treated = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
    model_control = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
    model_treated.fit(X_train[t_train == 1], y_train[t_train == 1])
    model_control.fit(X_train[t_train == 0], y_train[t_train == 0])
    return model_treated, model_control


def predict_uplift(model_treated, model_control, X):
    return model_treated.predict(X) - model_control.predict(X)


if __name__ == "__main__":
    from data_gen import generate_synthetic_experiment

    df = generate_synthetic_experiment(n=20000, true_ate=5.0, confounding_strength=0.0)
    covariates = ["X1", "X2", "X3"]
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

    model_treated, model_control = fit_t_learner(train_df, covariates)
    uplift_pred = predict_uplift(model_treated, model_control, test_df[covariates].values)
    true_effect = test_df.true_individual_effect.values

    corr, pval = spearmanr(uplift_pred, true_effect)
    print(f"Spearman corr(predicted uplift, true effect) = {corr:.3f} (p={pval:.2e})")
    print()

    order = np.argsort(-uplift_pred)
    sorted_true = true_effect[order]
    random_mean = true_effect.mean()
    print(f"Targeting curve (avg TRUE effect captured, vs. random targeting = {random_mean:.3f}):")
    for k in [0.1, 0.2, 0.3, 0.5, 1.0]:
        n_k = int(len(sorted_true) * k)
        print(f"  top {int(k*100):>3}% by predicted uplift: {sorted_true[:n_k].mean():.3f}")