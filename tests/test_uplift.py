from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from data_gen import generate_synthetic_experiment
from uplift import fit_t_learner, predict_uplift


def test_uplift_model_finds_heterogeneous_effect():
    df = generate_synthetic_experiment(n=20000, true_ate=5.0, confounding_strength=0.0, seed=42)
    covariates = ["X1", "X2", "X3"]
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

    model_treated, model_control = fit_t_learner(train_df, covariates)
    uplift_pred = predict_uplift(model_treated, model_control, test_df[covariates].values)

    corr, _ = spearmanr(uplift_pred, test_df.true_individual_effect.values)
    assert corr > 0.3, "predicted uplift should correlate meaningfully with the true individual effect"