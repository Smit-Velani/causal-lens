from data_gen import generate_synthetic_experiment
from matching import estimate_psm


def test_psm_reduces_bias_under_confounding():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=42)
    naive = (
        df.loc[df.treatment == 1, "post_period_metric"].mean()
        - df.loc[df.treatment == 0, "post_period_metric"].mean()
    )
    psm_est, n_matched, n_treated = estimate_psm(df)
    assert abs(psm_est - 5.0) < abs(naive - 5.0), "PSM should reduce bias relative to naive comparison"
    assert n_matched / n_treated > 0.9, "most treated units should find a good match"