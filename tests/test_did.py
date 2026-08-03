from data_gen import generate_synthetic_experiment
from did import estimate_did


def test_did_reduces_bias_under_confounding():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=42)
    naive = (
        df.loc[df.treatment == 1, "post_period_metric"].mean()
        - df.loc[df.treatment == 0, "post_period_metric"].mean()
    )
    did_est, lo, hi = estimate_did(df)
    assert abs(did_est - 5.0) < abs(naive - 5.0), "DiD should reduce bias relative to naive comparison"