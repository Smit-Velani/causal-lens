from data_gen import generate_synthetic_experiment


def test_confounded_data_shows_bias():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.5, seed=42)
    naive = (
        df.loc[df.treatment == 1, "post_period_metric"].mean()
        - df.loc[df.treatment == 0, "post_period_metric"].mean()
    )
    assert abs(naive - 5.0) > 1.0, "confounding should produce a clear naive bias"


def test_clean_experiment_shows_small_bias():
    df = generate_synthetic_experiment(true_ate=5.0, confounding_strength=0.0, seed=42)
    naive = (
        df.loc[df.treatment == 1, "post_period_metric"].mean()
        - df.loc[df.treatment == 0, "post_period_metric"].mean()
    )
    assert abs(naive - 5.0) < 0.6, "randomized experiment should have small bias"