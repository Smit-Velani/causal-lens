# Criteo Uplift Validation

The uplift model (T-Learner, see `uplift.py` and Phase 6) was validated a
second time against a real, independently collected experimental dataset
— the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/),
built from genuine incrementality tests where a random slice of users is
deliberately withheld from ad targeting.

## Setup

- **Sample size:** 223,673 rows, drawn evenly across the full dataset via
  chunked random sampling (not just the first N rows, which are not
  randomly ordered in the raw file).
- **Treatment split:** ~85% treated / ~15% control — consistent with how
  incrementality-test datasets are typically designed in industry (a
  small held-out control group, most users see the normal treatment).
- **Model:** the same T-Learner (two `GradientBoostingRegressor`s) used
  in Phase 6's synthetic validation — no retraining logic changed,
  only the data source.

## Results

| Metric      | Value |
|-------------|-------|
| AUUC, model | 27.48 |
| AUUC, random targeting baseline | 12.03 |
| **Model beats random targeting by** | **128.4%** |

## Why this matters

Phase 6 validated the same model against synthetic data with a *known*
ground-truth effect — a strong check, but one that can't fully rule out
the model exploiting quirks of how the synthetic data was generated. This
result is the independent confirmation: on real, messier experimental
data with no hidden answer key, the model still ranks users by treatment
benefit far better than random targeting would. Together, the two checks
cover both failure modes: "does it recover a known answer" and "does it
generalize to data it can't have overfit to."