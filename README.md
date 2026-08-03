# CausalLens

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-3776AB?style=flat)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Tests](https://github.com/Smit-Velani/causal-lens/actions/workflows/tests.yml/badge.svg)

> A causal inference & experimentation platform that goes beyond correlation to measure real treatment effects. Implements CUPED variance reduction, propensity score matching, difference-in-differences, Bayesian A/B testing, and uplift modeling — then validates every method against synthetic experiments with a known ground-truth effect, so the estimators are shown to recover the true effect, not just produce a number. Further validated on the real-world Criteo Uplift dataset. Built with Python, scikit-learn, statsmodels, and Streamlit.


**Live demo:** https://causal-lens-smit.streamlit.app/


## Features

**Ground-Truth Validation Harness**
- Synthetic experiment generator with an injected, known treatment effect
- Configurable confounding strength, heterogeneous effects, and a correlated pre-period covariate
- Every method below is checked against this known answer, not just run and trusted

**Confounding Correction**
- Difference-in-Differences (via `statsmodels` OLS with a treatment×post interaction)
- Propensity Score Matching (logistic regression propensity + nearest-neighbor caliper matching)
- Both benchmarked head-to-head against naive diff-in-means across 30 random seeds

**Variance Reduction & Experimentation**
- CUPED variance reduction using a pre-experiment covariate
- Bayesian A/B testing via a Normal-Normal conjugate model — posterior probability of a positive effect and a credible interval, not just a p-value

**Uplift Modeling**
- Manual T-Learner (two `GradientBoostingRegressor`s, one per arm) — built from scratch rather than a black-box library call
- Evaluated two ways: against known individual-level ground truth on synthetic data, and via a from-scratch Qini/AUUC curve on real data with no hidden answer key

**Real-World Validation**
- Independently validated on the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) — genuine incrementality-test data, not synthetic

**Interactive App**
- Streamlit app with three tabs: live ground-truth validation (drag sliders, watch bias change in real time), upload-your-own-CSV analysis with column mapping, and the Criteo real-data results

## Results

**Confounding correction, 30 random seeds, confounding_strength=0.5, true ATE=5.0:**

| Method | Mean Bias | Mean Abs. Bias |
|---|---|---|
| Naive diff-in-means | 1.816 | 1.816 |
| Difference-in-Differences | 0.396 | 0.396 |
| Propensity Score Matching | -0.170 | 0.195 |

PSM edges out DiD as the strongest bias-corrector on this data; both dramatically outperform the naive comparison.

**CUPED variance reduction:** 47.2%, with the ATE estimate essentially unchanged (5.369 → 5.189) — the CI tightens without moving the answer.

**Uplift modeling:** Spearman correlation of 0.663 between predicted and true individual treatment effect on synthetic data; on real Criteo data (223,673 rows), the model beat random targeting by **128.4%** in AUUC.

## Tech Stack

| Layer | Technology |
|---|---|
| Causal inference | statsmodels, scikit-learn, scipy |
| Data | pandas, numpy |
| App | Streamlit |
| Testing & CI | pytest, GitHub Actions |

## Project Structure

    causal-lens/
    |
    +-- src/causallens/
    |   +-- data_gen.py          Synthetic ground-truth experiment generator
    |   +-- cuped.py             CUPED variance reduction
    |   +-- did.py               Difference-in-Differences
    |   +-- matching.py          Propensity Score Matching
    |   +-- bayesian_ab.py       Bayesian A/B testing
    |   +-- uplift.py            T-Learner uplift modeling
    |   +-- validate.py          Validation harness (30-seed bias comparison)
    |   +-- criteo_validate.py   Real-world Qini/AUUC validation
    |
    +-- app/
    |   +-- streamlit_app.py     Interactive 3-tab app
    |
    +-- tests/
    |   +-- test_*.py            pytest suite (8 tests)
    |   +-- conftest.py
    |
    +-- reports/
    |   +-- criteo_validation.md
    |   +-- validation_results.csv
    |
    +-- data/raw/                Criteo dataset (not tracked in git -- see below)
    +-- .github/workflows/tests.yml   GitHub Actions CI
    +-- requirements.txt

## Setup & Installation

Clone the repository:

```
git clone https://github.com/Smit-Velani/causal-lens.git
cd causal-lens
```

Set up the environment:

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/streamlit_app.py
```

Open browser at: `http://localhost:8501`

**To reproduce the real-world validation:** download the [Criteo Uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (or the Kaggle mirror `criteo-uplift-v2-1`), place it at `data/raw/criteo-uplift.csv`, then run `python src/causallens/criteo_validate.py`. The file is excluded from git via `.gitignore` since it exceeds GitHub's size limit.

## Running Tests

```
pip install pytest
pytest tests/ -v
```

## ML Design Decisions

**Ground truth first, everything else second**
- Every method is checked against synthetic data with a known, injected effect before being trusted on real data — the project is built around proving correctness, not assuming it.

**Naive T-Learner over a library call**
- Uplift modeling uses a from-scratch two-model T-Learner rather than `causalml`, both to avoid a notoriously fragile dependency and because building it manually demonstrates the mechanism rather than hiding it behind a library.

**Closed-form Bayesian model over PyMC**
- Bayesian A/B testing uses a Normal-Normal conjugate model rather than MCMC sampling — fully analytic, easier to verify, and avoids a heavy dependency for a continuous-metric use case that has a clean closed form.

**Honest reporting of a real limitation**
- The validation harness surfaced that DiD's 95% CI covers the true effect only ~83% of the time, not the nominal 95%. Traced this to a genuine violation of DiD's parallel-trends assumption in how the confounder affects the pre- vs. post-period baseline — documented rather than quietly patched, since a validation harness that always reports "everything's perfect" is less convincing than one that catches real edge cases.

**Real Qini/AUUC, not a proxy**
- The Criteo validation uses the actual treated-vs-control-outcome Qini methodology (no hidden ground truth), separate from the simpler ground-truth-correlation check used on synthetic data — the two together cover both "does it recover a known answer" and "does it generalize to data it can't have overfit to."

## Known Limitations

- DiD's 95% CI has measured coverage of ~83%, not the nominal 95%, due to a parallel-trends assumption violation in the synthetic confounder design (see above) — a known, documented caveat rather than a silent inaccuracy.
- The "Upload Your Data" tab assumes numeric treatment (0/1) and outcome columns; it does not yet handle categorical treatment coding or missing data.
- Criteo validation uses a representative subsample (223,673 rows) rather than the full multi-million-row dataset, for local runtime practicality.
- No authentication layer; the uploaded-data tab processes files entirely client-session-side with no persistence.

## Author

**Smitkumar Velani**
MS Data Science - Northeastern University, Boston

[GitHub](https://github.com/Smit-Velani) | [LinkedIn](https://linkedin.com/in/smit-velani) | [Portfolio](https://smit-velani.github.io)

*Built with Python, Streamlit, scikit-learn, statsmodels, and SciPy*