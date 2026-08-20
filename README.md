# CausalLens

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-3776AB?style=flat)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Tests](https://github.com/Smit-Velani/causal-lens/actions/workflows/tests.yml/badge.svg)

> A causal inference & experimentation platform that goes beyond correlation to measure real treatment effects. Implements CUPED variance reduction, propensity score matching, difference-in-differences, Bayesian A/B testing, and uplift modeling — then validates every method against synthetic experiments with a known ground-truth effect, so the estimators are shown to recover the true effect, not just produce a number. Each estimator also ships its own assumption diagnostic, so a method is only trusted once the assumption it depends on has been tested. Further validated on the real-world Criteo Uplift dataset. Built with Python, scikit-learn, statsmodels, and Streamlit.


**Live demo:** https://causal-lens-smit.streamlit.app/


## Features

**Ground-Truth Validation Harness**
- Synthetic experiment generator with an injected, known treatment effect
- Configurable confounding strength, heterogeneous effects, a correlated pre-period covariate, and an optional multi-period pre-treatment panel
- Every method below is checked against this known answer, not just run and trusted

**Confounding Correction**
- Difference-in-Differences (via `statsmodels` OLS with a treatment×post interaction)
- Propensity Score Matching (logistic regression propensity + nearest-neighbor caliper matching)
- Both benchmarked head-to-head against naive diff-in-means across 30 random seeds

**Assumption Diagnostics**
- CUPED residual-correlation check — confirms the covariate's linear signal was actually removed, rather than inferring it from a narrower CI
- PSM standardized mean difference (SMD) balance, before vs after matching, with a Love plot and the conventional |SMD| < 0.1 threshold
- DiD parallel-trends placebo test — regresses outcome on treatment×period using pre-treatment data only, where no effect can exist by construction

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

**CUPED residual correlation:** corr(Y, X_pre) = 0.689 before adjustment, and -0.000000 (p = 1.000) after. The covariate's linear signal is removed exactly, which is the direct proof CUPED did what it claims rather than the indirect evidence of a narrower interval.

**PSM covariate balance (SMD, seed 0):**

| Covariate | SMD before | SMD after | Balanced (\|SMD\| < 0.1) |
|---|---|---|---|
| X1 (confounder) | 0.483 | 0.001 | yes |
| X2 (pure noise) | 0.005 | -0.036 | yes |
| X3 (effect modifier) | -0.001 | -0.011 | yes |

Matching collapsed the confounder X1 by ~99.8% while X2 and X3, already balanced, stayed balanced. The correction landed on the variable that actually drove treatment assignment, which is the result you want — not an indiscriminate reshuffle.

**Parallel-trends placebo test (3 pre-treatment periods):**

| Scenario | Pre-period gaps (treated − control) | Pre-trend coef | p-value | Verdict | Realised DiD bias |
|---|---|---|---|---|---|
| Default generator | 0.439 → 0.925 → 1.340 | +0.450 | 1.5e-04 | violated | +0.366 |
| Trend removed | 1.384 → 1.398 → 1.340 | -0.022 | 0.857 | holds | -0.106 |

The pre-trend is measured entirely before treatment exists, and it predicts the DiD bias observed afterwards to within ~0.08. Re-running the generator with the trend removed returns a flat gap and a null pre-trend — note the large *constant* gap of ~1.38 in that case, which the test correctly ignores, since DiD differences out level differences by design and only a difference in *slope* breaks it.

**Uplift modeling:** Spearman correlation of 0.663 between predicted and true individual treatment effect on synthetic data; on real Criteo data (223,673 rows), the model beat random targeting by **128.4%** in AUUC.

## Tech Stack

| Layer | Technology |
|---|---|
| Causal inference | statsmodels, scikit-learn, scipy |
| Data | pandas, numpy |
| App | Streamlit |
| Plots | matplotlib |
| Testing & CI | pytest, GitHub Actions |

## Project Structure

    causal-lens/
    |
    +-- src/causallens/
    |   +-- data_gen.py          Synthetic generator (single or multi pre-period)
    |   +-- cuped.py             CUPED + residual-correlation diagnostic
    |   +-- did.py               Difference-in-Differences + parallel-trends placebo test
    |   +-- matching.py          Propensity Score Matching + SMD balance + Love plot
    |   +-- bayesian_ab.py       Bayesian A/B testing
    |   +-- uplift.py            T-Learner uplift modeling
    |   +-- validate.py          Validation harness (30-seed comparison + diagnostics)
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
    |   +-- psm_balance.csv
    |   +-- psm_balance.png
    |   +-- cuped_diagnostics.csv
    |   +-- parallel_trends_test.csv
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
venv\Scripts\Activate.ps1        # Windows PowerShell
source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/streamlit_app.py
```

Open browser at: `http://localhost:8501`

Reproduce the full validation and all diagnostics:

```
python src/causallens/validate.py
```

**To reproduce the real-world validation:** download the [Criteo Uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (or the Kaggle mirror `criteo-uplift-v2-1`), place it at `data/raw/criteo-uplift.csv`, then run `python src/causallens/criteo_validate.py`. The file is excluded from git via `.gitignore` since it exceeds GitHub's size limit.

## Running Tests

```
pip install pytest
pytest tests/ -v
```

## ML Design Decisions

**Ground truth first, everything else second**
- Every method is checked against synthetic data with a known, injected effect before being trusted on real data — the project is built around proving correctness, not assuming it.

**Every estimator ships its own assumption diagnostic**
- A correct-looking point estimate is not evidence the method worked; it can be right by luck or wrong in a way the final number hides. So CUPED reports residual correlation, PSM reports SMD balance, and DiD reports a parallel-trends placebo test. Each one checks the specific assumption that estimator depends on, independently of whether the headline number looks reasonable.

**Diagnosing, not just detecting, a failed assumption**
- The validation harness surfaced that DiD's 95% CI covers the true effect only ~83% of the time, not the nominal 95%. Rather than assume a cause, the generator was extended to emit multiple pre-treatment periods and a placebo test was added: regress outcome on treatment×period using pre-treatment data only, where no treatment effect can exist by construction. The test measures a +0.450/period pre-trend (p = 1.5e-04) that predicts the realised DiD bias of +0.366 — so the parallel-trends violation is measured, not inferred. Running the same test on a version of the generator that satisfies parallel trends returns a null result (coef -0.022, p = 0.857), so the test discriminates rather than always failing.

**Naive T-Learner over a library call**
- Uplift modeling uses a from-scratch two-model T-Learner rather than `causalml`, both to avoid a notoriously fragile dependency and because building it manually demonstrates the mechanism rather than hiding it behind a library.

**Closed-form Bayesian model over PyMC**
- Bayesian A/B testing uses a Normal-Normal conjugate model rather than MCMC sampling — fully analytic, easier to verify, and avoids a heavy dependency for a continuous-metric use case that has a clean closed form.

**Real Qini/AUUC, not a proxy**
- The Criteo validation uses the actual treated-vs-control-outcome Qini methodology (no hidden ground truth), separate from the simpler ground-truth-correlation check used on synthetic data — the two together cover both "does it recover a known answer" and "does it generalize to data it can't have overfit to."

## Known Limitations

- DiD's 95% CI has measured coverage of ~83%, not the nominal 95%. The parallel-trends placebo test traces this to a +0.450/period pre-existing divergence between the groups, driven by the confounder's effect growing across periods in the generator design. This is a diagnosed and quantified failure mode kept deliberately in place — a validation harness that always reports "everything's perfect" is less convincing than one that catches, explains, and reproduces a real edge case.
- The parallel-trends test requires at least two pre-treatment periods. In a single-pre-period design the assumption is structurally untestable, since only a level difference is observable and DiD differences that away by construction.
- The "Upload Your Data" tab assumes numeric treatment (0/1) and outcome columns; it does not yet handle categorical treatment coding or missing data.
- PSM reports a point estimate without a confidence interval; matched-pair uncertainty would need a bootstrap, which is not yet implemented.
- No sensitivity analysis for unmeasured confounding — PSM only balances covariates it can observe.
- Criteo validation uses a representative subsample (223,673 rows) rather than the full multi-million-row dataset, for local runtime practicality.
- No authentication layer; the uploaded-data tab processes files entirely client-session-side with no persistence.

## Author

**Smitkumar Velani**
MS Data Science - Northeastern University, Boston

[GitHub](https://github.com/Smit-Velani) | [LinkedIn](https://linkedin.com/in/smit-velani) | [Portfolio](https://smit-velani.github.io)

*Built with Python, Streamlit, scikit-learn, statsmodels, and SciPy*