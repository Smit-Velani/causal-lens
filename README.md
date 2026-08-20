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

**Uncertainty & Robustness**
- Bootstrap confidence interval for PSM that re-runs the entire pipeline — resample, refit the propensity model, rematch — rather than resampling matched pairs and treating the matching as given
- Rosenbaum sensitivity bounds quantifying how strong an unmeasured confounder would have to be before the conclusion breaks

**Variance Reduction & Experimentation**
- CUPED variance reduction using a pre-experiment covariate
- Bayesian A/B testing via a Normal-Normal conjugate model — posterior probability of a positive effect and a credible interval, not just a p-value

**Uplift Modeling**
- Manual T-Learner (two `GradientBoostingRegressor`s, one per arm) — built from scratch rather than a black-box library call
- Evaluated two ways: against known individual-level ground truth on synthetic data, and via a from-scratch Qini/AUUC curve on real data with no hidden answer key

**Real-World Validation**
- Independently validated on the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) — genuine incrementality-test data, not synthetic

**Interactive App**
- Streamlit app with four tabs: live ground-truth validation (drag sliders, watch bias change in real time), upload-your-own-CSV analysis with column mapping and balance checking, the Criteo real-data results, and a full assumption-diagnostics panel

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

**PSM bootstrap confidence interval (seed 0, n=10,000, 500 replicates):**

| Interval | 95% CI | Width | SE |
|---|---|---|---|
| Naive paired | (4.548, 5.295) | 0.746 | 0.1904 |
| Bootstrap | (4.390, 5.543) | 1.154 | 0.3046 |

The bootstrap standard error is **60% larger**. That gap is not noise — it is the uncertainty introduced by having *estimated* the propensity model rather than known it, which the naive paired interval discards by conditioning on the matching as though it were given. The naive upper bound of 5.295 sits uncomfortably close to the true ATE of 5.0; on a less favourable draw it would exclude the truth entirely and report a significant deviation that does not exist.

**Rosenbaum sensitivity bounds (seed 0, n=10,000):** Γ* = **2.20**. An unmeasured confounder would have to shift the odds of treatment by more than 2.20×, between units identical on every measured covariate, before the result stops being significant at α = 0.05. Matching only balances what you measured; this quantifies how much hidden bias the conclusion can absorb.

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
    |   +-- matching.py          PSM + SMD balance + bootstrap CI + Rosenbaum bounds
    |   +-- bayesian_ab.py       Bayesian A/B testing
    |   +-- uplift.py            T-Learner uplift modeling
    |   +-- validate.py          Validation harness (30-seed comparison + all diagnostics)
    |   +-- criteo_validate.py   Real-world Qini/AUUC validation
    |
    +-- app/
    |   +-- streamlit_app.py     Interactive 4-tab app
    |
    +-- tests/
    |   +-- test_*.py            pytest suite (17 tests)
    |   +-- conftest.py
    |
    +-- reports/
    |   +-- criteo_validation.md
    |   +-- validation_results.csv
    |   +-- psm_balance.csv / psm_balance.png
    |   +-- psm_bootstrap.csv / psm_bootstrap.png
    |   +-- rosenbaum_bounds.csv
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

This takes a couple of minutes — the bootstrap alone refits the propensity model 500 times.

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

**Bootstrapping the whole pipeline, not the matched pairs**
- The bootstrap resamples original units and re-runs everything from the propensity fit onward. Resampling only the matched pairs is faster and gives a tighter interval, but it conditions on a matching that was itself estimated from the data — the resulting interval answers "how variable is this matched sample" rather than "how variable is this procedure." The 60% gap between the two standard errors is the size of that shortcut.

**Balance is necessary, not sufficient**
- On a deliberately unlucky draw (seed 0, n=5,000, shown in the app) every treated unit finds a match, all three covariates balance to |SMD| < 0.01, and the estimate still lands well below the true effect. Across seeds 1-9 the identical setup averages 4.96, so nothing is broken — but a single run can miss badly with every diagnostic looking clean. Balance confirms the covariates you matched on; it does not confirm the estimate, which is why the interval is reported alongside it.

**Sensitivity analysis over silence on hidden bias**
- Matching can only balance covariates that were measured, so "what about an unmeasured confounder?" is unanswerable from the data itself. Rosenbaum bounds sidestep this by asking a question the data *can* answer: how strong would such a confounder need to be to overturn the result. Reporting Γ* = 2.20 is more useful than either ignoring the objection or conceding it.

**Naive T-Learner over a library call**
- Uplift modeling uses a from-scratch two-model T-Learner rather than `causalml`, both to avoid a notoriously fragile dependency and because building it manually demonstrates the mechanism rather than hiding it behind a library.

**Closed-form Bayesian model over PyMC**
- Bayesian A/B testing uses a Normal-Normal conjugate model rather than MCMC sampling — fully analytic, easier to verify, and avoids a heavy dependency for a continuous-metric use case that has a clean closed form.

**Real Qini/AUUC, not a proxy**
- The Criteo validation uses the actual treated-vs-control-outcome Qini methodology (no hidden ground truth), separate from the simpler ground-truth-correlation check used on synthetic data — the two together cover both "does it recover a known answer" and "does it generalize to data it can't have overfit to."

## Known Limitations

- DiD's 95% CI has measured coverage of ~83%, not the nominal 95%. The parallel-trends placebo test traces this to a +0.450/period pre-existing divergence between the groups, driven by the confounder's effect growing across periods in the generator design. This is a diagnosed and quantified failure mode kept deliberately in place — a validation harness that always reports "everything's perfect" is less convincing than one that catches, explains, and reproduces a real edge case.
- The parallel-trends test requires at least two pre-treatment periods. In a single-pre-period design the assumption is structurally untestable, since only a level difference is observable and DiD differences that away by construction.
- The PSM bootstrap is an approximation, not a formally valid interval. Abadie & Imbens (2008) showed the standard bootstrap fails for nearest-neighbour matching with a fixed number of matches, because the matching function is not smooth enough for its asymptotic guarantees. It is a clear improvement on the naive paired interval and is reported as such; the Abadie-Imbens analytic variance estimator would be the rigorous alternative and is not implemented.
- Rosenbaum bounds use the normal approximation to the Wilcoxon signed-rank distribution and assume matched pairs. The lower p-value bound underflows to exactly zero here because the true effect is large relative to the noise, which is correct but not informative.
- The "Upload Your Data" tab does not impute missing values or encode categorical columns — it detects both and declines with a specific message, since dropping rows or choosing an encoding changes which units are comparable and is a modelling decision that belongs to the analyst rather than the tool.
- The synthetic generator is the same code the diagnostics are then validated against, so the parallel-trends result in particular is evidence that the test works, not independent evidence about real data. The Criteo validation partly addresses this for the uplift model; the confounding-correction methods have no equivalent external check.
- Criteo validation uses a representative subsample (223,673 rows) rather than the full multi-million-row dataset, for local runtime practicality.
- No authentication layer; the uploaded-data tab processes files entirely client-session-side with no persistence.

## Author

**Smitkumar Velani**
MS Data Science - Northeastern University, Boston

[GitHub](https://github.com/Smit-Velani) | [LinkedIn](https://linkedin.com/in/smit-velani) | [Portfolio](https://smit-velani.github.io)

*Built with Python, Streamlit, scikit-learn, statsmodels, and SciPy*