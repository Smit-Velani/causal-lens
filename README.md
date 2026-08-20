# CausalLens

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-3776AB?style=flat)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Tests](https://github.com/Smit-Velani/causal-lens/actions/workflows/tests.yml/badge.svg)

> A causal inference and experimentation platform covering the full lifecycle of an experiment: designing it, running it under continuous monitoring, and estimating the effect afterwards. Implements CUPED variance reduction, difference-in-differences, propensity score matching, doubly robust AIPW, Bayesian A/B testing, uplift modeling, always-valid sequential inference, and power analysis. Every estimator is validated against synthetic data with a known injected effect, ships its own assumption diagnostic, and is checked externally on the LaLonde/NSW and Criteo benchmarks. Built with Python, scikit-learn, statsmodels, and Streamlit.

**Live demo:** https://causal-lens-smit.streamlit.app/

---

## What this is trying to do differently

Most causal inference projects implement the methods and report the numbers they produce. That is not evidence the methods worked — an estimate can be right by luck, or wrong in a way the final number conceals. Three things follow from taking that seriously, and they shape the whole project:

**Every method is validated against a known answer.** The synthetic generator injects a treatment effect, so bias is measurable rather than assumed.

**Every estimator ships a diagnostic for the assumption it depends on.** CUPED reports residual correlation, PSM reports covariate balance and sensitivity to hidden bias, DiD reports a parallel-trends placebo test, AIPW reports overlap. These check the assumption, not the answer.

**The failures are kept and documented.** DiD's confidence interval undercovers at 83%, and the placebo test explains why. AIPW beats matching on synthetic data and loses badly on LaLonde, and the balance table explains why. A validation harness that only ever reports success is less informative than one that catches something.

---

## Features

**Ground-truth validation harness**
- Synthetic generator with an injected known effect, configurable confounding, heterogeneous effects, a correlated pre-period covariate, and an optional multi-period pre-treatment panel
- Every method benchmarked head-to-head across 30 random seeds

**Confounding correction**
- Difference-in-Differences (`statsmodels` OLS with a treatment×post interaction)
- Propensity Score Matching (logistic propensity, nearest-neighbour caliper matching on the logit scale)
- Augmented IPW / doubly robust, consistent if either the propensity or the outcome model is correct

**Assumption diagnostics**
- CUPED residual-correlation check
- PSM standardized mean difference (SMD) balance with a Love plot
- DiD parallel-trends placebo test on pre-treatment periods only
- AIPW propensity overlap and effective sample size

**Uncertainty and robustness**
- Bootstrap confidence interval for PSM that refits the propensity model in every replicate
- Rosenbaum sensitivity bounds quantifying tolerance to unmeasured confounding
- Analytic influence-function standard errors for AIPW

**Variance reduction and experimentation**
- CUPED using a pre-experiment covariate
- Bayesian A/B testing via a Normal-Normal conjugate model
- T-Learner uplift modeling, built from scratch rather than a library call

**Sequential inference**
- mSPRT always-valid p-values and confidence sequences
- Simulation quantifying how much peeking inflates the fixed-horizon error rate

**Experiment design**
- Sample size, minimum detectable effect, achieved power
- Duration planning with a whole-week calendar floor
- CUPED sample savings and the sequential-testing power penalty

**External validation**
- LaLonde / NSW benchmark with PSID and CPS observational controls
- Criteo Uplift Prediction Dataset

---

## Results

### Confounding correction
30 random seeds, `confounding_strength=0.5`, true ATE = 5.0:

| Method | Mean Bias | Mean Abs. Bias |
|---|---|---|
| Naive diff-in-means | 1.816 | 1.816 |
| Difference-in-Differences | 0.396 | 0.396 |
| Propensity Score Matching | -0.158 | 0.190 |
| AIPW (seed 0) | -0.048 | -- |

AIPW edges out PSM while using all 10,000 units rather than discarding unmatched controls, and comes with a formally valid analytic confidence interval where PSM needs a bootstrap.

### CUPED
**47.2% variance reduction**, ATE essentially unchanged (5.369 → 5.189) — the interval tightens without moving the answer.

**Residual correlation:** corr(Y, X_pre) = 0.689 before adjustment, **-0.000000** (p = 1.000) after. The covariate's linear signal is removed exactly. This is the direct proof CUPED worked, as opposed to the indirect evidence of a narrower interval.

### PSM covariate balance
Seed 0, `confounding_strength=0.5`:

| Covariate | SMD before | SMD after | Balanced |
|---|---|---|---|
| X1 (confounder) | 0.483 | 0.0006 | yes |
| X2 (pure noise) | 0.005 | -0.028 | yes |
| X3 (effect modifier) | -0.001 | -0.008 | yes |

Matching collapsed X1 by 99.9% while X2 and X3, already balanced, stayed balanced. The correction landed on the variable that actually drove assignment rather than reshuffling indiscriminately.

### PSM uncertainty
Seed 0, n = 10,000, 500 bootstrap replicates:

| Interval | 95% CI | Width | SE |
|---|---|---|---|
| Naive paired | (4.477, 5.225) | 0.748 | 0.1907 |
| Bootstrap | (4.402, 5.520) | 1.118 | 0.3011 |

The bootstrap standard error is **58% larger**. That gap is the uncertainty introduced by having *estimated* the propensity model rather than known it, which the naive paired interval discards by conditioning on the matching as though it were given. Note the naive upper bound of 5.225 sits close to the true ATE of 5.0; on a less favourable draw it would exclude the truth and report a significant deviation that does not exist.

**Rosenbaum sensitivity:** Γ* = **2.10**. An unmeasured confounder would have to shift the odds of treatment by more than 2.10×, between units identical on every measured covariate, before the result stops being significant at α = 0.05.

### Parallel-trends placebo test
Three pre-treatment periods:

| Scenario | Pre-period gaps | Pre-trend coef | p-value | Verdict | Realised DiD bias |
|---|---|---|---|---|---|
| Default generator | 0.439 → 0.925 → 1.340 | +0.450 | 1.5e-04 | violated | +0.366 |
| Trend removed | 1.384 → 1.398 → 1.340 | -0.022 | 0.857 | holds | -0.106 |

The pre-trend is measured entirely before treatment exists, and it predicts the DiD bias observed afterwards to within 0.08. Removing the trend returns a flat gap and a null result — note the large *constant* gap of 1.38 in that case, which the test correctly ignores, since DiD differences out level differences by design and only a difference in slope breaks it.

### Double robustness
AIPW is claimed to be consistent if either model is correct. That is testable by breaking each in turn:

| Scenario | ATE | Bias | Covers true effect |
|---|---|---|---|
| Both correct | 4.952 | -0.048 | yes |
| Outcome model broken | 4.945 | -0.055 | yes |
| Propensity model broken | 4.934 | -0.066 | yes |
| **Both broken** | **6.763** | **+1.763** | **no** |

The last row is the control. With the confounder absent from both models, AIPW lands at 6.7634 against a naive difference of 6.7630 — it reduces *exactly* to the unadjusted comparison, because there is nothing left to adjust with. Without that row, the first three could just mean the problem was easy.

### Sequential testing
1,000 simulated A/A tests, 20 interim looks each, true effect zero:

| Looks | Fixed-horizon FPR | Always-valid FPR |
|---|---|---|
| 1 | 6.0% | 0.0% |
| 5 | 13.0% | 0.3% |
| 20 | 22.3% | 0.3% |
| 50 | **27.3%** | 1.3% |

After 50 looks a fixed-horizon t-test declares a false positive more than a quarter of the time on data with no effect at all. The always-valid line stays flat, so the number of looks stops being a decision that has to be made correctly in advance.

The trade is real and paid in power: at a 0.15 SD effect the fixed-horizon test detected 99.8% of the time and mSPRT 90.5%, roughly 25% more traffic for the same sensitivity.

### Experiment design
α = 0.05, power = 0.80:

| MDE (SD) | n per arm | With CUPED | Saved |
|---|---|---|---|
| 0.05 | 6,280 | 3,316 | 47.2% |
| 0.10 | 1,570 | 829 | 47.2% |
| 0.20 | 393 | 208 | 47.1% |

Required sample size is linear in variance, so CUPED's 47.2% reduction passes through one-for-one rather than being square-rooted away. Same traffic, same design, roughly half the experiment.

### Uplift modeling
Spearman correlation of 0.663 between predicted and true individual treatment effect on synthetic data; on real Criteo data (223,673 rows) the model beat random targeting by **128.4%** in AUUC (27.48 vs 12.03).

### External validation — LaLonde / NSW
The canonical benchmark for observational causal methods. The National Supported Work Demonstration was a randomized trial, so the experimental comparison gives a credible effect. LaLonde then discarded the experimental controls and substituted survey respondents from PSID and CPS.

**Experimental benchmark: $1,794** (185 treated, 260 control) — matching the published Dehejia-Wahba figure.

| Method | PSID controls | CPS controls |
|---|---|---|
| Naive diff-in-means | -15,205 | -8,498 |
| **PSM** | **2,696** | **1,637** |
| AIPW (linear) | -8,194 | -3,638 |
| AIPW (GBM) | -12,565 | -9,468 |

Starting from -$8,498 with CPS controls, PSM recovers $1,637 against an experimental truth of $1,794 — an error of $157.

**AIPW fails here, and the reason is instructive.** Several covariates begin with |SMD| above 1.5; the two groups barely share support. Matching survives by discarding non-comparable controls. Weighting cannot, because it must keep them and assign them enormous weights. On the synthetic data, where the propensity range is [0.126, 0.847], AIPW wins. Same two methods, opposite ranking, and the balance table tells you which situation you are in.

PSM balanced only 4 of 8 covariates on PSID, so the close hit there is not a clean one. That is the Smith and Todd (2005) critique in miniature: the result is real but fragile to specification.

---

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
    |   +-- data_gen.py           Synthetic generator (single or multi pre-period)
    |   +-- cuped.py              CUPED + residual-correlation diagnostic
    |   +-- did.py                Difference-in-Differences + parallel-trends placebo test
    |   +-- matching.py           PSM + SMD balance + bootstrap CI + Rosenbaum bounds
    |   +-- aipw.py               Doubly robust AIPW + double-robustness check + overlap
    |   +-- bayesian_ab.py        Bayesian A/B testing
    |   +-- uplift.py             T-Learner uplift modeling
    |   +-- sequential.py         mSPRT, confidence sequences, peeking simulation
    |   +-- power.py              MDE, sample size, duration, CUPED savings
    |   +-- validate.py           Main harness: 30-seed comparison + all diagnostics
    |   +-- criteo_validate.py    External uplift benchmark
    |   +-- lalonde_validate.py   External NSW/PSID/CPS benchmark
    |
    +-- app/
    |   +-- streamlit_app.py      Interactive 6-tab app
    |
    +-- tests/
    |   +-- test_*.py             pytest suite (45 tests)
    |   +-- conftest.py
    |
    +-- reports/                  All generated CSVs and figures
    +-- data/raw/lalonde/         LaLonde benchmark files (cached)
    +-- .github/workflows/tests.yml
    +-- requirements.txt

## Setup

```
git clone https://github.com/Smit-Velani/causal-lens.git
cd causal-lens

python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/streamlit_app.py
```

## Reproducing the results

```
python src/causallens/validate.py           # main harness + all diagnostics (~3 min)
python src/causallens/sequential.py         # peeking simulation, mSPRT
python src/causallens/power.py              # MDE, duration, CUPED savings
python src/causallens/lalonde_validate.py   # external NSW benchmark
python src/causallens/criteo_validate.py    # external uplift benchmark
```

`validate.py` takes a few minutes — the bootstrap alone refits the propensity model 500 times. The LaLonde script downloads four files from NBER on first run and caches them locally.

For the Criteo validation, download the [Criteo Uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (or the Kaggle mirror `criteo-uplift-v2-1`) and place it at `data/raw/criteo-uplift.csv`. It is excluded from git for size.

## Tests

```
pytest tests/ -v
```

45 tests. The LaLonde tests skip cleanly if the benchmark data has not been fetched, so CI does not depend on a third-party host being up.

---

## Design decisions

**Ground truth first, everything else second**
Every method is checked against data with a known injected effect before being trusted on real data. The project is built around proving correctness rather than assuming it.

**Every estimator ships its own assumption diagnostic**
A correct-looking point estimate is not evidence a method worked. CUPED reports residual correlation, PSM reports SMD balance, DiD reports a placebo test, AIPW reports overlap and effective sample size. Each checks the specific assumption that estimator depends on, independently of whether the headline number looks reasonable.

**Diagnosing rather than just detecting a failed assumption**
The harness surfaced that DiD's 95% CI covers the true effect only ~83% of the time. Rather than assume a cause, the generator was extended to emit multiple pre-treatment periods and a placebo test was added: regress outcome on treatment×period using pre-treatment data only, where no effect can exist by construction. The test measures a +0.450/period pre-trend (p = 1.5e-04) that predicts the realised DiD bias of +0.366 — so the violation is measured, not inferred. Running the same test on a parallel-trends-satisfying version returns a null result, so the test discriminates rather than always failing.

**Bootstrapping the whole pipeline, not the matched pairs**
The bootstrap resamples original units and re-runs everything from the propensity fit onward. Resampling only the matched pairs is faster and gives a tighter interval, but it conditions on a matching that was itself estimated from the data — that interval answers "how variable is this matched sample" rather than "how variable is this procedure." The 58% gap between the two standard errors is the size of that shortcut.

**Balance is necessary, not sufficient**
On a deliberately unlucky draw (seed 0, n=5,000, shown in the app) every treated unit finds a match, all three covariates balance to |SMD| < 0.01, and the estimate still lands well below the true effect. Across seeds 1–9 the identical setup averages 4.975, so nothing is broken — but a single run can miss badly with every diagnostic looking clean. Balance confirms the covariates you matched on; it does not confirm the estimate.

**Sensitivity analysis over silence on hidden bias**
Matching can only balance covariates that were measured, so "what about an unmeasured confounder?" is unanswerable from the data itself. Rosenbaum bounds ask a question the data *can* answer: how strong would such a confounder have to be to overturn the result. Reporting Γ* = 2.10 is more useful than either ignoring the objection or conceding it.

**Testing double robustness properly requires breaking both models**
An earlier version of `aipw.py` "broke" the propensity model by fitting it on X1 alone — which is in fact the exactly-correct specification for this generator, making the whole check vacuous while appearing to pass. Both models are now broken the same way, by omitting X1, and the "both broken" row confirms the misspecification actually bites.

**Naive T-Learner over a library call**
Uplift modeling uses a from-scratch two-model T-Learner rather than `causalml`, both to avoid a fragile dependency and because building it manually demonstrates the mechanism rather than hiding it.

**Closed-form Bayesian model over PyMC**
Bayesian A/B testing uses a Normal-Normal conjugate model rather than MCMC — fully analytic, easier to verify, and avoids a heavy dependency for a continuous-metric case with a clean closed form.

**A calendar floor on experiment duration**
The planner never returns a sub-week duration even when the raw sample arrives sooner. A shorter run confounds the treatment effect with day-of-week seasonality; an experiment that only sees Tuesdays is measuring Tuesday.

---

## Known limitations

- **DiD undercovers.** The 95% CI covers the true effect ~83% of the time. The placebo test traces this to a +0.450/period pre-existing divergence driven by the confounder's effect growing across periods in the generator design. This is a diagnosed and quantified failure mode kept deliberately in place.
- **The parallel-trends test needs two or more pre-periods.** In a single-pre-period design the assumption is structurally untestable, since only a level difference is observable and DiD differences that away by construction.
- **The PSM bootstrap is an approximation.** Abadie & Imbens (2008) showed the standard bootstrap is not formally valid for nearest-neighbour matching with a fixed number of matches, because the matching function is not smooth enough for its asymptotic guarantees. It is a clear improvement on the naive paired interval and is reported as such; the Abadie-Imbens analytic variance estimator would be the rigorous alternative and is not implemented.
- **Rosenbaum bounds use the normal approximation** to the Wilcoxon signed-rank distribution and assume matched pairs. The lower p-value bound underflows to exactly zero here because the effect is large relative to the noise — correct but uninformative.
- **The mSPRT is conservative.** Its false-positive rate sits near 0.3% against a nominal 5%. The mixture prior τ is set for effects around 1 SD while the simulation uses 0 to 0.15 SD, so the test is more conservative than it needs to be. Tuning τ to the effect size of interest would recover power.
- **AIPW requires overlap and does not degrade gracefully without it.** The LaLonde results show it failing badly where the propensity distributions barely intersect. Trimming bounds the weights but does not fix a positivity violation.
- **The upload tab does not impute or encode.** Missing values and categorical columns are detected and reported rather than silently handled, since dropping rows or choosing an encoding changes which units are comparable and is a modelling decision that belongs to the analyst.
- **The synthetic generator is the same code the diagnostics are validated against.** The LaLonde and Criteo benchmarks address this externally, but the parallel-trends result in particular is evidence that the test works rather than independent evidence about real data.
- **Criteo validation uses a 223,673-row representative subsample** rather than the full multi-million-row dataset, for local runtime practicality.
- **No authentication layer.** The uploaded-data tab processes files entirely client-session-side with no persistence.

---

## References

- LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training Programs. *American Economic Review* 76(4).
- Dehejia, R. & Wahba, S. (1999, 2002). Propensity Score Matching Methods for Non-Experimental Causal Studies.
- Smith, J. & Todd, P. (2005). Does Matching Overcome LaLonde's Critique of Nonexperimental Estimators?
- Abadie, A. & Imbens, G. (2008). On the Failure of the Bootstrap for Matching Estimators. *Econometrica* 76(6).
- Rosenbaum, P. (2002). *Observational Studies*, 2nd ed.
- Deng, A., Xu, Y., Kohavi, R. & Walker, T. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED).
- Johari, R., Koomen, P., Pekelis, L. & Walsh, D. (2017). Peeking at A/B Tests: Why It Matters and What to Do About It.

## Author

**Smitkumar Velani**
MS Data Science — Northeastern University, Boston

[GitHub](https://github.com/Smit-Velani) | [LinkedIn](https://linkedin.com/in/smit-velani) | [Portfolio](https://smit-velani.github.io)