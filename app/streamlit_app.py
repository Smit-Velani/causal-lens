"""
streamlit_app.py — CausalLens: interactive causal inference & experimentation platform.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "causallens"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st

from data_gen import generate_synthetic_experiment
from cuped import cuped_adjust, estimate_ate_with_ci, check_residual_correlation
from did import estimate_did, pre_trends_test, pre_period_gaps
from matching import (estimate_psm, smd_before_after, plot_smd,
                      naive_paired_ci, bootstrap_psm_ci, rosenbaum_bounds,
                      plot_bootstrap)
from bayesian_ab import bayesian_ab_test
from aipw import estimate_aipw, double_robustness_check, plot_propensity_overlap
from sequential import (simulate_peeking, peeking_curve, plot_peeking_curve,
                        plot_confidence_sequence)
from power import (required_sample_size, minimum_detectable_effect,
                   achieved_power, experiment_duration, cuped_sample_savings,
                   plot_power_curves, plot_mde_curve)

st.set_page_config(page_title="CausalLens", layout="wide")
st.title("CausalLens — Causal Inference & Experimentation Platform")


# ---------------------------------------------------------------------------
# Cached generators — sliders re-run the whole script, so avoid regenerating
# and re-matching identical data on every widget interaction.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def make_data(n, true_ate, confounding_strength, seed=42, n_pre_periods=1, x1_trend=1.0):
    return generate_synthetic_experiment(
        n=n, true_ate=true_ate, confounding_strength=confounding_strength,
        seed=seed, n_pre_periods=n_pre_periods, x1_trend_per_period=x1_trend,
    )


@st.cache_data(show_spinner=False)
def cached_balance(n, true_ate, confounding_strength, seed=42):
    df = make_data(n, true_ate, confounding_strength, seed)
    return smd_before_after(df)


@st.cache_data(show_spinner=False)
def cached_bootstrap(n, true_ate, confounding_strength, n_boot, seed=0):
    df = make_data(n, true_ate, confounding_strength, seed)
    return bootstrap_psm_ci(df, n_boot=n_boot, seed=seed)


@st.cache_data(show_spinner=False)
def cached_rosenbaum(n, true_ate, confounding_strength, seed=0):
    df = make_data(n, true_ate, confounding_strength, seed)
    return rosenbaum_bounds(df)


@st.cache_data(show_spinner=False)
def cached_dr_check(n, true_ate, confounding_strength, learner, seed=0):
    df = make_data(n, true_ate, confounding_strength, seed)
    return double_robustness_check(df, true_ate=true_ate, learner=learner)


@st.cache_data(show_spinner=False)
def cached_peeking_curve(n_experiments, seed=0):
    return peeking_curve(n_experiments=n_experiments, seed=seed)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Ground Truth Validation",
    "Upload Your Data",
    "Real-World Benchmarks",
    "Assumption Diagnostics",
    "Doubly Robust (AIPW)",
    "Experiment Design",
])

# ---------------------------------------------------------------------------
# TAB 1 — Ground Truth Validation
# ---------------------------------------------------------------------------
with tab1:
    st.header("Validate every method against a known, injected treatment effect")
    st.write(
        "This generates synthetic data where the true treatment effect is "
        "known in advance, so you can see exactly how much each method's "
        "estimate deviates from the truth."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.slider("Sample size", 1000, 50000, 10000, step=1000)
    with col2:
        true_ate = st.slider("True average treatment effect", 1.0, 20.0, 5.0, step=0.5)
    with col3:
        confounding_strength = st.slider("Confounding strength", 0.0, 1.0, 0.5, step=0.1)

    df = make_data(n, true_ate, confounding_strength)

    naive = (
        df.loc[df.treatment == 1, "post_period_metric"].mean()
        - df.loc[df.treatment == 0, "post_period_metric"].mean()
    )
    did_est, did_lo, did_hi = estimate_did(df)
    psm_est, n_matched, n_treated = estimate_psm(df)

    results_df = pd.DataFrame({
        "Method": ["Naive diff-in-means", "Difference-in-Differences", "Propensity Score Matching"],
        "Estimate": [naive, did_est, psm_est],
        "Bias (estimate - true)": [naive - true_ate, did_est - true_ate, psm_est - true_ate],
    })

    st.subheader("Results")
    st.dataframe(results_df.style.format({"Estimate": "{:.3f}", "Bias (estimate - true)": "{:.3f}"}))
    st.bar_chart(results_df.set_index("Method")["Bias (estimate - true)"])

    st.caption(
        f"True ATE = {true_ate}. Bars closer to zero mean less bias from confounding. "
        f"Try dragging confounding to 0 -- all three should converge close to {true_ate}. "
        f"These are single-run estimates; the 30-seed averages reported in the README "
        f"are more stable."
    )

    st.subheader("CUPED variance reduction (clean experiment only)")
    df_clean = make_data(n, true_ate, 0.0)
    y = df_clean.post_period_metric.values
    x_pre = df_clean.pre_period_metric.values
    treatment = df_clean.treatment.values

    ate_before, se_before, lo_before, hi_before = estimate_ate_with_ci(y, treatment)
    y_adj, theta = cuped_adjust(y, x_pre)
    ate_after, se_after, lo_after, hi_after = estimate_ate_with_ci(y_adj, treatment)
    var_reduction = 1 - (y_adj.var() / y.var())

    width_before = hi_before - lo_before
    width_after = hi_after - lo_after

    c1, c2 = st.columns(2)
    c1.metric(
        "ATE before CUPED", f"{ate_before:.3f}",
        f"CI width {width_before:.3f}", delta_color="off",
    )
    c2.metric(
        "ATE after CUPED", f"{ate_after:.3f}",
        f"-{width_before - width_after:.3f} CI width", delta_color="inverse",
    )
    st.success(f"Variance reduction: {var_reduction*100:.1f}%")

    corr_before, corr_after, p_after = check_residual_correlation(y, y_adj, x_pre)
    st.caption(
        f"Residual correlation check: corr(Y, X_pre) = {corr_before:+.4f} before "
        f"adjustment, {corr_after:+.6f} after (p = {p_after:.3f}). The covariate's "
        f"linear signal is removed exactly -- direct proof CUPED worked, rather than "
        f"inferring it from the narrower interval. Full detail in the Assumption "
        f"Diagnostics tab."
    )

    st.subheader("Bayesian A/B test (clean experiment only)")
    diff_mean, diff_std, prob_pos, blo, bhi = bayesian_ab_test(y, treatment)
    st.write(f"Posterior mean effect: **{diff_mean:.3f}**, 95% credible interval ({blo:.3f}, {bhi:.3f})")
    if prob_pos > 0.9999:
        st.write("P(treatment > control) = **>0.9999**")
    else:
        st.write(f"P(treatment > control) = **{prob_pos:.4f}**")


# ---------------------------------------------------------------------------
# TAB 2 — Upload Your Own Data
# ---------------------------------------------------------------------------
with tab2:
    st.header("Run causal analysis on your own experiment data")
    st.write(
        "Upload a CSV with a treatment column (0/1) and an outcome column. "
        "A pre-experiment covariate column is optional but unlocks CUPED "
        "and Difference-in-Differences. Extra numeric columns are optional "
        "but unlock Propensity Score Matching and AIPW."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(user_df):,} rows, {len(user_df.columns)} columns.")
        st.dataframe(user_df.head())

        cols = user_df.columns.tolist()
        treatment_col = st.selectbox("Treatment column (0/1)", cols)
        outcome_col = st.selectbox("Outcome column", cols, index=min(1, len(cols) - 1))
        pre_period_col = st.selectbox(
            "Pre-experiment covariate column (optional, enables CUPED/DiD)",
            ["(none)"] + cols,
        )
        covariate_cols = st.multiselect(
            "Additional numeric covariate columns (optional, enables PSM/AIPW)",
            [c for c in cols if c not in (treatment_col, outcome_col, pre_period_col)],
        )

        if st.button("Run analysis"):
            try:
                check_cols = [treatment_col, outcome_col] + list(covariate_cols)
                if pre_period_col != "(none)":
                    check_cols.append(pre_period_col)

                na_counts = {c: int(user_df[c].isna().sum()) for c in check_cols}
                bad = {c: n for c, n in na_counts.items() if n > 0}
                if bad:
                    st.error(
                        "Missing values found in: "
                        + ", ".join(f"**{c}** ({n:,} rows)" for c, n in bad.items())
                    )
                    st.info(
                        "This tab does not impute missing data — dropping rows with "
                        "missing covariates would change which units are comparable, "
                        "which is a modelling decision that belongs to you, not the tool. "
                        "Either deselect these columns, or impute them upstream and "
                        "re-upload."
                    )
                    st.stop()

                t = user_df[treatment_col].values.astype(int)
                y = user_df[outcome_col].values.astype(float)

                naive = y[t == 1].mean() - y[t == 0].mean()
                rows = [{"Method": "Naive diff-in-means", "Estimate": f"{naive:.3f}", "Detail": "--"}]

                if pre_period_col != "(none)":
                    x_pre = user_df[pre_period_col].values.astype(float)

                    y_adj, theta = cuped_adjust(y, x_pre)
                    ate_after, se_after, lo_after, hi_after = estimate_ate_with_ci(y_adj, t)
                    cb, ca, pa = check_residual_correlation(y, y_adj, x_pre)
                    rows.append({
                        "Method": "CUPED-adjusted diff-in-means",
                        "Estimate": f"{ate_after:.3f}",
                        "Detail": f"95% CI ({lo_after:.3f}, {hi_after:.3f}); "
                                  f"residual corr {cb:+.3f} -> {ca:+.6f}",
                    })

                    did_df = pd.DataFrame({
                        "user_id": np.arange(len(user_df)),
                        "pre_period_metric": x_pre,
                        "treatment": t,
                        "post_period_metric": y,
                    })
                    did_est, did_lo, did_hi = estimate_did(did_df)
                    rows.append({
                        "Method": "Difference-in-Differences",
                        "Estimate": f"{did_est:.3f}",
                        "Detail": f"95% CI ({did_lo:.3f}, {did_hi:.3f})",
                    })

                r_up = None
                if covariate_cols:
                    psm_df = pd.DataFrame({c: user_df[c].values for c in covariate_cols})
                    psm_df["treatment"] = t
                    psm_df["post_period_metric"] = y
                    psm_est, n_matched, n_treated = estimate_psm(psm_df, covariates=covariate_cols)
                    rows.append({
                        "Method": "Propensity Score Matching",
                        "Estimate": f"{psm_est:.3f}",
                        "Detail": f"matched {n_matched}/{n_treated} treated units",
                    })

                    r_up = estimate_aipw(psm_df, covariates=covariate_cols)
                    rows.append({
                        "Method": "AIPW (doubly robust)",
                        "Estimate": f"{r_up['ate']:.3f}",
                        "Detail": f"95% CI ({r_up['ci_low']:.3f}, {r_up['ci_high']:.3f}); "
                                  f"effective n {r_up['effective_n']:,.0f}",
                    })

                diff_mean, diff_std, prob_pos, blo, bhi = bayesian_ab_test(y, t)
                prob_txt = ">0.9999" if prob_pos > 0.9999 else f"{prob_pos:.3f}"
                rows.append({
                    "Method": "Bayesian A/B",
                    "Estimate": f"{diff_mean:.3f}",
                    "Detail": f"P(treatment>control)={prob_txt}, CI ({blo:.3f}, {bhi:.3f})",
                })

                st.subheader("Results")
                st.dataframe(pd.DataFrame(rows))

                if pre_period_col == "(none)":
                    st.info(
                        "CUPED and Difference-in-Differences were skipped — both need a "
                        "pre-experiment covariate column, and none was selected."
                    )

                if covariate_cols:
                    st.subheader("PSM covariate balance")
                    st.write(
                        "A matched estimate is only trustworthy if the covariates it "
                        "was supposed to balance actually came out balanced."
                    )
                    user_balance = smd_before_after(psm_df, covariates=covariate_cols)
                    st.dataframe(user_balance.style.format({
                        "smd_before": "{:.4f}", "smd_after": "{:.4f}",
                        "abs_reduction": "{:.4f}",
                    }))
                    n_ok = int(user_balance.balanced_after.sum())
                    if n_ok == len(user_balance):
                        st.success(f"All {n_ok} covariates balanced after matching (|SMD| < 0.1).")
                    else:
                        st.warning(
                            f"Only {n_ok}/{len(user_balance)} covariates fall under "
                            f"|SMD| < 0.1 after matching. Treat the PSM estimate with "
                            f"caution -- residual imbalance means residual confounding."
                        )

                    if r_up is not None and r_up["effective_n"] < 0.7 * r_up["n"]:
                        st.warning(
                            f"AIPW's effective sample size is only "
                            f"{r_up['effective_n']:,.0f} of {r_up['n']:,} "
                            f"({r_up['effective_n']/r_up['n']*100:.0f}%), with a maximum "
                            f"weight of {r_up['max_weight']:.1f}. That indicates poor "
                            f"overlap between the arms — a few units are dominating the "
                            f"estimate, and the AIPW figure above should be treated with "
                            f"more suspicion than the matched one."
                        )

                    if st.checkbox("Run bootstrap confidence interval on this PSM estimate",
                                   help="Refits the propensity model in every replicate. "
                                        "Slow on large uploads."):
                        with st.spinner("Bootstrapping (100 replicates)..."):
                            try:
                                ate_u, lo_u, hi_u, boots_u = bootstrap_psm_ci(
                                    psm_df, covariates=covariate_cols, n_boot=100, seed=0
                                )
                                _, se_nu, lo_nu, hi_nu = naive_paired_ci(
                                    psm_df, covariates=covariate_cols
                                )
                                st.write(
                                    f"Bootstrap 95% CI: **({lo_u:.3f}, {hi_u:.3f})** "
                                    f"— width {hi_u-lo_u:.3f}, versus the naive paired "
                                    f"width of {hi_nu-lo_nu:.3f}."
                                )
                                st.pyplot(plot_bootstrap(boots_u, ate_u, lo_u, hi_u))
                            except Exception as be:
                                st.warning(f"Bootstrap did not complete: {be}")

            except ValueError as e:
                st.error(f"Could not run the analysis: {e}")
                st.info(
                    "Most common causes: the treatment column is not coded 0/1, or a "
                    "selected column is categorical text rather than numeric. Categorical "
                    "covariates need one-hot encoding before upload."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info(
                    "Check that the treatment column is 0/1 and that outcome and "
                    "covariate columns are numeric with no missing values."
                )


# ---------------------------------------------------------------------------
# TAB 3 — Real-world benchmarks
# ---------------------------------------------------------------------------
with tab3:
    st.header("Two external benchmarks, neither of them synthetic")
    st.write(
        "Everything on the other tabs is validated against a generator written "
        "for this project. That is a real check, but it has a circularity "
        "problem worth naming. These two datasets are the answer to it."
    )

    st.divider()

    st.subheader("Criteo Uplift — uplift model")
    st.write(
        "The same T-Learner uplift model was separately validated against a "
        "real advertising incrementality-test dataset (Criteo AI Lab) — no "
        "hidden ground truth this time, just real treated/control outcomes."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows sampled", "223,673")
    c2.metric("AUUC, model", "27.48")
    c3.metric("AUUC, random targeting", "12.03")

    st.success("Model beats random targeting by 128.4%")

    st.caption(
        "Full methodology and code: see reports/criteo_validation.md and "
        "src/causallens/criteo_validate.py."
    )

    st.divider()

    st.subheader("LaLonde / NSW — confounding correction")
    st.write(
        "The canonical benchmark in this field. The National Supported Work "
        "Demonstration was a genuine randomized trial, so the experimental "
        "comparison gives a credible causal effect on 1978 earnings. LaLonde "
        "then discarded the experimental controls and substituted survey "
        "respondents from PSID and CPS — people who never applied to the "
        "programme and differ on almost every dimension."
    )

    l1, l2, l3 = st.columns(3)
    l1.metric("Experimental ATE", "$1,794")
    l2.metric("Naive, PSID controls", "-$15,205")
    l3.metric("PSM, CPS controls", "$1,637")

    st.success(
        "Starting from -\\$8,498 with CPS controls, PSM recovers \\$1,637 "
        "against an experimental truth of \\$1,794 — an error of \\$157 on the "
        "benchmark the field has argued about since 1986."
    )

    st.error(
        "**AIPW fails badly here**, landing at -\\$3,638 with a linear outcome "
        "model and -\\$9,468 with gradient boosting. This is the overlap "
        "problem: several covariates start with |SMD| above 1.5, so the two "
        "groups barely share support. Matching survives by discarding the "
        "non-comparable controls; weighting cannot, because it must keep them "
        "and hand them enormous weights. On the synthetic data, where overlap "
        "is good, the ranking is reversed and AIPW wins."
    )

    st.caption(
        "PSM balanced only 4 of 8 covariates on PSID, so the close hit there is "
        "not a clean one. That is the Smith and Todd (2005) critique in "
        "miniature: the result is real but fragile to specification, and a "
        "single run is one data point in a forty-year argument rather than a "
        "resolution of it. Full output in reports/lalonde_psid.csv and "
        "reports/lalonde_cps.csv; reproduce with "
        "python src/causallens/lalonde_validate.py."
    )


# ---------------------------------------------------------------------------
# TAB 4 — Assumption Diagnostics
# ---------------------------------------------------------------------------
with tab4:
    st.header("Every estimator ships its own assumption diagnostic")
    st.write(
        "A correct-looking point estimate is not evidence a method worked -- it "
        "can be right by luck, or wrong in a way the final number hides. Each "
        "method below is checked against the specific assumption it depends on, "
        "independently of whether its headline estimate looks reasonable."
    )

    st.divider()

    # --- CUPED --------------------------------------------------------------
    st.subheader("1. CUPED — residual correlation")
    st.write(
        "CUPED subtracts the component of the outcome that is linearly "
        "predictable from a pre-experiment covariate. If it worked, the adjusted "
        "outcome should retain no linear association with that covariate. This "
        "is an algebraic guarantee, so the check is exact rather than approximate."
    )

    df_diag_clean = make_data(10000, 5.0, 0.0)
    yd = df_diag_clean.post_period_metric.values
    xd = df_diag_clean.pre_period_metric.values
    yd_adj, theta_d = cuped_adjust(yd, xd)
    cb, ca, pa = check_residual_correlation(yd, yd_adj, xd)

    d1, d2, d3 = st.columns(3)
    d1.metric("theta", f"{theta_d:.4f}")
    d2.metric("corr(Y, X_pre)", f"{cb:+.4f}")
    d3.metric("corr(Y_cuped, X_pre)", f"{ca:+.6f}")
    st.caption(f"p = {pa:.3f} on the post-adjustment correlation.")

    st.success(
        "Residual correlation is zero to machine precision — the covariate's "
        "linear signal is fully removed."
    )

    st.divider()

    # --- PSM balance --------------------------------------------------------
    st.subheader("2. PSM — covariate balance (SMD)")
    st.write(
        "Matching is only credible if the covariates it was supposed to balance "
        "actually came out balanced. Standardized Mean Difference makes imbalance "
        "comparable across covariates on different scales; |SMD| < 0.1 is the "
        "conventional threshold."
    )

    bal_conf = st.slider(
        "Confounding strength for the balance check", 0.0, 1.0, 0.5, step=0.1,
        key="bal_conf",
    )
    balance = cached_balance(10000, 5.0, bal_conf)

    st.dataframe(balance.style.format({
        "smd_before": "{:.4f}", "smd_after": "{:.4f}", "abs_reduction": "{:.4f}",
    }))
    n_ok = int(balance.balanced_after.sum())
    st.success(f"{n_ok}/{len(balance)} covariates balanced after matching.")
    st.pyplot(plot_smd(balance))

    st.caption(
        "X1 drives treatment assignment, so it starts imbalanced and is corrected "
        "by matching. X2 (pure noise) and X3 (effect modifier) do not drive "
        "assignment and stay balanced throughout — the correction lands on the "
        "variable that actually mattered, rather than reshuffling indiscriminately. "
        "Drag confounding to 0 and X1's initial imbalance disappears."
    )

    st.divider()

    # --- Parallel trends ----------------------------------------------------
    st.subheader("3. DiD — parallel-trends placebo test")
    st.write(
        "Difference-in-Differences assumes that, absent treatment, both groups "
        "would have trended in parallel. That is an assumption, not a guarantee. "
        "This test regresses outcome on treatment x period using **pre-treatment "
        "data only**, where no treatment effect can exist by construction. Any "
        "non-zero interaction is a pre-existing divergence — exactly what DiD "
        "assumes away."
    )

    p1, p2 = st.columns(2)
    with p1:
        trend = st.slider(
            "X1 effect growth per period", 0.0, 2.0, 1.0, step=0.1,
            help="How much the confounder's influence grows each period. "
                 "0.0 gives a world where parallel trends holds.",
        )
    with p2:
        n_pre = st.slider("Number of pre-treatment periods", 2, 6, 3)

    df_pt = make_data(10000, 5.0, 0.5, seed=0, n_pre_periods=n_pre, x1_trend=trend)
    res = pre_trends_test(df_pt)
    gaps = pre_period_gaps(df_pt)
    did_pt, _, _ = estimate_did(df_pt)

    g1, g2 = st.columns([1, 1])
    with g1:
        st.write("**Treated − control gap, by pre-period**")
        st.dataframe(gaps.style.format({"treated_minus_control": "{:.4f}"}))
    with g2:
        st.line_chart(gaps.set_index("period")["treated_minus_control"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Pre-trend coefficient", f"{res['pre_trend_coef']:+.4f}")
    m2.metric("p-value", f"{res['p_value']:.2e}")
    m3.metric("Realised DiD bias", f"{did_pt - 5.0:+.4f}")

    if res["parallel_trends_holds"]:
        st.success(
            f"**Parallel trends HOLDS** (p = {res['p_value']:.3f}). The gap is flat "
            f"across pre-periods, so DiD's key assumption is satisfied and its "
            f"estimate is roughly unbiased ({did_pt - 5.0:+.3f})."
        )
        st.caption(
            "Note the gap may still be large in *level* — DiD differences level "
            "differences away by design. Only a difference in slope breaks it, "
            "which is why this test looks at the trend and not the gap size."
        )
    else:
        st.error(
            f"**Parallel trends VIOLATED** (p = {res['p_value']:.2e}). The groups "
            f"diverge by {res['pre_trend_coef']:+.3f} per period before treatment "
            f"even exists — and that pre-trend predicts the realised DiD bias of "
            f"{did_pt - 5.0:+.3f}."
        )
        st.caption(
            "This is what drives the ~83% CI coverage reported in the README: the "
            "bias is systematic, not random, so the intervals undercover. Drag the "
            "growth slider to 0.0 to see the test correctly return a null result — "
            "a diagnostic that always fails would prove nothing."
        )

    st.divider()

    # --- PSM uncertainty ----------------------------------------------------
    st.subheader("4. PSM — uncertainty and hidden bias")
    st.write(
        "A point estimate without an interval is not a result, and an interval "
        "that ignores how it was produced is worse than none. These two checks "
        "cover the uncertainty PSM actually carries."
    )

    st.markdown("**Bootstrap confidence interval**")
    st.write(
        "The whole pipeline is re-run inside each replicate — resample units, "
        "refit the propensity model, rematch, recompute. Resampling only the "
        "matched pairs would treat the matching as fixed and understate "
        "uncertainty, since the propensity scores were estimated from the same "
        "data."
    )

    n_boot = st.select_slider(
        "Bootstrap replicates", options=[50, 100, 200, 300], value=100,
        help="More replicates give a smoother interval but take longer.",
    )

    with st.spinner(f"Running {n_boot} bootstrap replicates..."):
        df_boot = make_data(5000, 5.0, 0.5, seed=0)
        psm_pt, n_m, n_t = estimate_psm(df_boot)
        ate_n, se_n, lo_n, hi_n = naive_paired_ci(df_boot)
        ate_b, lo_b, hi_b, boots = cached_bootstrap(5000, 5.0, 0.5, n_boot)
        se_b = boots.std(ddof=1)

    st.caption(
        f"Run on n=5,000 rather than the 10,000 used elsewhere, since {n_boot} full "
        f"propensity refits would be too slow interactively. This is seed 0, and it is "
        f"deliberately an unlucky draw: every treated unit matched ({n_m:,}/{n_t:,}, no "
        f"caliper drops) and all three covariates balanced to |SMD| < 0.01, yet the "
        f"estimate ({psm_pt:.3f}) sits well below the true ATE of 5.0. Across seeds 1-9 "
        f"the identical setup averages 4.975, so nothing is broken — this is one bad "
        f"draw. That is the whole argument for reporting an interval: balance "
        f"diagnostics confirm the covariates you matched on, not the estimate, and a "
        f"single run can land far from the truth with every diagnostic looking clean."
    )

    u1, u2, u3 = st.columns(3)
    u1.metric("Naive paired CI width", f"{hi_n - lo_n:.3f}", f"SE {se_n:.4f}", delta_color="off")
    u2.metric("Bootstrap CI width", f"{hi_b - lo_b:.3f}", f"SE {se_b:.4f}", delta_color="off")
    u3.metric("SE inflation", f"{(se_b/se_n - 1)*100:.0f}%")

    st.pyplot(plot_bootstrap(boots, ate_b, lo_b, hi_b, true_ate=5.0))

    st.info(
        f"The naive paired interval is **{(1 - (hi_n-lo_n)/(hi_b-lo_b))*100:.0f}% "
        f"narrower** than the bootstrap. That gap is the uncertainty most PSM "
        f"implementations silently drop by conditioning on a propensity model "
        f"they estimated from the data — and on this draw it is the difference "
        f"between an interval that covers the true effect and one that barely does."
    )

    if not (lo_b <= 5.0 <= hi_b):
        st.warning(
            f"The bootstrap interval ({lo_b:.3f}, {hi_b:.3f}) does not cover the true "
            f"ATE of 5.0 on this draw. The naive interval ({lo_n:.3f}, {hi_n:.3f}) "
            f"misses it too, but reports the same wrong answer with more confidence."
        )

    st.caption(
        "Caveat, stated deliberately: Abadie & Imbens (2008) showed the standard "
        "bootstrap is not formally valid for nearest-neighbour matching with a "
        "fixed number of matches — the matching function is not smooth enough for "
        "its asymptotic guarantees. This is a clear improvement on the naive "
        "interval, not a formally correct one. The Abadie-Imbens analytic variance "
        "estimator would be the rigorous alternative."
    )

    st.markdown("**Rosenbaum sensitivity bounds**")
    st.write(
        "Matching only balances covariates you measured. The standing objection "
        "to any PSM result is \"what about a confounder you didn't observe?\" — "
        "which cannot be answered from data, since the confounder is by definition "
        "unobserved. Rosenbaum bounds instead quantify how strong such a confounder "
        "would have to be before the conclusion breaks."
    )

    bounds, gamma_star = cached_rosenbaum(5000, 5.0, 0.5)

    r1, r2 = st.columns([1, 1])
    with r1:
        st.dataframe(bounds.style.format({
            "gamma": "{:.2f}", "p_lower_bound": "{:.2e}", "p_upper_bound": "{:.2e}",
        }))
    with r2:
        st.metric("Gamma*", f"{gamma_star:.2f}")
        st.success(
            f"An unmeasured confounder would need to shift the odds of treatment "
            f"by more than **{gamma_star:.2f}x**, between units identical on every "
            f"measured covariate, before this result stops being significant."
        )
        st.caption(
            "Gamma = 1 means no hidden bias: within a matched pair, either unit was "
            "equally likely to be treated. Gamma = 2 means one could have been twice "
            "as likely despite being identical on X1, X2 and X3. Higher Gamma* means "
            "a more robust finding. Gamma* is lower here than the 2.20 reported in "
            "the README because this runs on n=5,000 rather than 10,000, and on an "
            "unlucky draw — less data and a weaker measured effect both mean less "
            "robustness to hidden bias, as they should. The lower bound underflows "
            "to zero because the effect is still large relative to the noise."
        )


# ---------------------------------------------------------------------------
# TAB 5 — Doubly Robust estimation
# ---------------------------------------------------------------------------
with tab5:
    st.header("Two chances to be right instead of one")
    st.write(
        "Propensity score matching depends entirely on the propensity model "
        "being correct, and throws away every control it cannot match. AIPW "
        "keeps all the data and combines two models — one for treatment "
        "assignment, one for the response surface. It is consistent if "
        "**either** is correctly specified, even when the other is wrong."
    )

    st.latex(r"""
    \text{ATE} = \mathbb{E}\left[\mu_1(X) - \mu_0(X)
    + \frac{T(Y - \mu_1(X))}{e(X)}
    - \frac{(1-T)(Y - \mu_0(X))}{1 - e(X)}\right]
    """)

    st.caption(
        "The first term is plain outcome regression. The rest is the "
        "augmentation: an IPW-weighted correction built from the outcome "
        "model's residuals. If the outcome model is perfect those residuals "
        "vanish and the propensity model stops mattering; if the propensity "
        "model is perfect the correction is unbiased however bad the outcome "
        "model is."
    )

    st.divider()

    st.subheader("Head to head with PSM")

    a1, a2 = st.columns(2)
    with a1:
        aipw_conf = st.slider("Confounding strength", 0.0, 1.0, 0.5, step=0.1,
                              key="aipw_conf")
    with a2:
        aipw_learner = st.radio("Outcome model", ["linear", "gbm"],
                                horizontal=True, key="aipw_learner")

    df_a = make_data(10000, 5.0, aipw_conf, seed=0)
    naive_a = (df_a.loc[df_a.treatment == 1, "post_period_metric"].mean()
               - df_a.loc[df_a.treatment == 0, "post_period_metric"].mean())
    psm_a, n_m_a, n_t_a = estimate_psm(df_a)
    r_a = estimate_aipw(df_a, learner=aipw_learner)

    comp = pd.DataFrame({
        "Method": ["Naive diff-in-means", "PSM", "AIPW"],
        "Estimate": [naive_a, psm_a, r_a["ate"]],
        "Bias": [naive_a - 5.0, psm_a - 5.0, r_a["ate"] - 5.0],
        "Analytic CI": [
            "--", "-- (bootstrap only)",
            f"({r_a['ci_low']:.3f}, {r_a['ci_high']:.3f})",
        ],
        "Data used": [
            f"{len(df_a):,}", f"{n_m_a:,} matched pairs", f"{len(df_a):,}",
        ],
    })
    st.dataframe(comp.style.format({"Estimate": "{:.4f}", "Bias": "{:+.4f}"}))

    st.caption(
        f"AIPW's influence function is known in closed form, so its standard "
        f"error comes straight from the per-unit scores — no bootstrap, and "
        f"unlike the matching bootstrap it is formally valid rather than an "
        f"approximation. Effective sample size after weighting: "
        f"{r_a['effective_n']:,.0f} of {r_a['n']:,} "
        f"({r_a['effective_n']/r_a['n']*100:.1f}%), max weight "
        f"{r_a['max_weight']:.2f}."
    )

    st.divider()

    st.subheader("Testing the double robustness claim")
    st.write(
        "The claim is checkable: break one model, then the other, then both. "
        "The fourth row is the control — without it the first three could "
        "just mean the problem is easy."
    )

    dr = cached_dr_check(10000, 5.0, aipw_conf, aipw_learner)
    st.dataframe(dr.style.format({
        "ate": "{:.4f}", "bias": "{:+.4f}",
        "ci_low": "{:.3f}", "ci_high": "{:.3f}",
    }))

    both_broken_bias = float(dr.loc[dr.scenario == "Both broken", "bias"].iloc[0])
    if abs(both_broken_bias) > 1.0:
        st.success(
            f"One correct model suffices. With both broken the estimate "
            f"collapses to a bias of {both_broken_bias:+.3f} — essentially the "
            f"naive difference of {naive_a - 5.0:+.3f}, since with the "
            f"confounder absent from both models there is nothing left to "
            f"adjust with."
        )
    else:
        st.warning(
            "The 'both broken' case is not showing bias, which means the "
            "misspecification is not severe enough for this test to "
            "demonstrate anything. Increase the confounding strength."
        )

    st.caption(
        "Both models are broken the same way: by omitting X1, the only "
        "confounder in the generator. An earlier version of this module broke "
        "the propensity model by fitting X1 *alone* — which is in fact the "
        "exactly-correct specification, and made the whole check vacuous. "
        "Worth noting, because the failure looked like a success."
    )

    st.divider()

    st.subheader("Overlap — the assumption AIPW actually needs")
    st.write(
        "Weighting by 1/e(X) blows up when propensity scores approach zero. "
        "Where the two arms do not share support there is no comparable unit "
        "to weight toward, and no estimator can recover an effect there "
        "without extrapolating past the data."
    )
    st.pyplot(plot_propensity_overlap(df_a))

    st.info(
        "**This is where AIPW loses to PSM on real data.** On the LaLonde "
        "benchmark, where NSW participants and PSID survey controls barely "
        "overlap (several covariates start with |SMD| above 1.5), PSM lands "
        "within \\$157 of the experimental answer while AIPW misses by "
        "thousands in the wrong direction. Matching survives by discarding "
        "the non-comparable units; weighting cannot, because it has to keep "
        "them and give them enormous weights. Same two methods, opposite "
        "ranking, and the balance table tells you which situation you are in."
    )


# ---------------------------------------------------------------------------
# TAB 6 — Experiment design
# ---------------------------------------------------------------------------
with tab6:
    st.header("Planning the experiment before it runs")
    st.write(
        "Every other tab analyses an experiment that already happened. This "
        "one answers what comes first: how long should it run, and what is "
        "the smallest effect it could possibly detect. An underpowered null "
        "does not mean there is no effect — it means the experiment could "
        "never have found one."
    )

    st.divider()

    st.subheader("Sample size and duration")

    d1, d2, d3 = st.columns(3)
    with d1:
        target_mde = st.number_input(
            "Target MDE (in SD)", min_value=0.001, max_value=1.0,
            value=0.05, step=0.005, format="%.3f",
        )
    with d2:
        daily = st.number_input("Daily traffic", min_value=100,
                                value=50000, step=1000)
    with d3:
        exposure = st.slider("Fraction eligible", 0.05, 1.0, 0.4, step=0.05)

    use_cuped = st.checkbox(
        "Apply CUPED (47.2% variance reduction, measured on this data)",
        value=False,
    )
    sigma_eff = np.sqrt(1 - 0.472) if use_cuped else 1.0

    plan = experiment_duration(target_mde, daily_traffic=daily,
                               sigma=sigma_eff, exposure_rate=exposure)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Per arm", f"{plan['n_per_arm']:,}")
    p2.metric("Total", f"{plan['n_total']:,}")
    p3.metric("Duration", f"{plan['weeks']} wk", f"{plan['days']:.0f} days",
              delta_color="off")
    p4.metric("Bound by",
              "sample size" if plan["sample_bound"] else "1-week floor")

    if plan["too_long"]:
        st.warning(
            f"At {plan['weeks']} weeks this is past the point where novelty "
            f"effects and seasonality start contaminating the comparison. A "
            f"plan this long is usually a signal to pick a more sensitive "
            f"metric rather than wait it out."
        )
    else:
        st.success(f"Feasible: {plan['weeks']} week(s).")

    st.caption(
        "Durations are rounded up to whole weeks with a one-week floor. Even "
        "when the raw sample arrives sooner, a shorter run confounds the "
        "treatment effect with day-of-week seasonality — an experiment that "
        "only sees Tuesdays is measuring Tuesday."
    )

    if use_cuped:
        sav = cuped_sample_savings(target_mde)
        st.info(
            f"CUPED cuts the requirement from {sav['n_per_arm_plain']:,} to "
            f"{sav['n_per_arm_cuped']:,} per arm — {sav['samples_saved']:,} "
            f"fewer users ({sav['pct_saved']:.1f}%). Required sample size is "
            f"linear in variance, so the reduction passes through one-for-one "
            f"rather than being square-rooted away. No extra traffic and no "
            f"design change; it only needs a pre-period covariate you were "
            f"already logging."
        )

    st.divider()

    st.subheader("What could this experiment detect?")

    n_check = st.number_input("Sample size per arm", min_value=10,
                              value=5000, step=500)
    eff_check = st.number_input("Effect size you care about (SD)",
                                min_value=0.001, max_value=2.0,
                                value=0.05, step=0.005, format="%.3f")

    mde_at_n = minimum_detectable_effect(n_check)
    pwr = achieved_power(n_check, eff_check)

    q1, q2 = st.columns(2)
    q1.metric("Smallest detectable effect", f"{mde_at_n:.4f} SD")
    q2.metric(f"Power for a {eff_check} SD effect", f"{pwr*100:.1f}%")

    if pwr < 0.5:
        st.error(
            f"At {pwr*100:.0f}% power this experiment is more likely to miss "
            f"a real {eff_check} SD effect than to find it. A null result "
            f"here would tell you almost nothing."
        )
    elif pwr < 0.8:
        st.warning(
            f"{pwr*100:.0f}% power is below the conventional 80% target. A "
            f"null result would be weak evidence of no effect."
        )
    else:
        st.success(f"{pwr*100:.0f}% power — adequately powered.")

    n_grid = np.logspace(2, 5.5, 60).astype(int)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_power_curves([0.05, 0.10, 0.20, 0.50], n_grid))
    with c2:
        st.pyplot(plot_mde_curve(n_grid))

    st.divider()

    st.subheader("The cost of peeking")
    st.write(
        "A fixed-horizon t-test is valid exactly once, at a sample size fixed "
        "in advance. Every real experiment gets watched — someone opens the "
        "dashboard on day three, sees p = 0.04, and ships. Each look is "
        "another chance for noise to cross the threshold."
    )

    n_exp = st.select_slider(
        "Simulated experiments per point", options=[100, 200, 400],
        value=200, help="More gives a smoother curve but takes longer.",
    )

    with st.spinner("Simulating A/A tests at each peeking frequency..."):
        curve = cached_peeking_curve(n_exp)

    st.dataframe(curve.style.format({
        "fixed_horizon_fpr": "{:.3f}", "always_valid_fpr": "{:.3f}",
        "nominal_alpha": "{:.2f}",
    }))
    st.pyplot(plot_peeking_curve(curve))

    worst = curve.fixed_horizon_fpr.max()
    worst_k = int(curve.loc[curve.fixed_horizon_fpr.idxmax(), "n_peeks"])
    st.error(
        f"After {worst_k} looks, a fixed-horizon test declares a false "
        f"positive **{worst*100:.0f}% of the time** on data with no effect at "
        f"all — against a nominal 5%."
    )
    st.success(
        "The always-valid mSPRT line stays flat. It is valid at every sample "
        "size simultaneously, so the number of looks stops being a decision "
        "you have to get right in advance."
    )

    st.caption(
        "The trade is real and paid in power: at a 0.15 SD effect the "
        "fixed-horizon test detected 99.8% of the time and mSPRT 90.5%, which "
        "translates to roughly 25% more traffic for the same sensitivity. "
        "Note also that always-valid sits well *below* 5% here rather than at "
        "it — the mixture prior tau is set for effects around 1 SD while the "
        "simulation uses 0 to 0.15 SD, so the test is more conservative than "
        "it needs to be. Tuning tau to the effect size you actually care "
        "about would recover some of that lost power."
    )

    st.divider()

    st.subheader("Anytime-valid intervals")
    st.write(
        "The interval form of the same idea: a sequence where **all** the "
        "intervals contain the true effect with probability 1 − α, rather "
        "than each one separately."
    )
    st.pyplot(plot_confidence_sequence(true_effect=0.5))
    st.caption(
        "The fixed-horizon band is narrower everywhere, which looks better "
        "and is not — it is only valid at one pre-chosen sample size, and "
        "reading it at any other point is the peeking problem in visual form."
    )