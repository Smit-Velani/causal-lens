"""
streamlit_app.py — CausalLens: interactive causal inference & experimentation platform.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "causallens"))

import numpy as np
import pandas as pd
import streamlit as st

from data_gen import generate_synthetic_experiment
from cuped import cuped_adjust, estimate_ate_with_ci
from did import estimate_did
from matching import estimate_psm
from bayesian_ab import bayesian_ab_test

st.set_page_config(page_title="CausalLens", layout="wide")
st.title("CausalLens — Causal Inference & Experimentation Platform")

tab1, tab2, tab3 = st.tabs(["Ground Truth Validation", "Upload Your Data", "Criteo Uplift (Real Data)"])

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

    df = generate_synthetic_experiment(
        n=n, true_ate=true_ate, confounding_strength=confounding_strength, seed=42
    )

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
        f"Try dragging confounding to 0 -- all three should converge close to {true_ate}."
    )

    st.subheader("CUPED variance reduction (clean experiment only)")
    df_clean = generate_synthetic_experiment(n=n, true_ate=true_ate, confounding_strength=0.0, seed=42)
    y = df_clean.post_period_metric.values
    x_pre = df_clean.pre_period_metric.values
    treatment = df_clean.treatment.values

    ate_before, se_before, lo_before, hi_before = estimate_ate_with_ci(y, treatment)
    y_adj, theta = cuped_adjust(y, x_pre)
    ate_after, se_after, lo_after, hi_after = estimate_ate_with_ci(y_adj, treatment)
    var_reduction = 1 - (y_adj.var() / y.var())

    c1, c2 = st.columns(2)
    c1.metric("ATE before CUPED", f"{ate_before:.3f}", f"CI width {hi_before-lo_before:.3f}")
    c2.metric("ATE after CUPED", f"{ate_after:.3f}", f"CI width {hi_after-lo_after:.3f}")
    st.success(f"Variance reduction: {var_reduction*100:.1f}%")

    st.subheader("Bayesian A/B test (clean experiment only)")
    diff_mean, diff_std, prob_pos, blo, bhi = bayesian_ab_test(y, treatment)
    st.write(f"Posterior mean effect: **{diff_mean:.3f}**, 95% credible interval ({blo:.3f}, {bhi:.3f})")
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
        "but unlock Propensity Score Matching."
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
            "Additional numeric covariate columns (optional, enables PSM)",
            [c for c in cols if c not in (treatment_col, outcome_col, pre_period_col)],
        )

        if st.button("Run analysis"):
            try:
                t = user_df[treatment_col].values.astype(int)
                y = user_df[outcome_col].values.astype(float)

                naive = y[t == 1].mean() - y[t == 0].mean()
                rows = [{"Method": "Naive diff-in-means", "Estimate": f"{naive:.3f}", "Detail": "--"}]

                if pre_period_col != "(none)":
                    x_pre = user_df[pre_period_col].values.astype(float)

                    y_adj, theta = cuped_adjust(y, x_pre)
                    ate_after, se_after, lo_after, hi_after = estimate_ate_with_ci(y_adj, t)
                    rows.append({
                        "Method": "CUPED-adjusted diff-in-means",
                        "Estimate": f"{ate_after:.3f}",
                        "Detail": f"95% CI ({lo_after:.3f}, {hi_after:.3f})",
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

                diff_mean, diff_std, prob_pos, blo, bhi = bayesian_ab_test(y, t)
                rows.append({
                    "Method": "Bayesian A/B",
                    "Estimate": f"{diff_mean:.3f}",
                    "Detail": f"P(treatment>control)={prob_pos:.3f}, CI ({blo:.3f}, {bhi:.3f})",
                })

                st.subheader("Results")
                st.dataframe(pd.DataFrame(rows))

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("Check that the treatment column is 0/1 and outcome/covariate columns are numeric.")


# ---------------------------------------------------------------------------
# TAB 3 — Criteo Uplift (Real Data) — static report from Phase 8
# ---------------------------------------------------------------------------
with tab3:
    st.header("Validated on real-world data: Criteo Uplift Dataset")
    st.write(
        "The same T-Learner uplift model (Phase 6) was separately validated "
        "against a real advertising incrementality-test dataset (Criteo AI "
        "Lab) -- no hidden ground truth this time, just real treated/control "
        "outcomes."
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