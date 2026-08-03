"""
criteo_validate.py — Validates the uplift model on real data (Criteo).

Unlike Phase 6's synthetic check, there's no hidden ground truth here --
this uses the real Qini/AUUC methodology: compare treated vs. control
OUTCOMES directly within each top-k% slice (rescaled for group-size
imbalance), instead of peeking at a known true effect.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def qini_curve(uplift_scores, outcome, treatment, n_bins=20):
    order = np.argsort(-uplift_scores)
    outcome_sorted = outcome[order]
    treatment_sorted = treatment[order]
    n = len(outcome_sorted)
    fractions = np.linspace(1 / n_bins, 1.0, n_bins)
    qini_values = []
    for frac in fractions:
        k = int(n * frac)
        treat_mask = treatment_sorted[:k] == 1
        ctrl_mask = treatment_sorted[:k] == 0
        n_t, n_c = treat_mask.sum(), ctrl_mask.sum()
        y_t, y_c = outcome_sorted[:k][treat_mask].sum(), outcome_sorted[:k][ctrl_mask].sum()
        qini = y_t - y_c * (n_t / n_c) if n_c > 0 else 0
        qini_values.append(qini)
    return fractions, np.array(qini_values)


def compute_auuc(fractions, qini_values):
    fracs = np.concatenate([[0], fractions])
    vals = np.concatenate([[0], qini_values])
    return np.sum(np.diff(fracs) * (vals[:-1] + vals[1:]) / 2)


if __name__ == "__main__":
    print("Scanning the full file in chunks to build a representative sample...", flush=True)
    chunk_size = 500_000
    frac = 0.016  # targets ~400k rows sampled across the whole ~25M-row file
    sampled_chunks = []
    for i, chunk in enumerate(pd.read_csv("data/raw/criteo-uplift.csv", chunksize=chunk_size)):
        sampled_chunks.append(chunk.sample(frac=frac, random_state=0))
        print(f"  chunk {i+1} done ({(i+1)*chunk_size:,} rows scanned so far)", flush=True)
    df = pd.concat(sampled_chunks, ignore_index=True)

    print(f"\nFinal sample: {len(df):,} rows.")
    print(f"Treatment distribution:\n{df.treatment.value_counts()}\n", flush=True)

    covariates = [c for c in df.columns if c.startswith("f")]
    outcome_col = "conversion"

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
    X_train, t_train, y_train = (
        train_df[covariates].values, train_df.treatment.values, train_df[outcome_col].values,
    )

    print("Training treated-group model...", flush=True)
    model_treated = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
    model_treated.fit(X_train[t_train == 1], y_train[t_train == 1])

    print("Training control-group model...", flush=True)
    model_control = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=0)
    model_control.fit(X_train[t_train == 0], y_train[t_train == 0])

    print("Scoring test set and computing Qini curves...", flush=True)
    X_test = test_df[covariates].values
    uplift_pred = model_treated.predict(X_test) - model_control.predict(X_test)
    outcome_test, treatment_test = test_df[outcome_col].values, test_df.treatment.values

    fractions, qini_vals = qini_curve(uplift_pred, outcome_test, treatment_test)
    auuc_model = compute_auuc(fractions, qini_vals)

    rng = np.random.default_rng(0)
    _, qini_random = qini_curve(rng.normal(0, 1, len(uplift_pred)), outcome_test, treatment_test)
    auuc_random = compute_auuc(fractions, qini_random)

    print(f"AUUC model:  {auuc_model:.2f}")
    print(f"AUUC random: {auuc_random:.2f}")
    print(f"Model beats random targeting by: {(auuc_model/auuc_random - 1)*100:.1f}%")