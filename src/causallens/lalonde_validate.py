"""
lalonde_validate.py — External validation on the LaLonde / NSW benchmark.

Every other validation in this project uses the synthetic generator in
data_gen.py. That is a real check -- the estimators recover an effect they
could not have seen -- but it has a circularity problem worth naming: the
same code that creates the confounding is the code the diagnostics are
tuned against. A method that works there has been shown to work on data
built by the person who wrote the method.

The LaLonde data is the standard answer to that objection, and has been
since 1986. The National Supported Work Demonstration was a genuine
randomized trial of a job-training programme, so the experimental
treated-vs-control comparison gives a credible causal effect on 1978
earnings. LaLonde then threw the experimental controls away and replaced
them with survey respondents from PSID and CPS -- people who never
applied to the programme and differ from participants on almost every
dimension.

That yields the benchmark: run an observational method on the
treated-plus-survey-controls data, and compare its answer to the
experimental one. The naive comparison is catastrophically wrong, usually
large and negative, because the survey controls earn far more to begin
with. Whether propensity methods can close that gap is the question the
literature has argued about for forty years.

Benchmarks (Dehejia-Wahba subsample, n=185 treated):
  Experimental ATE   ~ $1,794
  Naive vs PSID      ~ -$15,000
"""

import os

import numpy as np
import pandas as pd

COLUMNS = ["treat", "age", "education", "black", "hispanic",
           "married", "nodegree", "re74", "re75", "re78"]

COVARIATES = ["age", "education", "black", "hispanic",
              "married", "nodegree", "re74", "re75"]

BASE_URL = "http://users.nber.org/~rdehejia/data"
DATA_DIR = os.path.join("data", "raw", "lalonde")

FILES = {
    "treated": "nswre74_treated.txt",
    "control": "nswre74_control.txt",
    "psid": "psid_controls.txt",
    "cps": "cps_controls.txt",
}

# Published reference values, for orientation rather than exact replication.
EXPERIMENTAL_BENCHMARK = 1794.0


def _load_one(key):
    """
    Load one LaLonde file, from local cache if present, otherwise download.

    Files are whitespace-delimited with no header; column order is fixed and
    documented on Dehejia's page. Cached under data/raw/lalonde/ so the
    script runs offline after the first success.
    """
    fname = FILES[key]
    local = os.path.join(DATA_DIR, fname)

    if os.path.exists(local):
        return pd.read_csv(local, sep=r"\s+", header=None, names=COLUMNS)

    url = f"{BASE_URL}/{fname}"
    try:
        df = pd.read_csv(url, sep=r"\s+", header=None, names=COLUMNS)
    except Exception as e:
        raise RuntimeError(
            f"Could not read {url} ({e}).\n"
            f"Download the four files listed at "
            f"https://users.nber.org/~rdehejia/nswdata2.html manually and "
            f"place them in {DATA_DIR}/ , then re-run."
        ) from e

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(local, sep=" ", header=False, index=False)
    return df


def load_lalonde(control_source="psid"):
    """
    Build an analysis frame.

    control_source='experimental' returns the randomized comparison -- the
    ground truth. 'psid' or 'cps' returns the observational version: the same
    treated units, but with survey respondents standing in for the controls.

    Columns are renamed to `treatment` and `post_period_metric` so the
    existing estimators in matching.py and aipw.py work unmodified.
    """
    treated = _load_one("treated")

    if control_source == "experimental":
        controls = _load_one("control")
    elif control_source in ("psid", "cps"):
        controls = _load_one(control_source)
    else:
        raise ValueError("control_source must be 'experimental', 'psid', or 'cps'")

    df = pd.concat([treated, controls], ignore_index=True)
    df["treatment"] = df["treat"].astype(int)
    df["post_period_metric"] = df["re78"].astype(float)
    df["user_id"] = np.arange(len(df))
    df["pre_period_metric"] = df["re75"].astype(float)
    return df


def naive_difference(df):
    """Difference in mean 1978 earnings, no adjustment."""
    return float(df.loc[df.treatment == 1, "post_period_metric"].mean()
                 - df.loc[df.treatment == 0, "post_period_metric"].mean())


def experimental_benchmark():
    """
    The target: the ATE from the randomized comparison. Since assignment was
    random, the simple difference in means is unbiased and needs no
    adjustment -- which is exactly what makes it a usable benchmark.
    """
    df = load_lalonde("experimental")
    ate = naive_difference(df)
    n_t = int((df.treatment == 1).sum())
    n_c = int((df.treatment == 0).sum())

    t = df.loc[df.treatment == 1, "post_period_metric"].values
    c = df.loc[df.treatment == 0, "post_period_metric"].values
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))

    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - 1.96 * se,
        "ci_high": ate + 1.96 * se,
        "n_treated": n_t,
        "n_control": n_c,
    }


def run_observational_methods(control_source="psid", n_boot=100, seed=0):
    """
    Run every confounding-correction method on the observational version and
    score each against the experimental benchmark.

    This is the only place in the project where the answer was not put there
    by code in this repository.
    """
    from matching import estimate_psm, smd_before_after, bootstrap_psm_ci
    from aipw import estimate_aipw

    df = load_lalonde(control_source)
    truth = experimental_benchmark()["ate"]

    rows = []

    naive = naive_difference(df)
    rows.append({
        "method": "Naive diff-in-means",
        "estimate": naive,
        "error_vs_experimental": naive - truth,
        "ci_low": np.nan, "ci_high": np.nan,
    })

    psm_est, n_matched, n_treated = estimate_psm(df, covariates=COVARIATES)
    if control_source == 'cps':
        psm_lo = psm_hi = np.nan
    else:
        try:
            _, psm_lo, psm_hi, _ = bootstrap_psm_ci(
                df, covariates=COVARIATES, n_boot=n_boot, seed=seed
            )
        except Exception:
            psm_lo = psm_hi = np.nan
    rows.append({
        "method": f"PSM (matched {n_matched}/{n_treated})",
        "estimate": psm_est,
        "error_vs_experimental": psm_est - truth,
        "ci_low": psm_lo, "ci_high": psm_hi,
    })

    for learner in ("linear", "gbm"):
        r = estimate_aipw(df, covariates=COVARIATES, learner=learner)
        rows.append({
            "method": f"AIPW ({learner} outcome model)",
            "estimate": r["ate"],
            "error_vs_experimental": r["ate"] - truth,
            "ci_low": r["ci_low"], "ci_high": r["ci_high"],
        })

    results = pd.DataFrame(rows)
    results["covers_experimental"] = (
        (results.ci_low <= truth) & (truth <= results.ci_high)
    )
    balance = smd_before_after(df, covariates=COVARIATES)
    return results, balance, truth


if __name__ == "__main__":
    print("=" * 74)
    print("LALONDE / NSW EXTERNAL VALIDATION")
    print("=" * 74)
    print("The only validation in this project where the answer was not put")
    print("there by code in this repository.\n")

    bench = experimental_benchmark()
    print(f"Experimental benchmark (randomized comparison):")
    print(f"  ATE = ${bench['ate']:,.0f}  95% CI (${bench['ci_low']:,.0f}, "
          f"${bench['ci_high']:,.0f})")
    print(f"  n = {bench['n_treated']} treated, {bench['n_control']} control")
    print(f"  Published reference: ~${EXPERIMENTAL_BENCHMARK:,.0f}")

    for source in ("psid", "cps"):
        print("\n" + "=" * 74)
        print(f"OBSERVATIONAL VERSION -- {source.upper()} survey controls")
        print("=" * 74)

        results, balance, truth = run_observational_methods(source, n_boot=200)

        disp = results.copy()
        for col in ("estimate", "error_vs_experimental", "ci_low", "ci_high"):
            disp[col] = disp[col].map(
                lambda v: "--" if pd.isna(v) else f"{v:>10,.0f}"
            )
        print(disp.to_string(index=False))

        print("\nCovariate balance before vs after matching:")
        print(balance.round(4).to_string(index=False))
        n_ok = int(balance.balanced_after.sum())
        print(f"  {n_ok}/{len(balance)} covariates under |SMD| < 0.1 after matching")

        results.to_csv(f"reports/lalonde_{source}.csv", index=False)
        balance.to_csv(f"reports/lalonde_{source}_balance.csv", index=False)
        print(f"\nSaved to reports/lalonde_{source}.csv")

    print("\n" + "=" * 74)
    print("READING THIS")
    print("=" * 74)
    print("The naive estimate should be large and negative -- survey controls")
    print("earn far more than programme applicants to begin with, so an")
    print("unadjusted comparison makes job training look actively harmful.")
    print()
    print("Closing that gap is the actual test. This benchmark has been")
    print("contested since 1986: Dehejia and Wahba (1999) reported propensity")
    print("methods recovering the experimental answer closely, and Smith and")
    print("Todd (2005) showed the result is fragile to which subsample and")
    print("specification you pick. A single run here is one data point in that")
    print("argument, not a resolution of it.")