"""
test_lalonde.py — External validation tests.

These require the LaLonde data files, which are downloaded on first run of
lalonde_validate.py and cached under data/raw/lalonde/. They skip cleanly
when the cache is absent so CI does not depend on a third-party host being
up -- a test that fails because NBER is down is not telling you anything
about the code.
"""

import os

import numpy as np
import pytest

DATA_DIR = os.path.join("data", "raw", "lalonde")
REQUIRED = ["nswre74_treated.txt", "nswre74_control.txt", "psid_controls.txt"]

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED),
    reason="LaLonde data not cached; run lalonde_validate.py once to fetch it",
)

from lalonde_validate import (load_lalonde, naive_difference,
                              experimental_benchmark, COVARIATES)


def test_experimental_benchmark_matches_published_value():
    """
    The randomized comparison should reproduce the ~$1,794 figure reported
    by Dehejia and Wahba. If this drifts, the data loader is wrong and every
    downstream comparison is meaningless.
    """
    b = experimental_benchmark()

    assert abs(b["ate"] - 1794) < 50
    assert b["n_treated"] == 185
    assert b["n_control"] == 260
    assert b["ci_low"] < 1794 < b["ci_high"]


def test_observational_naive_is_catastrophically_wrong():
    """
    Replacing experimental controls with PSID survey respondents should make
    job training look actively harmful. This is the premise of the whole
    benchmark -- if the naive estimate were fine there would be nothing to
    correct.
    """
    df = load_lalonde("psid")
    naive = naive_difference(df)

    assert naive < -5000
    assert abs(naive - 1794) > 10000


def test_covariates_are_severely_imbalanced_before_matching():
    """
    The two groups barely inhabit the same population. Several SMDs should
    exceed 1.0, which is an order of magnitude past the 0.1 threshold.
    """
    from matching import smd_before_after

    df = load_lalonde("psid")
    balance = smd_before_after(df, covariates=COVARIATES)

    assert (balance.smd_before.abs() > 1.0).sum() >= 3
    assert (balance.smd_after.abs() < balance.smd_before.abs()).sum() >= 6


def test_psm_closes_most_of_the_gap():
    """
    PSM should land far closer to the experimental $1,794 than the naive
    estimate does. Not a claim that it is correct -- Smith and Todd (2005)
    showed this benchmark is fragile -- only that adjustment helps a lot.
    """
    from matching import estimate_psm

    df = load_lalonde("psid")
    naive = naive_difference(df)
    psm, _, _ = estimate_psm(df, covariates=COVARIATES)

    assert abs(psm - 1794) < abs(naive - 1794) / 5


def test_aipw_struggles_under_poor_overlap():
    """
    The finding worth documenting: AIPW beats PSM on the synthetic data,
    where overlap is good, and loses badly here, where it is not. Weighting
    by 1/e(X) breaks down when propensity scores approach zero, and no
    amount of double robustness repairs a positivity violation.

    Asserted as a property of this data rather than a defect, so that if a
    future change makes AIPW succeed here, the test flags it for
    re-examination rather than silently passing.
    """
    from aipw import estimate_aipw

    df = load_lalonde("psid")
    r = estimate_aipw(df, covariates=COVARIATES, learner="linear")

    assert r["ate"] < 0, "AIPW is expected to fail under this overlap violation"
    assert r["effective_n"] < 0.9 * r["n"], "extreme weights should shrink effective n"