"""
test_power.py — Tests for power analysis and duration planning.

Mostly checks against known closed-form values and structural properties,
since these functions have exact answers rather than simulated ones.
"""

import numpy as np
import pytest

from power import (required_sample_size, minimum_detectable_effect,
                   achieved_power, experiment_duration, cuped_sample_savings,
                   sequential_power_penalty)


def test_sample_size_matches_textbook_value():
    """
    alpha=0.05, power=0.80, MDE=0.10 SD gives 1,570 per arm. This is the
    standard reference figure and the fastest way to catch a formula error.
    """
    assert required_sample_size(0.10) == 1570


def test_sample_size_scales_quadratically():
    """Halving the MDE should roughly quadruple the required sample."""
    n_large = required_sample_size(0.20)
    n_small = required_sample_size(0.10)
    assert 3.9 < n_small / n_large < 4.1


def test_mde_and_sample_size_are_inverses():
    """Round-tripping n -> MDE -> n should return to where it started."""
    for n in (500, 5000, 50000):
        mde = minimum_detectable_effect(n)
        assert abs(required_sample_size(mde) - n) <= 2


def test_achieved_power_hits_target_at_required_n():
    """At the sample size solved for 80% power, power should be 80%."""
    n = required_sample_size(0.10, power=0.80)
    assert abs(achieved_power(n, 0.10) - 0.80) < 0.01


def test_unequal_allocation_costs_sample_size():
    """A 90/10 split needs more total sample than a 50/50 split."""
    balanced = required_sample_size(0.10, allocation=0.5)
    skewed = required_sample_size(0.10, allocation=0.9)
    assert skewed > balanced


def test_cuped_savings_match_variance_reduction():
    """
    Required n is linear in variance, so a 47.2% variance reduction should
    cut sample size by 47.2% -- not by its square root.
    """
    s = cuped_sample_savings(0.10, variance_reduction=0.472)
    assert abs(s["pct_saved"] - 47.2) < 0.5


def test_duration_respects_one_week_minimum():
    """
    Even with effectively unlimited traffic the planner should not return a
    sub-week duration, since a shorter run confounds the effect with
    day-of-week seasonality.
    """
    d = experiment_duration(0.5, daily_traffic=10_000_000)
    assert d["days"] >= 7
    assert not d["sample_bound"]


def test_duration_becomes_sample_bound_for_small_effects():
    """A very small MDE should take longer than the calendar floor."""
    d = experiment_duration(0.005, daily_traffic=50000, exposure_rate=0.4)
    assert d["sample_bound"]
    assert d["days"] > 28
    assert d["too_long"]


def test_sequential_penalty_increases_sample_size():
    s = sequential_power_penalty(0.10)
    assert s["n_per_arm_sequential"] > s["n_per_arm_fixed"]
    assert s["extra_samples"] > 0


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        required_sample_size(0)
    with pytest.raises(ValueError):
        required_sample_size(0.1, allocation=1.0)
    with pytest.raises(ValueError):
        minimum_detectable_effect(1)