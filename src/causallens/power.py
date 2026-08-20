"""
power.py — Power analysis, MDE, and experiment duration planning.

Everything else in this project analyses an experiment that already ran.
This module answers the question that comes first and gets asked most
often in practice: how long do we need to run it, and what is the smallest
effect we could actually detect?

Getting this wrong is the most common failure in experimentation. An
underpowered test that comes back null does not mean "no effect" -- it
means the experiment could never have found one. That distinction is
usually discovered after the fact, when it is too late.

Three connections to the rest of the project make this more than a
textbook calculator:

  - CUPED reduces variance by a measured 47.2% on this data, and required
    sample size scales linearly with variance, so the same MDE is reachable
    with roughly half the traffic.
  - Always-valid inference costs power, which converts into needing more
    samples -- the price of being allowed to stop whenever you like.
  - Effects are heterogeneous in the generator, so the average effect the
    power calculation targets is not the effect any individual sees.
"""

import numpy as np
import pandas as pd
from scipy import stats


def required_sample_size(mde, sigma=1.0, alpha=0.05, power=0.80, two_sided=True,
                         allocation=0.5):
    """
    Per-arm sample size needed to detect an effect of size `mde`.

        n = (z_alpha + z_beta)^2 * 2 * sigma^2 / mde^2   (equal allocation)

    Note the square: halving the MDE quadruples the sample size. That
    nonlinearity is why "let's also detect a 1% lift" is never a small ask.

    allocation: fraction assigned to treatment. 0.5 is optimal; skewed
    splits cost sample size, which the variance inflation factor captures.
    """
    if mde <= 0:
        raise ValueError("mde must be positive")
    if not 0 < allocation < 1:
        raise ValueError("allocation must be strictly between 0 and 1")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    # Variance inflation from unequal allocation: 1/(p(1-p)) vs 4 at p=0.5
    inflation = (1 / (allocation * (1 - allocation))) / 4.0

    n_per_arm = ((z_alpha + z_beta) ** 2 * 2 * sigma ** 2 / mde ** 2) * inflation
    return int(np.ceil(n_per_arm))


def minimum_detectable_effect(n_per_arm, sigma=1.0, alpha=0.05, power=0.80,
                              two_sided=True, allocation=0.5):
    """
    The inverse: given the traffic you have, the smallest effect you could
    detect with the stated power.

    This is the more useful direction in practice. Sample size is usually
    not a free parameter -- you get the traffic you get -- so the real
    question is whether the effect you are hoping for is even inside the
    range of things this experiment can see.
    """
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    inflation = (1 / (allocation * (1 - allocation))) / 4.0
    return float((z_alpha + z_beta) * sigma * np.sqrt(2 * inflation / n_per_arm))


def achieved_power(n_per_arm, effect, sigma=1.0, alpha=0.05, two_sided=True,
                   allocation=0.5):
    """
    Power actually achieved for a given n and effect -- the diagnostic for
    a null result. If a completed test had 30% power for the effect size
    you cared about, the null tells you almost nothing.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    inflation = (1 / (allocation * (1 - allocation))) / 4.0
    se = sigma * np.sqrt(2 * inflation / n_per_arm)
    ncp = abs(effect) / se
    return float(stats.norm.cdf(ncp - z_alpha))


def experiment_duration(mde, daily_traffic, sigma=1.0, alpha=0.05, power=0.80,
                        allocation=0.5, exposure_rate=1.0, min_days=7):
    """
    Days needed, given daily traffic.

    exposure_rate: fraction of traffic actually eligible for the experiment.
    Defaults to 1.0, but in practice a feature behind a login, a locale, or
    a device type sees far less than total traffic, and forgetting that is
    how a two-week plan becomes two months.

    min_days: a floor, defaulting to one full week. Even when the raw sample
    arrives sooner, a shorter run confounds the treatment effect with
    day-of-week seasonality -- weekend and weekday users differ, and an
    experiment that only sees Tuesdays is measuring Tuesday. Running in
    whole weeks is standard practice for that reason, so the returned
    duration is also rounded up to a whole number of weeks.

    Returns a dict; `sample_bound` distinguishes the case where sample size
    is the binding constraint from the case where the calendar floor is.
    """
    n_per_arm = required_sample_size(mde, sigma, alpha, power, allocation=allocation)
    n_total = n_per_arm * 2
    usable_daily = daily_traffic * exposure_rate

    if usable_daily <= 0:
        raise ValueError("usable daily traffic must be positive")

    raw_days = n_total / usable_daily
    days = max(np.ceil(raw_days), min_days)
    weeks = int(np.ceil(days / 7))
    days = weeks * 7

    return {
        "mde": mde,
        "n_per_arm": n_per_arm,
        "n_total": n_total,
        "daily_traffic": daily_traffic,
        "exposure_rate": exposure_rate,
        "usable_daily": usable_daily,
        "raw_days": float(np.ceil(raw_days)),
        "days": float(days),
        "weeks": weeks,
        "sample_bound": bool(raw_days > min_days),
        "too_long": bool(days > 28),
    }


def power_curve(effects, n_grid, sigma=1.0, alpha=0.05, allocation=0.5):
    """Power at each (effect, n) combination -- long format for plotting."""
    rows = []
    for eff in effects:
        for n in n_grid:
            rows.append({
                "effect": eff,
                "n_per_arm": n,
                "power": achieved_power(n, eff, sigma, alpha, allocation=allocation),
            })
    return pd.DataFrame(rows)


def cuped_sample_savings(mde, sigma=1.0, variance_reduction=0.472, alpha=0.05,
                         power=0.80):
    """
    Sample size with and without CUPED.

    Required n scales linearly with variance, so a 47.2% variance reduction
    -- the figure measured in cuped.py on this generator -- cuts required
    sample size by 47.2%. CUPED needs no extra traffic and no change to the
    experiment design; it only needs a pre-period covariate that was already
    being logged.

    Note sigma_cuped = sigma * sqrt(1 - reduction): the reduction applies to
    variance, and sample size depends on variance directly, which is why the
    saving passes through one-for-one rather than being square-rooted away.
    """
    sigma_cuped = sigma * np.sqrt(1 - variance_reduction)

    n_plain = required_sample_size(mde, sigma, alpha, power)
    n_cuped = required_sample_size(mde, sigma_cuped, alpha, power)

    return {
        "mde": mde,
        "variance_reduction": variance_reduction,
        "n_per_arm_plain": n_plain,
        "n_per_arm_cuped": n_cuped,
        "samples_saved": n_plain - n_cuped,
        "pct_saved": (n_plain - n_cuped) / n_plain * 100,
    }


def sequential_power_penalty(mde, sigma=1.0, alpha=0.05, power=0.80,
                             inflation=1.25):
    """
    Extra sample size needed to recover fixed-horizon power under an
    always-valid test.

    The default 1.25x sits in the rule-of-thumb range reported for
    mSPRT-style tests at conventional power targets. An earlier version
    derived this from the 99.8% vs 90.5% figures measured in sequential.py
    and got 2.19x, which is badly overstated: those were measured at a
    0.15 SD effect where both methods sit in the flat tail of the power
    curve, and a small power gap there implies a large non-centrality gap
    that does not generalise down to an 80% target.

    Left as an explicit parameter rather than a derived one, because the
    honest answer is that it depends on the mixture prior tau and the
    effect size, and pretending otherwise would be false precision.
    """
    n_fixed = required_sample_size(mde, sigma, alpha, power)
    n_seq = int(np.ceil(n_fixed * inflation))

    return {
        "mde": mde,
        "n_per_arm_fixed": n_fixed,
        "n_per_arm_sequential": n_seq,
        "inflation_factor": inflation,
        "extra_samples": n_seq - n_fixed,
    }


def plot_power_curves(effects, n_grid, sigma=1.0, alpha=0.05, target_power=0.80,
                      save_path=None):
    """Power vs sample size, one line per effect size."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for eff in effects:
        powers = [achieved_power(n, eff, sigma, alpha) for n in n_grid]
        ax.plot(n_grid, powers, lw=2, label=f"Effect = {eff} SD")

    ax.axhline(target_power, ls="--", color="grey", lw=1.5,
               label=f"Target power = {target_power}")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.02)
    ax.set_title("Power by sample size and effect size")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mde_curve(n_grid, sigma=1.0, alpha=0.05, power=0.80,
                   variance_reduction=0.472, save_path=None):
    """MDE vs sample size, with and without CUPED."""
    import matplotlib.pyplot as plt

    sigma_cuped = sigma * np.sqrt(1 - variance_reduction)
    mde_plain = [minimum_detectable_effect(n, sigma, alpha, power) for n in n_grid]
    mde_cuped = [minimum_detectable_effect(n, sigma_cuped, alpha, power) for n in n_grid]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_grid, mde_plain, lw=2, label="Standard")
    ax.plot(n_grid, mde_cuped, lw=2, ls="--",
            label=f"With CUPED ({variance_reduction*100:.0f}% variance reduction)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Minimum detectable effect (SD)")
    ax.set_title("Smallest detectable effect by sample size")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("=" * 72)
    print("SAMPLE SIZE REQUIRED (alpha=0.05, power=0.80, two-sided)")
    print("=" * 72)
    print("Note the quadratic scaling: halving the MDE quadruples the cost.\n")

    rows = []
    for mde in [0.02, 0.05, 0.10, 0.15, 0.20, 0.50]:
        n = required_sample_size(mde)
        rows.append({"mde_sd": mde, "n_per_arm": n, "n_total": n * 2})
    size_df = pd.DataFrame(rows)
    print(size_df.to_string(index=False))
    print("\nSanity check: MDE 0.10 SD -> 1,570 per arm is the standard")
    print("textbook figure at these settings.")

    print("\n" + "=" * 72)
    print("MINIMUM DETECTABLE EFFECT (the more useful direction)")
    print("=" * 72)
    print("Traffic is rarely a free parameter, so the real question is what")
    print("you could detect with the traffic you already have.\n")

    rows = []
    for n in [500, 1000, 5000, 10000, 50000, 100000]:
        rows.append({
            "n_per_arm": n,
            "mde_sd": round(minimum_detectable_effect(n), 4),
            "power_for_0.10_effect": round(achieved_power(n, 0.10), 3),
        })
    mde_df = pd.DataFrame(rows)
    print(mde_df.to_string(index=False))
    print("\nThe third column is the diagnostic for a null result: at n=500")
    print("an experiment has 35% power for a 0.10 SD effect, so a null there")
    print("says almost nothing about whether the effect exists.")

    print("\n" + "=" * 72)
    print("CUPED SAMPLE SAVINGS (47.2% variance reduction, measured)")
    print("=" * 72)
    print("Variance reduction passes through to sample size one-for-one,")
    print("because required n scales linearly with variance.\n")

    rows = []
    for mde in [0.05, 0.10, 0.20]:
        s = cuped_sample_savings(mde)
        rows.append({
            "mde_sd": s["mde"],
            "n_plain": s["n_per_arm_plain"],
            "n_with_cuped": s["n_per_arm_cuped"],
            "saved": s["samples_saved"],
            "pct_saved": round(s["pct_saved"], 1),
        })
    cuped_df = pd.DataFrame(rows)
    print(cuped_df.to_string(index=False))
    cuped_df.to_csv("reports/cuped_sample_savings.csv", index=False)
    print("\nThis is the practical argument for CUPED: same traffic, same")
    print("design, roughly half the experiment.")

    print("\n" + "=" * 72)
    print("SEQUENTIAL TESTING PENALTY (1.25x rule of thumb)")
    print("=" * 72)
    print("Always-valid inference costs power, which converts into traffic.\n")

    rows = []
    for mde in [0.05, 0.10, 0.20]:
        s = sequential_power_penalty(mde)
        rows.append({
            "mde_sd": s["mde"],
            "n_fixed": s["n_per_arm_fixed"],
            "n_sequential": s["n_per_arm_sequential"],
            "extra": s["extra_samples"],
            "inflation": s["inflation_factor"],
        })
    seq_df = pd.DataFrame(rows)
    print(seq_df.to_string(index=False))
    print("\nThe trade: pay ~25% more traffic, and stop the experiment")
    print("whenever you like without inflating the false-positive rate --")
    print("which the peeking simulation put at 27% after 50 looks.")

    print("\n" + "=" * 72)
    print("EXPERIMENT DURATION -- 50,000 daily users, 40% eligible")
    print("=" * 72)
    print("Rounded up to whole weeks with a one-week floor, since a shorter")
    print("run confounds the treatment effect with day-of-week seasonality.\n")

    rows = []
    for mde in [0.005, 0.01, 0.02, 0.05, 0.10]:
        d = experiment_duration(mde, daily_traffic=50000, exposure_rate=0.4)
        rows.append({
            "mde_sd": d["mde"],
            "n_total": d["n_total"],
            "raw_days": d["raw_days"],
            "days": d["days"],
            "weeks": d["weeks"],
            "bound_by": "sample size" if d["sample_bound"] else "1-week minimum",
            "flag": "too long" if d["too_long"] else "ok",
        })
    dur_df = pd.DataFrame(rows)
    print(dur_df.to_string(index=False))
    dur_df.to_csv("reports/experiment_duration.csv", index=False)
    print("\nSmall MDEs are bound by sample size; larger ones hit the calendar")
    print("floor long before they run out of traffic. Past about four weeks,")
    print("novelty effects and seasonality start to contaminate the comparison,")
    print("so a long-duration plan is usually a signal to pick a more sensitive")
    print("metric rather than wait it out.")

    n_grid = np.logspace(2, 5.5, 60).astype(int)
    plot_power_curves([0.05, 0.10, 0.20, 0.50], n_grid,
                      save_path="reports/power_curves.png")
    plot_mde_curve(n_grid, save_path="reports/mde_curve.png")
    print("\nSaved plots to reports/power_curves.png and reports/mde_curve.png")