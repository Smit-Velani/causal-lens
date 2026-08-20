"""
sequential.py — Always-valid inference for continuously monitored experiments.

A fixed-horizon t-test is valid exactly once, at a sample size fixed in
advance. Every real experiment gets watched: someone opens the dashboard
on day 3, sees p = 0.04, and ships. Each look is another chance for noise
to cross the threshold, so the false-positive rate compounds -- 5% becomes
20% or 30% depending on how often you look.

The usual advice ("don't peek") loses to organizational reality. Always-
valid methods solve it properly: they are valid at EVERY sample size
simultaneously, so you can stop whenever you like without inflating alpha.

Two are implemented here:

  - mSPRT (mixture Sequential Probability Ratio Test): a likelihood ratio
    of the alternative against the null, mixed over a prior on the effect
    size. The mixture is what buys anytime-validity -- the resulting
    process is a martingale under the null, so Ville's inequality bounds
    P(ever crossing 1/alpha) at alpha across the whole run, not per look.

  - Confidence sequences: the interval form of the same idea. A sequence
    of intervals with the guarantee that ALL of them contain the true
    effect with probability 1-alpha, rather than each one separately.
"""

import numpy as np
import pandas as pd
from scipy import stats


def fixed_horizon_pvalue(y_treat, y_control):
    """Standard two-sample Welch t-test. Valid once, at a pre-set n."""
    if len(y_treat) < 2 or len(y_control) < 2:
        return 1.0
    _, p = stats.ttest_ind(y_treat, y_control, equal_var=False)
    return float(p)


def msprt_statistic(y_treat, y_control, tau=1.0):
    """
    mSPRT likelihood ratio for a difference in means.

    Under H0 the effect is 0; under H1 it is theta, with a N(0, tau^2)
    mixing prior over theta. Integrating the likelihood ratio against that
    prior gives a closed form for the normal case:

        LR_n = sqrt( sigma^2 / (sigma^2 + n_eff * tau^2) )
               * exp( n_eff^2 * tau^2 * delta^2
                      / (2 * sigma^2 * (sigma^2 + n_eff * tau^2)) )

    where delta is the observed difference in means, sigma^2 the pooled
    variance, and n_eff the effective per-arm sample size.

    Reject when LR >= 1/alpha. The always-valid p-value is min(1, 1/LR),
    taken as a running minimum over the experiment.

    tau encodes the effect size you care about. Larger tau puts prior mass
    on bigger effects, detecting those faster at the cost of sensitivity to
    small ones. It is a genuine design choice, not a nuisance parameter --
    the sequential analogue of powering a fixed-horizon test.
    """
    n1, n0 = len(y_treat), len(y_control)
    if n1 < 2 or n0 < 2:
        return 1.0

    delta = y_treat.mean() - y_control.mean()
    var_pooled = ((n1 - 1) * y_treat.var(ddof=1) + (n0 - 1) * y_control.var(ddof=1)) / (n1 + n0 - 2)
    if var_pooled <= 0:
        return 1.0

    n_eff = (n1 * n0) / (n1 + n0)

    denom = var_pooled + n_eff * tau ** 2
    coef = np.sqrt(var_pooled / denom)
    expo = (n_eff ** 2 * tau ** 2 * delta ** 2) / (2 * var_pooled * denom)

    # Cap the exponent so a decisive result returns inf rather than overflowing
    if expo > 700:
        return np.inf
    return float(coef * np.exp(expo))


def always_valid_pvalue(y_treat, y_control, tau=1.0):
    """Always-valid p-value from the mSPRT statistic: min(1, 1/LR)."""
    lr = msprt_statistic(y_treat, y_control, tau=tau)
    if not np.isfinite(lr):
        return 0.0
    return float(min(1.0, 1.0 / lr)) if lr > 0 else 1.0


def confidence_sequence(y_treat, y_control, alpha=0.05, tau=1.0):
    """
    Anytime-valid confidence interval for the difference in means.

    Wider than a fixed-horizon CI by a factor that grows slowly with n --
    that width is the price of being allowed to look whenever you want.
    The guarantee is uniform: with probability 1-alpha, EVERY interval in
    the sequence contains the true effect, so stopping at the first one
    that excludes zero does not break anything.

    Returns (delta, lo, hi).
    """
    n1, n0 = len(y_treat), len(y_control)
    if n1 < 2 or n0 < 2:
        return 0.0, -np.inf, np.inf

    delta = y_treat.mean() - y_control.mean()
    var_pooled = ((n1 - 1) * y_treat.var(ddof=1) + (n0 - 1) * y_control.var(ddof=1)) / (n1 + n0 - 2)
    n_eff = (n1 * n0) / (n1 + n0)

    # Mixture boundary: sqrt( (sigma^2 + n*tau^2) / n^2
    #                         * log( (sigma^2 + n*tau^2) / (alpha^2 * sigma^2) ) )
    denom = var_pooled + n_eff * tau ** 2
    radius = np.sqrt(
        (denom / n_eff ** 2) * np.log(denom / (alpha ** 2 * var_pooled))
    )
    return float(delta), float(delta - radius), float(delta + radius)


def simulate_peeking(n_experiments=1000, n_max=2000, n_peeks=20, true_effect=0.0,
                     alpha=0.05, tau=1.0, sigma=1.0, seed=0):
    """
    The demonstration: run many experiments, peek repeatedly at each, and
    count how often a method ever crosses significance.

    With true_effect=0 this is an A/A test, so every rejection is a false
    positive. A method controlling alpha at 0.05 should reject about 5% of
    the time no matter how often it is peeked at. The fixed-horizon t-test
    does not, and the gap widens with the number of looks.

    Returns (results_df, summary_dict).
    """
    rng = np.random.default_rng(seed)
    peek_points = np.linspace(n_max // n_peeks, n_max, n_peeks).astype(int)

    fixed_flags, av_flags = [], []
    fixed_first, av_first = [], []

    for _ in range(n_experiments):
        control = rng.normal(0, sigma, n_max)
        treat = rng.normal(true_effect, sigma, n_max)

        fixed_hit = av_hit = False
        fixed_at = av_at = None

        for n in peek_points:
            if not fixed_hit:
                if fixed_horizon_pvalue(treat[:n], control[:n]) < alpha:
                    fixed_hit, fixed_at = True, int(n)
            if not av_hit:
                if always_valid_pvalue(treat[:n], control[:n], tau=tau) < alpha:
                    av_hit, av_at = True, int(n)
            if fixed_hit and av_hit:
                break

        fixed_flags.append(fixed_hit)
        av_flags.append(av_hit)
        fixed_first.append(fixed_at)
        av_first.append(av_at)

    results = pd.DataFrame({
        "fixed_horizon_rejected": fixed_flags,
        "always_valid_rejected": av_flags,
        "fixed_horizon_first_n": fixed_first,
        "always_valid_first_n": av_first,
    })

    summary = {
        "n_experiments": n_experiments,
        "n_peeks": n_peeks,
        "n_max": n_max,
        "true_effect": true_effect,
        "nominal_alpha": alpha,
        "fixed_horizon_rate": float(np.mean(fixed_flags)),
        "always_valid_rate": float(np.mean(av_flags)),
    }
    return results, summary


def peeking_curve(n_peek_grid=(1, 2, 5, 10, 20, 50), n_experiments=400,
                  n_max=2000, alpha=0.05, tau=1.0, seed=0):
    """
    False-positive rate as a function of how many times you peek.

    Fixed-horizon should climb steadily away from alpha; always-valid should
    stay flat. The flat line is the whole point -- it means the number of
    looks stops being a decision you have to get right.
    """
    rows = []
    for k in n_peek_grid:
        _, s = simulate_peeking(n_experiments=n_experiments, n_max=n_max,
                                n_peeks=k, true_effect=0.0, alpha=alpha,
                                tau=tau, seed=seed)
        rows.append({
            "n_peeks": k,
            "fixed_horizon_fpr": s["fixed_horizon_rate"],
            "always_valid_fpr": s["always_valid_rate"],
            "nominal_alpha": alpha,
        })
    return pd.DataFrame(rows)


def plot_peeking_curve(curve_df, save_path=None):
    """False-positive rate vs number of looks, both methods."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve_df.n_peeks, curve_df.fixed_horizon_fpr, "o-",
            label="Fixed-horizon t-test", lw=2)
    ax.plot(curve_df.n_peeks, curve_df.always_valid_fpr, "D-",
            label="Always-valid (mSPRT)", lw=2)
    ax.axhline(curve_df.nominal_alpha.iloc[0], ls="--", color="grey", lw=1.5,
               label="Nominal alpha = 0.05")
    ax.set_xscale("log")
    ax.set_xlabel("Number of interim looks")
    ax.set_ylabel("False-positive rate (A/A test)")
    ax.set_title("Peeking inflates the fixed-horizon error rate")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confidence_sequence(true_effect=0.5, n_max=3000, alpha=0.05, tau=1.0,
                             sigma=1.0, seed=0, save_path=None):
    """
    A single experiment watched from start to finish: the confidence
    sequence narrowing around the true effect, alongside the fixed-horizon
    interval computed at the same points.

    The fixed-horizon band is narrower everywhere, which looks better and
    is not -- it is only valid at one pre-chosen n, and reading it at any
    other point is the peeking problem in visual form.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    control = rng.normal(0, sigma, n_max)
    treat = rng.normal(true_effect, sigma, n_max)

    ns = np.arange(50, n_max + 1, 25)
    cs_lo, cs_hi, fx_lo, fx_hi, deltas = [], [], [], [], []

    for n in ns:
        d, lo, hi = confidence_sequence(treat[:n], control[:n], alpha=alpha, tau=tau)
        deltas.append(d)
        cs_lo.append(lo)
        cs_hi.append(hi)

        se = np.sqrt(treat[:n].var(ddof=1) / n + control[:n].var(ddof=1) / n)
        z = stats.norm.ppf(1 - alpha / 2)
        fx_lo.append(d - z * se)
        fx_hi.append(d + z * se)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ns, cs_lo, cs_hi, alpha=0.25,
                    label="Confidence sequence (valid at every n)")
    ax.fill_between(ns, fx_lo, fx_hi, alpha=0.35,
                    label="Fixed-horizon CI (valid at one n)")
    ax.plot(ns, deltas, color="black", lw=1.5, label="Observed difference")
    ax.axhline(true_effect, color="crimson", ls=":", lw=2,
               label=f"True effect {true_effect}")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("Sample size per arm")
    ax.set_ylabel("Estimated effect")
    ax.set_title("Anytime-valid vs fixed-horizon intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    print("=" * 64)
    print("PEEKING SIMULATION -- 1,000 A/A tests, 20 looks each")
    print("=" * 64)
    print("True effect is zero, so every rejection is a false positive.")
    print("A method controlling alpha at 0.05 should sit near 5% regardless")
    print("of how often it is checked.\n")

    _, summary = simulate_peeking(n_experiments=1000, n_max=2000,
                                  n_peeks=20, true_effect=0.0, seed=0)

    fh = summary["fixed_horizon_rate"]
    av = summary["always_valid_rate"]
    print(f"Fixed-horizon t-test : {fh*100:5.1f}% false positives  "
          f"({fh/0.05:.1f}x the nominal rate)")
    print(f"Always-valid mSPRT   : {av*100:5.1f}% false positives  "
          f"({av/0.05:.1f}x the nominal rate)")

    print("\n" + "=" * 64)
    print("FALSE-POSITIVE RATE vs NUMBER OF LOOKS")
    print("=" * 64)
    print("Fixed-horizon should climb; always-valid should stay flat.\n")

    curve = peeking_curve(n_experiments=400, seed=0)
    print(curve.to_string(index=False))
    curve.to_csv("reports/peeking_curve.csv", index=False)
    plot_peeking_curve(curve, save_path="reports/peeking_curve.png")

    print("\n" + "=" * 64)
    print("POWER CHECK -- does always-valid still detect a real effect?")
    print("=" * 64)
    print("Controlling false positives is easy if you never reject anything.")
    print("This confirms the method has not simply gone silent.\n")

    _, s_eff = simulate_peeking(n_experiments=400, n_max=2000, n_peeks=20,
                                true_effect=0.15, seed=1)
    print(f"True effect = 0.15 (0.15 SD)")
    print(f"  Fixed-horizon detected: {s_eff['fixed_horizon_rate']*100:5.1f}%")
    print(f"  Always-valid detected : {s_eff['always_valid_rate']*100:5.1f}%")
    print("\nAlways-valid should detect somewhat less often -- that is the")
    print("cost of anytime-validity, paid in power rather than in error rate.")

    plot_confidence_sequence(true_effect=0.5,
                             save_path="reports/confidence_sequence.png")
    print("\nSaved plots to reports/peeking_curve.png and "
          "reports/confidence_sequence.png")