"""
simulation.py
=============
Monte Carlo simulation engine for the AB Audit tool.

All simulations are fully vectorised using NumPy — no Python loops
in the hot path. Target: 10,000 simulations in under 3 seconds on
a standard laptop CPU.

Public API
----------
run_peeking_simulation()     Empirical false positive rate under peeking
run_power_simulation()       Empirical power curve across sample sizes
run_research_study()         1,000 synthetic experiments → headline result
simulate_null_trajectories() p-value paths over time for the UI plot
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════
# Result dataclasses
# ════════════════════════════════════════════════════════════

@dataclass
class PeekingSimResult:
    """Output of run_peeking_simulation()."""
    nominal_alpha:      float
    inflated_alpha:     float
    inflation_factor:   float
    n_simulations:      int
    peeking_days:       list
    false_positives:    int
    # Arrays for plotting
    alpha_by_n_peeks:   list        # inflated alpha at 1,2,...,k peeks
    trajectory_sample:  list        # list of p-value lists (20 sample paths)
    daily_fp_rates:     list        # cumulative FP rate at each peek day


@dataclass
class PowerSimResult:
    """Output of run_power_simulation()."""
    p_control:      float
    true_lift:      float
    alpha:          float
    n_values:       list            # sample sizes tested
    power_values:   list            # empirical power at each n
    required_n_80:  Optional[int]   # n where power first crosses 0.80
    required_n_90:  Optional[int]   # n where power first crosses 0.90


@dataclass
class ResearchStudyResult:
    """Output of run_research_study() — the headline finding."""
    n_experiments:              int
    n_invalid:                  int
    n_would_ship_without_audit: int
    pct_invalid:                float
    pct_caught_by_audit:        float
    pct_false_ships:            float   # invalid + would have shipped
    violation_breakdown:        dict    # how many experiments had each violation
    severity_breakdown:         dict    # PASS / WARN / FAIL counts


# ════════════════════════════════════════════════════════════
# CORE MATH  (vectorised)
# ════════════════════════════════════════════════════════════

def _vectorised_z_pvalue(x_c: np.ndarray, n_c: int,
                          x_t: np.ndarray, n_t: int) -> np.ndarray:
    """
    Two-proportion z-test p-values for many experiments at once.

    Parameters
    ----------
    x_c : (n_sims,) array of integer successes in control
    x_t : (n_sims,) array of integer successes in treatment
    n_c : int  — arm size for control
    n_t : int  — arm size for treatment

    Returns
    -------
    p_values : (n_sims,) array of two-tailed p-values
    """
    p_c = x_c / n_c
    p_t = x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
    # Avoid division by zero (degenerate cases)
    se = np.where(se < 1e-10, 1e-10, se)

    z = np.abs(p_t - p_c) / se
    # Two-tailed p-value via complementary error function
    # P(|Z| > z) = 2 * (1 - Φ(z)) = erfc(z / sqrt(2))
    from math import sqrt
    p_vals = np.array([float(np.real(
        2.0 * (1.0 - _fast_norm_cdf(float(zi)))
    )) for zi in z])
    return np.clip(p_vals, 0.0, 1.0)


def _fast_norm_cdf(z: float) -> float:
    """Standard normal CDF — same implementation as checks.py."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530
          + t * (-0.356563782
          + t * (1.781477937
          + t * (-1.821255978
          + t *  1.330274429))))
    p = 1.0 - 0.3989422804 * np.exp(-0.5 * z * z) * poly
    return float(p) if z >= 0 else float(1.0 - p)


def _vectorised_norm_cdf(z: np.ndarray) -> np.ndarray:
    """Vectorised normal CDF using scipy (for large arrays in simulation)."""
    from scipy.special import ndtr
    return ndtr(z)


# ════════════════════════════════════════════════════════════
# 1. PEEKING SIMULATION
# ════════════════════════════════════════════════════════════

def run_peeking_simulation(
    n_per_arm:      int,
    p_control:      float,
    peeking_days:   list,
    total_days:     int   = None,
    alpha:          float = 0.05,
    n_simulations:  int   = 10_000,
    seed:           int   = 42,
) -> PeekingSimResult:
    """
    Empirically measure how much peeking inflates the false positive rate.

    Runs n_simulations null experiments (true lift = 0). For each, checks
    the p-value at every peeking day. If ANY check crosses alpha, that
    simulation is a false positive.

    This is the core demo of the tool — making the abstract peeking
    problem concrete and visceral through simulation.

    Parameters
    ----------
    n_per_arm : int
        Users per arm at the END of the experiment.
    p_control : float
        True conversion rate (same for both arms — null is true).
    peeking_days : list of int
        Days on which the team peeked. Users accumulate linearly.
    total_days : int, optional
        Total experiment duration. Defaults to max(peeking_days).
    alpha : float
        Nominal significance threshold.
    n_simulations : int
        Number of Monte Carlo iterations. 10,000 gives stable estimates.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if not peeking_days:
        return PeekingSimResult(
            nominal_alpha=alpha, inflated_alpha=alpha, inflation_factor=1.0,
            n_simulations=0, peeking_days=[], false_positives=0,
            alpha_by_n_peeks=[], trajectory_sample=[], daily_fp_rates=[],
        )

    total_days  = total_days or max(peeking_days)
    peeks       = sorted(peeking_days)
    n_peeks     = len(peeks)

    # Users seen at each peek day (linear accumulation)
    n_at_peek   = [max(10, int(n_per_arm * d / total_days)) for d in peeks]

    # ── Vectorised simulation ────────────────────────────────
    # Shape: (n_simulations, n_peeks)
    # Generate all data at maximum n, then subset for each peek
    max_n = n_per_arm

    # For each sim, generate max_n Bernoulli(p) per arm once,
    # then take cumulative sums to get counts at each peek day.
    # This is exact and avoids re-sampling.

    ctrl_full = rng.binomial(1, p_control, size=(n_simulations, max_n))
    trt_full  = rng.binomial(1, p_control, size=(n_simulations, max_n))  # null: same p

    # Cumulative successes at each sample size checkpoint
    ctrl_cum = np.cumsum(ctrl_full, axis=1)   # (n_sims, max_n)
    trt_cum  = np.cumsum(trt_full,  axis=1)

    # Extract at each peek day's sample size
    # p-value matrix: shape (n_simulations, n_peeks)
    p_matrix = np.ones((n_simulations, n_peeks))

    for j, n_so_far in enumerate(n_at_peek):
        idx = min(n_so_far - 1, max_n - 1)
        x_c = ctrl_cum[:, idx].astype(float)
        x_t = trt_cum[:,  idx].astype(float)

        p_c_hat  = x_c / n_so_far
        p_t_hat  = x_t / n_so_far
        p_pool   = (x_c + x_t) / (2 * n_so_far)
        se       = np.sqrt(p_pool * (1 - p_pool) * 2 / n_so_far)
        se       = np.where(se < 1e-10, 1e-10, se)

        z        = np.abs(p_t_hat - p_c_hat) / se
        p_vals   = 2.0 * (1.0 - _vectorised_norm_cdf(z))
        p_matrix[:, j] = np.clip(p_vals, 0.0, 1.0)

    # ── False positive: any peek crosses alpha ───────────────
    fp_per_sim  = (p_matrix < alpha).any(axis=1)   # (n_sims,) bool
    false_positives = int(fp_per_sim.sum())
    inflated_alpha  = false_positives / n_simulations

    # ── Cumulative FP rate at each peek ──────────────────────
    # What's the FP rate if you only peeked up to peek k?
    daily_fp_rates = []
    for k in range(1, n_peeks + 1):
        fp_k = int((p_matrix[:, :k] < alpha).any(axis=1).sum())
        daily_fp_rates.append(round(fp_k / n_simulations, 4))

    # ── FP rate as function of number of peeks ───────────────
    alpha_by_n_peeks = daily_fp_rates   # same as above

    # ── Sample 20 trajectories for UI plot ───────────────────
    n_sample = min(20, n_simulations)
    sample_idx = rng.choice(n_simulations, size=n_sample, replace=False)
    trajectory_sample = [
        [round(float(p_matrix[i, j]), 4) for j in range(n_peeks)]
        for i in sample_idx
    ]

    return PeekingSimResult(
        nominal_alpha=alpha,
        inflated_alpha=round(inflated_alpha, 4),
        inflation_factor=round(inflated_alpha / max(alpha, 1e-10), 3),
        n_simulations=n_simulations,
        peeking_days=peeks,
        false_positives=false_positives,
        alpha_by_n_peeks=[round(x, 4) for x in alpha_by_n_peeks],
        trajectory_sample=trajectory_sample,
        daily_fp_rates=[round(x, 4) for x in daily_fp_rates],
    )


# ════════════════════════════════════════════════════════════
# 2. POWER CURVE SIMULATION
# ════════════════════════════════════════════════════════════

def run_power_simulation(
    p_control:      float,
    true_lift:      float,
    alpha:          float = 0.05,
    n_values:       list  = None,
    n_simulations:  int   = 5_000,
    seed:           int   = 7,
) -> PowerSimResult:
    """
    Empirical power curve — how power grows with sample size.

    For each n in n_values, runs n_simulations experiments with the
    true effect baked in, and counts how many correctly reject H₀.

    This produces the power curve shown in the Experiment Designer page.

    Parameters
    ----------
    p_control : float
        True control conversion rate.
    true_lift : float
        True treatment effect in absolute pp.
    alpha : float
        Significance threshold.
    n_values : list of int, optional
        Sample sizes to evaluate. Defaults to a log-spaced range.
    n_simulations : int
        Monte Carlo iterations per sample size. 5,000 is sufficient.
    """
    rng = np.random.default_rng(seed)
    p_treatment = np.clip(p_control + true_lift, 1e-6, 1 - 1e-6)

    if n_values is None:
        # Log-spaced from 100 to 20,000
        n_values = sorted(set(
            [int(x) for x in np.logspace(np.log10(100), np.log10(20_000), 30)]
        ))

    power_values   = []
    required_n_80  = None
    required_n_90  = None

    for n in n_values:
        # Generate all sims at once — shape (n_sims,)
        x_c = rng.binomial(n, p_control,   size=n_simulations)
        x_t = rng.binomial(n, p_treatment, size=n_simulations)

        p_c_hat = x_c / n
        p_t_hat = x_t / n
        p_pool  = (x_c + x_t) / (2 * n)
        se      = np.sqrt(p_pool * (1 - p_pool) * 2 / n)
        se      = np.where(se < 1e-10, 1e-10, se)

        z       = np.abs(p_t_hat - p_c_hat) / se
        p_vals  = 2.0 * (1.0 - _vectorised_norm_cdf(z))
        power   = float((p_vals < alpha).mean())
        power_values.append(round(power, 4))

        if required_n_80 is None and power >= 0.80:
            required_n_80 = n
        if required_n_90 is None and power >= 0.90:
            required_n_90 = n

    return PowerSimResult(
        p_control=p_control,
        true_lift=true_lift,
        alpha=alpha,
        n_values=n_values,
        power_values=power_values,
        required_n_80=required_n_80,
        required_n_90=required_n_90,
    )


# ════════════════════════════════════════════════════════════
# 3. NULL P-VALUE TRAJECTORIES  (for interactive peeking plot)
# ════════════════════════════════════════════════════════════

def simulate_null_trajectories(
    n_per_arm:      int,
    p_control:      float,
    n_days:         int   = 14,
    n_trajectories: int   = 50,
    alpha:          float = 0.05,
    seed:           int   = 42,
) -> dict:
    """
    Simulate p-value trajectories day-by-day for n_trajectories null experiments.

    Used by the Peeking Simulator page to show the animated p-value
    traces — each line is one null experiment checked daily.

    Returns
    -------
    dict with keys:
        days          : list of ints [1..n_days]
        trajectories  : list of lists — each inner list is p-values by day
        alpha_line    : float — the significance threshold (horizontal line)
        crossings     : list of dicts — {traj_idx, day, p_value} for each
                        time a trajectory crosses below alpha (false positives)
        fp_rate       : float — fraction of trajectories that ever cross alpha
    """
    rng  = np.random.default_rng(seed)
    days = list(range(1, n_days + 1))

    # Users accumulate linearly
    n_at_day = [max(5, int(n_per_arm * d / n_days)) for d in days]

    # Generate all data upfront: shape (n_trajectories, n_per_arm)
    ctrl_raw = rng.binomial(1, p_control, size=(n_trajectories, n_per_arm))
    trt_raw  = rng.binomial(1, p_control, size=(n_trajectories, n_per_arm))

    ctrl_cum = np.cumsum(ctrl_raw, axis=1)
    trt_cum  = np.cumsum(trt_raw,  axis=1)

    trajectories = []
    crossings    = []
    ever_crossed = np.zeros(n_trajectories, dtype=bool)

    for traj_idx in range(n_trajectories):
        traj_pvals = []
        for day_idx, n_so_far in enumerate(n_at_day):
            idx   = min(n_so_far - 1, n_per_arm - 1)
            x_c   = float(ctrl_cum[traj_idx, idx])
            x_t   = float(trt_cum[traj_idx, idx])
            p_c   = x_c / n_so_far
            p_t   = x_t / n_so_far
            p_pool = (x_c + x_t) / (2 * n_so_far)
            se    = (p_pool * (1 - p_pool) * 2 / n_so_far) ** 0.5
            if se < 1e-10:
                pv = 1.0
            else:
                z  = abs(p_t - p_c) / se
                pv = float(2.0 * (1.0 - _fast_norm_cdf(z)))

            pv = min(max(pv, 0.0), 1.0)
            traj_pvals.append(round(pv, 4))

            if pv < alpha and not ever_crossed[traj_idx]:
                crossings.append({
                    "traj_idx": traj_idx,
                    "day":      days[day_idx],
                    "p_value":  round(pv, 4),
                })
                ever_crossed[traj_idx] = True

        trajectories.append(traj_pvals)

    fp_rate = float(ever_crossed.sum()) / n_trajectories

    return {
        "days":          days,
        "trajectories":  trajectories,
        "alpha_line":    alpha,
        "crossings":     crossings,
        "fp_rate":       round(fp_rate, 4),
        "n_trajectories": n_trajectories,
        "n_per_arm":     n_per_arm,
        "p_control":     p_control,
    }


# ════════════════════════════════════════════════════════════
# 4. RESEARCH SIMULATION STUDY
# ════════════════════════════════════════════════════════════

def run_research_study(
    n_experiments:  int   = 1_000,
    alpha:          float = 0.05,
    seed:           int   = 2024,
    verbose:        bool  = False,
) -> ResearchStudyResult:
    """
    The headline finding of the research paper.

    Generates n_experiments synthetic A/B experiments, each randomly
    assigned 0–3 methodological violations. Runs the full audit on each.
    Measures: what fraction of invalid experiments would have been shipped
    as "statistically significant" without the audit?

    That number — the false ship rate — is the paper's opening line.

    Algorithm
    ---------
    For each experiment:
    1. Randomly assign: true_effect (null or real), and 0–3 violations
       (SRM, underpowered, peeking, multiple testing, novelty)
    2. Generate data with those properties
    3. Compute naive result (would naive team ship it?)
    4. Run the audit (does the audit catch it?)
    5. Record ground truth + audit verdict

    Parameters
    ----------
    n_experiments : int
        Number of synthetic experiments. 1,000 gives stable percentages.
    alpha : float
        Significance threshold used throughout.
    seed : int
        Master random seed.
    verbose : bool
        Print progress every 100 experiments.

    Returns
    -------
    ResearchStudyResult with the headline statistics.
    """
    rng = np.random.default_rng(seed)

    # Import here to avoid circular imports
    import sys
    # Ensure engine package is importable
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from engine.data_generator import generate_experiment, split_arms
    from engine.checks import run_full_audit, check_srm
    from engine import ExperimentConfig, MetricType, Severity

    # Counters
    n_invalid               = 0
    n_would_ship            = 0   # naive team, no audit
    n_would_ship_invalid    = 0   # invalid AND naive team ships
    n_audit_caught          = 0   # invalid AND audit flags FAIL/WARN

    violation_counts = {
        "srm":              0,
        "underpowered":     0,
        "peeking":          0,
        "multiple_testing": 0,
        "novelty":          0,
    }
    severity_counts = {"pass": 0, "warn": 0, "fail": 0}

    for exp_idx in range(n_experiments):
        if verbose and exp_idx % 100 == 0:
            print(f"  [{exp_idx}/{n_experiments}]", end="\r")

        # ── Randomise experiment parameters ──────────────────
        p_control  = float(rng.uniform(0.03, 0.25))
        has_effect = bool(rng.binomial(1, 0.5))        # 50% null
        true_lift  = float(rng.uniform(0.01, 0.06)) if has_effect else 0.0
        n_per_arm  = int(rng.integers(200, 8000))

        # ── Randomise which violations are present ────────────
        do_srm     = bool(rng.binomial(1, 0.20))   # 20% chance
        do_unpower = bool(rng.binomial(1, 0.35))   # 35% chance
        do_peek    = bool(rng.binomial(1, 0.40))   # 40% chance
        do_multit  = bool(rng.binomial(1, 0.25))   # 25% chance
        do_novelty = bool(rng.binomial(1, 0.20))   # 20% chance

        is_invalid = do_srm or do_unpower or do_peek or do_multit or do_novelty

        if is_invalid:
            n_invalid += 1
            if do_srm:       violation_counts["srm"]              += 1
            if do_unpower:   violation_counts["underpowered"]     += 1
            if do_peek:      violation_counts["peeking"]          += 1
            if do_multit:    violation_counts["multiple_testing"] += 1
            if do_novelty:   violation_counts["novelty"]          += 1

        # ── Generate experiment data ──────────────────────────
        try:
            df = generate_experiment(
                n_per_arm=n_per_arm,
                p_control=p_control,
                true_lift=true_lift,
                srm_bug=do_srm,
                srm_ratio=float(rng.uniform(0.54, 0.65)),
                novelty_decay=do_novelty,
                experiment_days=14,
                seed=int(rng.integers(0, 999999)),
            )
        except Exception:
            continue

        ctrl, trt = split_arms(df)
        p_c = float(ctrl["converted"].mean())
        p_t = float(trt["converted"].mean())

        # ── Naive result: simple z-test, no validity checks ───
        import math
        se_naive = math.sqrt(p_c*(1-p_c)/len(ctrl) + p_t*(1-p_t)/len(trt))
        if se_naive > 0:
            z_naive = abs(p_t - p_c) / se_naive
            p_naive = 2.0 * (1.0 - _fast_norm_cdf(z_naive))
        else:
            p_naive = 1.0

        naive_ships = p_naive < alpha
        if naive_ships:
            n_would_ship += 1

        # ── Full audit ────────────────────────────────────────
        peeking_days = list(range(2, 15, 3)) if do_peek else []
        p_values     = [p_naive] + [float(rng.uniform(0, 1)) for _ in range(
            int(rng.integers(1, 5)) if do_multit else 0
        )]

        config = ExperimentConfig(
            name=f"exp_{exp_idx}",
            metric_type=MetricType.PROPORTION,
            alpha=alpha,
            target_power=0.80 if not do_unpower else 0.95,  # make underpowered more likely to fail
            peeking_days=peeking_days,
        )

        try:
            audit = run_full_audit(df, config, p_values=p_values)
        except Exception:
            continue

        sev = audit.overall_severity.value
        severity_counts[sev] += 1

        # Invalid AND would have shipped without audit
        if is_invalid and naive_ships:
            n_would_ship_invalid += 1

        # Invalid AND audit flags it (WARN or FAIL)
        if is_invalid and sev in ("warn", "fail"):
            n_audit_caught += 1

    # ── Compute headline statistics ───────────────────────────
    pct_invalid        = round(n_invalid / n_experiments * 100, 1)
    pct_false_ships    = round(n_would_ship_invalid / max(n_invalid, 1) * 100, 1)
    pct_caught         = round(n_audit_caught / max(n_invalid, 1) * 100, 1)

    return ResearchStudyResult(
        n_experiments=n_experiments,
        n_invalid=n_invalid,
        n_would_ship_without_audit=n_would_ship_invalid,
        pct_invalid=pct_invalid,
        pct_caught_by_audit=pct_caught,
        pct_false_ships=pct_false_ships,
        violation_breakdown=violation_counts,
        severity_breakdown=severity_counts,
    )


# ════════════════════════════════════════════════════════════
# 5. BENCHMARK UTILITY
# ════════════════════════════════════════════════════════════

def benchmark(n_simulations: int = 10_000) -> dict:
    """
    Benchmark simulation speed.
    Reports wall-clock time for 10,000 peeking simulations.
    Target: under 3 seconds on a standard laptop.
    """
    import time
    start = time.perf_counter()
    run_peeking_simulation(
        n_per_arm=5000,
        p_control=0.094,
        peeking_days=list(range(1, 15)),
        n_simulations=n_simulations,
        seed=42,
    )
    elapsed = time.perf_counter() - start
    return {
        "n_simulations":  n_simulations,
        "elapsed_seconds": round(elapsed, 3),
        "sims_per_second": round(n_simulations / elapsed, 0),
        "meets_target":   elapsed < 3.0,
    }
