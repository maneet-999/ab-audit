"""
data_generator.py
=================
Synthetic A/B experiment data generator.

Generates realistic experiment DataFrames with known ground truth —
used for unit testing all 8 checks and for the research simulation study.

Key design decisions
--------------------
- Every function accepts a `seed` for full reproducibility.
- SRM bugs, peeking data, and novelty effects can all be injected
  as optional parameters — so we can test each check in isolation.
- The generated DataFrame schema is fixed and matches what the
  Streamlit file uploader will accept from real data.

DataFrame Schema (output of generate_experiment)
-------------------------------------------------
user_id      : int       — unique user identifier
arm          : str       — "control" or "treatment"
converted    : int       — 0 or 1 (for proportion metrics)
metric_value : float     — raw metric (revenue, time, score)
day          : int       — experiment day (1-indexed)
pre_metric   : float     — same metric measured pre-experiment (for CUPED)
"""

import numpy as np
import pandas as pd
from typing import Optional


# --------------------------------------------------
# Main generator
# --------------------------------------------------

def generate_experiment(
    n_per_arm:       int   = 5_000,
    p_control:       float = 0.094,       # 9.4% baseline (Zepto add-to-cart)
    true_lift:       float = 0.018,       # True treatment effect in pp
    metric_type:     str   = "proportion",
    # Bugs / violations — inject these to test checks
    srm_bug:         bool  = False,       # Skew assignment to 55/45 instead of 50/50
    srm_ratio:       float = 0.55,        # How skewed when srm_bug=True
    novelty_decay:   bool  = False,       # Lift fades over time (novelty effect)
    experiment_days: int   = 14,
    # Continuous metric params (used when metric_type="continuous")
    mu_control:      float = 340.0,       # Mean revenue in control (Rs)
    sigma_control:   float = 180.0,       # Std dev of revenue
    true_diff:       float = 15.0,        # True mean difference for continuous
    unequal_variance:bool  = False,       # Inject Levene violation
    # Reproducibility
    seed:            Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic A/B experiment DataFrame with known ground truth.

    Parameters
    ----------
    n_per_arm : int
        Number of users assigned to each arm (control and treatment).
        Total experiment size = 2 * n_per_arm.

    p_control : float
        True conversion rate in the control arm (proportion metrics).
        Realistic Indian e-commerce baseline: 0.03 – 0.12.

    true_lift : float
        True treatment effect in absolute percentage points.
        Set to 0.0 to generate a pure null experiment.

    metric_type : str
        "proportion" → Bernoulli outcomes (click, convert, signup)
        "continuous" → Normal outcomes (revenue, time-on-site, score)

    srm_bug : bool
        If True, assign srm_ratio to control instead of 0.5.
        Injects a Sample Ratio Mismatch for Check 1 testing.

    novelty_decay : bool
        If True, treatment lift decreases linearly across days.
        Day 1 lift = 2 × true_lift, Day N lift → 0. Tests Check 7.

    experiment_days : int
        How many days the experiment ran. Used for day-level data.

    seed : int or None
        Random seed. None = non-reproducible.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, arm, converted, metric_value, day, pre_metric
    """
    rng = np.random.default_rng(seed)

    # ── Determine per-arm sample sizes ─────────────────────────
    if srm_bug:
        # Deliberately skew the split to simulate a pipeline bug
        n_control   = int(n_per_arm * 2 * srm_ratio)
        n_treatment = int(n_per_arm * 2 * (1 - srm_ratio))
    else:
        n_control   = n_per_arm
        n_treatment = n_per_arm

    n_total = n_control + n_treatment

    # ── Assign users to arms ────────────────────────────────────
    arms = np.array(["control"] * n_control + ["treatment"] * n_treatment)
    user_ids = np.arange(1, n_total + 1)

    # ── Assign days (uniform across experiment window) ──────────
    days = rng.integers(1, experiment_days + 1, size=n_total)

    # ── Generate pre-experiment metric (for CUPED) ──────────────
    # Pre-metric correlates with the outcome (r ≈ 0.4–0.6)
    pre_metric = rng.normal(
        loc=mu_control if metric_type == "continuous" else p_control * 10,
        scale=sigma_control * 0.3 if metric_type == "continuous" else 1.2,
        size=n_total
    )

    # ── Generate primary metric ─────────────────────────────────
    if metric_type == "proportion":
        converted, metric_value = _generate_proportion(
            rng, n_control, n_treatment, days, experiment_days,
            p_control, true_lift, novelty_decay, pre_metric
        )

    elif metric_type == "continuous":
        converted, metric_value = _generate_continuous(
            rng, n_control, n_treatment,
            mu_control, sigma_control, true_diff, unequal_variance
        )

    else:
        raise ValueError(f"metric_type must be 'proportion' or 'continuous', got '{metric_type}'")

    # ── Assemble DataFrame ──────────────────────────────────────
    df = pd.DataFrame({
        "user_id":      user_ids,
        "arm":          arms,
        "converted":    converted,
        "metric_value": metric_value,
        "day":          days,
        "pre_metric":   np.clip(pre_metric, 0, None),
    })

    return df


# --------------------------------------------------
# Internal generators for each metric type
# --------------------------------------------------

def _generate_proportion(rng, n_control, n_treatment, days, experiment_days,
                         p_control, true_lift, novelty_decay, pre_metric):
    """Generate Bernoulli outcomes for proportion metrics."""
    p_treatment = p_control + true_lift
    p_treatment = np.clip(p_treatment, 0.001, 0.999)

    if not novelty_decay:
        # Standard case: constant treatment effect
        converted_c = rng.binomial(1, p_control,   size=n_control)
        converted_t = rng.binomial(1, p_treatment, size=n_treatment)
    else:
        # Novelty effect: lift decays linearly from 2x on day 1 to 0 at end
        converted_c = rng.binomial(1, p_control, size=n_control)
        # Treatment lift is day-dependent
        day_scale = 1 - (days[n_control:] - 1) / max(experiment_days - 1, 1)
        day_scale = np.clip(day_scale, 0, 1)
        p_t_daily = np.clip(p_control + true_lift * 2 * day_scale, 0.001, 0.999)
        converted_t = rng.binomial(1, p_t_daily)

    converted    = np.concatenate([converted_c, converted_t]).astype(int)
    metric_value = converted.astype(float)

    return converted, metric_value


def _generate_continuous(rng, n_control, n_treatment,
                         mu_control, sigma_control, true_diff, unequal_variance):
    """Generate Normal outcomes for continuous metrics (e.g. revenue)."""
    if unequal_variance:
        # Inject variance heterogeneity — treatment group has 3x variance
        sigma_treatment = sigma_control * 3.0
    else:
        sigma_treatment = sigma_control

    metric_c = rng.normal(mu_control,             sigma_control,   size=n_control)
    metric_t = rng.normal(mu_control + true_diff, sigma_treatment, size=n_treatment)

    # For proportion columns, threshold at mean (not meaningful but keeps schema)
    converted_c = (metric_c > mu_control).astype(int)
    converted_t = (metric_t > mu_control).astype(int)

    metric_value = np.concatenate([metric_c, metric_t])
    converted    = np.concatenate([converted_c, converted_t]).astype(int)

    return converted, metric_value


# --------------------------------------------------
# Convenience: generate pre-built scenario data
# --------------------------------------------------

SCENARIOS = {
    "zepto_scarcity_badge": {
        "description": "Zepto: scarcity badge on product listing page",
        "n_per_arm":       5_000,
        "p_control":       0.094,
        "true_lift":       0.018,
        "srm_bug":         False,
        "novelty_decay":   False,
        "experiment_days": 7,
        "seed":            42,
    },
    "fintech_cashback_srm": {
        "description": "Fintech: cashback offer — SRM bug injected",
        "n_per_arm":       3_000,
        "p_control":       0.062,
        "true_lift":       0.012,
        "srm_bug":         True,
        "srm_ratio":       0.57,
        "novelty_decay":   False,
        "experiment_days": 10,
        "seed":            7,
    },
    "edtech_notification": {
        "description": "EdTech: push notification format — clean experiment",
        "n_per_arm":       8_000,
        "p_control":       0.210,
        "true_lift":       0.025,
        "srm_bug":         False,
        "novelty_decay":   False,
        "experiment_days": 14,
        "seed":            99,
    },
    "null_experiment": {
        "description": "Null experiment — true lift = 0, for peeking demo",
        "n_per_arm":       5_000,
        "p_control":       0.094,
        "true_lift":       0.000,
        "srm_bug":         False,
        "novelty_decay":   False,
        "experiment_days": 14,
        "seed":            123,
    },
    "novelty_effect": {
        "description": "EdTech tooltip — lift fades after day 3",
        "n_per_arm":       6_000,
        "p_control":       0.170,
        "true_lift":       0.020,
        "srm_bug":         False,
        "novelty_decay":   True,
        "experiment_days": 14,
        "seed":            55,
    },
}


def load_scenario(name: str) -> pd.DataFrame:
    """
    Generate data for one of the three pre-built demo scenarios.

    Parameters
    ----------
    name : str
        One of: "zepto_scarcity_badge", "fintech_cashback_srm",
                "edtech_notification", "null_experiment", "novelty_effect"

    Returns
    -------
    pd.DataFrame — same schema as generate_experiment()
    """
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )
    params = {k: v for k, v in SCENARIOS[name].items() if k != "description"}
    return generate_experiment(**params)


def split_arms(df: pd.DataFrame) -> tuple:
    """
    Split a full experiment DataFrame into control and treatment DataFrames.
    Convenience function for passing to individual checks.

    Returns
    -------
    (df_control, df_treatment) : tuple of DataFrames
    """
    return (
        df[df["arm"] == "control"].copy(),
        df[df["arm"] == "treatment"].copy(),
    )


# --------------------------------------------------
# Quick summary stats helper
# --------------------------------------------------

def experiment_summary(df: pd.DataFrame) -> dict:
    """
    Return a concise summary of an experiment DataFrame.
    Used for the Experiment Designer page header.
    """
    ctrl, trt = split_arms(df)
    return {
        "n_total":            len(df),
        "n_control":          len(ctrl),
        "n_treatment":        len(trt),
        "split_ratio":        round(len(ctrl) / len(df), 4),
        "p_control":          round(ctrl["converted"].mean(), 4),
        "p_treatment":        round(trt["converted"].mean(), 4),
        "observed_lift_pp":   round((trt["converted"].mean() - ctrl["converted"].mean()) * 100, 3),
        "n_days":             int(df["day"].max()),
        "mean_metric_ctrl":   round(ctrl["metric_value"].mean(), 2),
        "mean_metric_trt":    round(trt["metric_value"].mean(), 2),
    }
