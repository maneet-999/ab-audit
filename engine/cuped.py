"""
cuped.py
========
CUPED — Controlled-experiment Using Pre-Experiment Data.
(Deng, Xu, Kohavi & Walker, KDD 2013)

Core idea
---------
Users vary naturally in their baseline behaviour. This variance is
unrelated to the treatment, but inflates the variance of our estimator,
making experiments less sensitive.

CUPED removes this noise by adjusting each user's outcome by how much
their pre-experiment behaviour predicts it:

    Ỹ_i = Y_i - θ × (X_i - X̄)

where:
    Y_i  = outcome metric during the experiment
    X_i  = same metric in a pre-experiment window (e.g. prior week)
    X̄    = mean of X across all users
    θ    = Cov(Y, X) / Var(X)   ← minimises Var(Ỹ)

Variance reduction:
    Var(Ỹ) = Var(Y) × (1 - ρ²)

where ρ = Corr(Y, X).  If ρ = 0.5 → 25% variance reduction.

The adjusted estimator is unbiased because:
    E[X_i - X̄] = 0 under randomisation (pre-experiment data
    cannot be affected by a treatment that hasn't happened yet).

This is equivalent to ANCOVA: regress Y on treatment + X,
the OLS coefficient on treatment equals the CUPED estimate.

Public API
----------
cuped_adjust()       Core adjustment — returns adjusted metrics
run_cuped_analysis() Full analysis with before/after comparison
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════
# Result dataclass
# ════════════════════════════════════════════════════════════

@dataclass
class CUPEDResult:
    """
    Full output of run_cuped_analysis().

    Fields
    ------
    theta : float
        Covariate coefficient — Cov(Y, X) / Var(X).
        Tells you how strongly pre-experiment behaviour
        predicts experiment outcome.

    rho : float
        Pearson correlation between X and Y (across all users).
        The key driver of variance reduction: reduction = rho².

    variance_reduction_pct : float
        Percentage reduction in outcome variance: rho² × 100.

    sample_size_equivalent : float
        The CUPED-adjusted experiment is equivalent to running
        the original experiment with this many users per arm.
        = original_n / (1 - rho²)

    # Before CUPED
    lift_unadjusted : float
        Raw treatment - control mean difference.
    se_unadjusted : float
        Standard error of the unadjusted estimator.
    ci_unadjusted : tuple
        95% CI on the unadjusted lift.
    p_unadjusted : float
        Two-tailed p-value, unadjusted.

    # After CUPED
    lift_adjusted : float
        CUPED-adjusted treatment effect estimate.
    se_adjusted : float
        Standard error of the adjusted estimator (narrower).
    ci_adjusted : tuple
        95% CI on the adjusted lift (tighter).
    p_adjusted : float
        Two-tailed p-value, adjusted (typically smaller).

    # Diagnostics
    ci_width_reduction_pct : float
        How much narrower the CI became after CUPED.
    significant_before : bool
    significant_after  : bool
    """
    # Covariate info
    theta:              float
    rho:                float
    variance_reduction_pct: float
    sample_size_equivalent: float

    # Before CUPED
    lift_unadjusted:    float
    se_unadjusted:      float
    ci_unadjusted:      tuple
    p_unadjusted:       float

    # After CUPED
    lift_adjusted:      float
    se_adjusted:        float
    ci_adjusted:        tuple
    p_adjusted:         float

    # Diagnostics
    ci_width_reduction_pct: float
    significant_before: bool
    significant_after:  bool
    alpha:              float = 0.05

    # Raw data for plotting
    ctrl_Y:   list = field(default_factory=list)
    trt_Y:    list = field(default_factory=list)
    ctrl_Y_adj: list = field(default_factory=list)
    trt_Y_adj:  list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"── CUPED Analysis ──────────────────────────",
            f"  Pre/post correlation ρ  : {self.rho:+.4f}",
            f"  Variance reduction      : {self.variance_reduction_pct:.1f}%",
            f"  Sample size equivalent  : {self.sample_size_equivalent:.0f} users/arm",
            f"",
            f"  Before CUPED:",
            f"    Lift   : {self.lift_unadjusted*100:+.3f} pp",
            f"    SE     : {self.se_unadjusted*100:.4f} pp",
            f"    95% CI : [{self.ci_unadjusted[0]*100:+.3f}, {self.ci_unadjusted[1]*100:+.3f}] pp",
            f"    p-val  : {self.p_unadjusted:.4f}  {'✅ Significant' if self.significant_before else '❌ Not significant'}",
            f"",
            f"  After CUPED:",
            f"    Lift   : {self.lift_adjusted*100:+.3f} pp",
            f"    SE     : {self.se_adjusted*100:.4f} pp",
            f"    95% CI : [{self.ci_adjusted[0]*100:+.3f}, {self.ci_adjusted[1]*100:+.3f}] pp",
            f"    p-val  : {self.p_adjusted:.4f}  {'✅ Significant' if self.significant_after else '❌ Not significant'}",
            f"",
            f"  CI width reduction      : {self.ci_width_reduction_pct:.1f}%",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# CORE MATH
# ════════════════════════════════════════════════════════════

def _compute_theta(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Compute the CUPED covariate coefficient θ = Cov(Y, X) / Var(X).

    This minimises the variance of Ỹ = Y - θ(X - X̄).
    Equivalent to OLS coefficient of X in a regression of Y on X.

    Parameters
    ----------
    Y : array-like
        Outcome metric during the experiment (all users, both arms).
    X : array-like
        Pre-experiment metric (same users, same metric, prior window).

    Returns
    -------
    float : the optimal θ
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)

    var_X = float(np.var(X, ddof=1))
    if var_X < 1e-10:
        # No variance in covariate — CUPED provides no benefit
        return 0.0

    cov_YX = float(np.cov(Y, X, ddof=1)[0, 1])
    return cov_YX / var_X


def _cuped_adjust_arm(Y: np.ndarray, X: np.ndarray,
                       theta: float, X_bar: float) -> np.ndarray:
    """
    Apply CUPED adjustment to one arm.

        Ỹ_i = Y_i - θ × (X_i - X̄)

    Parameters
    ----------
    Y     : outcome metrics for this arm
    X     : pre-experiment metrics for this arm
    theta : global θ (computed across BOTH arms combined)
    X_bar : global mean of X (computed across BOTH arms combined)

    Note: θ and X̄ must be computed on the combined dataset,
    not per-arm — otherwise the adjustment introduces bias.
    """
    return Y - theta * (X - X_bar)


def _two_sample_z_test(y1: np.ndarray,
                        y2: np.ndarray) -> tuple:
    """
    Two-sample z-test (or Welch-style t-test for continuous).
    Returns (lift, se, ci_lower, ci_upper, p_value, z_stat).

    Uses Welch's unpooled SE for generality.
    """
    n1, n2   = len(y1), len(y2)
    mean1    = float(y1.mean())
    mean2    = float(y2.mean())
    lift     = mean2 - mean1   # treatment - control

    se = float(np.sqrt(
        np.var(y1, ddof=1) / n1 +
        np.var(y2, ddof=1) / n2
    ))

    if se < 1e-10:
        return lift, se, lift, lift, 1.0, 0.0

    z_crit = 1.959964   # z_{0.025}
    ci_lo  = lift - z_crit * se
    ci_hi  = lift + z_crit * se

    z_stat = lift / se
    # Two-tailed p-value
    from scipy.special import ndtr
    p_val  = float(2.0 * (1.0 - ndtr(abs(z_stat))))

    return lift, se, ci_lo, ci_hi, p_val, z_stat


# ════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════

def cuped_adjust(
    Y_control:   np.ndarray,
    Y_treatment: np.ndarray,
    X_control:   np.ndarray,
    X_treatment: np.ndarray,
) -> tuple:
    """
    Core CUPED adjustment — returns adjusted metric arrays.

    Computes θ on the combined dataset, then adjusts each arm.

    Parameters
    ----------
    Y_control, Y_treatment : array-like
        Outcome metric during the experiment.
    X_control, X_treatment : array-like
        Pre-experiment metric (same metric, prior window).

    Returns
    -------
    (Y_ctrl_adj, Y_trt_adj, theta, rho) : tuple
        Adjusted outcome arrays + the coefficients.

    Example
    -------
    >>> Y_c_adj, Y_t_adj, theta, rho = cuped_adjust(
    ...     Y_control, Y_treatment, X_control, X_treatment
    ... )
    >>> lift_adj = Y_t_adj.mean() - Y_c_adj.mean()
    """
    Yc = np.asarray(Y_control,   dtype=float)
    Yt = np.asarray(Y_treatment, dtype=float)
    Xc = np.asarray(X_control,   dtype=float)
    Xt = np.asarray(X_treatment, dtype=float)

    # ── Compute θ and X̄ on combined data ─────────────────────
    Y_all = np.concatenate([Yc, Yt])
    X_all = np.concatenate([Xc, Xt])

    theta = _compute_theta(Y_all, X_all)
    X_bar = float(X_all.mean())

    # ── Correlation (determines variance reduction) ───────────
    if np.std(Y_all) < 1e-10 or np.std(X_all) < 1e-10:
        rho = 0.0
    else:
        rho = float(np.corrcoef(Y_all, X_all)[0, 1])

    # ── Apply adjustment to each arm ──────────────────────────
    Yc_adj = _cuped_adjust_arm(Yc, Xc, theta, X_bar)
    Yt_adj = _cuped_adjust_arm(Yt, Xt, theta, X_bar)

    return Yc_adj, Yt_adj, theta, rho


def run_cuped_analysis(
    df:    pd.DataFrame,
    alpha: float = 0.05,
    Y_col: str   = "metric_value",
    X_col: str   = "pre_metric",
) -> CUPEDResult:
    """
    Full CUPED analysis with before/after comparison.

    Takes the standard experiment DataFrame (from data_generator or
    real CSV upload) and returns a CUPEDResult with all statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: arm, {Y_col}, {X_col}.
        arm values must be "control" and "treatment".
    alpha : float
        Significance level. Default 0.05.
    Y_col : str
        Name of the outcome metric column. Default "metric_value".
    X_col : str
        Name of the pre-experiment covariate column. Default "pre_metric".

    Returns
    -------
    CUPEDResult

    Raises
    ------
    ValueError if required columns are missing.

    Notes
    -----
    The X (pre-experiment) column should be:
    - The same metric as Y, measured before the experiment started
    - Available for all users (no missingness)
    - Correlated with Y (ρ > 0.2 for meaningful reduction)

    If ρ ≈ 0, CUPED provides no benefit and the adjusted/unadjusted
    results will be nearly identical.
    """
    # ── Validate inputs ───────────────────────────────────────
    required = {"arm", Y_col, X_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    ctrl = df[df["arm"] == "control"].copy()
    trt  = df[df["arm"] == "treatment"].copy()

    if len(ctrl) == 0 or len(trt) == 0:
        raise ValueError("Both 'control' and 'treatment' arms must be present.")

    Yc = ctrl[Y_col].values.astype(float)
    Yt = trt[Y_col].values.astype(float)
    Xc = ctrl[X_col].values.astype(float)
    Xt = trt[X_col].values.astype(float)

    # ── Unadjusted analysis ───────────────────────────────────
    lift_u, se_u, ci_lo_u, ci_hi_u, p_u, _ = _two_sample_z_test(Yc, Yt)

    # ── CUPED adjustment ──────────────────────────────────────
    Yc_adj, Yt_adj, theta, rho = cuped_adjust(Yc, Yt, Xc, Xt)

    # ── Adjusted analysis ─────────────────────────────────────
    lift_a, se_a, ci_lo_a, ci_hi_a, p_a, _ = _two_sample_z_test(Yc_adj, Yt_adj)

    # ── Derived statistics ────────────────────────────────────
    var_reduction_pct   = round(rho ** 2 * 100, 2)
    n_arm               = (len(Yc) + len(Yt)) // 2
    sample_size_equiv   = round(n_arm / max(1 - rho**2, 1e-6), 1)

    ci_width_before     = ci_hi_u - ci_lo_u
    ci_width_after      = ci_hi_a - ci_lo_a
    ci_width_reduction  = round(
        (1 - ci_width_after / max(ci_width_before, 1e-10)) * 100, 2
    )

    return CUPEDResult(
        theta=round(theta, 6),
        rho=round(rho, 4),
        variance_reduction_pct=var_reduction_pct,
        sample_size_equivalent=sample_size_equiv,
        lift_unadjusted=round(lift_u, 6),
        se_unadjusted=round(se_u, 6),
        ci_unadjusted=(round(ci_lo_u, 6), round(ci_hi_u, 6)),
        p_unadjusted=round(p_u, 6),
        lift_adjusted=round(lift_a, 6),
        se_adjusted=round(se_a, 6),
        ci_adjusted=(round(ci_lo_a, 6), round(ci_hi_a, 6)),
        p_adjusted=round(p_a, 6),
        ci_width_reduction_pct=ci_width_reduction,
        significant_before=(p_u < alpha),
        significant_after=(p_a < alpha),
        alpha=alpha,
        ctrl_Y=Yc.tolist(),
        trt_Y=Yt.tolist(),
        ctrl_Y_adj=Yc_adj.tolist(),
        trt_Y_adj=Yt_adj.tolist(),
    )


# ════════════════════════════════════════════════════════════
# VALIDATION UTILITY
# ════════════════════════════════════════════════════════════

def validate_against_ancova(
    df:    pd.DataFrame,
    Y_col: str = "metric_value",
    X_col: str = "pre_metric",
) -> dict:
    """
    Cross-validate CUPED against OLS ANCOVA.

    CUPED is theoretically equivalent to:
        Y ~ treatment + X   (OLS regression)

    The treatment coefficient from OLS should equal the CUPED
    lift estimate. This function checks that equality.

    Used in test_cuped.py to verify our implementation is correct.

    Returns
    -------
    dict with keys: cuped_lift, ancova_lift, difference, match (bool)
    """
    from scipy import stats as sp

    ctrl = df[df["arm"] == "control"]
    trt  = df[df["arm"] == "treatment"]

    Yc = ctrl[Y_col].values.astype(float)
    Yt = trt[Y_col].values.astype(float)
    Xc = ctrl[X_col].values.astype(float)
    Xt = trt[X_col].values.astype(float)

    # ── CUPED estimate ────────────────────────────────────────
    Yc_adj, Yt_adj, theta, rho = cuped_adjust(Yc, Yt, Xc, Xt)
    cuped_lift = float(Yt_adj.mean() - Yc_adj.mean())

    # ── ANCOVA estimate ───────────────────────────────────────
    # OLS: Y = β₀ + β₁×treatment + β₂×X
    Y_all  = np.concatenate([Yc, Yt])
    X_all  = np.concatenate([Xc, Xt])
    T_all  = np.concatenate([
        np.zeros(len(Yc)),
        np.ones(len(Yt))
    ])
    # Design matrix [intercept, treatment, X]
    design = np.column_stack([np.ones(len(Y_all)), T_all, X_all])
    # OLS: β = (X'X)⁻¹ X'y
    try:
        beta    = np.linalg.lstsq(design, Y_all, rcond=None)[0]
        ancova_lift = float(beta[1])   # coefficient on treatment
    except Exception:
        ancova_lift = float("nan")

    diff  = abs(cuped_lift - ancova_lift)
    match = diff < 1e-4 or (
        abs(ancova_lift) > 1e-10 and diff / abs(ancova_lift) < 0.01
    )

    return {
        "cuped_lift":  round(cuped_lift,  6),
        "ancova_lift": round(ancova_lift, 6),
        "difference":  round(diff, 8),
        "match":       match,
        "theta":       round(theta, 6),
        "rho":         round(rho, 4),
    }


__all__ = [
    "CUPEDResult",
    "cuped_adjust",
    "run_cuped_analysis",
    "validate_against_ancova",
]
