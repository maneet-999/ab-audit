"""
checks.py
=========
All 8 statistical validity checks for the AB Audit Engine.

Every function returns a CheckResult (see engine/__init__.py).
Core math is implemented from scratch in NumPy.
SciPy is used ONLY for validation — not as the primary logic.

Check inventory
---------------
1. check_srm()               Sample Ratio Mismatch
2. check_power()             Statistical Power
3. check_variance()          Variance Homogeneity (Levene → Welch)
4. check_normality()         Normality (Shapiro-Wilk → Mann-Whitney)
5. check_multiple_testing()  Multiple Testing (FWER, Bonferroni, BH)
6. check_peeking()           Peeking / Optional Stopping (Monte Carlo)
7. check_novelty()           Novelty Effect (linear regression on daily lift)
8. check_sutva()             SUTVA / Network Effects (heuristic)

run_full_audit()             Runs all 8, returns AuditResult
"""

import numpy as np
import pandas as pd
from typing import Optional
from engine import CheckResult, AuditResult, ExperimentConfig, Severity, MetricType


# ════════════════════════════════════════════════════════════
# MATH PRIMITIVES  (implemented from scratch — no scipy)
# ════════════════════════════════════════════════════════════

def _normal_cdf(z: float) -> float:
    """
    Standard normal CDF via Horner-form rational approximation.
    Abramowitz & Stegun 26.2.17 — max error < 7.5e-8.
    """
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530
          + t * (-0.356563782
          + t * (1.781477937
          + t * (-1.821255978
          + t *  1.330274429))))
    p = 1.0 - 0.3989422804 * np.exp(-0.5 * z * z) * poly
    return p if z >= 0 else 1.0 - p


def _normal_pdf(z: float) -> float:
    return 0.3989422804 * np.exp(-0.5 * z * z)


def _chi2_cdf(x: float, df: int) -> float:
    """
    Chi-square CDF via regularised lower incomplete gamma function.
    Uses the series expansion for small x, continued fraction for large x.
    Only needed for df=1 and df=k-1 (small values in our context).
    """
    if x <= 0:
        return 0.0
    return _regularised_gamma_lower(df / 2.0, x / 2.0)


def _regularised_gamma_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    # Series: P(a,x) = e^(-x) * x^a * Σ x^n / Γ(a+n+1)
    MAX_ITER = 200
    TOL      = 1e-10
    term = 1.0 / a
    total = term
    for n in range(1, MAX_ITER):
        term *= x / (a + n)
        total += term
        if abs(term) < TOL * abs(total):
            break
    log_gamma_a = _log_gamma(a)
    return np.exp(-x + a * np.log(x) - log_gamma_a) * total


def _log_gamma(z: float) -> float:
    """Lanczos approximation for log-Gamma, accurate to ~15 significant figures."""
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if z < 0.5:
        return np.log(np.pi) - np.log(abs(np.sin(np.pi * z))) - _log_gamma(1 - z)
    z -= 1
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)


def _t_cdf(t: float, df: float) -> float:
    """
    Student's t CDF via regularised incomplete beta function.
    P(T ≤ t | df) for two-tailed use: p_two = 2 * min(cdf, 1-cdf).
    """
    x = df / (df + t * t)
    ib = _regularised_beta(df / 2.0, 0.5, x)
    p = 0.5 * ib
    return p if t < 0 else 1.0 - p


def _regularised_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) via continued fraction (Lentz)."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    lbeta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
    front = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta) / a
    # Continued fraction via modified Lentz
    TINY = 1e-30
    MAX_ITER = 200
    TOL = 1e-10
    f = TINY
    C = f
    D = 0.0
    for m in range(0, MAX_ITER + 1):
        for step in range(2):
            if step == 0:
                if m == 0:
                    num = 1.0
                else:
                    num = -(a + m - 1) * (a + b + m - 1) * x / ((a + 2*m - 2) * (a + 2*m - 1))
            else:
                num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
            D = 1.0 + num * D
            if abs(D) < TINY: D = TINY
            C = 1.0 + num / C
            if abs(C) < TINY: C = TINY
            D = 1.0 / D
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < TOL:
                return front * f
    return front * f


def _two_tailed_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def _two_tailed_p_from_t(t: float, df: float) -> float:
    return 2.0 * min(_t_cdf(t, df), 1.0 - _t_cdf(t, df))


def _z_from_alpha_power(alpha: float, power: float) -> tuple:
    """
    Return (z_alpha2, z_beta) critical values for power formula.
    Inverse normal via Newton-Raphson on our own _normal_cdf.
    Accurate to ~1e-9.
    """
    import math

    def inv_norm(p, tol=1e-9, max_iter=60):
        if p <= 0: return -8.0
        if p >= 1: return  8.0
        # Seed: log-based approximation
        if abs(p - 0.5) < 0.49:
            x = (p - 0.5) * 2.5
        else:
            x = math.copysign(
                math.sqrt(-2.0 * math.log(min(p, 1.0 - p))),
                p - 0.5
            )
        for _ in range(max_iter):
            fx  = _normal_cdf(x) - p
            dfx = math.exp(-0.5 * x * x) * 0.3989422804   # φ(x)
            if dfx < 1e-15:
                break
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        return x

    z_alpha2 = inv_norm(1.0 - alpha / 2.0)
    z_beta   = inv_norm(power)
    return z_alpha2, z_beta


def _compute_power_proportion(n: int, p1: float, p2: float,
                               alpha: float = 0.05) -> float:
    """
    Exact power for two-proportion z-test (two-tailed).
    Uses the unpooled SE under H₁.

    Power = Φ(z - z_{α/2}) + Φ(-z - z_{α/2})
    where z = |p2-p1| / SE_unpooled
    """
    if n <= 0 or p1 <= 0 or p2 <= 0:
        return 0.0
    z_alpha2 = 1.959964  # z_{0.025}
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    if se == 0:
        return 1.0
    ncp = abs(p2 - p1) / se           # non-centrality parameter
    power = (1.0 - _normal_cdf(z_alpha2 - ncp) +
             _normal_cdf(-z_alpha2 - ncp))
    return float(np.clip(power, 0.0, 1.0))


def _required_n_proportion(p1: float, delta: float,
                            alpha: float = 0.05,
                            power: float = 0.80) -> int:
    """
    Required sample size per arm for two-proportion z-test.
    n = (z_{α/2} + z_β)² × [p1(1-p1) + p2(1-p2)] / δ²
    """
    p2 = p1 + delta
    z_a, z_b = _z_from_alpha_power(alpha, power)
    num = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denom = delta ** 2
    return int(np.ceil(num / denom))


# ════════════════════════════════════════════════════════════
# CHECK 1 — SAMPLE RATIO MISMATCH (SRM)
# ════════════════════════════════════════════════════════════

def check_srm(
    n_control:      int,
    n_treatment:    int,
    expected_ratio: float = 0.5,
    alpha:          float = 0.01,   # stricter — SRM is almost always a bug
) -> CheckResult:
    """
    Check 1: Sample Ratio Mismatch

    Tests whether the observed assignment split matches the designed ratio
    using a chi-square goodness-of-fit test.

    A significant result (p < 0.01) almost always indicates a pipeline
    bug — not sampling noise — so we use a stricter threshold than usual.

    Parameters
    ----------
    n_control, n_treatment : int
        Observed user counts in each arm.
    expected_ratio : float
        Expected fraction in control. Default 0.5 (equal split).
    alpha : float
        Significance threshold. Default 0.01 (stricter than usual).

    Returns
    -------
    CheckResult with statistic = chi2, p_value = p
    """
    n_total = n_control + n_treatment
    expected_c = n_total * expected_ratio
    expected_t = n_total * (1.0 - expected_ratio)

    # Chi-square statistic (df = 1)
    chi2 = ((n_control   - expected_c) ** 2 / expected_c +
            (n_treatment - expected_t) ** 2 / expected_t)

    p_val = 1.0 - _chi2_cdf(chi2, df=1)

    actual_ratio = n_control / n_total
    passed = p_val >= alpha

    if passed:
        severity = Severity.PASS
        verdict  = (f"Randomisation looks clean — "
                    f"observed split {actual_ratio:.1%}/{1-actual_ratio:.1%} "
                    f"is consistent with expected {expected_ratio:.0%}/{1-expected_ratio:.0%}.")
        cost     = "N/A"
        rec      = "Proceed to the next check."
    else:
        severity = Severity.FAIL
        verdict  = (f"SRM detected — observed split {actual_ratio:.1%}/{1-actual_ratio:.1%} "
                    f"significantly differs from expected "
                    f"{expected_ratio:.0%}/{1-expected_ratio:.0%} "
                    f"(χ²={chi2:.3f}, p={p_val:.4f}).")
        cost     = ("Your confidence intervals cannot be trusted. "
                    "The skewed split indicates a randomisation pipeline bug — "
                    "bot filtering after assignment, logging errors, or "
                    "redirect/cache issues are common causes.")
        rec      = ("Do NOT ship based on this result. "
                    "Investigate your assignment pipeline. "
                    "Fix the bug and re-run the experiment from scratch.")

    return CheckResult(
        name="Sample Ratio Mismatch",
        severity=severity,
        statistic=round(chi2, 6),
        p_value=round(p_val, 6),
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "n_control":    n_control,
            "n_treatment":  n_treatment,
            "n_total":      n_total,
            "expected_c":   expected_c,
            "expected_t":   expected_t,
            "actual_ratio": round(actual_ratio, 4),
            "chi2":         round(chi2, 6),
            "p_value":      round(p_val, 6),
            "threshold":    alpha,
        }
    )


# ════════════════════════════════════════════════════════════
# CHECK 2 — STATISTICAL POWER
# ════════════════════════════════════════════════════════════

def check_power(
    n_control:     int,
    n_treatment:   int,
    p_control:     float,
    p_treatment:   float,
    alpha:         float = 0.05,
    target_power:  float = 0.80,
) -> CheckResult:
    """
    Check 2: Statistical Power

    Computes the achieved statistical power given observed sample sizes
    and estimated conversion rates. Flags if power < target (default 0.80).

    Also computes: required sample size for target power, and the
    minimum detectable effect at the observed sample size.

    Parameters
    ----------
    n_control, n_treatment : int
        Observed arm sizes.
    p_control, p_treatment : float
        Observed (or prior estimated) conversion rates.
    alpha : float
        Significance level.
    target_power : float
        Minimum acceptable power. Default 0.80.
    """
    n_arm = min(n_control, n_treatment)   # conservative: use smaller arm
    delta = abs(p_treatment - p_control)

    achieved_power = _compute_power_proportion(n_arm, p_control, p_treatment, alpha)

    # Required n for target power at observed delta
    if delta > 0:
        required_n = _required_n_proportion(p_control, delta, alpha, target_power)
    else:
        required_n = 999_999   # infinite if no effect assumed

    # Minimum detectable effect at observed n and target power
    # Solve numerically: find smallest delta s.t. power >= target_power
    mde = _compute_mde(n_arm, p_control, alpha, target_power)

    if achieved_power >= target_power:
        severity = Severity.PASS
        verdict  = (f"Experiment is well-powered — "
                    f"achieved power {achieved_power:.1%} ≥ target {target_power:.0%}.")
        cost     = "N/A"
        rec      = "Proceed."
    elif achieved_power >= 0.60:
        severity = Severity.WARN
        verdict  = (f"Borderline power — achieved {achieved_power:.1%}, "
                    f"target is {target_power:.0%}. "
                    f"You needed {required_n:,} users per arm; "
                    f"you ran {n_arm:,}.")
        cost     = (f"There was a {1-achieved_power:.0%} chance of missing a real "
                    f"effect of {delta*100:.2f} pp. Results may be a false negative.")
        rec      = (f"Treat result with caution. "
                    f"For future runs, collect {required_n:,} users per arm.")
    else:
        severity = Severity.FAIL
        verdict  = (f"Experiment is severely underpowered — "
                    f"achieved power {achieved_power:.1%}, "
                    f"target is {target_power:.0%}. "
                    f"Required {required_n:,} users per arm; ran {n_arm:,}.")
        cost     = (f"There was a {1-achieved_power:.0%} chance of missing a real "
                    f"effect of {delta*100:.2f} pp. "
                    f"A non-significant result here is uninformative — "
                    f"you may have just killed a feature that actually worked.")
        rec      = (f"Do not conclude there is no effect. "
                    f"Re-run with at least {required_n:,} users per arm.")

    return CheckResult(
        name="Statistical Power",
        severity=severity,
        statistic=round(achieved_power, 6),
        p_value=None,
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "achieved_power":  round(achieved_power, 4),
            "target_power":    target_power,
            "n_per_arm":       n_arm,
            "required_n":      required_n,
            "observed_delta":  round(delta, 6),
            "mde_pp":          round(mde * 100, 3),
            "p_control":       round(p_control, 4),
            "p_treatment":     round(p_treatment, 4),
        }
    )


def _compute_mde(n: int, p: float, alpha: float, power: float) -> float:
    """Binary search for minimum detectable effect."""
    lo, hi = 1e-6, min(p, 1 - p) - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if _compute_power_proportion(n, p, p + mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ════════════════════════════════════════════════════════════
# CHECK 3 — VARIANCE HOMOGENEITY
# ════════════════════════════════════════════════════════════

def check_variance(
    control_values:   np.ndarray,
    treatment_values: np.ndarray,
    alpha:            float = 0.05,
) -> CheckResult:
    """
    Check 3: Variance Homogeneity (Levene's Test)

    Tests whether control and treatment groups have equal variances.
    If violated, flags that the pooled t-test is inappropriate and
    Welch's correction should be used instead.

    Uses the mean-based Levene test statistic:
        W = (N-2) / (k-1) × [Σ nᵢ(Z̄ᵢ - Z̄)²] / [Σ Σ (Zᵢⱼ - Z̄ᵢ)²]
    where Zᵢⱼ = |Yᵢⱼ - Ȳᵢ|  (absolute deviations from group mean)
    """
    c = np.asarray(control_values,   dtype=float)
    t = np.asarray(treatment_values, dtype=float)

    n1, n2 = len(c), len(t)
    N  = n1 + n2
    k  = 2

    # Absolute deviations from group means (Levene's Z scores)
    z1 = np.abs(c - c.mean())
    z2 = np.abs(t - t.mean())

    z1_bar = z1.mean()
    z2_bar = z2.mean()
    z_bar  = np.concatenate([z1, z2]).mean()

    # Between-group sum of squares
    ssbg = n1 * (z1_bar - z_bar)**2 + n2 * (z2_bar - z_bar)**2

    # Within-group sum of squares
    sswg = np.sum((z1 - z1_bar)**2) + np.sum((z2 - z2_bar)**2)

    if sswg == 0:
        W = 0.0
    else:
        W = ((N - k) / (k - 1)) * ssbg / sswg

    df1, df2 = k - 1, N - k
    # F-distribution p-value via scipy (Levene is a flagging check,
    # not the core analysis — scipy use is acceptable here)
    try:
        from scipy.stats import f as f_dist
        p_val = float(f_dist.sf(W, df1, df2))
    except Exception:
        p_val = float(1.0 - _chi2_cdf(W * df1, df=df1)) if W > 0 else 1.0

    var_c = float(np.var(c, ddof=1))
    var_t = float(np.var(t, ddof=1))
    # Report ratio as max/min (always >= 1) for interpretability
    var_ratio = max(var_c, var_t) / max(min(var_c, var_t), 1e-10)
    equal_var = p_val >= alpha

    if equal_var:
        severity = Severity.PASS
        verdict  = (f"Variances are homogeneous — Levene W={W:.3f}, p={p_val:.4f}. "
                    f"Variance ratio {var_ratio:.2f}:1. "
                    f"Pooled t-test is appropriate.")
        cost     = "N/A"
        rec      = "Use the pooled t-test for continuous metrics."
    else:
        severity = Severity.WARN
        verdict  = (f"Variance heterogeneity detected — Levene W={W:.3f}, p={p_val:.4f}. "
                    f"Variance ratio {var_ratio:.2f}:1. "
                    f"The treatment group has substantially different spread.")
        cost     = ("The pooled t-test assumes equal variances. "
                    "Using it here inflates Type I error. "
                    "Welch's t-test is automatically applied in the main analysis.")
        rec      = ("Welch's t-test has been substituted. "
                    "Investigate why variance differs — outliers or "
                    "a segment responding very differently to the treatment.")

    return CheckResult(
        name="Variance Homogeneity",
        severity=severity,
        statistic=round(W, 6),
        p_value=round(p_val, 6),
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "levene_W":   round(W, 6),
            "p_value":    round(p_val, 6),
            "var_control":   round(var_c, 4),
            "var_treatment": round(var_t, 4),
            "var_ratio":     round(float(var_ratio), 4),
            "equal_var":     equal_var,
            "use_welch":     not equal_var,
        }
    )


# ════════════════════════════════════════════════════════════
# CHECK 4 — NORMALITY ASSESSMENT
# ════════════════════════════════════════════════════════════

def check_normality(
    control_values:   np.ndarray,
    treatment_values: np.ndarray,
    alpha:            float = 0.05,
    max_n_shapiro:    int   = 5_000,
) -> CheckResult:
    """
    Check 4: Normality Assessment (Shapiro-Wilk)

    Tests whether the metric distribution in each arm is approximately
    normal. If not, recommends Mann-Whitney U as a non-parametric
    alternative to the t-test.

    Note: CLT means normality matters less as n grows.
    For n > 5,000 the t-test is robust even to moderate skew.
    We apply a subsample for Shapiro-Wilk (which has O(n²) complexity).
    """
    c = np.asarray(control_values,   dtype=float)
    t = np.asarray(treatment_values, dtype=float)

    # Subsample for large n (Shapiro-Wilk is reliable up to ~5,000)
    rng  = np.random.default_rng(42)
    c_sw = rng.choice(c, size=min(len(c), max_n_shapiro), replace=False)
    t_sw = rng.choice(t, size=min(len(t), max_n_shapiro), replace=False)

    # Shapiro-Wilk via scipy (there is no clean from-scratch version
    # that's numerically stable — we use scipy here as the primary,
    # since this check is about flagging, not the core test statistic)
    try:
        from scipy.stats import shapiro
        W_c, p_c = shapiro(c_sw)
        W_t, p_t = shapiro(t_sw)
    except Exception:
        # Fallback: use skewness as a normality proxy
        W_c = float(1 - abs(_skewness(c_sw)) / 10)
        W_t = float(1 - abs(_skewness(t_sw)) / 10)
        p_c = 0.5 if abs(_skewness(c_sw)) < 0.5 else 0.01
        p_t = 0.5 if abs(_skewness(t_sw)) < 0.5 else 0.01

    normal_c = float(p_c) >= alpha
    normal_t = float(p_t) >= alpha
    both_normal = normal_c and normal_t

    n_c, n_t = len(c), len(t)
    clt_ok = n_c >= 30 and n_t >= 30   # CLT kicks in

    skew_c = round(float(_skewness(c)), 3)
    skew_t = round(float(_skewness(t)), 3)

    if both_normal or clt_ok:
        severity = Severity.PASS if both_normal else Severity.WARN
        if clt_ok and not both_normal:
            verdict = (f"Normality rejected (control p={p_c:.4f}, "
                       f"treatment p={p_t:.4f}) but n is large "
                       f"(n_ctrl={n_c:,}, n_trt={n_t:,}). "
                       f"CLT justifies the t-test.")
            cost = ("Mild concern. Heavy tails or extreme skew "
                    f"(skew_ctrl={skew_c}, skew_trt={skew_t}) "
                    "may still inflate variance estimates.")
            rec  = ("Consider a log-transform or trimmed mean "
                    "if skewness is extreme (|skew| > 2). "
                    "Mann-Whitney U is available as an alternative.")
        else:
            verdict = (f"Both arms are approximately normal "
                       f"(control W={W_c:.4f} p={p_c:.4f}, "
                       f"treatment W={W_t:.4f} p={p_t:.4f}). "
                       f"t-test assumptions are satisfied.")
            cost = "N/A"
            rec  = "Proceed with t-test or z-test."
    else:
        severity = Severity.WARN
        verdict  = (f"Normality rejected in one or both arms — "
                    f"control: W={W_c:.4f}, p={p_c:.4f}; "
                    f"treatment: W={W_t:.4f}, p={p_t:.4f}. "
                    f"Skewness: control={skew_c}, treatment={skew_t}.")
        cost     = ("The t-test p-value may be unreliable for small, "
                    "non-normal samples. Type I error could be inflated.")
        rec      = ("Use Mann-Whitney U test instead of t-test. "
                    "Mann-Whitney is valid without normality and nearly "
                    "as powerful for large samples.")

    return CheckResult(
        name="Normality Assessment",
        severity=severity,
        statistic=round(float(min(W_c, W_t)), 6),
        p_value=round(float(min(p_c, p_t)), 6),
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "shapiro_W_control":    round(float(W_c), 6),
            "shapiro_p_control":    round(float(p_c), 6),
            "shapiro_W_treatment":  round(float(W_t), 6),
            "shapiro_p_treatment":  round(float(p_t), 6),
            "skewness_control":     skew_c,
            "skewness_treatment":   skew_t,
            "normal_control":       normal_c,
            "normal_treatment":     normal_t,
            "clt_applies":          clt_ok,
            "recommend_mwu":        not (both_normal or clt_ok),
        }
    )


def _skewness(x: np.ndarray) -> float:
    """Fisher-Pearson skewness coefficient."""
    n  = len(x)
    if n < 3:
        return 0.0
    mu = x.mean()
    s  = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - mu) / s) ** 3))


# ════════════════════════════════════════════════════════════
# CHECK 5 — MULTIPLE TESTING
# ════════════════════════════════════════════════════════════

def check_multiple_testing(
    p_values:    list,
    alpha:       float = 0.05,
    method:      str   = "both",   # "bonferroni", "bh", or "both"
) -> CheckResult:
    """
    Check 5: Multiple Testing Correction

    When multiple hypotheses are tested simultaneously, the probability
    of at least one false positive (FWER) inflates rapidly:
        FWER = 1 - (1 - α)^k

    Applies:
    - Bonferroni correction: αᵢ = α / k  (controls FWER, conservative)
    - Benjamini-Hochberg:   step-up procedure (controls FDR, preferred)

    Parameters
    ----------
    p_values : list of float
        All p-values from this experiment's hypothesis tests.
        Include all metrics and subgroup analyses.
    alpha : float
        Nominal significance level.
    method : str
        Which correction to show: "bonferroni", "bh", or "both".
    """
    k = len(p_values)
    p = np.array(p_values, dtype=float)

    fwer = 1.0 - (1.0 - alpha) ** k

    # ── Bonferroni ───────────────────────────────────────────
    alpha_bonf       = alpha / k
    rejected_bonf    = p <= alpha_bonf
    n_sig_bonf       = int(rejected_bonf.sum())

    # ── Benjamini-Hochberg step-up ───────────────────────────
    order       = np.argsort(p)
    p_sorted    = p[order]
    bh_thresh   = (np.arange(1, k + 1) / k) * alpha
    # Find largest j where p_(j) <= (j/k)*alpha
    below       = p_sorted <= bh_thresh
    if below.any():
        cutoff_idx = int(np.where(below)[0].max())
    else:
        cutoff_idx = -1

    rejected_bh_sorted = np.zeros(k, dtype=bool)
    if cutoff_idx >= 0:
        rejected_bh_sorted[:cutoff_idx + 1] = True

    # Map back to original order
    rejected_bh = np.zeros(k, dtype=bool)
    rejected_bh[order] = rejected_bh_sorted
    n_sig_bh = int(rejected_bh.sum())

    n_sig_naive = int((p <= alpha).sum())

    if k == 1:
        severity = Severity.PASS
        verdict  = "Only one hypothesis tested — no multiple testing concern."
        cost     = "N/A"
        rec      = "No correction needed."
    elif fwer <= 0.10:
        severity = Severity.PASS
        verdict  = (f"{k} tests run. FWER = {fwer:.1%} — acceptable. "
                    f"Naive: {n_sig_naive} significant. "
                    f"After BH correction: {n_sig_bh} significant.")
        cost     = "N/A"
        rec      = "Apply BH correction as good practice regardless."
    elif fwer <= 0.30:
        severity = Severity.WARN
        verdict  = (f"{k} simultaneous tests at α={alpha} → "
                    f"FWER = {fwer:.1%}. "
                    f"Naive: {n_sig_naive} significant; "
                    f"after BH correction: {n_sig_bh}.")
        cost     = (f"You have a {fwer:.1%} chance of at least one false positive "
                    f"across these {k} tests — not {alpha:.0%}.")
        rec      = (f"Use BH-corrected results. "
                    f"{n_sig_naive - n_sig_bh} of your 'significant' results "
                    f"may be false positives.")
    else:
        severity = Severity.FAIL
        verdict  = (f"Severe multiple testing problem — "
                    f"{k} tests at α={alpha} → FWER = {fwer:.1%}. "
                    f"Naive: {n_sig_naive} significant; "
                    f"after correction: {n_sig_bh}.")
        cost     = (f"Your actual false positive rate is {fwer:.1%}, "
                    f"not {alpha:.0%}. Nearly guaranteed to have at least "
                    f"one spurious result.")
        rec      = (f"Use only BH-corrected results. "
                    f"Pre-register your primary metric next time to "
                    f"avoid the multiple testing trap.")

    return CheckResult(
        name="Multiple Testing",
        severity=severity,
        statistic=round(fwer, 6),
        p_value=None,
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "k":              k,
            "fwer":           round(fwer, 6),
            "alpha_nominal":  alpha,
            "alpha_bonf":     round(alpha_bonf, 6),
            "n_sig_naive":    n_sig_naive,
            "n_sig_bonf":     n_sig_bonf,
            "n_sig_bh":       n_sig_bh,
            "p_values":       [round(x, 6) for x in p_values],
            "rejected_bonf":  rejected_bonf.tolist(),
            "rejected_bh":    rejected_bh.tolist(),
            "bh_thresholds":  [round(x, 6) for x in bh_thresh.tolist()],
        }
    )


# ════════════════════════════════════════════════════════════
# CHECK 6 — PEEKING / OPTIONAL STOPPING
# ════════════════════════════════════════════════════════════

def check_peeking(
    peeking_days:     list,
    n_per_arm:        int,
    p_control:        float,
    alpha:            float = 0.05,
    n_simulations:    int   = 5_000,
    seed:             int   = 42,
) -> CheckResult:
    """
    Check 6: Peeking / Optional Stopping

    If a team checks results before the planned experiment end and stops
    when p < alpha, the true Type I error rate inflates significantly.

    This check:
    1. Asks whether the team peeked (via peeking_days parameter)
    2. Runs a Monte Carlo simulation of the exact peeking strategy
       under the NULL (no real effect) to estimate the actual α
    3. Reports the inflated false positive rate empirically

    The simulation IS the proof — we show it happening, not just
    assert it theoretically.
    """
    if not peeking_days:
        return CheckResult(
            name="Peeking / Optional Stopping",
            severity=Severity.PASS,
            statistic=alpha,
            p_value=None,
            verdict=(f"No early peeking reported. "
                     f"Fixed-horizon test at α={alpha} is valid."),
            cost_of_violation="N/A",
            recommendation="Continue using pre-planned sample size and end date.",
            details={"peeking_days": [], "n_peeks": 0,
                     "inflated_alpha": alpha, "n_simulations": 0}
        )

    rng = np.random.default_rng(seed)
    n_checks = len(peeking_days)

    # ── Monte Carlo under H₀ (no real effect) ───────────────
    # For each simulation: generate cumulative data at each peek day
    # and check if p < alpha at ANY check. If yes → false positive.

    false_positives = 0
    p_trajectory_sample = []   # store one sample trajectory for the UI

    # Precompute: max users needed
    max_n = n_per_arm  # we assume users accumulate linearly

    for sim_idx in range(n_simulations):
        fp_this_sim = False
        traj = []

        for i, day in enumerate(sorted(peeking_days)):
            # Users seen so far (proportional to day fraction)
            n_so_far = max(10, int(n_per_arm * day / max(peeking_days)))

            # Generate data under null: both arms have same p_control
            ctrl = rng.binomial(n_so_far, p_control)
            trt  = rng.binomial(n_so_far, p_control)   # NULL: same rate

            # Two-proportion z-test
            p_hat = (ctrl + trt) / (2 * n_so_far)
            se    = np.sqrt(2 * p_hat * (1 - p_hat) / n_so_far)
            if se > 0:
                z   = abs(ctrl / n_so_far - trt / n_so_far) / se
                pv  = _two_tailed_p_from_z(z)
            else:
                pv = 1.0

            traj.append(round(pv, 4))

            if pv < alpha and not fp_this_sim:
                fp_this_sim = True

        if fp_this_sim:
            false_positives += 1

        if sim_idx < 20:   # save first 20 trajectories for UI
            p_trajectory_sample.append(traj)

    inflated_alpha = false_positives / n_simulations
    inflation_factor = inflated_alpha / alpha

    if not peeking_days:
        severity = Severity.PASS
    elif inflated_alpha <= alpha * 1.2:
        severity = Severity.PASS
    elif inflated_alpha <= alpha * 2.0:
        severity = Severity.WARN
    else:
        severity = Severity.FAIL

    verdict = (f"Team peeked {n_checks} time(s) "
               f"(days: {sorted(peeking_days)}). "
               f"Monte Carlo simulation ({n_simulations:,} null experiments) "
               f"shows your actual false positive rate is "
               f"~{inflated_alpha:.1%} — "
               f"not the nominal {alpha:.0%}. "
               f"That is {inflation_factor:.1f}× inflation.")

    cost = (f"Your stated confidence level is {1-alpha:.0%}. "
            f"Your actual confidence level is ~{1-inflated_alpha:.0%}. "
            f"Every time you saw p < {alpha} while peeking, "
            f"there was a {inflated_alpha:.0%} chance it was a false alarm.")

    rec = ("Next experiment: pre-commit to a sample size AND an end date, "
           "then do not look until both are reached. "
           "If early stopping is needed, use a sequential test "
           "(O'Brien-Fleming boundaries or always-valid p-values).")

    return CheckResult(
        name="Peeking / Optional Stopping",
        severity=severity,
        statistic=round(inflated_alpha, 6),
        p_value=None,
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "peeking_days":          sorted(peeking_days),
            "n_peeks":               n_checks,
            "nominal_alpha":         alpha,
            "inflated_alpha":        round(inflated_alpha, 4),
            "inflation_factor":      round(inflation_factor, 2),
            "n_simulations":         n_simulations,
            "false_positives":       false_positives,
            "p_trajectory_sample":   p_trajectory_sample,
        }
    )


# ════════════════════════════════════════════════════════════
# CHECK 7 — NOVELTY EFFECT
# ════════════════════════════════════════════════════════════

def check_novelty(
    df:          pd.DataFrame,
    alpha:       float = 0.05,
    min_days:    int   = 5,
) -> CheckResult:
    """
    Check 7: Novelty Effect Detection

    If the treatment lift declines significantly over the experiment
    window, the observed effect may be inflated by novelty — users
    engage more simply because something is new, not because it's better.

    Detection: fit a linear regression of daily lift on day index.
    A significantly negative slope (p < alpha) flags novelty risk.

    The experiment needs at least min_days of daily data to run this check.

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment DataFrame with columns: arm, converted, day.
    alpha : float
        Significance level for the slope test.
    min_days : int
        Minimum days of data required to run this check.
    """
    required_cols = {"arm", "converted", "day"}
    if not required_cols.issubset(df.columns):
        return _novelty_insufficient("Missing required columns (arm, converted, day).")

    days = sorted(df["day"].unique())
    if len(days) < min_days:
        return _novelty_insufficient(
            f"Only {len(days)} days of data — need at least {min_days} to detect novelty."
        )

    # Compute daily lift: treatment rate - control rate per day
    daily = []
    for d in days:
        day_df = df[df["day"] == d]
        ctrl_d = day_df[day_df["arm"] == "control"]["converted"]
        trt_d  = day_df[day_df["arm"] == "treatment"]["converted"]
        if len(ctrl_d) > 0 and len(trt_d) > 0:
            daily.append({
                "day":  d,
                "lift": float(trt_d.mean() - ctrl_d.mean()),
                "n":    len(ctrl_d) + len(trt_d),
            })

    if len(daily) < min_days:
        return _novelty_insufficient(
            f"Insufficient days with data in both arms (found {len(daily)}, need {min_days})."
        )

    day_arr  = np.array([r["day"]  for r in daily], dtype=float)
    lift_arr = np.array([r["lift"] for r in daily], dtype=float)

    # Simple OLS: lift = β₀ + β₁ × day
    n_d  = len(day_arr)
    x_bar = day_arr.mean()
    y_bar = lift_arr.mean()

    Sxx = np.sum((day_arr  - x_bar) ** 2)
    Sxy = np.sum((day_arr  - x_bar) * (lift_arr - y_bar))

    if Sxx == 0:
        return _novelty_insufficient("All observations on the same day.")

    beta1 = Sxy / Sxx              # slope
    beta0 = y_bar - beta1 * x_bar  # intercept

    # Residuals and SE of slope
    y_hat = beta0 + beta1 * day_arr
    resid = lift_arr - y_hat
    s2    = np.sum(resid ** 2) / max(n_d - 2, 1)
    se_b1 = np.sqrt(s2 / Sxx) if Sxx > 0 else 1e-10

    t_stat = beta1 / se_b1 if se_b1 > 0 else 0.0
    p_val  = _two_tailed_p_from_t(t_stat, df=n_d - 2)

    # Novelty: significant NEGATIVE slope
    novelty_detected = (p_val < alpha) and (beta1 < 0)

    avg_lift_early = float(lift_arr[:len(lift_arr)//3].mean())
    avg_lift_late  = float(lift_arr[-(len(lift_arr)//3):].mean())

    if novelty_detected:
        severity = Severity.WARN
        verdict  = (f"Novelty effect likely — treatment lift declines over time. "
                    f"Slope β₁ = {beta1*100:+.3f} pp/day (p={p_val:.4f}). "
                    f"Early lift: {avg_lift_early*100:+.2f} pp → "
                    f"Late lift: {avg_lift_late*100:+.2f} pp.")
        cost     = ("The observed average lift may overstate the true long-run effect. "
                    "Users are engaging with the treatment because it is new, "
                    "not because it is better.")
        rec      = ("Do not ship yet. Extend the experiment until the lift stabilises. "
                    "Use the late-period lift as a more conservative estimate.")
    else:
        severity = Severity.PASS
        verdict  = (f"No significant novelty effect — "
                    f"lift is stable across the experiment window. "
                    f"Slope β₁ = {beta1*100:+.3f} pp/day (p={p_val:.4f}). "
                    f"Early: {avg_lift_early*100:+.2f} pp, "
                    f"Late: {avg_lift_late*100:+.2f} pp.")
        cost     = "N/A"
        rec      = "Observed lift appears stable — proceed."

    return CheckResult(
        name="Novelty Effect",
        severity=severity,
        statistic=round(float(t_stat), 6),
        p_value=round(float(p_val), 6),
        verdict=verdict,
        cost_of_violation=cost,
        recommendation=rec,
        details={
            "slope_beta1":      round(float(beta1), 8),
            "intercept_beta0":  round(float(beta0), 8),
            "se_slope":         round(float(se_b1), 8),
            "t_stat":           round(float(t_stat), 6),
            "p_value":          round(float(p_val), 6),
            "novelty_detected": novelty_detected,
            "avg_lift_early_pp": round(avg_lift_early * 100, 3),
            "avg_lift_late_pp":  round(avg_lift_late  * 100, 3),
            "daily_lifts":       [round(r["lift"] * 100, 3) for r in daily],
            "days":              [r["day"] for r in daily],
        }
    )


def _novelty_insufficient(reason: str) -> CheckResult:
    return CheckResult(
        name="Novelty Effect",
        severity=Severity.WARN,
        statistic=0.0,
        p_value=None,
        verdict=f"Could not run novelty check — {reason}",
        cost_of_violation="Cannot assess novelty risk without multi-day data.",
        recommendation="Ensure the experiment DataFrame includes a 'day' column "
                       "with at least 5 distinct days.",
        details={"reason": reason}
    )


# ════════════════════════════════════════════════════════════
# CHECK 8 — SUTVA / NETWORK EFFECTS
# ════════════════════════════════════════════════════════════

def check_sutva(
    is_social_feature:    bool = False,
    is_referral_feature:  bool = False,
    is_comms_feature:     bool = False,
    is_marketplace:       bool = False,
    n_variants:           int  = 2,
    uses_cluster_rand:    bool = False,
) -> CheckResult:
    """
    Check 8: SUTVA / Network Effect Assessment

    The Stable Unit Treatment Value Assumption (SUTVA) requires that
    the treatment of one unit does not affect the outcomes of another.
    This is violated whenever users interact — social features,
    referrals, communications, marketplaces.

    This check is a structured heuristic (no p-value) — it flags
    design problems that statistics alone cannot fix.

    Parameters
    ----------
    is_social_feature : bool
        Feature involves user-to-user interaction (feed, likes, follows).
    is_referral_feature : bool
        Feature involves referring other users (referral codes, invite flows).
    is_comms_feature : bool
        Feature involves communication between users (chat, notifications
        that trigger other users' actions).
    is_marketplace : bool
        Two-sided marketplace where supply/demand interact.
    n_variants : int
        Number of treatment arms.
    uses_cluster_rand : bool
        Whether cluster randomisation was used (e.g., by city, cohort).
    """
    violations = []
    if is_social_feature:   violations.append("social interaction")
    if is_referral_feature: violations.append("referral mechanics")
    if is_comms_feature:    violations.append("user-to-user communication")
    if is_marketplace:      violations.append("two-sided marketplace dynamics")

    n_violations = len(violations)

    if n_violations == 0:
        severity = Severity.PASS
        verdict  = ("No SUTVA violations detected. "
                    "This feature does not appear to create "
                    "treatment–control contamination pathways.")
        cost     = "N/A"
        rec      = "Standard randomisation is appropriate."

    elif uses_cluster_rand:
        severity = Severity.WARN
        verdict  = (f"Network effect risk detected "
                    f"({', '.join(violations)}) "
                    f"but cluster randomisation is in use. "
                    f"This is the correct design choice.")
        cost     = ("Cluster randomisation reduces contamination "
                    "but reduces effective sample size. "
                    "Ensure your power analysis accounts for "
                    "the design effect (DEFF).")
        rec      = ("Verify your design effect calculation. "
                    "DEFF = 1 + (m-1)×ICC where m = cluster size, "
                    "ICC = intra-cluster correlation.")

    else:
        severity = Severity.FAIL
        verdict  = (f"SUTVA violation likely — "
                    f"feature involves {', '.join(violations)}. "
                    f"Treatment and control users can interact, "
                    f"contaminating the control group.")
        cost     = ("Your control group is not a clean counterfactual. "
                    "Treatment effects are biased — likely underestimated "
                    "(treated users spill positive effects to control) "
                    "or overestimated (network effects inflate treatment). "
                    "The direction depends on your specific feature.")
        rec      = ("Redesign the experiment with cluster randomisation: "
                    "assign entire social clusters, cities, or cohorts "
                    "to a single arm. "
                    "Alternatively, use a holdout design with a geo split "
                    "if user-level randomisation is unavoidable.")

    return CheckResult(
        name="SUTVA / Network Effects",
        severity=severity,
        statistic=float(n_violations),
        p_value=None,
        verdict=verdict,
        cost_of_violation=cost if n_violations > 0 else "N/A",
        recommendation=rec,
        details={
            "violations_detected":   violations,
            "n_violations":          n_violations,
            "is_social":             is_social_feature,
            "is_referral":           is_referral_feature,
            "is_comms":              is_comms_feature,
            "is_marketplace":        is_marketplace,
            "uses_cluster_rand":     uses_cluster_rand,
            "n_variants":            n_variants,
        }
    )


# ════════════════════════════════════════════════════════════
# FULL AUDIT RUNNER
# ════════════════════════════════════════════════════════════

def run_full_audit(
    df:             pd.DataFrame,
    config:         ExperimentConfig,
    p_values:       Optional[list]  = None,
    sutva_flags:    Optional[dict]  = None,
) -> AuditResult:
    """
    Run all 8 validity checks and return a rolled-up AuditResult.

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment DataFrame (from data_generator or real upload).
        Required columns: user_id, arm, converted, metric_value, day, pre_metric

    config : ExperimentConfig
        Experiment configuration (alpha, target_power, peeking_days, etc.)

    p_values : list of float, optional
        All p-values from this experiment's tests (for multiple testing check).
        If None, only the primary metric p-value is used.

    sutva_flags : dict, optional
        Keys: is_social_feature, is_referral_feature, is_comms_feature,
              is_marketplace, uses_cluster_rand.
        If None, all flags default to False.
    """
    ctrl = df[df["arm"] == "control"]
    trt  = df[df["arm"] == "treatment"]

    n_c = len(ctrl)
    n_t = len(trt)
    p_c = float(ctrl["converted"].mean())
    p_t = float(trt["converted"].mean())

    results = []

    # ── Check 1: SRM ─────────────────────────────────────────
    results.append(check_srm(n_c, n_t, config.expected_ratio, config.alpha))

    # ── Check 2: Power ───────────────────────────────────────
    results.append(check_power(n_c, n_t, p_c, p_t, config.alpha, config.target_power))

    # ── Check 3: Variance (continuous metrics only) ──────────
    if config.metric_type.value == "continuous":
        results.append(check_variance(
            ctrl["metric_value"].values,
            trt["metric_value"].values,
            config.alpha,
        ))
    else:
        results.append(CheckResult(
            name="Variance Homogeneity",
            severity=Severity.PASS,
            statistic=0.0,
            p_value=None,
            verdict="Skipped — applies to continuous metrics only. "
                    "Proportion metrics use z-test (variance is determined by p).",
            cost_of_violation="N/A",
            recommendation="N/A",
            details={"skipped": True, "reason": "proportion metric"}
        ))

    # ── Check 4: Normality ───────────────────────────────────
    results.append(check_normality(
        ctrl["metric_value"].values,
        trt["metric_value"].values,
        config.alpha,
    ))

    # ── Check 5: Multiple Testing ────────────────────────────
    if p_values is None:
        # Compute primary metric p-value on the spot
        se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
        z  = abs(p_t - p_c) / se if se > 0 else 0.0
        pv = _two_tailed_p_from_z(z)
        p_values = [pv]
    results.append(check_multiple_testing(p_values, config.alpha))

    # ── Check 6: Peeking ─────────────────────────────────────
    results.append(check_peeking(
        config.peeking_days,
        n_per_arm=min(n_c, n_t),
        p_control=p_c,
        alpha=config.alpha,
    ))

    # ── Check 7: Novelty Effect ──────────────────────────────
    results.append(check_novelty(df, config.alpha))

    # ── Check 8: SUTVA ───────────────────────────────────────
    sf = sutva_flags or {}
    results.append(check_sutva(
        is_social_feature=sf.get("is_social_feature",   config.is_social_feature),
        is_referral_feature=sf.get("is_referral_feature", False),
        is_comms_feature=sf.get("is_comms_feature",       False),
        is_marketplace=sf.get("is_marketplace",           False),
        n_variants=config.n_variants,
        uses_cluster_rand=sf.get("uses_cluster_rand",     False),
    ))

    # ── Roll up overall verdict ───────────────────────────────
    n_fail = sum(1 for r in results if r.severity == Severity.FAIL)
    n_warn = sum(1 for r in results if r.severity == Severity.WARN)

    if n_fail == 0 and n_warn == 0:
        overall = Severity.PASS
        summary = ("All 8 validity checks passed. "
                   "This experiment meets statistical standards for a trusted result. "
                   "You can make a shipping decision with confidence.")
    elif n_fail == 0:
        overall = Severity.WARN
        summary = (f"{n_warn} warning(s) detected. "
                   "The result is likely valid but warrants caution. "
                   "Review the flagged checks before shipping.")
    else:
        overall = Severity.FAIL
        summary = (f"{n_fail} critical failure(s) and {n_warn} warning(s). "
                   "This experiment has methodological violations that "
                   "undermine the validity of the result. "
                   "Do not make a shipping decision based on this data.")

    return AuditResult(
        checks=results,
        overall_severity=overall,
        overall_verdict=summary,
        experiment_meta={
            "name":        config.name,
            "n_control":   n_c,
            "n_treatment": n_t,
            "p_control":   round(p_c, 4),
            "p_treatment": round(p_t, 4),
            "lift_pp":     round((p_t - p_c) * 100, 3),
            "metric_type": config.metric_type.value,
        }
    )
