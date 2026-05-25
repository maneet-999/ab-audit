"""
bayesian.py
===========
Bayesian A/B testing via Beta-Binomial conjugate model.

The frequentist approach asks:
    "If there is no effect, how often would I see data this extreme?"

The Bayesian approach asks:
    "Given the data I observed, what is the probability that
     treatment is actually better than control?"

Theory
------
Prior:      p ~ Beta(alpha_prior, beta_prior)
            Default: Beta(1, 1) = uniform (no prior belief)

Likelihood: X | p ~ Binomial(n, p)

Posterior:  p | X ~ Beta(alpha_prior + X, beta_prior + n - X)

Inference:
    P(p_treatment > p_control) via Monte Carlo integration.
    Credible interval: [2.5%, 97.5%] quantiles of the posterior lift.

Why Bayesian matters here
--------------------------
1. No fixed sample size required — check results any time without
   inflating Type I error. This is the correct answer to peeking.
2. Results are directly interpretable:
   "There is a 94.3% probability treatment is better than control."
3. Credible intervals mean what most people wrongly think confidence
   intervals mean.

Public API
----------
run_bayesian_ab()        Full analysis, returns BayesianResult
bayesian_from_dataframe() Wrapper that takes a standard experiment df
bayesian_sequential()    Track posterior update day by day
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════
# Result dataclasses
# ════════════════════════════════════════════════════════════

@dataclass
class BayesianResult:
    """Full output of run_bayesian_ab()."""

    prob_treatment_better:    float   # P(treatment > control)
    prob_control_better:      float   # 1 - above
    expected_lift:            float   # E[p_t - p_c] under posterior
    credible_interval:        tuple   # 95% CI on lift
    posterior_control_mean:   float   # posterior mean for control
    posterior_treatment_mean: float   # posterior mean for treatment
    control_alpha:            float   # posterior Beta params
    control_beta:             float
    treatment_alpha:          float
    treatment_beta:           float
    n_samples:                int
    decision:                 str     # plain-English verdict
    prior_alpha:              float
    prior_beta:               float
    # Plot data (first 2000 samples — enough for smooth charts)
    posterior_samples_control:   list = field(default_factory=list)
    posterior_samples_treatment: list = field(default_factory=list)
    lift_samples:                list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Bayesian A/B Analysis",
            f"  Prior                  : Beta({self.prior_alpha}, {self.prior_beta})",
            f"  Posterior control      : Beta({self.control_alpha:.1f}, {self.control_beta:.1f})",
            f"  Posterior treatment    : Beta({self.treatment_alpha:.1f}, {self.treatment_beta:.1f})",
            f"  Control rate (post.)   : {self.posterior_control_mean*100:.3f}%",
            f"  Treatment rate (post.) : {self.posterior_treatment_mean*100:.3f}%",
            "",
            f"  P(treatment > control) : {self.prob_treatment_better:.1%}",
            f"  Expected lift          : {self.expected_lift*100:+.3f} pp",
            f"  95% Credible interval  : [{self.credible_interval[0]*100:+.3f},"
            f" {self.credible_interval[1]*100:+.3f}] pp",
            "",
            f"  Decision               : {self.decision}",
        ]
        return "\n".join(lines)


@dataclass
class BayesianSequentialResult:
    """Output of bayesian_sequential() — daily posterior updates."""
    days:                 list
    prob_better_by_day:   list   # P(treatment > control) at each day
    expected_lift_by_day: list
    ci_lower_by_day:      list
    ci_upper_by_day:      list
    n_control_by_day:     list
    n_treatment_by_day:   list
    final_result:         BayesianResult


# ════════════════════════════════════════════════════════════
# CORE MATH
# ════════════════════════════════════════════════════════════

def _update_posterior(n: int, x: int,
                       alpha_prior: float,
                       beta_prior: float) -> tuple:
    """
    Beta-Binomial conjugate update.
    Prior Beta(a, b) + data (n trials, x successes)
    = Posterior Beta(a + x, b + n - x)
    """
    return (alpha_prior + x, beta_prior + (n - x))


def _beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _prob_treatment_better(a_c: float, b_c: float,
                             a_t: float, b_t: float,
                             n_samples: int,
                             rng: np.random.Generator) -> tuple:
    """
    P(p_treatment > p_control) via Monte Carlo.
    Returns (prob, samples_control, samples_treatment, lift_samples).
    """
    s_c = rng.beta(a_c, b_c, size=n_samples)
    s_t = rng.beta(a_t, b_t, size=n_samples)
    prob = float((s_t > s_c).mean())
    return prob, s_c, s_t, (s_t - s_c)


def _credible_interval(lift_samples: np.ndarray,
                         ci: float = 0.95) -> tuple:
    lo = (1 - ci) / 2
    return (float(np.quantile(lift_samples, lo)),
            float(np.quantile(lift_samples, 1 - lo)))


def _decision(prob: float,
               threshold_ship: float = 0.95,
               threshold_stop: float = 0.05) -> str:
    if prob >= threshold_ship:
        return (f"Ship treatment. {prob:.1%} probability it outperforms control. "
                f"Risk of wrong decision: {1-prob:.1%}.")
    elif prob <= threshold_stop:
        return (f"Keep control. {1-prob:.1%} probability control is better. "
                f"Treatment shows no meaningful improvement.")
    else:
        return (f"Inconclusive — {prob:.1%} probability treatment is better. "
                f"Collect more data before deciding.")


# ════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════

def run_bayesian_ab(
    n_control:      int,
    x_control:      int,
    n_treatment:    int,
    x_treatment:    int,
    alpha_prior:    float = 1.0,
    beta_prior:     float = 1.0,
    n_samples:      int   = 50_000,
    ci:             float = 0.95,
    threshold_ship: float = 0.95,
    threshold_stop: float = 0.05,
    seed:           int   = 42,
) -> BayesianResult:
    """
    Full Bayesian A/B test for proportion metrics.

    Parameters
    ----------
    n_control, x_control : int
        Users and conversions in control.
    n_treatment, x_treatment : int
        Users and conversions in treatment.
    alpha_prior, beta_prior : float
        Prior Beta parameters. Beta(1,1) = uniform (default).
    n_samples : int
        Monte Carlo draws. 50k gives stable results to 3 decimal places.
    ci : float
        Credible interval width. Default 0.95.
    threshold_ship : float
        Ship treatment if P(treatment > control) >= this.
    threshold_stop : float
        Call it for control if P(treatment > control) <= this.
    seed : int
        Random seed.

    Returns
    -------
    BayesianResult

    Example
    -------
    >>> r = run_bayesian_ab(5000, 470, 5000, 558)
    >>> print(f"{r.prob_treatment_better:.1%}")
    97.4%
    """
    rng = np.random.default_rng(seed)

    a_c, b_c = _update_posterior(n_control,   x_control,   alpha_prior, beta_prior)
    a_t, b_t = _update_posterior(n_treatment, x_treatment, alpha_prior, beta_prior)

    prob, s_c, s_t, lift = _prob_treatment_better(
        a_c, b_c, a_t, b_t, n_samples, rng
    )

    ci_lo, ci_hi = _credible_interval(lift, ci)

    return BayesianResult(
        prob_treatment_better=round(prob, 4),
        prob_control_better=round(1 - prob, 4),
        expected_lift=round(float(lift.mean()), 6),
        credible_interval=(round(ci_lo, 6), round(ci_hi, 6)),
        posterior_control_mean=round(_beta_mean(a_c, b_c), 6),
        posterior_treatment_mean=round(_beta_mean(a_t, b_t), 6),
        control_alpha=round(a_c, 2),
        control_beta=round(b_c, 2),
        treatment_alpha=round(a_t, 2),
        treatment_beta=round(b_t, 2),
        n_samples=n_samples,
        decision=_decision(prob, threshold_ship, threshold_stop),
        prior_alpha=alpha_prior,
        prior_beta=beta_prior,
        posterior_samples_control=s_c[:2000].tolist(),
        posterior_samples_treatment=s_t[:2000].tolist(),
        lift_samples=lift[:2000].tolist(),
    )


def bayesian_from_dataframe(df,
                              alpha_prior: float = 1.0,
                              beta_prior:  float = 1.0,
                              n_samples:   int   = 50_000,
                              seed:        int   = 42) -> BayesianResult:
    """
    Convenience wrapper — takes the standard experiment DataFrame
    (from data_generator or real upload) and runs the Bayesian analysis.

    Requires columns: arm, converted.
    """
    ctrl = df[df["arm"] == "control"]
    trt  = df[df["arm"] == "treatment"]

    return run_bayesian_ab(
        n_control=len(ctrl),
        x_control=int(ctrl["converted"].sum()),
        n_treatment=len(trt),
        x_treatment=int(trt["converted"].sum()),
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        n_samples=n_samples,
        seed=seed,
    )


def bayesian_sequential(
    df,
    alpha_prior: float = 1.0,
    beta_prior:  float = 1.0,
    n_samples:   int   = 20_000,
    seed:        int   = 42,
) -> BayesianSequentialResult:
    """
    Track P(treatment > control) as data accumulates day by day.

    This is the core Bayesian answer to the peeking problem.
    You can look at this curve every day — unlike frequentist
    p-values, the posterior probability does not inflate with
    repeated inspection.

    Requires columns: arm, converted, day.
    """
    rng  = np.random.default_rng(seed)
    days = sorted(df["day"].unique())

    prob_by_day  = []
    lift_by_day  = []
    ci_lo_by_day = []
    ci_hi_by_day = []
    n_c_by_day   = []
    n_t_by_day   = []

    for d in days:
        subset = df[df["day"] <= d]
        ctrl   = subset[subset["arm"] == "control"]
        trt    = subset[subset["arm"] == "treatment"]

        n_c = len(ctrl);  x_c = int(ctrl["converted"].sum())
        n_t = len(trt);   x_t = int(trt["converted"].sum())

        if n_c == 0 or n_t == 0:
            prob_by_day.append(0.5); lift_by_day.append(0.0)
            ci_lo_by_day.append(0.0); ci_hi_by_day.append(0.0)
            n_c_by_day.append(n_c); n_t_by_day.append(n_t)
            continue

        a_c, b_c = _update_posterior(n_c, x_c, alpha_prior, beta_prior)
        a_t, b_t = _update_posterior(n_t, x_t, alpha_prior, beta_prior)
        prob, _, _, lift = _prob_treatment_better(
            a_c, b_c, a_t, b_t, n_samples, rng
        )
        ci_lo, ci_hi = _credible_interval(lift, 0.95)

        prob_by_day.append(round(float(prob), 4))
        lift_by_day.append(round(float(lift.mean()), 6))
        ci_lo_by_day.append(round(float(ci_lo), 6))
        ci_hi_by_day.append(round(float(ci_hi), 6))
        n_c_by_day.append(n_c)
        n_t_by_day.append(n_t)

    final = bayesian_from_dataframe(
        df, alpha_prior, beta_prior, n_samples * 2, seed
    )

    return BayesianSequentialResult(
        days=list(days),
        prob_better_by_day=prob_by_day,
        expected_lift_by_day=lift_by_day,
        ci_lower_by_day=ci_lo_by_day,
        ci_upper_by_day=ci_hi_by_day,
        n_control_by_day=n_c_by_day,
        n_treatment_by_day=n_t_by_day,
        final_result=final,
    )


__all__ = [
    "BayesianResult",
    "BayesianSequentialResult",
    "run_bayesian_ab",
    "bayesian_from_dataframe",
    "bayesian_sequential",
]
