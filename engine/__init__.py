"""
AB Audit Engine
===============
Statistical Validity Audit Engine for A/B experiments.

Package structure:
    engine/
        __init__.py       <- Core dataclasses and enums (you are here)
        checks.py         <- All 8 validity checks
        simulation.py     <- Monte Carlo simulation engine
        cuped.py          <- CUPED variance reduction
        bayesian.py       <- Beta-Binomial Bayesian A/B
        data_generator.py <- Synthetic experiment data generator
        report.py         <- PDF/markdown report generator
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# --------------------------------------------------
# Enums
# --------------------------------------------------

class Severity(str, Enum):
    """Severity level for a validity check result."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MetricType(str, Enum):
    """Type of primary metric being analyzed."""
    PROPORTION = "proportion"
    CONTINUOUS = "continuous"
    COUNT      = "count"


class TestType(str, Enum):
    """Which statistical test was used for the primary analysis."""
    Z_TEST        = "z_test"
    T_TEST_POOLED = "t_test_pooled"
    T_TEST_WELCH  = "t_test_welch"
    MANN_WHITNEY  = "mann_whitney"
    FISHER_EXACT  = "fisher_exact"


# --------------------------------------------------
# Core result dataclass — every check returns this
# --------------------------------------------------

@dataclass
class CheckResult:
    """
    Standardised result object returned by every validity check.

    Fields
    ------
    name : str
        Human-readable name of the check.

    severity : Severity
        PASS | WARN | FAIL — drives UI colour and overall verdict.

    statistic : float
        The raw test statistic computed by this check.

    p_value : Optional[float]
        Two-tailed p-value where applicable. None if not applicable.

    verdict : str
        One plain-English sentence summarising the outcome.

    cost_of_violation : str
        What it concretely means if this check fails.

    recommendation : str
        Exactly what the user should do next.

    details : dict
        Optional bag for extra computed values (power curves,
        adjusted p-values, observed vs expected counts, etc.)
    """
    name:              str
    severity:          Severity
    statistic:         float
    p_value:           Optional[float]
    verdict:           str
    cost_of_violation: str
    recommendation:    str
    details:           dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity == Severity.PASS

    @property
    def emoji(self) -> str:
        return {"pass": "✅", "warn": "⚠️", "fail": "❌"}[self.severity.value]

    def __repr__(self) -> str:
        p = f"{self.p_value:.4f}" if self.p_value is not None else "N/A"
        return (
            f"CheckResult({self.emoji} {self.name} | "
            f"stat={self.statistic:.4f} | p={p} | "
            f"{self.severity.value.upper()})"
        )


# --------------------------------------------------
# Audit summary — wraps all 8 CheckResults
# --------------------------------------------------

@dataclass
class AuditResult:
    """
    Full audit output containing all 8 CheckResults plus a rolled-up verdict.
    Returned by engine.checks.run_full_audit().
    """
    checks:           list
    overall_severity: Severity
    overall_verdict:  str
    experiment_meta:  dict = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.PASS)

    @property
    def n_warned(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.WARN)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.FAIL)

    @property
    def score_summary(self) -> str:
        return (
            f"{self.n_passed}/8 passed · "
            f"{self.n_warned} warnings · "
            f"{self.n_failed} failures"
        )

    def __repr__(self) -> str:
        return (
            f"AuditResult({self.overall_severity.value.upper()} | "
            f"{self.score_summary})"
        )


# --------------------------------------------------
# Experiment config — typed input to the audit
# --------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Typed input configuration for an A/B experiment audit.
    Pass this to run_full_audit() instead of loose keyword args.
    """
    name:              str        = "Unnamed Experiment"
    metric_type:       MetricType = MetricType.PROPORTION
    alpha:             float      = 0.05
    target_power:      float      = 0.80
    n_variants:        int        = 2
    expected_ratio:    float      = 0.50
    peeking_days:      list       = field(default_factory=list)
    is_social_feature: bool       = False


# --------------------------------------------------
# Exports
# --------------------------------------------------

__all__ = [
    "Severity",
    "MetricType",
    "TestType",
    "CheckResult",
    "AuditResult",
    "ExperimentConfig",
]
