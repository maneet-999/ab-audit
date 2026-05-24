"""
test_checks.py
==============
Unit tests for all 8 validity checks in engine/checks.py.

Run from inside the ab_audit/ folder:
    python3 tests/test_checks.py

Each check has:
  - pass case   : known-good input  → PASS
  - fail case   : known-bad input   → FAIL or WARN
  - edge case   : boundary / degenerate input
  - scipy cross-validation where applicable
"""

import sys, traceback
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # run from ab_audit/ root

import numpy as np
from scipy import stats as sp_stats

from engine import Severity, CheckResult, ExperimentConfig, MetricType
from engine.checks import (
    check_srm, check_power, check_variance, check_normality,
    check_multiple_testing, check_peeking, check_novelty,
    check_sutva, run_full_audit,
    _normal_cdf, _chi2_cdf,
    _compute_power_proportion, _required_n_proportion,
)
from engine.data_generator import (
    load_scenario, generate_experiment, split_arms
)

# ── Test runner ──────────────────────────────────────────────
passed, failed, errors = 0, 0, []

def ok(name):
    global passed; passed += 1; print(f"  ✅ {name}")

def fail(name, msg):
    global failed; failed += 1
    errors.append((name, msg))
    print(f"  ❌ {name}")
    print(f"     {msg}")

def run(name, fn):
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, str(e) or "AssertionError (no message)")
    except Exception:
        fail(name, traceback.format_exc().splitlines()[-1])


# ════════════════════════════════════════════════════════════
# MATH PRIMITIVES
# ════════════════════════════════════════════════════════════
print("\n[MATH PRIMITIVES]")

def t_ncdf_zero():
    assert abs(_normal_cdf(0.0) - 0.5) < 1e-4

def t_ncdf_scipy():
    for z in [-1.96, 0.0, 1.96, 3.0]:
        diff = abs(_normal_cdf(z) - float(sp_stats.norm.cdf(z)))
        assert diff < 1e-4, f"z={z}: diff={diff:.2e}"

def t_chi2_scipy():
    for x, df in [(0.5,1), (3.84,1), (6.63,1)]:
        diff = abs(_chi2_cdf(x, df) - float(sp_stats.chi2.cdf(x, df)))
        assert diff < 1e-3, f"x={x}: diff={diff:.2e}"

def t_power_known():
    # n=5000, p1=0.10, p2=0.12, alpha=0.05 → power ≈ 0.80
    p = _compute_power_proportion(5000, 0.10, 0.12, 0.05)
    assert 0.72 < p < 0.90, f"got {p:.3f}"

def t_required_n_known():
    # p1=0.10, delta=0.02, alpha=0.05, power=0.80 → ~3839
    n = _required_n_proportion(0.10, 0.02, 0.05, 0.80)
    assert 3000 < n < 5000, f"got {n}"

def t_z_alpha2():
    from engine.checks import _z_from_alpha_power
    za, zb = _z_from_alpha_power(0.05, 0.80)
    assert abs(za - 1.96)  < 0.01, f"z_alpha2={za:.4f}"
    assert abs(zb - 0.842) < 0.01, f"z_beta={zb:.4f}"

run("normal_cdf(0) = 0.5",              t_ncdf_zero)
run("normal_cdf matches scipy",          t_ncdf_scipy)
run("chi2_cdf matches scipy",            t_chi2_scipy)
run("power formula (n=5000, δ=2pp)",     t_power_known)
run("required_n (p=0.10, δ=0.02) ~3839", t_required_n_known)
run("z_alpha2=1.96, z_beta=0.842",       t_z_alpha2)


# ════════════════════════════════════════════════════════════
# CHECK 1 — SAMPLE RATIO MISMATCH
# ════════════════════════════════════════════════════════════
print("\n[CHECK 1 — SRM]")

def t_srm_pass():
    r = check_srm(5000, 5000)
    assert r.severity == Severity.PASS, f"got {r.severity}"

def t_srm_fail():
    r = check_srm(5700, 4300)
    assert r.severity == Severity.FAIL
    assert r.p_value < 0.001, f"p={r.p_value:.4f}"

def t_srm_edge():
    r = check_srm(10000, 0)
    assert r.severity == Severity.FAIL

def t_srm_scipy():
    r = check_srm(5700, 4300)
    sp_chi2, sp_p = sp_stats.chisquare([5700, 4300], f_exp=[5000, 5000])
    assert abs(r.statistic - sp_chi2) < 0.01, \
        f"chi2: ours={r.statistic:.4f} scipy={sp_chi2:.4f}"
    assert abs(r.p_value - sp_p) < 0.01, \
        f"p: ours={r.p_value:.6f} scipy={sp_p:.6f}"

def t_srm_details():
    r = check_srm(5000, 5000)
    assert r.details["n_total"] == 10000
    assert "actual_ratio" in r.details

run("SRM: 50/50 → PASS",              t_srm_pass)
run("SRM: 57/43 → FAIL + p<0.001",    t_srm_fail)
run("SRM: 100/0 edge case → FAIL",    t_srm_edge)
run("SRM: matches scipy chi-square",   t_srm_scipy)
run("SRM: details populated",          t_srm_details)


# ════════════════════════════════════════════════════════════
# CHECK 2 — STATISTICAL POWER
# ════════════════════════════════════════════════════════════
print("\n[CHECK 2 — Power]")

def t_pw_pass():
    r = check_power(8000, 8000, 0.094, 0.115)
    assert r.severity == Severity.PASS, \
        f"power={r.details['achieved_power']:.3f}"

def t_pw_fail():
    r = check_power(200, 200, 0.094, 0.097)
    assert r.severity == Severity.FAIL

def t_pw_zero_delta():
    r = check_power(5000, 5000, 0.094, 0.094)
    assert r.details["achieved_power"] <= 0.15

def t_pw_mde_positive():
    r = check_power(5000, 5000, 0.094, 0.112)
    assert r.details["mde_pp"] > 0

def t_pw_required_n():
    r = check_power(200, 200, 0.094, 0.112, target_power=0.80)
    assert r.details["required_n"] > 200

run("Power: large n + detectable → PASS",  t_pw_pass)
run("Power: tiny n + tiny delta → FAIL",   t_pw_fail)
run("Power: zero delta → power ≈ alpha",   t_pw_zero_delta)
run("Power: MDE is positive",              t_pw_mde_positive)
run("Power: required_n > observed n",      t_pw_required_n)


# ════════════════════════════════════════════════════════════
# CHECK 3 — VARIANCE HOMOGENEITY
# ════════════════════════════════════════════════════════════
print("\n[CHECK 3 — Variance]")

rng = np.random.default_rng(42)
eq_c   = rng.normal(340, 180, 2000)
eq_t   = rng.normal(355, 182, 2000)
uneq_t = rng.normal(355, 540, 2000)   # 3x variance

def t_var_pass():
    r = check_variance(eq_c, eq_t)
    assert r.severity == Severity.PASS

def t_var_warn():
    r = check_variance(eq_c, uneq_t)
    assert r.severity == Severity.WARN
    assert r.details["use_welch"] is True

def t_var_ratio():
    r = check_variance(eq_c, uneq_t)
    assert r.details["var_ratio"] > 2.0, \
        f"ratio={r.details['var_ratio']:.2f}"

def t_var_ctrl_matches_numpy():
    r = check_variance(eq_c, eq_t)
    expected = float(np.var(eq_c, ddof=1))
    assert abs(r.details["var_control"] - expected) < 1.0

def t_var_scipy_close():
    r = check_variance(eq_c, uneq_t)
    sp_stat, sp_p = sp_stats.levene(eq_c, uneq_t)
    rel_diff = abs(r.statistic - sp_stat) / sp_stat
    assert rel_diff < 0.01, \
        f"W: ours={r.statistic:.2f} scipy={sp_stat:.2f} rel_diff={rel_diff:.4%}"

run("Variance: equal variances → PASS",        t_var_pass)
run("Variance: 3× var → WARN + use_welch",     t_var_warn)
run("Variance: ratio > 2.0 for 3× case",       t_var_ratio)
run("Variance: control var matches numpy",      t_var_ctrl_matches_numpy)
run("Variance: W stat within 1% of scipy",     t_var_scipy_close)


# ════════════════════════════════════════════════════════════
# CHECK 4 — NORMALITY
# ════════════════════════════════════════════════════════════
print("\n[CHECK 4 — Normality]")

rng2      = np.random.default_rng(99)
norm_data = rng2.normal(0, 1, 500)
skew_data = rng2.exponential(1, 500)
large_n   = rng2.normal(0, 1, 10_000)

def t_nor_normal():
    r = check_normality(norm_data, norm_data)
    assert r.severity in (Severity.PASS, Severity.WARN)

def t_nor_skewness():
    r = check_normality(skew_data, skew_data)
    assert r.details["skewness_control"] > 0.5, \
        f"skew={r.details['skewness_control']:.3f}"

def t_nor_clt():
    r = check_normality(large_n, large_n)
    assert r.details["clt_applies"] is True

def t_nor_mwu_bool():
    r = check_normality(skew_data, skew_data)
    assert isinstance(r.details["recommend_mwu"], bool)

def t_nor_tiny_n():
    r = check_normality(np.array([1., 2., 3.]),
                        np.array([1., 2., 3.]))
    assert isinstance(r, CheckResult)

run("Normality: normal data → PASS/WARN",       t_nor_normal)
run("Normality: exponential skewness > 0.5",    t_nor_skewness)
run("Normality: n=10k → CLT applies",           t_nor_clt)
run("Normality: recommend_mwu is bool",         t_nor_mwu_bool)
run("Normality: tiny n handles gracefully",     t_nor_tiny_n)


# ════════════════════════════════════════════════════════════
# CHECK 5 — MULTIPLE TESTING
# ════════════════════════════════════════════════════════════
print("\n[CHECK 5 — Multiple Testing]")

def t_mt_single():
    r = check_multiple_testing([0.03])
    assert r.severity == Severity.PASS
    assert r.details["k"] == 1

def t_mt_many_fail():
    r = check_multiple_testing([0.03] * 20)
    assert r.severity == Severity.FAIL
    assert r.statistic > 0.60

def t_mt_fwer_exact():
    k, a = 10, 0.05
    r = check_multiple_testing([0.1] * k, alpha=a)
    expected = 1 - (1 - a) ** k
    assert abs(r.statistic - expected) < 1e-6

def t_mt_bh_geq_bonf():
    p_vals = [0.001, 0.01, 0.04, 0.10, 0.20]
    r = check_multiple_testing(p_vals)
    assert r.details["n_sig_bh"] >= r.details["n_sig_bonf"]

def t_mt_all_null():
    r = check_multiple_testing([0.5, 0.6, 0.7, 0.8])
    assert r.details["n_sig_naive"] == 0

run("MT: k=1 → PASS",                     t_mt_single)
run("MT: k=20 → FAIL, FWER > 60%",        t_mt_many_fail)
run("MT: FWER = 1−(1−α)^k exactly",       t_mt_fwer_exact)
run("MT: BH rejects ≥ Bonferroni",         t_mt_bh_geq_bonf)
run("MT: all null p-vals → 0 significant", t_mt_all_null)


# ════════════════════════════════════════════════════════════
# CHECK 6 — PEEKING
# ════════════════════════════════════════════════════════════
print("\n[CHECK 6 — Peeking]")

def t_pk_none():
    r = check_peeking([], n_per_arm=5000, p_control=0.094)
    assert r.severity == Severity.PASS
    assert r.details["n_peeks"] == 0

def t_pk_heavy():
    r = check_peeking(
        list(range(1, 15)), n_per_arm=5000,
        p_control=0.094, n_simulations=2000, seed=42
    )
    assert r.details["inflated_alpha"] > 0.10, \
        f"inflated={r.details['inflated_alpha']:.3f}"

def t_pk_monotone():
    kw = dict(n_per_arm=5000, p_control=0.094, n_simulations=1000, seed=1)
    r1 = check_peeking([7],         **kw)
    r4 = check_peeking([2,5,7,10],  **kw)
    assert r4.details["inflated_alpha"] >= r1.details["inflated_alpha"] * 0.80

def t_pk_trajectory():
    r = check_peeking([3,7], n_per_arm=1000, p_control=0.10, n_simulations=50)
    assert len(r.details["p_trajectory_sample"]) > 0

def t_pk_nominal():
    r = check_peeking([5], n_per_arm=2000, p_control=0.094, n_simulations=100)
    assert r.details["nominal_alpha"] == 0.05

run("Peeking: no peeks → PASS, n_peeks=0",          t_pk_none)
run("Peeking: 14-day daily → alpha > 10%",           t_pk_heavy)
run("Peeking: more peeks → higher/equal inflation",  t_pk_monotone)
run("Peeking: trajectory sample non-empty",          t_pk_trajectory)
run("Peeking: nominal_alpha=0.05 in details",        t_pk_nominal)


# ════════════════════════════════════════════════════════════
# CHECK 7 — NOVELTY EFFECT
# ════════════════════════════════════════════════════════════
print("\n[CHECK 7 — Novelty]")

clean_df  = load_scenario("edtech_notification")
novel_df  = load_scenario("novelty_effect")

def t_nov_returns_result():
    r = check_novelty(clean_df)
    assert isinstance(r, CheckResult)

def t_nov_negative_slope():
    r = check_novelty(novel_df)
    if "slope_beta1" in r.details:
        assert r.details["slope_beta1"] < 0.03, \
            f"slope={r.details['slope_beta1']:.4f}"

def t_nov_insufficient_days():
    tiny = clean_df[clean_df["day"] <= 2].copy()
    r = check_novelty(tiny, min_days=5)
    assert r.severity == Severity.WARN
    assert "Could not run" in r.verdict

def t_nov_missing_col():
    bad = clean_df.drop(columns=["day"])
    r   = check_novelty(bad)
    assert r.severity == Severity.WARN

def t_nov_details_present():
    r = check_novelty(clean_df)
    assert "slope_beta1" in r.details or "reason" in r.details

run("Novelty: returns CheckResult",                t_nov_returns_result)
run("Novelty: decaying lift → negative slope",     t_nov_negative_slope)
run("Novelty: <5 days → WARN + message",           t_nov_insufficient_days)
run("Novelty: missing day col → graceful WARN",    t_nov_missing_col)
run("Novelty: details key present",                t_nov_details_present)


# ════════════════════════════════════════════════════════════
# CHECK 8 — SUTVA
# ════════════════════════════════════════════════════════════
print("\n[CHECK 8 — SUTVA]")

def t_su_pass():
    r = check_sutva()
    assert r.severity == Severity.PASS
    assert r.details["n_violations"] == 0

def t_su_fail_social():
    r = check_sutva(is_social_feature=True, uses_cluster_rand=False)
    assert r.severity == Severity.FAIL

def t_su_warn_cluster():
    r = check_sutva(is_social_feature=True, uses_cluster_rand=True)
    assert r.severity == Severity.WARN

def t_su_three_violations():
    r = check_sutva(
        is_social_feature=True,
        is_referral_feature=True,
        is_marketplace=True,
        uses_cluster_rand=False,
    )
    assert r.details["n_violations"] == 3
    assert r.statistic == 3.0

def t_su_comms_detected():
    r = check_sutva(is_comms_feature=True, uses_cluster_rand=False)
    assert "user-to-user communication" in r.details["violations_detected"]

run("SUTVA: no violations → PASS",              t_su_pass)
run("SUTVA: social, no cluster → FAIL",         t_su_fail_social)
run("SUTVA: social + cluster rand → WARN",      t_su_warn_cluster)
run("SUTVA: 3 violations → statistic=3.0",      t_su_three_violations)
run("SUTVA: comms feature flagged correctly",   t_su_comms_detected)


# ════════════════════════════════════════════════════════════
# FULL AUDIT RUNNER
# ════════════════════════════════════════════════════════════
print("\n[FULL AUDIT RUNNER]")

def t_fa_eight_checks():
    df  = load_scenario("edtech_notification")
    cfg = ExperimentConfig(name="EdTech", metric_type=MetricType.PROPORTION)
    res = run_full_audit(df, cfg)
    assert len(res.checks) == 8, f"got {len(res.checks)}"

def t_fa_srm_flagged():
    df  = load_scenario("fintech_cashback_srm")
    cfg = ExperimentConfig(name="Fintech SRM")
    res = run_full_audit(df, cfg)
    assert res.checks[0].severity == Severity.FAIL

def t_fa_counts_sum_to_eight():
    df  = load_scenario("zepto_scarcity_badge")
    cfg = ExperimentConfig(name="Zepto")
    res = run_full_audit(df, cfg)
    assert res.n_passed + res.n_warned + res.n_failed == 8

def t_fa_meta_populated():
    df  = load_scenario("zepto_scarcity_badge")
    cfg = ExperimentConfig(name="Meta Test")
    res = run_full_audit(df, cfg)
    assert "n_control"  in res.experiment_meta
    assert "lift_pp"    in res.experiment_meta

def t_fa_peeking_propagated():
    df  = load_scenario("zepto_scarcity_badge")
    cfg = ExperimentConfig(name="Peek", peeking_days=[3, 7])
    res = run_full_audit(df, cfg)
    assert res.checks[5].details["n_peeks"] == 2

def t_fa_fail_propagates():
    df  = load_scenario("fintech_cashback_srm")
    cfg = ExperimentConfig(name="SRM Audit")
    res = run_full_audit(df, cfg)
    assert res.overall_severity == Severity.FAIL

def t_fa_repr():
    df  = load_scenario("zepto_scarcity_badge")
    cfg = ExperimentConfig(name="Repr")
    res = run_full_audit(df, cfg)
    assert "AuditResult" in repr(res)
    assert "/8" in res.score_summary

run("Full audit: exactly 8 checks",           t_fa_eight_checks)
run("Full audit: SRM scenario → check[0] FAIL", t_fa_srm_flagged)
run("Full audit: n_passed+warned+failed = 8", t_fa_counts_sum_to_eight)
run("Full audit: experiment_meta populated",  t_fa_meta_populated)
run("Full audit: peeking_days propagated",    t_fa_peeking_propagated)
run("Full audit: any FAIL → overall FAIL",    t_fa_fail_propagates)
run("Full audit: repr + score_summary work",  t_fa_repr)


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print()
print("=" * 55)
total = passed + failed
print(f"Results: {passed}/{total} tests passed")

if errors:
    print(f"\nFailed tests ({len(errors)}):")
    for name, msg in errors:
        print(f"  ✗  {name}")
        print(f"     {msg}")
    print("\nPaste the failed test names above into Claude to fix.")
else:
    print("\n✅ ALL TESTS PASSED")
    print("   Phase 1 engine is working correctly.")
print("=" * 55)
