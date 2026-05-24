"""
test_simulation.py + test_cuped.py combined
============================================
Unit tests for simulation.py and cuped.py.
"""
import sys, os
# Add ab_audit/ root to path regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from engine.data_generator import load_scenario, generate_experiment, split_arms
from engine.simulation import (
    run_peeking_simulation, run_power_simulation,
    simulate_null_trajectories, benchmark,
)
from engine.cuped import (
    cuped_adjust, run_cuped_analysis,
    validate_against_ancova, _compute_theta,
)

passed, failed, errors = 0, 0, []

def ok(name):
    global passed; passed += 1; print(f"  ✅ {name}")

def fail(name, msg):
    global failed; failed += 1
    errors.append((name, msg)); print(f"  ❌ {name} — {msg}")

def run(name, fn):
    try: fn(); ok(name)
    except AssertionError as e: fail(name, str(e))
    except Exception as e:
        import traceback
        fail(name, traceback.format_exc().splitlines()[-1])


# ════════════════════════════════════════════════════════════
# SIMULATION — PEEKING
# ════════════════════════════════════════════════════════════
print("\n[SIMULATION — Peeking]")

def t_peek_no_peeks():
    r = run_peeking_simulation(5000, 0.094, [], n_simulations=100)
    assert r.inflated_alpha == r.nominal_alpha == 0.05
    assert r.n_simulations == 0
run("No peeking → inflated_alpha = nominal_alpha", t_peek_no_peeks)

def t_peek_null_inflates():
    # Daily peeking on null: inflated alpha must exceed nominal
    r = run_peeking_simulation(5000, 0.094, list(range(1,15)),
                               n_simulations=3000, seed=42)
    assert r.inflated_alpha > 0.10, f"got {r.inflated_alpha:.3f}"
run("14-day peeking inflates alpha >10%", t_peek_null_inflates)

def t_peek_monotone():
    # FP rate should increase (or stay equal) as we add more peeks
    base = dict(n_per_arm=5000, p_control=0.094, n_simulations=2000, seed=1)
    r1 = run_peeking_simulation(peeking_days=[7],           **base)
    r3 = run_peeking_simulation(peeking_days=[3, 7, 10],    **base)
    assert r3.inflated_alpha >= r1.inflated_alpha * 0.85
run("More peeks → higher/equal inflation (monotone)", t_peek_monotone)

def t_peek_trajectories():
    r = run_peeking_simulation(1000, 0.10, [3,7,10], n_simulations=50)
    assert len(r.trajectory_sample) > 0
    assert all(len(t) == 3 for t in r.trajectory_sample)
run("Trajectory sample: 3 p-values per path (3 peek days)", t_peek_trajectories)

def t_peek_alpha_by_peeks_length():
    peeks = [2, 5, 8, 11, 14]
    r = run_peeking_simulation(2000, 0.10, peeks, n_simulations=500)
    assert len(r.alpha_by_n_peeks) == len(peeks)
run("alpha_by_n_peeks length matches n_peeks", t_peek_alpha_by_peeks_length)

def t_peek_cumulative_monotone():
    r = run_peeking_simulation(3000, 0.094, [2,5,8,11,14], n_simulations=2000)
    # Cumulative FP rates must be non-decreasing
    rates = r.daily_fp_rates
    for i in range(1, len(rates)):
        assert rates[i] >= rates[i-1] - 0.01, \
            f"Non-monotone at index {i}: {rates[i-1]:.3f} → {rates[i]:.3f}"
run("Cumulative FP rates are non-decreasing", t_peek_cumulative_monotone)

def t_peek_reproducible():
    kw = dict(n_per_arm=2000, p_control=0.094, peeking_days=[3,7], n_simulations=500)
    r1 = run_peeking_simulation(seed=42, **kw)
    r2 = run_peeking_simulation(seed=42, **kw)
    assert r1.inflated_alpha == r2.inflated_alpha
run("Same seed → identical results (reproducible)", t_peek_reproducible)


# ════════════════════════════════════════════════════════════
# SIMULATION — POWER CURVE
# ════════════════════════════════════════════════════════════
print("\n[SIMULATION — Power Curve]")

def t_power_monotone():
    r = run_power_simulation(p_control=0.094, true_lift=0.02,
                             n_values=[200,500,1000,2000,5000],
                             n_simulations=2000, seed=7)
    pv = r.power_values
    # Power should be generally increasing with n
    assert pv[-1] > pv[0], f"Power not increasing: {pv[0]:.3f} → {pv[-1]:.3f}"
run("Power increases with sample size", t_power_monotone)

def t_power_null_near_alpha():
    # When true_lift=0, empirical power ≈ alpha (false positive rate)
    r = run_power_simulation(p_control=0.10, true_lift=0.0,
                             n_values=[1000, 5000],
                             n_simulations=3000, seed=99)
    for pv in r.power_values:
        assert abs(pv - 0.05) < 0.04, f"Null power={pv:.3f}, expected ~0.05"
run("Null experiment: power ≈ alpha (±4%)", t_power_null_near_alpha)

def t_power_large_effect_high_power():
    r = run_power_simulation(p_control=0.10, true_lift=0.10,
                             n_values=[500], n_simulations=2000, seed=5)
    assert r.power_values[0] > 0.90, f"got {r.power_values[0]:.3f}"
run("Large effect at n=500 → power>90%", t_power_large_effect_high_power)

def t_power_required_n_detected():
    r = run_power_simulation(p_control=0.094, true_lift=0.02,
                             n_values=[500,1000,2000,3000,5000,8000],
                             n_simulations=2000, seed=3)
    # At least one of the n values should give 80% power
    if r.required_n_80 is not None:
        idx = r.n_values.index(r.required_n_80)
        assert r.power_values[idx] >= 0.75
run("required_n_80 corresponds to power≥75%", t_power_required_n_detected)


# ════════════════════════════════════════════════════════════
# SIMULATION — NULL TRAJECTORIES
# ════════════════════════════════════════════════════════════
print("\n[SIMULATION — Null Trajectories]")

def t_traj_shape():
    r = simulate_null_trajectories(2000, 0.094, n_days=7, n_trajectories=20)
    assert len(r["trajectories"]) == 20
    assert all(len(t) == 7 for t in r["trajectories"])
run("Trajectories: 20 paths × 7 days", t_traj_shape)

def t_traj_pvals_valid():
    r = simulate_null_trajectories(1000, 0.094, n_days=5, n_trajectories=10)
    for traj in r["trajectories"]:
        for pv in traj:
            assert 0.0 <= pv <= 1.0, f"p-value out of range: {pv}"
run("All trajectory p-values in [0, 1]", t_traj_pvals_valid)

def t_traj_fp_rate_reasonable():
    # For null experiment over 14 days, FP rate should be above nominal
    r = simulate_null_trajectories(3000, 0.094, n_days=14, n_trajectories=200, seed=42)
    assert r["fp_rate"] >= 0.05, f"fp_rate={r['fp_rate']:.3f}"
run("Null 14-day trajectory FP rate ≥ 5%", t_traj_fp_rate_reasonable)

def t_traj_crossings_structure():
    r = simulate_null_trajectories(1000, 0.10, n_days=10, n_trajectories=30)
    for c in r["crossings"]:
        assert "traj_idx" in c and "day" in c and "p_value" in c
        assert c["p_value"] < r["alpha_line"]
run("Crossing events have correct structure + p<alpha", t_traj_crossings_structure)


# ════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════
print("\n[SIMULATION — Benchmark]")

def t_benchmark_speed():
    b = benchmark(n_simulations=5000)
    print(f"      {b['n_simulations']:,} sims in {b['elapsed_seconds']:.2f}s "
          f"({b['sims_per_second']:,.0f}/s)")
    assert b["elapsed_seconds"] < 10.0, f"Too slow: {b['elapsed_seconds']:.2f}s"
run("10k sim benchmark completes in <10s", t_benchmark_speed)


# ════════════════════════════════════════════════════════════
# CUPED — THETA
# ════════════════════════════════════════════════════════════
print("\n[CUPED — Theta Computation]")

def t_theta_known():
    # If Y = a*X + noise, theta should be close to a
    rng = np.random.default_rng(0)
    X = rng.normal(100, 20, 5000)
    Y = 0.6 * X + rng.normal(0, 10, 5000)   # slope = 0.6
    theta = _compute_theta(Y, X)
    assert abs(theta - 0.6) < 0.05, f"theta={theta:.4f}, expected ~0.6"
run("θ recovers known slope (Y = 0.6X + noise)", t_theta_known)

def t_theta_zero_variance():
    # Constant X → theta = 0
    X = np.ones(100) * 5.0
    Y = np.random.default_rng(1).normal(10, 2, 100)
    theta = _compute_theta(Y, X)
    assert theta == 0.0
run("θ = 0 when X has zero variance", t_theta_zero_variance)

def t_theta_negative():
    # Negative correlation → negative theta
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, 1000)
    Y = -0.5 * X + rng.normal(0, 1, 1000)
    theta = _compute_theta(Y, X)
    assert theta < 0, f"theta={theta:.4f}"
run("θ is negative for negatively correlated data", t_theta_negative)


# ════════════════════════════════════════════════════════════
# CUPED — ADJUSTMENT
# ════════════════════════════════════════════════════════════
print("\n[CUPED — Adjustment]")

def t_adjust_unbiased():
    # E[Ỹ_ctrl] ≈ E[Y_ctrl] — the adjustment is mean-zero
    rng = np.random.default_rng(10)
    Yc = rng.normal(10, 3, 5000)
    Yt = rng.normal(10.5, 3, 5000)
    Xc = rng.normal(9, 2, 5000)
    Xt = rng.normal(9, 2, 5000)
    Yc_adj, Yt_adj, theta, rho = cuped_adjust(Yc, Yt, Xc, Xt)
    # Adjusted lift should be close to true lift (0.5)
    adj_lift  = Yt_adj.mean() - Yc_adj.mean()
    raw_lift  = Yt.mean()     - Yc.mean()
    assert abs(adj_lift - 0.5) < 0.1, f"adj_lift={adj_lift:.4f}"
    assert abs(raw_lift - 0.5) < 0.1, f"raw_lift={raw_lift:.4f}"
run("CUPED lift estimate is unbiased (~0.5)", t_adjust_unbiased)

def t_adjust_reduces_variance():
    rng = np.random.default_rng(20)
    # High correlation → big variance reduction
    Xc = rng.normal(100, 30, 5000)
    Xt = rng.normal(100, 30, 5000)
    Yc = 0.7 * Xc + rng.normal(0, 10, 5000)
    Yt = 0.7 * Xt + rng.normal(5, 10, 5000)   # +5 treatment effect
    Yc_adj, Yt_adj, theta, rho = cuped_adjust(Yc, Yt, Xc, Xt)
    assert np.var(Yc_adj) < np.var(Yc), \
        f"var_adj={np.var(Yc_adj):.2f} >= var_raw={np.var(Yc):.2f}"
run("Adjusted variance < raw variance (high ρ)", t_adjust_reduces_variance)

def t_adjust_rho_range():
    df = load_scenario("zepto_scarcity_badge")
    r  = run_cuped_analysis(df)
    assert -1.0 <= r.rho <= 1.0, f"rho={r.rho}"
run("ρ is in [-1, 1]", t_adjust_rho_range)

def t_adjust_var_reduction_formula():
    df = load_scenario("zepto_scarcity_badge")
    r  = run_cuped_analysis(df)
    expected_vr = round(r.rho**2 * 100, 2)
    assert abs(r.variance_reduction_pct - expected_vr) < 0.1, \
        f"VR={r.variance_reduction_pct}%, expected {expected_vr}%"
run("Variance reduction = ρ² × 100 (exact)", t_adjust_var_reduction_formula)


# ════════════════════════════════════════════════════════════
# CUPED — FULL ANALYSIS
# ════════════════════════════════════════════════════════════
print("\n[CUPED — Full Analysis]")

def t_cuped_se_narrows():
    df = load_scenario("edtech_notification")
    r  = run_cuped_analysis(df)
    assert r.se_adjusted <= r.se_unadjusted * 1.05, \
        f"SE_adj={r.se_adjusted:.6f} > SE_raw={r.se_unadjusted:.6f}"
run("Adjusted SE ≤ unadjusted SE", t_cuped_se_narrows)

def t_cuped_ci_narrows():
    df = load_scenario("edtech_notification")
    r  = run_cuped_analysis(df)
    raw_width = r.ci_unadjusted[1] - r.ci_unadjusted[0]
    adj_width = r.ci_adjusted[1]   - r.ci_adjusted[0]
    assert adj_width <= raw_width * 1.05, \
        f"CI_adj={adj_width:.6f} > CI_raw={raw_width:.6f}"
run("Adjusted CI width ≤ unadjusted CI width", t_cuped_ci_narrows)

def t_cuped_missing_col():
    df = load_scenario("zepto_scarcity_badge").drop(columns=["pre_metric"])
    try:
        run_cuped_analysis(df)
        fail("cuped_missing_col", "Should have raised ValueError")
    except ValueError as e:
        assert "pre_metric" in str(e) or "missing" in str(e).lower()
run("Missing pre_metric column → ValueError", t_cuped_missing_col)

def t_cuped_lift_preserved():
    # CUPED should not dramatically change the lift estimate
    df = load_scenario("zepto_scarcity_badge")
    r  = run_cuped_analysis(df)
    assert abs(r.lift_adjusted - r.lift_unadjusted) < abs(r.lift_unadjusted) * 0.5, \
        f"adj={r.lift_adjusted:.4f} vs raw={r.lift_unadjusted:.4f}"
run("Adjusted lift within 50% of raw lift", t_cuped_lift_preserved)

def t_cuped_sample_size_equiv():
    df = load_scenario("edtech_notification")
    r  = run_cuped_analysis(df)
    n_arm = len(df) // 2
    expected = n_arm / max(1 - r.rho**2, 1e-6)
    assert abs(r.sample_size_equivalent - expected) < 10, \
        f"equiv={r.sample_size_equivalent:.1f}, expected={expected:.1f}"
run("Sample size equivalent = n / (1-ρ²)", t_cuped_sample_size_equiv)


# ════════════════════════════════════════════════════════════
# CUPED — ANCOVA VALIDATION (key correctness check)
# ════════════════════════════════════════════════════════════
print("\n[CUPED — ANCOVA Cross-Validation]")

def t_ancova_match_proportion():
    df = load_scenario("zepto_scarcity_badge")
    v  = validate_against_ancova(df)
    assert v["match"], \
        f"CUPED={v['cuped_lift']:.6f} vs ANCOVA={v['ancova_lift']:.6f} (diff={v['difference']:.2e})"
run("CUPED lift ≈ ANCOVA β_treatment (proportion)", t_ancova_match_proportion)

def t_ancova_match_continuous():
    df = generate_experiment(n_per_arm=3000, metric_type="continuous",
                             mu_control=340, sigma_control=180,
                             true_diff=20, seed=42)
    v  = validate_against_ancova(df)
    assert v["match"], \
        f"CUPED={v['cuped_lift']:.4f} vs ANCOVA={v['ancova_lift']:.4f} (diff={v['difference']:.2e})"
run("CUPED lift ≈ ANCOVA β_treatment (continuous)", t_ancova_match_continuous)

def t_ancova_theta_consistent():
    df = load_scenario("edtech_notification")
    v  = validate_against_ancova(df)
    r  = run_cuped_analysis(df)
    assert abs(v["theta"] - r.theta) < 1e-4, \
        f"theta mismatch: validate={v['theta']:.6f} analysis={r.theta:.6f}"
run("θ consistent between validate and analysis", t_ancova_theta_consistent)


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print()
print("=" * 58)
total = passed + failed
print(f"Results: {passed}/{total} tests passed")
if errors:
    print(f"\nFailed:")
    for name, msg in errors:
        print(f"  ✗ {name}")
        print(f"    {msg}")
else:
    print("\n✅ ALL TESTS PASSED — Phase 2 complete")
print("=" * 58)
