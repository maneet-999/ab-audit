# Research Reading Guide
## Statistical Validity Audit Engine — Phase 0 Study Notes

Before writing any Phase 1 code, read each paper below and extract
the specific equations noted. This file tracks what to take from each
source and exactly where it maps to in the engine.

---

## Paper 1 — Kohavi, Tang & Xu (2020)
**"Trustworthy Online Controlled Experiments"**
Cambridge University Press
Chapters to read: **3, 5, 8, 19**

### Chapter 3 — Sample Ratio Mismatch (SRM)
Maps to → `checks.py :: check_srm()`

Key concepts to extract:
- Definition: SRM occurs when the observed assignment ratio
  differs significantly from the designed ratio
- Detection: chi-square goodness-of-fit test on assignment counts
- Threshold: p < 0.001 (stricter than usual 0.05 — because SRM
  is almost always a pipeline bug, not sampling noise)
- Common causes: bot filtering after assignment, logging bugs,
  browser redirect issues, cache effects

Equation to note:
```
χ² = Σ (O_i - E_i)² / E_i
   = (n_ctrl - N/2)² / (N/2) + (n_trt - N/2)² / (N/2)

where N = n_ctrl + n_treatment
      E_i = N × expected_ratio_i
      df = k - 1 = 1  (two arms)
```

---

### Chapter 5 — Multiple Testing
Maps to → `checks.py :: check_multiple_testing()`

Key concepts:
- Family-Wise Error Rate (FWER): probability of at least one
  false positive across k simultaneous tests
- Bonferroni correction: conservative, controls FWER
- Benjamini-Hochberg: controls False Discovery Rate (FDR),
  less conservative — preferred when k is large

Equations:
```
FWER = 1 - (1 - α)^k

Bonferroni corrected threshold:
  α_i = α / k

Benjamini-Hochberg step-up procedure:
  1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(k)
  2. Find largest j such that p_(j) ≤ (j/k) × α
  3. Reject H₀ for all i ≤ j
```

---

### Chapter 8 — Novelty & Primacy Effects
Maps to → `checks.py :: check_novelty_effect()`

Key concepts:
- Novelty effect: users engage more with new features initially
  simply because they are new — lift inflates early then decays
- Primacy effect: control users may perform worse initially due to
  disruption (inverse of novelty)
- Detection: plot day-by-day treatment effect; fit a linear trend
- A significant negative slope = novelty effect likely
- Recommendation: run experiment until effect stabilises

---

### Chapter 19 — CUPED
Maps to → `cuped.py :: cuped_adjust()`

This chapter is the most important one for your differentiation.

Key equation:
```
Ỹ_i = Y_i - θ × (X_i - X̄)

where:
  Y_i  = outcome metric for user i during experiment
  X_i  = same metric for user i in pre-experiment period
  X̄    = mean of pre-experiment metric across all users
  θ    = Cov(Y, X) / Var(X)   ← estimated via OLS

Variance of adjusted estimator:
  Var(Ỹ) = Var(Y) × (1 - ρ²)

where ρ = Corr(Y, X)

Variance reduction:
  Reduction % = ρ² × 100

So if pre/post correlation ρ = 0.5:
  Variance reduces by 25% → CI narrows by ~13%
  Equivalent to running the experiment on 33% more users
```

---

## Paper 2 — Deng, Xu, Kohavi & Walker (2013)
**"Improving the Sensitivity of Online Controlled Experiments
 by Utilizing Pre-Experiment Data"**
KDD 2013, Microsoft Research

Maps to → `cuped.py` (primary theoretical foundation)

### Key derivation to understand

The goal: reduce Var(δ̂) where δ̂ = ȳ_T - ȳ_C

Without CUPED:
```
Var(δ̂) = Var(Y_T)/n_T + Var(Y_C)/n_C
        ≈ 2σ²/n  (balanced design)
```

With CUPED:
```
Var(δ̃) = Var(Ỹ_T)/n_T + Var(Ỹ_C)/n_C
        = 2σ²(1-ρ²)/n

Reduction factor = (1 - ρ²)
```

The CUPED estimator is unbiased because E[X_i - X̄] = 0 under
randomisation — treatment does not affect the pre-experiment period.

Equivalence to regression:
CUPED is equivalent to running an ANCOVA regression:
```
Y_i = α + τ × Treatment_i + β × X_i + ε_i

The OLS estimate of τ is identical to the CUPED estimator.
```

This is your soundbite for interviews:
*"CUPED is ANCOVA with pre-experiment data as the covariate.
It's a control variate method borrowed from Monte Carlo theory."*

---

## Paper 3 — Johari, Pekelis & Walsh (2015)
**"Always Valid Inference: Bringing Sequential Analysis
 to A/B Testing"**
arXiv:1512.04922

Maps to → `checks.py :: check_peeking()` and `simulation.py`

### The peeking problem (what your Monte Carlo proves empirically)

Standard frequentist testing assumes a fixed sample size decided
before the experiment. If you check results k times and stop when
significant, the true Type I error rate inflates:

```
Approximate inflated α when checking at n, 2n, 3n, ... kn:

α_actual ≈ α × (1 + 0.5 × log(k))   [rough approximation]

More precisely — via simulation:
  k=1  checks → α_actual ≈ 0.050
  k=5  checks → α_actual ≈ 0.142
  k=10 checks → α_actual ≈ 0.193
  k=∞  checks → α_actual → 1.000
```

This is exactly what your Monte Carlo in check_peeking() will
demonstrate empirically. The simulation IS the proof.

### Always-Valid p-values (the fix)
The paper derives a sequential test statistic that remains valid
at any stopping time. Key idea: use a mixture of sequential
probability ratio tests (mSPRT).

For implementation in Phase 2:
```
Λ_t = p(data | H₁) / p(data | H₀)

Under H₀, E[1/Λ_t] ≤ 1 at all times t (optional stopping theorem)
Reject when Λ_t > 1/α

This gives valid α-level test at ANY stopping time.
```

Your tool surfaces this in the peeking simulator:
show classical p-value trajectory vs. always-valid p-value —
the classical one zigzags past 0.05 randomly; the always-valid
one has a proper boundary.

---

## Paper 4 — Benjamini & Hochberg (1995)
**"Controlling the False Discovery Rate: A Practical and Powerful
 Approach to Multiple Testing"**
Journal of the Royal Statistical Society, Series B

Maps to → `checks.py :: check_multiple_testing()`

This paper is 12 pages. Read the full thing.

### The BH Procedure (step by step)

```
Given m hypothesis tests with p-values p_1, ..., p_m:

Step 1: Sort p-values in ascending order
        p_(1) ≤ p_(2) ≤ ... ≤ p_(m)

Step 2: For each p_(i), compute the BH threshold
        BH_threshold(i) = (i / m) × α

Step 3: Find the largest k such that
        p_(k) ≤ (k / m) × α

Step 4: Reject H₀ for all tests i = 1, 2, ..., k

FDR control guarantee:
  E[FDP] ≤ (m₀/m) × α ≤ α

where FDP = false discoveries / total discoveries
      m₀  = true null hypotheses (unknown, bounded by m)
```

When to use BH vs Bonferroni:
- Bonferroni: use when ANY false positive is unacceptable
  (e.g., clinical trials, policy decisions)
- BH: use when you can tolerate a controlled proportion of
  false positives (e.g., feature experiments, product testing)
- For A/B testing in tech: BH is almost always the right choice

---

## Paper 5 — Fisher (1935)
**"The Design of Experiments"**
Chapters 1, 2, 3

Maps to → conceptual foundation; cite in paper introduction

### Key ideas to absorb

1. The Lady Tasting Tea experiment — the original motivation
   for randomisation as the foundation of valid inference.
   Know this story. You will tell it in interviews.

2. The null hypothesis was Fisher's invention.
   "We may speak of this hypothesis as the 'null hypothesis',
   and it should be noted that the null hypothesis is never
   proved or established, but is possibly disproved."

3. p-value interpretation (Fisher's original):
   "The p-value is the probability of observing data at least
   as extreme as what was observed, assuming H₀ is true."
   NOT: "the probability that H₀ is true." This distinction
   matters and is commonly confused.

4. Randomisation as the physical basis of inference:
   Without randomisation, there is no valid statistical test.
   This is why SRM (Check 1) invalidates everything downstream.

---

## Reading Schedule

| Day | Paper | Sections | Output |
|-----|-------|----------|--------|
| 1 | Kohavi 2020 | Ch. 3, 5 | Notes on SRM + multiple testing equations |
| 2 | Deng 2013 | Full | CUPED θ formula written in your notebook |
| 3 | Johari 2015 | Sections 1-4 | Peeking inflation table, mSPRT concept |
| 3 | BH 1995 | Full | Step-up procedure written out by hand |
| 4 | Kohavi 2020 | Ch. 8, 19 | Novelty effect criteria, CUPED from industry POV |
| 5 | Fisher 1935 | Ch. 1-3 | Lady Tasting Tea story, null hypothesis origin |

Total reading time: ~14–18 hours across 5 days.
Do NOT skip any. You will be asked about these in interviews.

---

## Interview Soundbites — Memorise These

**On SRM:**
"Sample Ratio Mismatch is the most common silent bug in A/B testing.
If your groups aren't the size you intended, your randomisation pipeline
is broken and all statistical guarantees are void. We test for it first."

**On peeking:**
"Every time you check a p-value mid-experiment, you burn some of your
Type I error budget. Check 10 times at α=0.05 and your actual false
positive rate is nearly 20%. Our Monte Carlo proves this empirically —
you can watch it happen."

**On CUPED:**
"CUPED uses each user's pre-experiment behaviour to absorb variance
unrelated to the treatment. It's mathematically equivalent to ANCOVA.
In practice it lets you detect the same effect with 20-40% fewer users —
or equivalently, run a more sensitive experiment on the same traffic."

**On multiple testing:**
"Running 10 A/B tests simultaneously at α=0.05 gives you a 40% chance
of at least one false positive. Benjamini-Hochberg controls the
False Discovery Rate — a less conservative correction than Bonferroni
that's appropriate for product experimentation."
