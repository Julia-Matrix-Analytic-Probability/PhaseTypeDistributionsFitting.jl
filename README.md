# PhaseTypeDistributionsFitting.jl

[![CI](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributionsFitting.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributionsFitting.jl/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributionsFitting.jl/)

Maximum-likelihood fitting of phase-type (PH) and multi-absorbing phase-type
(MAPH) distributions. Companion package to
[PhaseTypeDistributions.jl](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl),
which provides the distribution types this package fits, and to the paper
*Inference for Multi-Absorbing Phase Type Distributions, Algorithms, and
Applications* (Qiao, Surya, Asanjarani, Nazarathy — in preparation), whose
algorithms it implements.

## What it does

- **PH fitting, `fit_mle(PHDist, data; m)`** — the classic
  Asmussen–Nerman–Olsson EM algorithm for absorption-time data, with an *exact*
  E-step computed through a Van Loan block-matrix exponential (one `2m × 2m`
  matrix exponential per observation yields every expected sufficient
  statistic). Also targets the structured families — `CoxianDist`,
  `HyperExponentialDist`, `HypoExponentialDist`, `ErlangPHDist` — with fits
  that stay in-family and return the specialised type.
- **MAPH fitting, `fit_mle(MAPHDist, data; m)`** — the approximate EM of the
  accompanying paper for competing-risks observations `(t, k)` (absorption time
  and cause): exact E-step via Van Loan matrix exponentials, a closed-form
  *relaxed* M-step in the inference-oriented `(α, q, R, U)` parameterization,
  and an ℓ1-projection linear program (solved with the open-source
  [HiGHS](https://highs.dev) solver) that restores feasibility whenever the
  relaxed update leaves the valid parameter region. Includes the two
  initialization heuristics of the paper (moment-based and simplified), a
  stopping rule robust to the non-monotone iteration, and a guard against
  degenerate (non-absorbing) parameter configurations.
- **`fit(...; method = :mle)`** — a Distributions.jl-style router over the
  estimation methods. (`fit_mm`, moment matching, is reserved and currently a
  stub that throws.)

## Installation

```julia
using Pkg
Pkg.add("PhaseTypeDistributionsFitting")
```

Until the registration in the Julia General registry is merged, install
directly from GitHub (all dependencies are registered):

```julia
Pkg.add(url = "https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributionsFitting.jl")
```

## Quick start: PH

```julia
using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions

# Synthetic data from a known hyperexponential.
data = rand(HyperExponentialDist([0.7, 0.3], [1.0, 0.2]), 5000)

# General PH fit by EM (give the number of phases m, or an `init` distribution):
phd = fit_mle(PHDist, data; m = 3)

# Structured fits stay in-family and return the specialised type:
cox = fit_mle(CoxianDist, data; m = 3)               # a CoxianDist
hyp = fit_mle(HyperExponentialDist, data; m = 2)     # a HyperExponentialDist
hpo = fit_mle(HypoExponentialDist, data; m = 3)      # a HypoExponentialDist
erl = fit_mle(ErlangPHDist, data; m = 3)             # closed-form rate = m / mean(data)

# PH distributions are not identifiable — compare fits by moments / CDF:
moments_isapprox(phd, hyp)
distribution_isapprox(phd, hyp)
```

Provide your own starting point with `init` (any `AbstractPHDist`), and tune
the loop with `maxiter`, `tol`, `verbose`, and `rng`:

```julia
fit_mle(PHDist, data; init = CoxianDist([3.0, 2.0, 1.0], [0.3, 0.4]),
        maxiter = 500, tol = 1e-8, verbose = true)
```

## Quick start: MAPH (competing risks)

An MAPH law describes the pair `(τ, κ)` — time to absorption together with the
cause of absorption (one of `n` competing causes). The data are pairs `(t, k)`,
exactly what `rand(::MAPHDist, L)` returns:

```julia
truth = MAPHDist([0.5, 0.3, 0.2],
                 [-3.0  1.0  0.5;  0.8 -2.5  0.7;  0.4  0.6 -2.0],   # T
                 [ 1.0  0.5;  0.4  0.6;  0.3  0.7])                  # D
data = rand(truth, 2000)                       # Vector of (t, k)

fitted = fit_mle(MAPHDist, data; m = 3)        # number of causes read off data
marginal_absorption(fitted)                    # ≈ empirical cause frequencies
conditional_time(fitted, 1)                    # τ | κ = 1, as a PHDist
```

MAPH fits accept `init_method = :moment` (default) or `:simplified`, an
explicit `init::MAPHDist`, and the usual `maxiter`/`tol`/`verbose` controls.
Because MAPH distributions are not identifiable, compare a fit to a reference
through distributional summaries (`marginal_absorption`, `conditional_time`,
`cdf`), not through the parameters themselves.

## Structure preservation

EM updates here are *zero-preserving*. The fitted parameters are stored as the
fixed-sparsity arrays from
[FixedSparsityMatrices.jl](https://github.com/yoninazarathy/FixedSparsityMatrices.jl)
(re-exported through PhaseTypeDistributions.jl): `α` is a `FixedSparsityVector`
and `T` a `FixedSparsityMatrix`, each carrying a **fixed sparsity pattern**. The
EM is given that pattern at the start, and its pattern-aware M-step only ever
writes into the allowed positions — a structural zero in `α`, in an off-diagonal
of `T`, or in the exit vector `t⁰` is reproduced exactly each iteration, and the
result type *enforces* it (writing a nonzero into a fixed-zero position throws).

So a Coxian stays Coxian, a hyperexponential stays diagonal, a hypoexponential
absorbs only from its last phase, and any zeros you supply via an `init`
distribution are respected throughout the fit:

```julia
# Fit a general PHDist but seed it with a Coxian: the bidiagonal structure and
# α = [1, 0, 0] are locked in for the whole run.
cox = fit_mle(CoxianDist, data; m = 3)
phd = fit_mle(PHDist, data; init = cox)

pattern(initial_prob(phd))   # Bool[1, 0, 0]
pattern(subgenerator(phd))   # Bool[1 1 0; 0 1 1; 0 0 1]

subgenerator(phd)[3, 1] = 0.5   # ERROR: that position is fixed to zero
```

A dense start (`fit_mle(PHDist, data; m = 3)`) instead has a full pattern, so
the EM is free to fill the whole matrix.

## Current limitations

- **Fully observed data only.** Both the PH and the MAPH EM require every
  observation to be an exact absorption time (and cause). Right-censored
  observations are not yet supported; extending the E-step to censoring is
  planned.
- **`fit_mm` is a stub.** Moment matching as a standalone estimator is not yet
  implemented (the moment-based construction is currently used internally, as
  an EM initialization).
- **The MAPH EM is approximate.** Its M-step maximizes a relaxed surrogate and
  then projects back to the feasible region, so the log-likelihood is not
  monotone; the stopping rule accounts for this, but convergence theory is an
  open question (see the paper).

## Documentation

Full documentation — background, the EM algorithm, the fitting API, and worked
examples — is at
<https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributionsFitting.jl/>.

## Relationship to EMpht.jl

An earlier Julia package for PH-EM is
[EMpht.jl](https://github.com/Pat-Laub/EMpht.jl), a port of Asmussen's
`EMpht.c`. PhaseTypeDistributionsFitting.jl extends its scope (PH *and* MAPH,
integrated with Distributions.jl via PhaseTypeDistributions.jl, with
structure-preserving updates), though EMpht.jl handles censored and binned
data, which this package does not yet. We draw on its ideas freely and credit
them as we go.

## Citing

If you use this package in academic work, please cite the accompanying paper:

> Zhihao Qiao, Budhi Surya, Azam Asanjarani, Yoni Nazarathy. *Inference for
> Multi-Absorbing Phase Type Distributions, Algorithms, and Applications*. (In
> preparation.)
