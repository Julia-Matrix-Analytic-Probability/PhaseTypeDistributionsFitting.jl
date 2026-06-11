# PhaseTypeDistributionsFitting.jl

[![CI](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributionsFitting.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributionsFitting.jl/actions/workflows/CI.yml)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributionsFitting.jl/)

Fitting algorithms for phase-type (PH) and multi-absorbing phase-type (MAPH)
distributions. Companion package to
[PhaseTypeDistributions.jl](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl),
which provides the distribution types this package fits.

> **Status:** under construction. Built alongside the accompanying paper
> *Inference for Multi-Absorbing Phase Type Distributions, Algorithms, and
> Applications* (Qiao, Surya, Asanjarani, Nazarathy — in preparation).

## Scope

The package covers **fitting PH distributions** (the classic EM) and **fitting
MAPH distributions** (the approximate EM of the accompanying paper, for
competing-risks data).

- **Maximum likelihood via EM (`fit_mle`)** — *implemented for PH.* The classic
  Asmussen–Nerman–Olsson EM algorithm, with an *exact* E-step computed through a
  Van Loan block-matrix exponential (one `2m × 2m` matrix exponential per
  observation gives every expected sufficient statistic). Targets the general
  `PHDist` as well as the structured families (`CoxianDist`,
  `HyperExponentialDist`, `HypoExponentialDist`, `ErlangPHDist`).
- **MAPH fitting (`fit_mle(MAPHDist, data)`)** — *implemented.* The approximate
  EM of the paper for competing-risks observations `(t, k)` (absorption time and
  cause): exact E-step via Van Loan matrix exponentials, a closed-form *relaxed*
  M-step in the inference-oriented `(α, q, R, U)` parameterization, and an
  ℓ1-projection linear program (HiGHS) that restores feasibility whenever the
  relaxed update leaves the valid parameter region. Includes the two
  initialization heuristics of the paper (moment-based and simplified).
- **Moment matching (`fit_mm`)** — *planned* (currently a stub). Will produce a
  PH distribution from empirical moments, and serve as an initialization
  strategy for the EM.

The two estimation approaches are exposed as separate functions in the
Distributions.jl style, with a `fit(...; method=)` router over them.

## Installation

Not yet registered. During development, clone alongside its sibling packages and
`Pkg.develop` them:

```julia
using Pkg
Pkg.develop(path = "path/to/FixedSparsityMatrices.jl")
Pkg.develop(path = "path/to/PhaseTypeDistributions.jl")
Pkg.develop(path = "path/to/PhaseTypeDistributionsFitting.jl")
```

## Quick start

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

# Router form:
phd2 = fit(PHDist, data; method = :mle, m = 3)

# PH distributions are not identifiable — compare fits by moments / CDF:
moments_isapprox(phd, hyp)
distribution_isapprox(phd, hyp)
```

Provide your own starting point with `init` (any `AbstractPHDist`), and tune the
loop with `maxiter`, `tol`, `verbose`, and `rng`:

```julia
fit_mle(PHDist, data; init = CoxianDist([3.0, 2.0, 1.0], [0.3, 0.4]),
        maxiter = 500, tol = 1e-8, verbose = true)
```

Fit an MAPH to competing-risks data — pairs `(t, k)` of absorption time and
cause, exactly what `rand(::MAPHDist, L)` returns:

```julia
truth = MAPHDist([0.5, 0.3, 0.2],
                 [-3.0  1.0  0.5;  0.8 -2.5  0.7;  0.4  0.6 -2.0],   # T
                 [ 1.0  0.5;  0.4  0.6;  0.3  0.7])                  # D
data = rand(truth, 2000)                       # Vector of (t, k)

fitted = fit_mle(MAPHDist, data; m = 3)        # number of causes read off data
marginal_absorption(fitted)                    # ≈ empirical cause frequencies
conditional_time(fitted, 1)                    # τ | κ = 1, as a PHDist
```

MAPH fits accept `init_method = :moment` (default) or `:simplified`, an explicit
`init::MAPHDist`, and the usual `maxiter`/`tol`/`verbose` controls.

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

A dense start (`fit_mle(PHDist, data; m = 3)`) instead has a full pattern, so the
EM is free to fill the whole matrix.

## Documentation

Full documentation — background, the EM algorithm, the fitting API, and worked
examples — is at
<https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributionsFitting.jl/>.

## Relationship to EMpht.jl

An earlier Julia package for PH-EM is
[EMpht.jl](https://github.com/Pat-Laub/EMpht.jl), a port of Asmussen's
`EMpht.c`. PhaseTypeDistributionsFitting.jl supersedes it in scope (PH and MAPH,
integrated with Distributions.jl via PhaseTypeDistributions.jl) and in active
maintenance. We draw on its ideas freely (structured-PH parameterizations,
censoring) and credit them as we go.

## Accompanying Paper

> Zhihao Qiao, Budhi Surya, Azam Asanjarani, Yoni Nazarathy. *Inference for
> Multi-Absorbing Phase Type Distributions, Algorithms, and Applications*. (In
> preparation.)
