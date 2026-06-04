# PhaseTypeDistributionsFitting.jl

Fitting algorithms for phase-type (PH) and multi-absorbing phase-type (MAPH)
distributions. Companion package to
[PhaseTypeDistributions.jl](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl),
which provides the distribution types this package fits.

> **Status:** under construction. Built alongside the accompanying paper
> *Inference for Multi-Absorbing Phase Type Distributions, Algorithms, and
> Applications* (Qiao, Surya, Asanjarani, Nazarathy — in preparation).

## Scope

The package is built in two stages: **(1) fitting PH distributions** (current
focus), then **(2) fitting MAPH distributions** (the subject of the accompanying
paper).

- **Maximum likelihood via EM (`fit_mle`)** — *implemented for PH.* The classic
  Asmussen–Nerman–Olsson EM algorithm (as popularised by
  [EMpht.jl](https://github.com/Pat-Laub/EMpht.jl) / `EMpht.c`), with an exact
  E-step computed through a Van Loan block-matrix exponential. Targets the
  general `PHDist` as well as the structured families
  (`CoxianDist`, `HyperExponentialDist`, `HypoExponentialDist`, `ErlangPHDist`).
- **Moment matching (`fit_mm`)** — *planned* (currently a stub). Will produce a
  PH distribution from empirical moments, and serve as an initialization
  strategy for the EM.
- **MAPH fitting** — *planned*, following the accompanying paper.

The two estimation approaches are exposed as separate functions in the
Distributions.jl style, with a `fit(...; method=)` router over them.

### Structure preservation

EM updates here are *zero-preserving*: a structural zero in `α`, in an
off-diagonal of `T`, or in the exit vector is reproduced exactly each iteration.
So a Coxian stays Coxian, a hyperexponential stays diagonal, and any zeros you
supply explicitly via an `init` distribution are respected throughout the fit.

## Usage

```julia
using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions

data = rand(HyperExponentialDist([0.7, 0.3], [1.0, 0.2]), 5000)  # synthetic data

# General PH fit by EM (give the number of phases m, or an `init` distribution):
phd = fit_mle(PHDist, data; m = 3)

# Structured fits stay in-family:
cox = fit_mle(CoxianDist, data; m = 3)               # returns a CoxianDist
hyp = fit_mle(HyperExponentialDist, data; m = 2)     # returns a HyperExponentialDist
erl = fit_mle(ErlangPHDist, data; m = 3)             # closed-form rate = m / mean(data)

# Router form:
phd2 = fit(PHDist, data; method = :mle, m = 3)

# PH distributions are not identifiable — compare fits by moments / CDF:
moments_isapprox(phd, hyp); distribution_isapprox(phd, hyp)
```

## Relationship to EMpht.jl

An earlier — and no longer actively maintained — Julia package for PH-EM is
[EMpht.jl](https://github.com/Pat-Laub/EMpht.jl) (Patrick Laub), itself a port
of Asmussen's `EMpht.c`. Our PH `fit_mle` is based on the same Asmussen EM
approach, and we gratefully build on that work. PhaseTypeDistributionsFitting.jl
aims to supersede EMpht.jl in scope (PH **and**, eventually, MAPH; integrated
with `Distributions.jl` via PhaseTypeDistributions.jl) and in active
maintenance. We borrow ideas freely (the EM E-step, structured-PH
parameterizations, censoring) and credit them as we go.

## Installation

Not yet registered. During development, clone alongside
PhaseTypeDistributions.jl and `Pkg.develop` both:

```julia
using Pkg
Pkg.develop(path="path/to/PhaseTypeDistributions.jl")
Pkg.develop(path="path/to/PhaseTypeDistributionsFitting.jl")
```

## Accompanying Paper

> Zhihao Qiao, Budhi Surya, Azam Asanjarani, Yoni Nazarathy. *Inference for
> Multi-Absorbing Phase Type Distributions, Algorithms, and Applications*. (In
> preparation.)
