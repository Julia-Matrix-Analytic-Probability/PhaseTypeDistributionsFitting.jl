```@meta
CurrentModule = PhaseTypeDistributionsFitting
```

# PhaseTypeDistributionsFitting.jl

Fitting algorithms for
[phase-type (PH)](https://en.wikipedia.org/wiki/Phase-type_distribution) and
multi-absorbing phase-type (MAPH) distributions. Companion package to
[PhaseTypeDistributions.jl](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl),
which provides the distribution types this package fits.

The package accompanies the paper *Inference for Multi-Absorbing Phase Type
Distributions, Algorithms, and Applications* (Qiao, Surya, Asanjarani,
Nazarathy; in preparation).

## Background

A continuous-time PH distribution is the distribution of the absorption time of
a finite-state continuous-time Markov chain with one absorbing state. Given an
initial distribution `α` over `m` transient phases and a sub-generator matrix
`T`, the absorption time `τ` has density and tail

```math
f_τ(x) = α^⊤ \exp(Tx)\, t^0, \qquad
\bar F_τ(x) = α^⊤ \exp(Tx)\, \mathbf 1, \qquad
t^0 = -T \mathbf 1 .
```

Given i.i.d. samples `x₁, …, x_L` of `τ`, this package estimates `(α, T)` by
maximum likelihood using the **EM algorithm**, with the structured PH families
(`CoxianDist`, `HyperExponentialDist`, `HypoExponentialDist`, `ErlangPHDist`)
available as constrained targets.

## Scope

| Estimator | Status | Notes |
|-----------|--------|-------|
| [`fit_mle`](@ref) (EM)            | **implemented for PH** | exact E-step via a Van Loan block-matrix exponential |
| [`fit_mm`](@ref) (moment matching) | planned (stub)         | will also seed the EM |
| MAPH fitting                      | planned                | follows the accompanying paper |

## Installation

Not yet registered. During development, `Pkg.develop` it alongside its sibling
packages:

```julia
using Pkg
Pkg.develop(path = "path/to/FixedSparsityMatrices.jl")
Pkg.develop(path = "path/to/PhaseTypeDistributions.jl")
Pkg.develop(path = "path/to/PhaseTypeDistributionsFitting.jl")
```

## Quick start

```@example quick
using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions, Random, Statistics
rng = MersenneTwister(1)

# Synthetic data from a known hyperexponential.
data = rand(rng, HyperExponentialDist([0.7, 0.3], [1.0, 0.2]), 5000)

phd = fit_mle(PHDist, data; m = 3)          # general PH fit
hyp = fit_mle(HyperExponentialDist, data; m = 2)

(mean_data = Statistics.mean(data), mean_phd = mean(phd), mean_hyp = mean(hyp))
```

```@example quick
# PH distributions are not identifiable — compare fits by moments / CDF.
moments_isapprox(phd, hyp; rtol = 0.1), distribution_isapprox(phd, hyp; atol = 0.05)
```

## Navigation

- [Fitting PH distributions](fitting.md) — the full fitting API and worked examples.
- [The EM algorithm](em.md) — how the exact E-step and M-step work, and structure preservation.
- [API reference](api.md) — index of every documented name.

## Relationship to EMpht.jl

An earlier Julia package for PH-EM is
[EMpht.jl](https://github.com/Pat-Laub/EMpht.jl), a port of Asmussen's
`EMpht.c`. PhaseTypeDistributionsFitting.jl supersedes it in scope (PH and MAPH,
integrated with Distributions.jl via PhaseTypeDistributions.jl) and in active
maintenance. We draw on its ideas freely (structured-PH parameterizations,
censoring) and credit them as we go.
