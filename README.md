# PhaseTypeDistributionsFitting.jl

Fitting algorithms for phase-type (PH) and multi-absorbing phase-type (MAPH)
distributions. Companion package to
[PhaseTypeDistributions.jl](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl),
which provides the distribution types this package fits.

> **Status:** under construction. Built alongside the accompanying paper
> *Inference for Multi-Absorbing Phase Type Distributions, Algorithms, and
> Applications* (Qiao, Surya, Asanjarani, Nazarathy — in preparation).

## Scope

- **Moment matching (`fit_mm`)** — produce a PH or MAPH distribution from
  empirical moments of data. Useful as a stand-alone fit, and as the default
  initialization strategy for maximum-likelihood fits.
- **Maximum likelihood via EM (`fit_mle`)** — for both PH and MAPH, including
  structured PH families (Coxian, canonical forms) and censored / interval
  data.
- **Initialization helpers** — pluggable strategies (moment-matched,
  EMpht-style, random) consumed by the EM routines.

The package is positioned as the natural fitting layer for the
PhaseTypeDistributions.jl ecosystem.

## Relationship to EMpht.jl

An earlier Julia package for PH-EM is
[EMpht.jl](https://github.com/Pat-Laub/EMpht.jl), a port of Asmussen's
`EMpht.c`. PhaseTypeDistributionsFitting.jl supersedes it in scope (PH **and**
MAPH, integrated with `Distributions.jl` via PhaseTypeDistributions.jl) and in
active maintenance. We borrow ideas freely (uniformization-based E-step,
structured-PH parameterizations, censoring) and credit them as we go.

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
