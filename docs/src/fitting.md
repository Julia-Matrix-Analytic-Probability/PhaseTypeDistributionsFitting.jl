```@meta
CurrentModule = PhaseTypeDistributionsFitting
```

# Fitting PH distributions

All estimators follow the Distributions.jl convention: dispatch on the *target
type* and return a new distribution of that type. The target is either the
general `PHDist` or one of the structured families (`CoxianDist`,
`HyperExponentialDist`, `HypoExponentialDist`, `ErlangPHDist`), which constrains
the fit to that family.

```@example fit
using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions, Random, Statistics
rng = MersenneTwister(2024)

truth = HyperExponentialDist([0.6, 0.4], [1.0, 0.25])   # mean 1.0, SCV > 1
data  = rand(rng, truth, 5000)
(empirical_mean = Statistics.mean(data), empirical_scv = var(data) / Statistics.mean(data)^2)
```

## General PH fit

Give the number of phases `m` for a random dense start, or an `init`
distribution to start from (and to impose structure — see below).

```@example fit
phd = fit_mle(PHDist, data; m = 3)
(mean = mean(phd), scv = scv(phd))
```

## Structured fits stay in-family

Each structured target returns its own type and keeps that family's structure
throughout the fit:

```@example fit
cox = fit_mle(CoxianDist, data; m = 3)
hyp = fit_mle(HyperExponentialDist, data; m = 2)
hpo = fit_mle(HypoExponentialDist, data; m = 3)
erl = fit_mle(ErlangPHDist, data; m = 3)     # closed form: rate = m / mean(data)
typeof.((cox, hyp, hpo, erl))
```

```@example fit
mean.((cox, hyp, hpo, erl))
```

Because PH distributions are not identifiable, compare fits by moments or CDF
rather than by `(α, T)`:

```@example fit
moments_isapprox(phd, hyp; rtol = 0.1), distribution_isapprox(phd, hyp; atol = 0.05)
```

## Supplying an initial distribution

Any `AbstractPHDist` can be the starting point. Its **sparsity pattern is
preserved** for the whole run (see [The EM algorithm](em.md)), so seeding a
general `PHDist` fit with a structured distribution imposes that structure:

```@example fit
phd_cox = fit_mle(PHDist, data; init = CoxianDist([3.0, 2.0, 1.0], [0.3, 0.4]))
pattern(subgenerator(phd_cox))     # bidiagonal — the Coxian structure is locked
```

The loop is controlled with `maxiter`, `tol`, `verbose`, and `rng`:

```@example fit
res = fit_mle(PHDist, data; m = 2, maxiter = 500, tol = 1e-9)
mean(res)
```

## The `fit` router

[`fit`](@ref) routes to [`fit_mle`](@ref) (`method = :mle`, the default) or
[`fit_mm`](@ref) (`method = :mm`, method of moments):

```@example fit
f = fit(PHDist, data; method = :mle, m = 2)
mean(f)
```

## API

```@docs
fit_mle(::Type{PhaseTypeDistributions.PHDist}, ::AbstractVector{<:Real})
fit_mle(::Type{PhaseTypeDistributions.CoxianDist}, ::AbstractVector{<:Real})
fit_mle(::Type{PhaseTypeDistributions.HyperExponentialDist}, ::AbstractVector{<:Real})
fit_mle(::Type{PhaseTypeDistributions.HypoExponentialDist}, ::AbstractVector{<:Real})
fit_mle(::Type{PhaseTypeDistributions.ErlangPHDist}, ::AbstractVector{<:Real})
fit(::Type{D}, ::AbstractVector{<:Real}) where {D<:PhaseTypeDistributions.AbstractPHDist}
fit_mm
```
