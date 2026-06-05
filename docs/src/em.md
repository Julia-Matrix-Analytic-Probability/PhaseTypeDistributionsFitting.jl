```@meta
CurrentModule = PhaseTypeDistributionsFitting
```

# The EM algorithm

The maximum-likelihood fit uses the classic Asmussen–Nerman–Olsson EM algorithm
for phase-type distributions.

## The complete-data model

The complete data for one observation `x` is the trajectory of the underlying
Markov jump process up to absorption: which phase it starts in, how long it
spends in each phase, and which transitions it makes. The complete-data
log-likelihood is linear in four sufficient statistics:

- ``B_i`` — number of starts in phase ``i`` (informs ``α``),
- ``Z_i`` — total time spent in phase ``i`` (the denominators of the rates),
- ``M_{ij}`` — number of ``i \to j`` transitions, ``i \neq j`` (informs ``T``),
- ``N_i`` — number of ``i \to`` absorption transitions (informs ``t^0``).

## Exact E-step via a Van Loan block exponential

For ``f(x) = α^⊤ \exp(Tx)\, t^0`` define the row and column vectors
``a(x) = α^⊤\exp(Tx)`` and ``b(x) = \exp(Tx)\,t^0``, and

```math
C(x) = \int_0^x \exp(T^⊤ u)\,(α\, t^{0\,⊤})\,\exp(T^⊤(x-u))\,\mathrm{d}u .
```

By Van Loan's identity, ``C(x)`` is exactly the top-right ``m \times m`` block of

```math
\exp\!\left( \begin{bmatrix} T^⊤ & α\,t^{0\,⊤} \\ 0 & T^⊤ \end{bmatrix} x \right),
```

so a **single ``2m \times 2m`` matrix exponential per observation** yields every
expected sufficient statistic, conditional on the observed absorption time:

```math
\bar B_i = \frac{α_i\, b_i(x)}{f(x)}, \quad
\bar Z_i = \frac{C_{ii}}{f(x)}, \quad
\bar M_{ij} = \frac{T_{ij}\, C_{ij}}{f(x)}, \quad
\bar N_i = \frac{a_i(x)\, t^0_i}{f(x)} .
```

This is *exact* — no quadrature or ODE integration — and the observed-data
log-likelihood ``\sum_\ell \log f(x_\ell)`` falls out at the same time.

## M-step and convergence

Summing the per-observation statistics over the data set, the M-step is
closed-form:

```math
α_i = \frac{\bar B_i}{L}, \qquad
t^0_i = \frac{\bar N_i}{\bar Z_i}, \qquad
T_{ij} = \frac{\bar M_{ij}}{\bar Z_i}\ (i \neq j), \qquad
T_{ii} = -\Big(t^0_i + \sum_{j \neq i} T_{ij}\Big) .
```

The loop alternates E- and M-steps until the increase in log-likelihood drops
below `tol` (or `maxiter` is reached). EM increases the observed-data likelihood
monotonically, a useful internal sanity check.

## Structure preservation

Because ``\bar B \propto α``, ``\bar M \propto T_{\text{off}}`` and
``\bar N \propto t^0``, a `0` factor produces a `0` statistic. On top of that the
M-step is **pattern-aware**: it is handed the sparsity pattern of the starting
parameters — the allowed positions of ``α``, of the off-diagonal of ``T``, and
of the exit vector ``t^0`` — and writes *only* into those positions.

The fitted parameters are returned as the fixed-sparsity arrays from
[FixedSparsityMatrices.jl](https://github.com/yoninazarathy/FixedSparsityMatrices.jl):
`α` as a `FixedSparsityVector` and `T` as a `FixedSparsityMatrix`. The pattern is
therefore not merely preserved by arithmetic but **enforced by the type** —
writing a nonzero into a fixed-zero entry throws.

```@example em
using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions, Random
rng = MersenneTwister(7)
data = rand(rng, HyperExponentialDist([0.5, 0.5], [1.0, 0.3]), 2000)

# Seed a general PHDist fit with a Coxian; the structure is locked in.
phd = fit_mle(PHDist, data; init = CoxianDist([3.0, 2.0, 1.0], [0.3, 0.4]))
pattern(initial_prob(phd)), pattern(subgenerator(phd))
```

```@example em
# The fixed zeros are enforced: this is an error.
try
    subgenerator(phd)[3, 1] = 0.5
catch err
    err
end
```

A dense start has a full pattern, leaving the EM free to fill the whole matrix:

```@example em
phd_dense = fit_mle(PHDist, data; m = 3)
all(pattern(subgenerator(phd_dense)))
```
