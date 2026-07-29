```@meta
CurrentModule = PhaseTypeDistributionsFitting
```

# Fitting MAPH distributions

A multi-absorbing phase-type (MAPH) distribution is the joint law of the pair
`(τ, κ)` — the absorption time and the index of the absorbing state reached —
of a continuous-time Markov chain with `m` transient phases and `n ≥ 1`
absorbing states. It is the natural phase-type model for **competing-risks
data**: `τ` is the time of the event and `κ` its cause. The distribution type
is [`MAPHDist`](https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributions.jl/)
from PhaseTypeDistributions.jl, parameterized by `(α, T, D)` with `α` the
initial distribution, `T` the `m × m` sub-generator and `D` the `m × n` matrix
of absorption rates.

Given i.i.d. observations of `(τ, κ)` alone — the trajectory of the underlying
chain is latent — `fit_mle(MAPHDist, data; ...)` estimates `(α, T, D)` with the
approximate EM algorithm of the accompanying paper.

## Quick start

```julia
using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions

# Synthetic competing-risks data from a known MAPH (m = 3 phases, n = 2 causes).
T = [-3.0  1.0  0.5;
      0.8 -2.5  0.7;
      0.4  0.6 -2.0]
D = [1.0 0.5;
     0.4 0.6;
     0.3 0.7]
truth = MAPHDist([0.5, 0.3, 0.2], T, D)
data = rand(truth, 2000)        # Vector of (t, k) pairs

# Fit with 3 phases. The number of causes is read off the data.
fitted = fit_mle(MAPHDist, data; m = 3)

# MAPH distributions are not identifiable — compare through summaries:
marginal_absorption(fitted)      # ≈ empirical cause frequencies
conditional_time(fitted, 1)      # conditional τ | κ = 1, as a PHDist
pdf(fitted, 1.3, 2)              # joint sub-density f(t, k)
```

The data format is a vector of `(t, k)` tuples with `t > 0` and `k` a positive
cause index — exactly what `rand(::MAPHDist, L)` returns.

## Right-censored observations

Real competing-risks data are rarely complete: some subjects leave the study
before their event, and all that is known about them is that they were still
event-free at some time `c`. Pass those censoring times as `censored`:

```julia
# 25% of subjects are censored at a fixed administrative horizon.
sim = rand_censored(truth, 2000, 1.5)         # (events, censored)

fitted = fit_mle(MAPHDist, sim.events; m = 3, censored = sim.censored)
```

A censored record at `c` is the observation `{τ > c}`: it contributes the
survival `α exp(Tc) 1` to the likelihood, and in the E-step its latent path is
completed *through its eventual absorption* — both before and after `c` — with
its sufficient statistics spread over every possible eventual cause in
proportion to the current model's posterior probability of reaching it. The
complete-data likelihood and the M-step are therefore exactly the ones used for
exact events; only the E-step changes.

Two assumptions matter. The censoring must be **independent** of `(τ, κ)` and
non-informative, so that its law contributes no MAPH parameters; and at least
one exact event must be present, since the number of causes and the reference
cause of the second parameterization are read off the events.

!!! warning
    Do not encode a censoring time as an event time. Doing so both shortens the
    fitted time scale and attributes the record to a cause it may never have
    reached — the fitted absorption probabilities and conditional means are
    then badly biased.

## The algorithm

Each iteration performs the steps of the paper's approximate EM:

1. **E-step.** For each exact observation `(t, k)`, the conditional expectations
   of the complete-data sufficient statistics — the start-phase indicator `B̄`,
   holding times `Z̄`, transient jump counts `M̄`, and absorbing-jump indicator
   `Ē` — are computed *exactly* from one `2m × 2m` Van Loan block-matrix
   exponential, the same device as the PH E-step (see
   [The EM algorithm](em.md)) with the exit vector replaced by the observed
   cause's column `D[:, k]`. Each right-censored record at `c` costs two such
   exponentials — not one per cause: because the columns of `R = -T⁻¹D` sum to
   `1`, the all-cause totals collapse onto a single evaluation and only the
   reference-cause slice needs a second.
2. **Relaxed M-step.** In the second parameterization `(α, q, R, U)` — exit
   rates `q`, absorption-probability matrix `R = -T⁻¹D`, and the conditional
   one-step jump matrix `U` taken with respect to a *reference cause* (the most
   frequent one in the data) — the relaxed complete-data likelihood maximizer
   is available in closed form.
3. **Constraint enforcement.** The relaxed `U` may leave the polytope on which
   `(α, q, R, U)` corresponds to a valid MAPH (equivalently, on which
   `D = -T·R ≥ 0`). Violating rows are returned to it by an ℓ1-projection — a
   small row-separable linear program solved with
   [HiGHS](https://github.com/jump-dev/HiGHS.jl). How often this fires is
   reported by the internal EM result (`nprojections`) and is a useful
   diagnostic for how binding the constraints are.
4. **Conversion.** `(α, q, R, U)` is mapped back to `(α, T, D)` for the next
   E-step.

Because the M-step maximizes a relaxed surrogate and then projects, the
observed-data log-likelihood is **not guaranteed to increase monotonically**;
convergence is declared when its change stays below `tol` for a few consecutive
iterations.

## Initialization

Two data-driven starting points are provided (Appendix C of the paper):

- [`maph_moment_init`](@ref) (`init_method = :moment`): routes an exponential
  front end into per-cause hyper-/hypo-exponential blocks so that the empirical
  absorption probabilities and the per-cause conditional means and SCVs are
  matched (exactly, up to a small regularization that restores reachability and
  strict positivity of `α`).
- [`maph_simplified_init`](@ref) (`init_method = :simplified`): matches the
  absorption probabilities and the overall mean only, with every parameter
  strictly positive. A deterministic per-phase rate jitter breaks the phase
  exchangeability of the textbook construction — without it the EM could never
  differentiate the phases.

The default `init_method = :auto` picks `:moment` for complete data and
`:simplified` when there is censoring. The reason is that the moment
initialization targets the *conditional* mean and SCV of `τ | κ = k`, which the
raw event-time moments no longer estimate once records are censored — they see
only the short times. The simplified initialization instead uses the
observed-event proportions and the exposure-per-event scale
`(Σ yℓ)/#events`, both of which stay valid under independent censoring. To use
the moment initialization on censored data anyway, supply externally adjusted
targets:

```julia
fit_mle(MAPHDist, events; m = 4, censored = cens, init_method = :moment,
        init_kwargs = (cond_means = μ̂, cond_scvs = ĉ²))
```

You can also pass any `MAPHDist` as `init`. Structural zeros of `α` and of the
off-diagonal of `T` are EM-invariant and survive the fit; zeros of `D` are
*not* preserved (it is regenerated from `(q, R, U)` each iteration).

## API

```@docs
fit_mle(::Type{PhaseTypeDistributions.MAPHDist}, ::AbstractVector{<:Tuple{<:Real, <:Integer}})
maph_moment_init
maph_simplified_init
```

The [`fit`](@ref) router also accepts `MAPHDist` with `method = :mle`.
