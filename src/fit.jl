# Public fitting API.
#
# Two estimation approaches, exposed as separate functions in the Distributions.jl
# style (type-as-target dispatch, returning a new distribution):
#
#   fit_mle(T, data; …)   maximum likelihood via the EM algorithm (implemented)
#   fit_mm(T, data; …)    method of moments                       (stub, planned)
#   fit(T, data; method)  convenience router over the two
#
# `T` is the target distribution type — `PHDist` for a general fit, or one of the
# structured families (`CoxianDist`, `HyperExponentialDist`, `HypoExponentialDist`,
# `ErlangPHDist`) to constrain the fit to that family. Structured fits initialise
# with the family's sparsity pattern and rely on the EM's zero-preservation
# (see `em.jl`) to stay in-family, returning the specialised type.

# ---- helpers ----

function _check_data(data::AbstractVector{<:Real})
    isempty(data) && throw(ArgumentError("data must be non-empty"))
    all(>(0), data) || throw(ArgumentError("all observations must be strictly positive"))
    return nothing
end

# Renormalise α defensively against floating-point drift (the EM update already
# yields a (near) probability vector).
function _safe_alpha(α::AbstractVector{<:Real})
    s = sum(α)
    s > 0 || throw(ArgumentError("fitted α summed to $s; cannot normalise"))
    return collect(Float64, α) ./ s
end

# Obtain the starting (α, T) for an EM run from either an explicit `init`
# distribution or a structure-specific initializer `initfn(m, data; rng)`. For an
# `init` distribution the FixedSparsity α/T of the resulting `PHDist` are returned
# directly, so the EM picks up its exact sparsity pattern (see `_em`).
function _starting_point(initfn, m, init, data, rng)
    if init !== nothing
        phd = PHDist(init)            # accepts any AbstractPHDist; keeps its zeros
        m === nothing || m == nphases(phd) ||
            throw(ArgumentError("m=$m conflicts with init of size $(nphases(phd))"))
        return initial_prob(phd), subgenerator(phd)
    end
    m === nothing && throw(ArgumentError("provide either `m` (number of phases) or `init`"))
    return initfn(m, data; rng=rng)
end

# ---- reconstruction of structured types from a fitted (α, T) ----
# `α`/`T` arrive as FixedSparsity arrays from the EM; these read through the
# standard AbstractArray interface. The EM preserved the family's structure (by
# the pattern-aware M-step), so clamps guard only against floating-point drift.

function _to_coxian(α::AbstractVector{<:Real}, T::AbstractMatrix{<:Real})
    m = length(α)
    rates = -diag(T)
    all(rates .> 0) || throw(ArgumentError("fitted Coxian has a non-positive rate: $rates"))
    t0 = -vec(sum(T; dims=2))
    exit_probs = [clamp(t0[i] / rates[i], 0.0, 1.0) for i in 1:(m - 1)]
    return CoxianDist(rates, exit_probs)
end

function _to_hyper(α::AbstractVector{<:Real}, T::AbstractMatrix{<:Real})
    rates = -diag(T)
    all(rates .> 0) || throw(ArgumentError("fitted hyperexponential has a non-positive rate: $rates"))
    return HyperExponentialDist(_safe_alpha(α), rates)
end

function _to_hypo(α::AbstractVector{<:Real}, T::AbstractMatrix{<:Real})
    rates = -diag(T)
    all(rates .> 0) || throw(ArgumentError("fitted hypoexponential has a non-positive rate: $rates"))
    return HypoExponentialDist(rates)
end

# ===========================================================================
# fit_mle  (EM)
# ===========================================================================

"""
    fit_mle(PHDist, data; m, init, maxiter=1000, tol=1e-7, verbose=false, rng)

Fit a general phase-type distribution to positive observations `data` by maximum
likelihood using the EM algorithm (Asmussen–Nerman–Olsson). Provide the number
of phases `m` for a dense random start, or an `init`
distribution (any `AbstractPHDist`) — in which case the **sparsity pattern of
`init` is preserved**, so explicit zeros stay zero throughout the fit.

Returns a `PHDist` whose `α` and `T` are `FixedSparsity` arrays carrying the
preserved sparsity pattern. Phase-type distributions are not identifiable, so
compare fits by moments or CDF (`moments_isapprox`, `distribution_isapprox`)
rather than by `(α, T)` directly.
"""
function Distributions.fit_mle(::Type{PHDist}, data::AbstractVector{<:Real};
                               m::Union{Integer,Nothing}=nothing, init=nothing,
                               maxiter::Int=1000, tol::Real=1e-7, verbose::Bool=false,
                               rng::AbstractRNG=Random.default_rng())
    _check_data(data)
    α0, T0 = _starting_point(default_init, m, init, data, rng)
    res = _em(α0, T0, data; maxiter=maxiter, tol=tol, verbose=verbose)
    # Renormalise α by scalar division so its FixedSparsity pattern is preserved;
    # PHDist then keeps both α's and T's patterns (see PHDist's _phpattern).
    s = sum(res.α)
    s > 0 || throw(ArgumentError("fitted α summed to $s; cannot normalise"))
    return PHDist(res.α / s, res.T)
end

"""
    fit_mle(CoxianDist, data; m, init, maxiter=1000, tol=1e-7, verbose=false, rng)

Fit a Coxian PH distribution (α = e₁, bidiagonal T, exit possible at each phase).
The Coxian structure is preserved by the EM and a `CoxianDist` is returned.
"""
function Distributions.fit_mle(::Type{CoxianDist}, data::AbstractVector{<:Real};
                               m::Union{Integer,Nothing}=nothing, init=nothing,
                               maxiter::Int=1000, tol::Real=1e-7, verbose::Bool=false,
                               rng::AbstractRNG=Random.default_rng())
    _check_data(data)
    α0, T0 = _starting_point(coxian_init, m, init, data, rng)
    res = _em(α0, T0, data; maxiter=maxiter, tol=tol, verbose=verbose)
    return _to_coxian(res.α, res.T)
end

"""
    fit_mle(HyperExponentialDist, data; m, init, maxiter=1000, tol=1e-7, verbose=false, rng)

Fit a hyperexponential (mixture of `m` exponentials; diagonal T). Returns a
`HyperExponentialDist`. Note SCV ≥ 1 is intrinsic to this family.
"""
function Distributions.fit_mle(::Type{HyperExponentialDist}, data::AbstractVector{<:Real};
                               m::Union{Integer,Nothing}=nothing, init=nothing,
                               maxiter::Int=1000, tol::Real=1e-7, verbose::Bool=false,
                               rng::AbstractRNG=Random.default_rng())
    _check_data(data)
    α0, T0 = _starting_point(hyper_init, m, init, data, rng)
    res = _em(α0, T0, data; maxiter=maxiter, tol=tol, verbose=verbose)
    return _to_hyper(res.α, res.T)
end

"""
    fit_mle(HypoExponentialDist, data; m, init, maxiter=1000, tol=1e-7, verbose=false, rng)

Fit a hypoexponential (convolution of `m` exponentials; α = e₁, absorption only
from the last phase). Returns a `HypoExponentialDist`. Note SCV ≤ 1 is intrinsic.
"""
function Distributions.fit_mle(::Type{HypoExponentialDist}, data::AbstractVector{<:Real};
                               m::Union{Integer,Nothing}=nothing, init=nothing,
                               maxiter::Int=1000, tol::Real=1e-7, verbose::Bool=false,
                               rng::AbstractRNG=Random.default_rng())
    _check_data(data)
    α0, T0 = _starting_point(hypo_init, m, init, data, rng)
    res = _em(α0, T0, data; maxiter=maxiter, tol=tol, verbose=verbose)
    return _to_hypo(res.α, res.T)
end

"""
    fit_mle(ErlangPHDist, data; m)

Fit an Erlang (`m` phases of equal rate) by maximum likelihood. The shape `m` is
fixed; the rate has the closed form λ̂ = m / mean(data), so no EM is required.
Returns an `ErlangPHDist`.
"""
function Distributions.fit_mle(::Type{ErlangPHDist}, data::AbstractVector{<:Real};
                               m::Union{Integer,Nothing}=nothing, kwargs...)
    _check_data(data)
    m === nothing && throw(ArgumentError("provide `m` (the Erlang shape)"))
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    return ErlangPHDist(Int(m), m / Statistics.mean(data))
end

"""
    fit_mle(MAPHDist, data; m, init, censored=Float64[], init_method=:auto,
            init_kwargs=(), maxiter=500, tol=1e-7, verbose=false)

Fit a multi-absorbing phase-type distribution to competing-risks observations
`data` — a vector of pairs `(t, k)` with `t > 0` the absorption time and
`k ∈ 1..n` the index of the absorbing cause (as returned by
`rand(::MAPHDist, L)`) — by the approximate EM algorithm of the paper: an exact
E-step via Van Loan matrix exponentials, a closed-form relaxed M-step in the
`(α, q, R, U)` parameterization, and an ℓ1-projection linear program (HiGHS)
that re-enforces the feasibility constraints whenever the relaxed update leaves
the valid parameter region.

Provide either the number of transient phases `m` — the starting point is then
built by [`maph_moment_init`](@ref) (`init_method = :moment`, matching per-cause
means and SCVs) or [`maph_simplified_init`](@ref) (`init_method = :simplified`)
— or an explicit `init::MAPHDist`. The default `init_method = :auto` picks
`:moment` for uncensored data and `:simplified` when there is censoring (see
below). Extra keyword arguments for the chosen initializer go in the named tuple
`init_kwargs`. The number of absorbing states is read off the data (the largest
cause index observed) unless `init` provides more. The reference cause of the
second parameterization is the most frequent *observed event* cause; it must be
reachable from every phase of the starting distribution (both built-in
initializers guarantee this).

## Right-censored observations

Pass the right-censoring times as `censored`. A censored record at time `c`
contributes the observation `{τ > c}` — the subject is known to have survived
past `c`, and its eventual cause is unobserved — so it adds `log(α exp(Tc) 1)`
to the log-likelihood and enters the E-step through the cause-resolved censored
expectations of the paper, which complete the latent path on both sides of `c`
and spread it fractionally over every possible eventual cause. The censoring is
assumed independent of `(τ, κ)` and non-informative, so its law contributes no
MAPH parameters.

```julia
fit_mle(MAPHDist, events; m = 3, censored = [3.0, 5.5, 2.1])
```

At least one exact event is required, since the cause count and the reference
cause are read off the events. Because the per-cause event-time moments are
biased under censoring, `init_method = :moment` then needs censoring-adjusted
targets, supplied as `init_kwargs = (cond_means = ..., cond_scvs = ...)`; the
`:auto` default sidesteps this by using the censoring-compatible
[`maph_simplified_init`](@ref).

Returns a `MAPHDist`. Because the M-step optimizes a relaxed surrogate followed
by a projection, the log-likelihood is not guaranteed to increase at every
iteration; convergence is declared when its change drops below `tol`. MAPH
distributions are not identifiable, so compare fits through distributional
summaries (absorption probabilities, conditional moments, cdfs) rather than
through `(α, T, D)` directly.
"""
function Distributions.fit_mle(::Type{MAPHDist},
                               data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                               m::Union{Integer, Nothing}=nothing, init=nothing,
                               censored::AbstractVector{<:Real}=Float64[],
                               init_method::Symbol=:auto, init_kwargs::NamedTuple=NamedTuple(),
                               maxiter::Int=500, tol::Real=1e-7, verbose::Bool=false)
    isempty(data) && throw(ArgumentError(
        "data must contain at least one exact event (censored times go in `censored`)"))
    for c in censored
        c > 0 || throw(ArgumentError("censoring times must be strictly positive, got $c"))
    end
    n = maximum(Int(k) for (_, k) in data)

    if init !== nothing
        init isa MAPHDist || throw(ArgumentError("init must be a MAPHDist, got $(typeof(init))"))
        m === nothing || m == nphases(init) ||
            throw(ArgumentError("m=$m conflicts with init of size $(nphases(init))"))
        nabsorbing(init) >= n || throw(ArgumentError(
            "init has $(nabsorbing(init)) absorbing states but the data contains cause $n"))
        d0 = init
    else
        m === nothing && throw(ArgumentError("provide either `m` (number of phases) or `init`"))
        # Step 1 of the algorithm: the moment initialization needs conditional
        # per-cause moments, which the event times alone do not estimate under
        # censoring, so :auto falls back to the censoring-compatible one.
        method = init_method === :auto ?
            (isempty(censored) ? :moment : :simplified) : init_method
        d0 = if method === :moment
            maph_moment_init(m, data; censored=censored, init_kwargs...)
        elseif method === :simplified
            maph_simplified_init(m, data; censored=censored, init_kwargs...)
        else
            throw(ArgumentError("unknown init_method $(repr(init_method)); " *
                                "use :auto, :moment or :simplified"))
        end
    end

    # Reference cause for the second parameterization: the most frequent cause
    # among the observed events (censored records have no observed cause).
    counts = zeros(Int, nabsorbing(d0))
    for (_, k) in data
        counts[k] += 1
    end
    ref = argmax(counts)

    res = _maph_em(Vector(initial_prob(d0)), Matrix(subgenerator(d0)),
                   Matrix(exit_rate_matrix(d0)), data;
                   ref=ref, censored=censored, maxiter=maxiter, tol=tol, verbose=verbose)
    return MAPHDist(res.α, res.T, res.D)
end

# ===========================================================================
# fit_mm  (method of moments) — stub, planned for a future iteration
# ===========================================================================

"""
    fit_mm(T, data; …)

Moment-based fitting of a phase-type distribution. **Not yet implemented** — this
is a placeholder for the moment-matching approach, which will land alongside the
EM-based [`fit_mle`](@ref). Use `fit_mle` in the meantime.
"""
function fit_mm(::Type{<:AbstractPHDist}, data::AbstractVector{<:Real}; kwargs...)
    error("fit_mm (moment matching) is not yet implemented; planned for a future " *
          "iteration. Use fit_mle (EM) for now.")
end

# ===========================================================================
# fit  (router)
# ===========================================================================

"""
    fit(T, data; method=:mle, kwargs...)

Convenience entry point routing to [`fit_mle`](@ref) (`method=:mle`, the default,
maximum likelihood via EM) or [`fit_mm`](@ref) (`method=:mm`, method of moments).
Extra keyword arguments are forwarded to the chosen estimator.
"""
function Distributions.fit(::Type{D}, data::AbstractVector{<:Real};
                           method::Symbol=:mle, kwargs...) where {D<:AbstractPHDist}
    if method === :mle
        return fit_mle(D, data; kwargs...)
    elseif method === :mm
        return fit_mm(D, data; kwargs...)
    else
        throw(ArgumentError("unknown method $(repr(method)); use :mle or :mm"))
    end
end

function Distributions.fit(::Type{MAPHDist}, data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                           method::Symbol=:mle, kwargs...)
    method === :mle || throw(ArgumentError("only method=:mle is available for MAPHDist"))
    return fit_mle(MAPHDist, data; kwargs...)
end
