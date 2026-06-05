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
