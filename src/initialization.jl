# Initialization strategies for the EM algorithm.
#
# EM for PH distributions is sensitive to its starting point, so callers may
# always supply their own `init` distribution. These helpers provide sensible
# defaults. Every initializer returns a concrete `(α, T)` pair whose mean is
# scaled to the empirical mean of the data (a cheap and effective heuristic),
# and whose sparsity pattern encodes the desired structure — the EM then
# preserves that pattern (see `em.jl`).

"""
    _mean_phasetype(α, T) -> Float64

Mean of PH(α, T): μ = -α' T⁻¹ 1.
"""
_mean_phasetype(α::AbstractVector, T::AbstractMatrix) = -dot(α, T \ ones(length(α)))

"""
    _scale_to_mean!(T, α, target_mean) -> T

Rescale the sub-generator `T` in place so that PH(α, T) has mean `target_mean`.
Scaling T by a factor c divides the mean by c, so c = current_mean / target_mean.
Preserves the sparsity pattern (all entries scale by the same factor).
"""
function _scale_to_mean!(T::Matrix{Float64}, α::Vector{Float64}, target_mean::Real)
    target_mean > 0 || throw(ArgumentError("target mean must be positive"))
    μ = _mean_phasetype(α, T)
    (isfinite(μ) && μ > 0) || throw(ArgumentError("initial distribution has non-positive/non-finite mean"))
    T .*= μ / target_mean
    return T
end

"""
    default_init(m, data; rng) -> (α, T)

Default dense (general) PH initializer with `m` phases: uniform α, random
non-negative off-diagonals and exit rates, scaled to the empirical mean. No
structural zeros, so `fit_mle(PHDist, …)` is free to fill the whole matrix.
"""
function default_init(m::Integer, data::AbstractVector{<:Real};
                      rng::AbstractRNG=Random.default_rng())
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    α = fill(1.0 / m, m)
    T = zeros(m, m)
    for i in 1:m
        exit_i = rand(rng) + 0.1
        off = 0.0
        for j in 1:m
            i == j && continue
            T[i, j] = rand(rng) + 0.1
            off += T[i, j]
        end
        T[i, i] = -(exit_i + off)
    end
    _scale_to_mean!(T, α, Statistics.mean(data))
    return α, T
end

"""
    coxian_init(m, data; rng) -> (α, T)

Coxian-structured initializer: α = e₁, bidiagonal T, an exit possible at every
phase. Built from a `CoxianDist` and scaled to the empirical mean.
"""
function coxian_init(m::Integer, data::AbstractVector{<:Real};
                     rng::AbstractRNG=Random.default_rng())
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    rates = [1.0 + rand(rng) for _ in 1:m]
    exit_probs = m > 1 ? fill(0.5, m - 1) : Float64[]
    d = CoxianDist(rates, exit_probs)
    α = collect(Float64, initial_prob(d))
    T = Matrix{Float64}(subgenerator(d))
    _scale_to_mean!(T, α, Statistics.mean(data))
    return α, T
end

"""
    hyper_init(m, data; rng) -> (α, T)

Hyperexponential-structured initializer: diagonal T (no transient transitions),
full α. Built from a `HyperExponentialDist` and scaled to the empirical mean.
"""
function hyper_init(m::Integer, data::AbstractVector{<:Real};
                    rng::AbstractRNG=Random.default_rng())
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    probs = fill(1.0 / m, m)
    rates = [Float64(i) for i in 1:m]   # distinct
    d = HyperExponentialDist(probs, rates)
    α = collect(Float64, initial_prob(d))
    T = Matrix{Float64}(subgenerator(d))
    _scale_to_mean!(T, α, Statistics.mean(data))
    return α, T
end

"""
    hypo_init(m, data; rng) -> (α, T)

Hypoexponential-structured initializer: α = e₁, bidiagonal T, absorption only
from the last phase. Built from a `HypoExponentialDist` and scaled to the
empirical mean.
"""
function hypo_init(m::Integer, data::AbstractVector{<:Real};
                   rng::AbstractRNG=Random.default_rng())
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    rates = [Float64(i) for i in 1:m]   # distinct
    d = HypoExponentialDist(rates)
    α = collect(Float64, initial_prob(d))
    T = Matrix{Float64}(subgenerator(d))
    _scale_to_mean!(T, α, Statistics.mean(data))
    return α, T
end
