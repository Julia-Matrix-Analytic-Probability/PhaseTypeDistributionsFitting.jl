# Initialization heuristics for the MAPH EM algorithm, following Appendix C of
# the paper. Both initializers consume competing-risks data (pairs `(t, k)`)
# and return a concrete `MAPHDist`.
#
# - `maph_simplified_init`: matches the empirical absorption probabilities π̂
#   and the overall mean exactly, with *every* parameter strictly positive — a
#   robust default that satisfies the reachability and positivity requirements
#   of the EM machinery with no minimum number of phases.
# - `maph_moment_init`: additionally matches the conditional mean and SCV of
#   the absorption time, cause by cause, by routing an exponential front end
#   into per-cause hyper-/hypo-exponential blocks; a small (ε, θ)
#   regularization restores reachability and strict positivity of α.

"""
    _maph_cause_statistics(data) -> (n, counts, π̂, μ̂, ĉ²)

Per-cause empirical statistics: `n` is the largest cause index observed,
`counts[k]` the number of observations of cause `k`, `π̂[k] = counts[k]/L`,
`μ̂[k]` the conditional mean and `ĉ²[k]` the conditional squared coefficient of
variation of the absorption times of cause `k`. A cause with a single
observation gets `ĉ²[k] = 1.0` (no variance information — treated as
exponential); a cause with no observations gets `μ̂[k] = ĉ²[k] = NaN` and is
skipped by the constructions.
"""
function _maph_cause_statistics(data::AbstractVector{<:Tuple{<:Real, <:Integer}})
    isempty(data) && throw(ArgumentError("data must be non-empty"))
    n = 0
    for (t, k) in data
        t > 0 || throw(ArgumentError("absorption times must be strictly positive, got $t"))
        k >= 1 || throw(ArgumentError("cause indices must be >= 1, got $k"))
        n = max(n, Int(k))
    end
    L = length(data)
    counts = zeros(Int, n)
    sums = zeros(n)
    for (t, k) in data
        counts[k] += 1
        sums[k] += t
    end
    π̂ = counts ./ L
    μ̂ = fill(NaN, n)
    ĉ² = fill(NaN, n)
    for k in 1:n
        counts[k] == 0 && continue
        μ̂[k] = sums[k] / counts[k]
        if counts[k] >= 2
            ss = 0.0
            for (t, kk) in data
                kk == k && (ss += (t - μ̂[k])^2)
            end
            σ̂² = ss / (counts[k] - 1)
            ĉ²[k] = σ̂² / μ̂[k]^2
            # A degenerate sample (all times equal) gives ĉ² = 0, which no
            # finite-phase block can match; floor it at a small positive value.
            ĉ²[k] = max(ĉ²[k], 1e-4)
        else
            ĉ²[k] = 1.0
        end
    end
    return n, counts, π̂, μ̂, ĉ²
end

"""
    maph_simplified_init(m, data; beta=0.5, jitter=0.5) -> MAPHDist

Simplified initialization (Appendix C of the paper): every phase has sojourn
rate `ω`; after a sojourn the chain moves to another transient phase with
probability `beta` (uniformly) and absorbs with probability `1 - beta`, into
cause `k` with probability `π̂ₖ`. Since a geometric sum of i.i.d. exponentials
is exponential, `τ ~ Exp((1-beta)·ω)` independently of `κ`, and choosing
`ω = 1/((1-beta)·μ̄)` matches the overall sample mean `μ̄` exactly, while the
absorption probabilities match `π̂` exactly. All parameters are strictly
positive (for causes present in the data), so the reachability assumption
holds and no parameter starts on the boundary. For `m = 1`, `beta` is forced
to 0 (there is no other phase to route to).

At `jitter = 0` the construction is exactly as above — but then all phases are
*exchangeable*, and since the EM update preserves exchangeability, an EM run
started there can never differentiate the phases (it is confined to fits with
`τ` exponential and independent of `κ`). The default `jitter > 0` therefore
applies a deterministic symmetry-breaking perturbation: the sojourn rate of
phase `i` is scaled by `1 + jitter·zᵢ` with `zᵢ` spread evenly over `[-1, 1]`.
The perturbation leaves the embedded jump chain — and hence the absorption
probabilities `π̂` — untouched, and the generator is rescaled afterwards so the
overall mean stays exactly `μ̄`; only the exponential shape of `τ` (and its
independence of `κ`) becomes approximate.
"""
function maph_simplified_init(m::Integer, data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                              beta::Real=0.5, jitter::Real=0.5)
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    0 <= beta < 1 || throw(ArgumentError("beta must be in [0, 1), got $beta"))
    0 <= jitter < 1 || throw(ArgumentError("jitter must be in [0, 1), got $jitter"))
    n, _, π̂, _, _ = _maph_cause_statistics(data)
    β = m == 1 ? 0.0 : Float64(beta)
    μ̄ = Statistics.mean(first.(data))
    ω = 1 / ((1 - β) * μ̄)

    α = fill(1.0 / m, m)
    T = fill(m == 1 ? 0.0 : β * ω / (m - 1), m, m)
    for i in 1:m
        T[i, i] = -ω
    end
    D = (1 - β) * ω .* repeat(π̂', m)

    if jitter > 0 && m > 1
        # Scale row i of [T | D] by 1 + jitter·zᵢ: per-phase sojourn rates
        # differ, but the routing probabilities (the jump chain) do not, so the
        # law of κ is unchanged. Then rescale time so the mean is exact again.
        for i in 1:m
            s = 1 + jitter * (2 * (i - 1) / (m - 1) - 1)
            T[i, :] .*= s
            D[i, :] .*= s
        end
        scale = -dot(α, T \ ones(m)) / μ̄     # current mean / target mean
        T .*= scale
        D .*= scale
    end
    return MAPHDist(α, T, D)
end

# ω threshold ω*ₖ of Appendix C: the smallest rate beyond which the corrected
# block targets (μₖ, c²ₖ) are positive and c²ₖ - 1 has the same sign as ĉ²ₖ - 1.
function _maph_omega_threshold(μ̂k::Float64, ĉ²k::Float64)
    if ĉ²k >= 1
        return 1 / μ̂k
    elseif ĉ²k < 0.5
        return 1 / (μ̂k * sqrt(ĉ²k))
    else
        return (1 + sqrt(2ĉ²k - 1)) / ((1 - ĉ²k) * μ̂k)
    end
end

"""
    maph_moment_init(m, data; omega_factor=10.0, epsilon=0.01, theta=nothing,
                     beta=0.5) -> MAPHDist

Moment-based initialization (Appendix C of the paper). The transient phases
split into a front end `S₁` (each phase exits after an `Exp(ω)` sojourn) and
per-cause phase-type blocks: leaving `S₁`, the chain routes into the block of
cause `k` — a two-phase hyper-exponential (`c²ₖ > 1`), a hypo-exponential
(`c²ₖ < 1`) or a single exponential (`c²ₖ ≈ 1`), built by the corresponding
`PhaseTypeDistributions` constructors — or, for causes the phase budget cannot
accommodate, directly into the absorbing state. At `epsilon = theta = 0` the
construction matches the empirical absorption probabilities exactly, and the
conditional mean and SCV of every block-fitted cause exactly.

`ω` is set to `omega_factor` (> 1) times the threshold of Appendix C, which
guarantees the corrected block targets are valid. Larger factors shrink the
front end's share `1/ω` of each conditional mean, keeping the construction
closer to the exact per-cause moment match (empirically a better EM starting
point), at the price of a stiffer generator in the E-step matrix exponentials. The `(epsilon, theta)`
regularization mixes `epsilon` of a uniform distribution into `α` and adds a
total feedback rate `theta` from every block phase to the front end (default
`0.01/μ̄`), making every phase carry positive initial mass and every absorbing
state reachable from every phase — as the EM machinery requires — at the cost
of an `O(epsilon + theta)` perturbation of the matched moments.

Causes are prioritized for blocks in decreasing order of empirical frequency.
If not even the most frequent cause fits the budget (`p = 0`, e.g. `m ≤ 2`),
this falls back to [`maph_simplified_init`](@ref) with routing constant `beta`.
"""
function maph_moment_init(m::Integer, data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                          omega_factor::Real=10.0, epsilon::Real=0.01,
                          theta::Union{Nothing, Real}=nothing, beta::Real=0.5)
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    omega_factor > 1 || throw(ArgumentError("omega_factor must be > 1, got $omega_factor"))
    0 <= epsilon < 1 || throw(ArgumentError("epsilon must be in [0, 1), got $epsilon"))
    n, counts, π̂, μ̂, ĉ² = _maph_cause_statistics(data)
    observed = findall(>(0), counts)
    μ̄ = Statistics.mean(first.(data))
    θ = theta === nothing ? 0.01 / μ̄ : Float64(theta)
    θ >= 0 || throw(ArgumentError("theta must be non-negative, got $θ"))

    ω = omega_factor * maximum(_maph_omega_threshold(μ̂[k], ĉ²[k]) for k in observed)

    # Corrected block targets (Proposition: mean and SCV of ξ + τₖ match the
    # empirical values when the block has mean μₖ and SCV c²ₖ).
    μ = Dict{Int, Float64}()
    c² = Dict{Int, Float64}()
    for k in observed
        μ[k] = μ̂[k] - 1 / ω
        c²[k] = (ĉ²[k] * (ω * μ[k] + 1)^2 - 1) / (ω * μ[k])^2
        μ[k] > 0 && c²[k] > 0 || error(
            "internal error: corrected targets invalid for cause $k (μ=$(μ[k]), c²=$(c²[k]))")
    end

    # Block sizes, and the number of fitted causes p that the budget m-1 allows,
    # prioritized by decreasing empirical frequency.
    scvtol = 1e-8
    mk = Dict(k => (c²[k] > 1 + scvtol ? 2 :
                    abs(c²[k] - 1) <= scvtol ? 1 :
                    Int(ceil(1 / c²[k]))) for k in observed)
    priority = sort(observed; by=k -> π̂[k], rev=true)
    fitted = Int[]
    budget = m - 1
    for k in priority
        mk[k] <= budget || break
        push!(fitted, k)
        budget -= mk[k]
    end
    isempty(fitted) && return maph_simplified_init(m, data; beta=beta)

    blocks = Dict{Int, Any}()
    for k in fitted
        blocks[k] = if c²[k] > 1 + scvtol
            HyperExponentialDist(μ[k], c²[k])
        elseif abs(c²[k] - 1) <= scvtol
            CoxianDist(1 / μ[k])                    # single-phase exponential
        else
            HypoExponentialDist(μ[k], c²[k])
        end
        nphases(blocks[k]) == mk[k] || error(
            "internal error: block for cause $k has $(nphases(blocks[k])) phases, expected $(mk[k])")
    end

    # Assembly: front end S₁ = phases 1..m₁, then the blocks in priority order.
    m₁ = m - sum(mk[k] for k in fitted)
    α = zeros(m)
    α[1:m₁] .= 1 / m₁
    T = zeros(m, m)
    D = zeros(m, n)
    for i in 1:m₁
        T[i, i] = -ω
    end
    offset = m₁
    for k in fitted
        rows = (offset + 1):(offset + mk[k])
        Tk = Matrix{Float64}(subgenerator(blocks[k]))
        T[rows, rows] = Tk
        D[rows, k] = -vec(sum(Tk; dims=2))          # block exit rates dₖ = -Tₖ·1
        αk = collect(Float64, initial_prob(blocks[k]))
        for i in 1:m₁
            T[i, rows] = ω * π̂[k] .* αk
        end
        offset += mk[k]
    end
    for k in setdiff(observed, fitted)              # unfitted causes: absorb from S₁
        for i in 1:m₁
            D[i, k] = ω * π̂[k]
        end
    end

    # (ε, θ) regularization: positive initial mass everywhere, and feedback from
    # every block phase to the front end so every cause is reachable everywhere.
    α .= (1 - epsilon) .* α .+ epsilon / m
    for r in (m₁ + 1):m
        for i in 1:m₁
            T[r, i] += θ / m₁
        end
        T[r, r] -= θ
    end

    return MAPHDist(α, T, D)
end
