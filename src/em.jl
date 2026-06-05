# Core EM engine for phase-type distributions.
#
# This implements the classic Asmussen–Nerman–Olsson EM algorithm for fitting a
# PH(α, T) distribution to fully-observed absorption times x₁, …, x_L.
#
# The E-step computes the *exact* expected sufficient statistics using a Van Loan
# block-matrix exponential, rather than quadrature or ODE integration. For a PH
# distribution with sub-generator T, exit vector t⁰ = -T·1 and density
# f(x) = α' exp(Tx) t⁰, define
#
#     a(x)        = α' exp(Tx)                      (row m-vector)
#     b(x)        = exp(Tx) t⁰                      (column m-vector)
#     c_j(x; i)   = ∫₀ˣ (α' exp(Tu))_i (exp(T(x-u)) t⁰)_j du
#
# Stacking the c_j(x; i) into a matrix C(x) with C[i, j] = c_j(x; i) gives
#
#     C(x) = ∫₀ˣ exp(Tᵀu) (αᵀ t⁰ᵀ) exp(Tᵀ(x-u)) du,
#
# which is exactly the top-right block of exp( [Tᵀ  G; 0  Tᵀ] · x ) with
# G = αᵀ t⁰ᵀ (Van Loan, 1978). One (2m × 2m) matrix exponential per observation
# therefore yields every statistic needed:
#
#     B̄ᵢ(x)  = αᵢ bᵢ(x) / f(x)            expected # starts in i
#     Z̄ᵢ(x)  = C[i, i]  / f(x)            expected time spent in i
#     M̄ᵢⱼ(x) = Tᵢⱼ C[i, j] / f(x)        expected # i→j transitions (i ≠ j)
#     N̄ᵢ(x)  = aᵢ(x) t⁰ᵢ / f(x)          expected # i→absorption transitions
#
# Crucially, B̄ ∝ α, M̄ ∝ T_offdiag and N̄ ∝ t⁰, so a 0.0 factor yields a 0.0
# statistic. The M-step on top of that is made *pattern-aware*: it is given the
# sparsity pattern of the starting parameters — the allowed positions of α, of
# the off-diagonal of T, and of the exit vector t⁰ — and only ever writes into
# those positions. Off-pattern entries stay exactly 0.0, and the result is
# wrapped in a [`FixedSparsityVector`](@ref) / [`FixedSparsityMatrix`](@ref)
# carrying that pattern, so the structural zeros are not just preserved by lucky
# arithmetic but *enforced by the type* (writing a nonzero into one throws).
#
# This is what keeps a Coxian a Coxian, a hyperexponential diagonal, a
# hypoexponential absorbing only from its last phase, and respects any zeros the
# caller supplies via an `init` distribution.

# ---- sparsity patterns of the starting parameters -------------------------
#
# A `FixedSparsity*` input (e.g. from a PHDist's accessors) carries an explicit
# pattern; a plain array's pattern is read off its current nonzeros. The
# sub-generator pattern always includes the diagonal (the phase rates are free),
# and the exit pattern is read from the nonzeros of t⁰ = -T·1.

_alpha_pattern(α::FixedSparsityVector) = BitVector(collect(pattern(α)))
_alpha_pattern(α::AbstractVector) = BitVector(.!iszero.(α))

function _subgen_pattern(T::AbstractMatrix)
    P = T isa FixedSparsityMatrix ? BitMatrix(collect(pattern(T))) : BitMatrix(.!iszero.(T))
    @inbounds for i in axes(P, 1)
        P[i, i] = true                  # diagonal phase rates are always free
    end
    return P
end

_exit_dense(T::AbstractMatrix) = -vec(sum(Matrix(T); dims = 2))
_exit_pattern(T::AbstractMatrix) = BitVector(.!iszero.(_exit_dense(T)))

"""
    PHEMStats

Accumulated expected sufficient statistics over a whole data set for one E-step.
`B`, `Z`, `N` are length-m vectors; `M` is the m×m transition-count matrix (its
diagonal is unused). `loglik` is the observed-data log-likelihood Σ_ℓ log f(xℓ)
evaluated at the parameters used for the E-step.
"""
struct PHEMStats
    B::Vector{Float64}
    Z::Vector{Float64}
    M::Matrix{Float64}
    N::Vector{Float64}
    loglik::Float64
end

"""
    _ph_estep(α, T, t0, data) -> PHEMStats

Compute the expected sufficient statistics and log-likelihood for PH(α, T) over
`data` (a vector of positive absorption times). `t0 = -T * 1` is passed in to
avoid recomputation. Operates on dense working arrays — the sparsity pattern is
applied in the M-step.
"""
function _ph_estep(α::Vector{Float64}, T::Matrix{Float64}, t0::Vector{Float64},
                   data::AbstractVector{<:Real})
    m = length(α)
    B = zeros(m)
    Z = zeros(m)
    M = zeros(m, m)
    N = zeros(m)
    loglik = 0.0

    Tt = permutedims(T)                 # Tᵀ
    G = α * t0'                         # G[i,j] = αᵢ t⁰ⱼ
    # Van Loan block matrix [Tᵀ G; 0 Tᵀ]; reused (scaled by x) per observation.
    VL = zeros(2m, 2m)
    @views VL[1:m, 1:m] .= Tt
    @views VL[m+1:2m, m+1:2m] .= Tt
    @views VL[1:m, m+1:2m] .= G

    for x in data
        x > 0 || throw(ArgumentError("all observations must be strictly positive, got $x"))
        W = exp(VL .* x)
        expTt = @view W[1:m, 1:m]       # exp(Tᵀx)
        C = @view W[1:m, m+1:2m]        # C[i,j] = c_j(x; i)

        # a(x) = α' exp(Tx); as a column vector aᵢ = (exp(Tᵀx) α)ᵢ
        a = expTt * α
        # b(x) = exp(Tx) t⁰; bᵢ = (exp(Tᵀx)ᵀ t⁰)ᵢ = (t⁰ᵀ exp(Tᵀx))ᵢ
        b = expTt' * t0
        f = dot(α, b)
        f > 0 || continue              # skip numerically-degenerate points
        loglik += log(f)

        invf = 1.0 / f
        @inbounds for i in 1:m
            B[i] += α[i] * b[i] * invf
            Z[i] += C[i, i] * invf
            N[i] += a[i] * t0[i] * invf
            for j in 1:m
                i == j && continue
                M[i, j] += T[i, j] * C[i, j] * invf
            end
        end
    end

    return PHEMStats(B, Z, M, N, loglik)
end

"""
    _ph_mstep!(α, T, stats, L, αpat, Tpat, t0pat) -> (α, T)

Pattern-aware closed-form M-step, writing the updated parameters into the dense
working arrays `α` and `T` in place. `L` is the number of observations and
`αpat` / `Tpat` / `t0pat` are the sparsity patterns (see `_subgen_pattern`,
`_exit_pattern`). Only in-pattern positions are written, so off-pattern entries
stay exactly `0.0`. A state with no accumulated holding time (`Z ≈ 0`, i.e. never
visited) keeps a self-absorbing-like row to avoid division by zero; such rows are
inert under the dynamics anyway.
"""
function _ph_mstep!(α::Vector{Float64}, T::Matrix{Float64}, stats::PHEMStats, L::Int,
                    αpat::BitVector, Tpat::BitMatrix, t0pat::BitVector)
    m = length(stats.B)
    fill!(α, 0.0)
    fill!(T, 0.0)
    @inbounds for i in 1:m
        αpat[i] && (α[i] = stats.B[i] / L)
    end
    @inbounds for i in 1:m
        Zi = stats.Z[i]
        if Zi <= 0
            T[i, i] = -1.0             # inert; row is never reached (αᵢ≈0, unreachable)
            continue
        end
        t0i = t0pat[i] ? stats.N[i] / Zi : 0.0
        rowsum_off = 0.0
        for j in 1:m
            (i == j || !Tpat[i, j]) && continue
            T[i, j] = stats.M[i, j] / Zi
            rowsum_off += T[i, j]
        end
        T[i, i] = -(t0i + rowsum_off)
    end
    return α, T
end

"""
    EMResult

Result of an EM run: the fitted `(α, T)` — as a [`FixedSparsityVector`](@ref) and
a [`FixedSparsityMatrix`](@ref) carrying the preserved sparsity pattern — the
per-iteration log-likelihood trace (`loglik[k]` is the observed-data
log-likelihood at the *start* of iteration k, i.e. evaluated at the parameters
produced by iteration k-1), whether the run converged, and the number of
iterations performed.
"""
struct EMResult
    α::FixedSparsityVector{Float64, Vector{Float64}, BitVector}
    T::FixedSparsityMatrix{Float64, Matrix{Float64}, BitMatrix}
    loglik::Vector{Float64}
    converged::Bool
    iterations::Int
end

"""
    _em(α0, T0, data; maxiter, tol, verbose) -> EMResult

Run the EM algorithm from initial parameters `(α0, T0)`, which may be plain
arrays or `FixedSparsity*` arrays (e.g. from a `PHDist`'s accessors). Iterates
until the absolute increase in log-likelihood drops below `tol` or `maxiter` is
reached. The sparsity pattern of `(α0, T0)` — including the exit-vector zeros
implied by `T0`'s row sums — is preserved throughout and carried into the
returned `FixedSparsity*` parameters.
"""
function _em(α0::AbstractVector{<:Real}, T0::AbstractMatrix{<:Real},
             data::AbstractVector{<:Real};
             maxiter::Int=1000, tol::Real=1e-7, verbose::Bool=false)
    length(data) >= 1 || throw(ArgumentError("need at least one observation"))
    α = Vector{Float64}(α0)
    T = Matrix{Float64}(T0)
    m = length(α)
    size(T) == (m, m) || throw(DimensionMismatch("T must be $m × $m"))
    L = length(data)

    # Capture the structural sparsity before iterating; the M-step respects it.
    αpat = _alpha_pattern(α0)
    Tpat = _subgen_pattern(T0)
    t0pat = _exit_pattern(T0)

    loglik = Float64[]
    converged = false
    iter = 0
    prev_ll = -Inf
    for outer iter in 1:maxiter
        t0 = -vec(sum(T; dims=2))
        stats = _ph_estep(α, T, t0, data)
        push!(loglik, stats.loglik)
        if verbose
            @info "EM iteration" iter loglik = stats.loglik Δ = stats.loglik - prev_ll
        end
        if abs(stats.loglik - prev_ll) < tol && iter > 1
            converged = true
            break
        end
        prev_ll = stats.loglik
        _ph_mstep!(α, T, stats, L, αpat, Tpat, t0pat)
    end

    αfs = FixedSparsityVector(α, αpat)
    Tfs = FixedSparsityMatrix(T, Tpat)
    return EMResult(αfs, Tfs, loglik, converged, iter)
end
