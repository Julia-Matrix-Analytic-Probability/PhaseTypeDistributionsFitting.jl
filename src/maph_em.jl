# EM engine for MAPH distributions (competing-risks data), with support for
# independently right-censored records.
#
# An exact-event datum is a pair (t, k): the absorption time t > 0 and the index
# k ∈ 1..n of the absorbing state (the cause). Only (t, k) is observed — the
# trajectory of the underlying CTMC is latent — so the complete-data sufficient
# statistics are replaced by conditional expectations (the E-step), computed
# exactly with a Van Loan block-matrix exponential exactly as in the PH engine
# (`em.jl`), with the exit vector t⁰ replaced by the observed cause's column
# D[:, k]:
#
#     a(t,k)      = α' exp(Tt)                       (row m-vector)
#     b(t,k)      = exp(Tt) D[:,k]                   (column m-vector)
#     C(t,k)      = ∫₀ᵗ exp(Tᵀu) (αᵀ D[:,k]ᵀ) exp(Tᵀ(t-u)) du
#     f(t,k)      = α' exp(Tt) D[:,k]                (observed density)
#
#     E[Bᵢ]   = αᵢ bᵢ / f          expected start-in-i indicator
#     E[Zᵢ]   = C[i,i] / f         expected holding time in i
#     E[Mᵢⱼ]  = Tᵢⱼ C[i,j] / f     expected i→j transient jumps (i ≠ j)
#     E[Eᵢₖ]  = aᵢ D[i,k] / f      expected absorbing jump i → cause k
#     E[Nᵢ]   = Σⱼ E[Mᵢⱼ] + Σₖ E[Eᵢₖ]   expected exits from i
#
# A right-censored datum is a time c > 0 at which the subject is known only to
# be still alive: the observation is the event {τ > c} and the eventual cause is
# unobserved. The complete datum is still the whole path through its eventual
# absorption, so the E-step completes the path on either side of c and averages
# over the unobserved cause. With
#
#     a(c)   = α' exp(Tc),      S(c) = a(c)·1        (survival, the likelihood)
#     g(c)   = a(c)(-T)⁻¹                            (occupation times after c)
#     C(c;1) = ∫₀ᶜ exp(Tᵀu) (αᵀ 1ᵀ) exp(Tᵀ(c-u)) du
#
# the censored expectations are
#
#     E[Bᵢ  | τ>c] = αᵢ (exp(Tc) 1)ᵢ / S
#     E[Zᵢ  | τ>c] = (C(c;1)[i,i] + gᵢ) / S
#     E[Mᵢⱼ | τ>c] = Tᵢⱼ (C(c;1)[i,j] + gᵢ) / S      (i ≠ j)
#     E[Eᵢₖ | τ>c] = gᵢ Dᵢₖ / S
#     E[Nᵢ  | τ>c] = Σⱼ E[Mᵢⱼ | τ>c] + Σₖ E[Eᵢₖ | τ>c]
#
# and the record contributes log S(c) to the observed-data log-likelihood. Only
# the terminal weight b = 1 is needed, so each censored record costs *one* Van
# Loan exponential whatever the number of causes: summing the cause-resolved
# formulas over k collapses the cause-weighted terminal vectors through
# Σₖ D[:,k] = -T·1.
#
# Both kinds of record are packaged into the same per-record contributions
# (bᵢ, zᵢ, mᵢⱼ, nᵢ, eᵢₖ) and the M-step is identical in the two cases. It is the
# *exact* maximizer of the expected complete-data log-likelihood over the
# parameter set — α on the simplex, q > 0, and each row of [Pλ | Pμ] a
# probability vector — and is available in closed form:
#
#     α̂ᵢ = Σℓ bᵢ / L                  q̂ᵢ  = Σℓ nᵢ / Σℓ zᵢ
#     p̂λᵢⱼ = Σℓ mᵢⱼ / Σℓ nᵢ  (i≠j)     p̂μᵢₖ = Σℓ eᵢₖ / Σℓ nᵢ
#
# where `L` counts *all* records (events and censored). The flow balance
# Σⱼ mᵢⱼ + Σₖ eᵢₖ = nᵢ holds pathwise, hence in expectation, so each fitted row
# sums to one automatically: the update lands in the parameter set with no
# feasibility step, and the conversion back to generator form is immediate,
#
#     Tᵢⱼ = q̂ᵢ p̂λᵢⱼ (i≠j),   Tᵢᵢ = -q̂ᵢ,   Dᵢₖ = q̂ᵢ p̂μᵢₖ.
#
# Because the M-step is exact, the iteration is a genuine EM iteration and the
# observed-data log-likelihood is non-decreasing; convergence is declared when
# its change drops below `tol`.
#
# Structural zeros: E[Bᵢ] ∝ αᵢ, E[Mᵢⱼ] ∝ Tᵢⱼ and E[Eᵢₖ] ∝ Dᵢₖ, so zeros of α and
# of the off-diagonal of T *and of D* are all EM-invariant and survive
# automatically.

"""
    MAPHEMStats

Expected sufficient statistics accumulated over the whole data set in one
E-step. `B`, `Z`, `N` are length-m totals over all observations, `M` is the
m×m matrix of expected transient jump counts (zero diagonal), and `E` is the
m×n matrix of expected absorbing jumps, `E[i, k]` being the expected number of
exits from phase `i` into cause `k`. `loglik` is the observed-data
log-likelihood Σℓ log f(tℓ, kℓ) + Σℓ log S(cℓ) at the E-step parameters (the
second sum over the right-censored records).

An exact-event record puts all of its absorbing-jump mass on its observed cause,
so it contributes to the single column `E[:, kℓ]`; a censored record has no
observed cause and spreads that mass over every column in proportion to its
posterior probability of eventual absorption there. Every row satisfies the
flow balance `sum(M[i, :]) + sum(E[i, :]) ≈ N[i]`.
"""
struct MAPHEMStats
    B::Vector{Float64}
    Z::Vector{Float64}
    N::Vector{Float64}
    M::Matrix{Float64}
    E::Matrix{Float64}
    loglik::Float64
end

"""
    _maph_estep(α, T, D, data; censored = Float64[]) -> MAPHEMStats

One E-step: exact conditional expectations of the complete-data sufficient
statistics for MAPH(α, T, D), aggregated over the exact-event `data` (pairs
`(t, k)`) and the right-censored times `censored`.

Each record — exact or censored — costs one (2m × 2m) Van Loan matrix
exponential, whatever the number of causes (see the file header).
"""
function _maph_estep(α::Vector{Float64}, T::Matrix{Float64}, D::Matrix{Float64},
                     data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                     censored::AbstractVector{<:Real}=Float64[])
    m = length(α)
    n = size(D, 2)
    B = zeros(m)
    Z = zeros(m)
    N = zeros(m)
    M = zeros(m, m)
    E = zeros(m, n)
    loglik = 0.0

    Tt = permutedims(T)
    G = [α * D[:, k]' for k in 1:n]     # G[k][i,j] = αᵢ D[j,k]
    VL = zeros(2m, 2m)
    @views VL[1:m, 1:m] .= Tt
    @views VL[m+1:2m, m+1:2m] .= Tt

    for (t, k) in data
        @views VL[1:m, m+1:2m] .= G[k]
        W = exp(VL .* t)
        expTt = @view W[1:m, 1:m]       # exp(Tᵀt)
        C = @view W[1:m, m+1:2m]

        a = expTt * α                   # aᵢ = (α' exp(Tt))ᵢ
        b = expTt' * view(D, :, k)      # bᵢ = (exp(Tt) D[:,k])ᵢ
        f = dot(α, b)
        f > 0 || continue               # skip numerically-degenerate points
        loglik += log(f)

        invf = 1.0 / f
        @inbounds for i in 1:m
            B[i] += α[i] * b[i] * invf
            Z[i] += C[i, i] * invf
            Mi = 0.0
            for j in 1:m
                i == j && continue
                Mij = T[i, j] * C[i, j] * invf
                M[i, j] += Mij
                Mi += Mij
            end
            Ei = a[i] * D[i, k] * invf  # the observed cause takes all the mass
            E[i, k] += Ei
            N[i] += Mi + Ei
        end
    end

    isempty(censored) && return MAPHEMStats(B, Z, N, M, E, loglik)

    # ---- right-censored records ----
    # The factorization of (-T)ᵀ (used for g(c) = a(c)(-T)⁻¹, i.e. (-T)ᵀ g = a)
    # depends only on the parameters, so it is formed once. Only the terminal
    # weight b = 1 is needed, so one Van Loan exponential per record suffices.
    negTt = lu(-Tt)
    ones_m = ones(m)
    G1 = α * ones_m'                    # b = 1_m  → C(c; 1)

    for c in censored
        @views VL[1:m, m+1:2m] .= G1
        W1 = exp(VL .* c)
        expTt = @view W1[1:m, 1:m]      # exp(Tᵀc)
        C1 = @view W1[1:m, m+1:2m]      # C(c; 1)

        a = expTt * α                   # aᵢ = (α' exp(Tc))ᵢ
        S = sum(a)                      # S(c) = α' exp(Tc) 1
        S > 0 || continue               # skip numerically-degenerate points
        loglik += log(S)

        g = negTt \ a                   # g(c) = a(c)(-T)⁻¹
        sv = expTt' * ones_m            # svᵢ = (exp(Tc) 1)ᵢ

        invS = 1.0 / S
        @inbounds for i in 1:m
            B[i] += α[i] * sv[i] * invS
            Z[i] += (C1[i, i] + g[i]) * invS
            Mi = 0.0
            for j in 1:m
                i == j && continue
                Mij = T[i, j] * (C1[i, j] + g[i]) * invS
                M[i, j] += Mij
                Mi += Mij
            end
            Ei = 0.0                    # mass spread over all causes
            for k in 1:n
                Eik = g[i] * D[i, k] * invS
                E[i, k] += Eik
                Ei += Eik
            end
            N[i] += Mi + Ei
        end
    end

    return MAPHEMStats(B, Z, N, M, E, loglik)
end

"""
    _maph_mstep(stats, L, qprev, Plprev, Pmprev) -> (α, q, Pλ, Pμ)

Closed-form M-step: the exact maximizer of the expected complete-data
log-likelihood over the parameter set (α on the simplex, q > 0, each row of
[Pλ | Pμ] a probability vector with zero diagonal in Pλ). Rows whose
denominators carry no information fall back to the previous iterate: a phase
never visited (`Z[i] ≈ 0`) keeps its previous exit rate, and one with no
expected exits (`N[i] ≈ 0`) keeps its previous jump and absorption rows.

The returned `Pλ`, `Pμ` need no feasibility step: by the flow balance the fitted
row sums are one by construction.
"""
function _maph_mstep(stats::MAPHEMStats, L::Int, qprev::Vector{Float64},
                     Plprev::Matrix{Float64}, Pmprev::Matrix{Float64})
    m, n = size(stats.E)
    tiny = 1e-12

    α = stats.B ./ L

    q = Vector{Float64}(undef, m)
    for i in 1:m
        q[i] = stats.Z[i] > tiny ? stats.N[i] / stats.Z[i] : qprev[i]
    end

    Pl = Matrix{Float64}(undef, m, m)
    Pm = Matrix{Float64}(undef, m, n)
    for i in 1:m
        if stats.N[i] > tiny
            invN = 1.0 / stats.N[i]
            Pl[i, :] = stats.M[i, :] .* invN
            Pl[i, i] = 0.0
            Pm[i, :] = stats.E[i, :] .* invN
        else
            Pl[i, :] = Plprev[i, :]
            Pm[i, :] = Pmprev[i, :]
        end
    end

    return α, q, Pl, Pm
end

"""
    MAPHEMResult

Result of an MAPH EM run: the fitted generator parameters `(α, T, D)`, the
per-iteration observed-data log-likelihood trace (`loglik[k]` is evaluated at
the parameters entering iteration k), whether the run converged, and the number
of iterations performed. The M-step is exact, so `loglik` is non-decreasing.
"""
struct MAPHEMResult
    α::Vector{Float64}
    T::Matrix{Float64}
    D::Matrix{Float64}
    loglik::Vector{Float64}
    converged::Bool
    iterations::Int
end

"""
    _maph_em(α0, T0, D0, data; censored=Float64[], maxiter=500, tol=1e-7,
             verbose=false) -> MAPHEMResult

Run the EM algorithm for MAPH(α, T, D) from the initial parameters, on
competing-risks `data` (exact-event pairs `(t, k)`) and the independently
right-censored times `censored`. Convergence is declared when the increase in
the observed-data log-likelihood drops below `tol`; the M-step being exact, the
sequence is non-decreasing, so a single threshold is sound.

At least one exact event is required — the cause count is read off the events —
and the M-step divides by the total record count
`length(data) + length(censored)`.
"""
function _maph_em(α0::AbstractVector{<:Real}, T0::AbstractMatrix{<:Real},
                  D0::AbstractMatrix{<:Real},
                  data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                  censored::AbstractVector{<:Real}=Float64[],
                  maxiter::Int=500, tol::Real=1e-7, verbose::Bool=false)
    length(data) >= 1 || throw(ArgumentError("need at least one exact-event observation"))
    α = Vector{Float64}(α0)
    T = Matrix{Float64}(T0)
    D = Matrix{Float64}(D0)
    m = length(α)
    n = size(D, 2)
    size(T) == (m, m) || throw(DimensionMismatch("T must be $m × $m"))
    size(D, 1) == m || throw(DimensionMismatch("D must have $m rows"))
    for (t, k) in data
        t > 0 || throw(ArgumentError("absorption times must be strictly positive, got $t"))
        1 <= k <= n || throw(ArgumentError("cause index $k out of range 1..$n"))
    end
    for c in censored
        c > 0 || throw(ArgumentError("censoring times must be strictly positive, got $c"))
    end
    cens = collect(Float64, censored)
    L = length(data) + length(cens)

    loglik = Float64[]
    converged = false
    iter = 0
    prev_ll = -Inf
    for outer iter in 1:maxiter
        stats = _maph_estep(α, T, D, data; censored=cens)
        push!(loglik, stats.loglik)
        if verbose
            @info "MAPH EM iteration" iter loglik = stats.loglik Δ = stats.loglik - prev_ll
        end
        if iter > 1 && stats.loglik - prev_ll < tol
            converged = true
            break
        end
        prev_ll = stats.loglik

        qprev, Plprev, Pmprev = _jump_parameters(T, D)
        αnew, q, Pl, Pm = _maph_mstep(stats, L, qprev, Plprev, Pmprev)
        α, T, D = _generator_from_jump(αnew, q, Pl, Pm)
    end

    return MAPHEMResult(α, T, D, loglik, converged, iter)
end
