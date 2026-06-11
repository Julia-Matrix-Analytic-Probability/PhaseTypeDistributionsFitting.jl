# Approximate EM engine for MAPH distributions (competing-risks data).
#
# Data are pairs (t, k): the absorption time t > 0 and the index k ∈ 1..n of the
# absorbing state (the cause). Only (t, k) is observed — the trajectory of the
# underlying CTMC is latent — so the complete-data sufficient statistics are
# replaced by conditional expectations (the E-step), computed exactly with a
# Van Loan block-matrix exponential exactly as in the PH engine (`em.jl`), with
# the exit vector t⁰ replaced by the observed cause's column D[:, k]:
#
#     a(t,k)      = α' exp(Tt)                       (row m-vector)
#     b(t,k)      = exp(Tt) D[:,k]                   (column m-vector)
#     C(t,k)      = ∫₀ᵗ exp(Tᵀu) (αᵀ D[:,k]ᵀ) exp(Tᵀ(t-u)) du
#     f(t,k)      = α' exp(Tt) D[:,k]                (observed density)
#
#     B̄ᵢ = αᵢ bᵢ / f          expected start-in-i indicator
#     Z̄ᵢ = C[i,i] / f          expected holding time in i
#     M̄ᵢⱼ = Tᵢⱼ C[i,j] / f    expected i→j transient jumps (i ≠ j)
#     Ēᵢ = aᵢ D[i,k] / f       expected absorbing-jump-from-i indicator
#     N̄ᵢ = Σⱼ M̄ᵢⱼ + Ēᵢ       expected exits from i
#
# The M-step maximizes the *relaxed* complete-data likelihood in the second
# parameterization (α, q, R, U) of maph_parameterization.jl — all closed form:
#
#     α̂ᵢ = Σℓ B̄ᵢ / L                       ρ̂ᵢₖ = Σ_{ℓ: kℓ=k} B̄ᵢ / Σℓ B̄ᵢ
#     q̂ᵢ = Σℓ N̄ᵢ / Σℓ Z̄ᵢ                 ûᵢⱼ = Σ_{ℓ: kℓ=ref} M̄ᵢⱼ / Σ_{ℓ: kℓ=ref} N̄ᵢ
#
# where `ref` is the reference cause (most frequent in the data). The relaxed
# Û may violate the feasibility constraints that keep D = -T·R non-negative;
# Step 4 projects the violating rows back (an ℓ1-projection LP, see
# `_project_U!`), and Step 5 converts (α̂, q̂, R̂, U*) back to (α, T, D).
#
# Because the M-step optimizes a relaxed surrogate and then projects, the
# observed-data log-likelihood is *not* guaranteed to increase monotonically
# (the procedure is approximate); convergence is declared when its change drops
# below `tol`.
#
# Structural zeros: B̄ ∝ αᵢ and M̄ᵢⱼ ∝ Tᵢⱼ, so zeros of α and of the
# off-diagonal of T are EM-invariant and survive automatically. Zeros of D are
# *not* preserved — D is regenerated from (q, R, U) each iteration by design.

"""
    MAPHEMStats

Expected sufficient statistics accumulated over the whole data set in one
E-step. `B`, `Z`, `N` are length-m totals over all observations; `Bk[i, k]`
restricts `B` to observations absorbed in cause `k`; `Mref` (m×m) and `Nref`
restrict `M` and `N` to observations absorbed in the reference cause. `loglik`
is the observed-data log-likelihood Σℓ log f(tℓ, kℓ) at the E-step parameters.
"""
struct MAPHEMStats
    B::Vector{Float64}
    Bk::Matrix{Float64}
    Z::Vector{Float64}
    N::Vector{Float64}
    Mref::Matrix{Float64}
    Nref::Vector{Float64}
    loglik::Float64
end

"""
    _maph_estep(α, T, D, data, ref) -> MAPHEMStats

One E-step: exact conditional expectations of the complete-data sufficient
statistics for MAPH(α, T, D), aggregated over `data` (pairs `(t, k)`), via one
(2m × 2m) Van Loan matrix exponential per observation.
"""
function _maph_estep(α::Vector{Float64}, T::Matrix{Float64}, D::Matrix{Float64},
                     data::AbstractVector{<:Tuple{<:Real, <:Integer}}, ref::Int)
    m = length(α)
    n = size(D, 2)
    B = zeros(m)
    Bk = zeros(m, n)
    Z = zeros(m)
    N = zeros(m)
    Mref = zeros(m, m)
    Nref = zeros(m)
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
        isref = (k == ref)
        @inbounds for i in 1:m
            Bi = α[i] * b[i] * invf
            B[i] += Bi
            Bk[i, k] += Bi
            Z[i] += C[i, i] * invf
            Mi = 0.0
            for j in 1:m
                i == j && continue
                Mij = T[i, j] * C[i, j] * invf
                Mi += Mij
                isref && (Mref[i, j] += Mij)
            end
            Ei = a[i] * D[i, k] * invf
            Ni = Mi + Ei
            N[i] += Ni
            isref && (Nref[i] += Ni)
        end
    end

    return MAPHEMStats(B, Bk, Z, N, Mref, Nref, loglik)
end

"""
    _maph_mstep(stats, L, ref, qprev, Rprev, Uprev) -> (α, q, R, U)

Closed-form relaxed M-step in the second parameterization. Rows whose
denominators carry no information fall back to the previous iterate's values:
a phase never started in (`B[i] ≈ 0`), or one from which the reference cause
was never reached (`Bk[i, ref] ≈ 0`, which the conversion would divide by),
keeps its previous R row; a phase never visited (`Z[i] ≈ 0`) keeps its previous
exit rate; and a phase never visited on reference-cause paths (`Nref[i] ≈ 0`)
keeps its previous U row. The returned U is *relaxed* — it still needs
`_project_U!` before conversion.
"""
function _maph_mstep(stats::MAPHEMStats, L::Int, ref::Int, qprev::Vector{Float64},
                     Rprev::Matrix{Float64}, Uprev::Matrix{Float64})
    m, n = size(stats.Bk)
    tiny = 1e-12

    α = stats.B ./ L

    R = Matrix{Float64}(undef, m, n)
    for i in 1:m
        if stats.B[i] > tiny && stats.Bk[i, ref] / stats.B[i] > tiny
            R[i, :] = stats.Bk[i, :] ./ stats.B[i]
        else
            R[i, :] = Rprev[i, :]
        end
    end

    q = Vector{Float64}(undef, m)
    for i in 1:m
        q[i] = stats.Z[i] > tiny ? stats.N[i] / stats.Z[i] : qprev[i]
    end

    U = Matrix{Float64}(undef, m, m)
    for i in 1:m
        if stats.Nref[i] > tiny
            U[i, :] = stats.Mref[i, :] ./ stats.Nref[i]
            U[i, i] = 0.0
        else
            U[i, :] = Uprev[i, :]
        end
    end

    return α, q, R, U
end

"""
    MAPHEMResult

Result of an MAPH EM run: the fitted generator parameters `(α, T, D)`, the
per-iteration observed-data log-likelihood trace (`loglik[k]` is evaluated at
the parameters entering iteration k), whether the run converged, the number of
iterations performed, and `nprojections` — the total number of U-rows returned
to the feasible polytope by the constraint-enforcement LP across all
iterations (a diagnostic for how binding the constraints were).
"""
struct MAPHEMResult
    α::Vector{Float64}
    T::Matrix{Float64}
    D::Matrix{Float64}
    loglik::Vector{Float64}
    converged::Bool
    iterations::Int
    nprojections::Int
end

"""
    _maph_em(α0, T0, D0, data; ref, maxiter=500, tol=1e-7, verbose=false) -> MAPHEMResult

Run the approximate EM algorithm for MAPH(α, T, D) from the initial parameters,
on competing-risks `data` (pairs `(t, k)`). `ref` is the reference absorbing
cause used by the second parameterization; it must be observed in the data and
reachable from every phase of the initial distribution, and is *not* relabeled
between iterations. Convergence is declared when the absolute change in the
observed-data log-likelihood drops below `tol` (the relaxed-plus-projected
M-step does not guarantee monotone increase).
"""
function _maph_em(α0::AbstractVector{<:Real}, T0::AbstractMatrix{<:Real},
                  D0::AbstractMatrix{<:Real},
                  data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                  ref::Int, maxiter::Int=500, tol::Real=1e-7, verbose::Bool=false)
    length(data) >= 1 || throw(ArgumentError("need at least one observation"))
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
    any(obs -> obs[2] == ref, data) ||
        throw(ArgumentError("reference cause $ref does not appear in the data"))
    L = length(data)

    loglik = Float64[]
    converged = false
    nprojections = 0
    iter = 0
    prev_ll = -Inf
    smallsteps = 0          # consecutive iterations with |Δ loglik| < tol
    for outer iter in 1:maxiter
        stats = _maph_estep(α, T, D, data, ref)
        push!(loglik, stats.loglik)
        if verbose
            @info "MAPH EM iteration" iter loglik = stats.loglik Δ = stats.loglik - prev_ll projections = nprojections
        end
        # The relaxed-plus-projected update is not monotone, so |Δ| can dip
        # below tol transiently (e.g. when the trace wiggles around the
        # constraint boundary); require a short streak before stopping.
        smallsteps = abs(stats.loglik - prev_ll) < tol ? smallsteps + 1 : 0
        if smallsteps >= 3 && iter > 1
            converged = true
            break
        end
        prev_ll = stats.loglik

        qprev, Rprev, Uprev = _second_parameterization(T, D; ref=ref)
        αnew, q, R, U = _maph_mstep(stats, L, ref, qprev, Rprev, Uprev)
        nprojections += _project_U!(U, R, ref)
        α, T, D = _generator_from_second(αnew, q, R, U; ref=ref)
    end

    return MAPHEMResult(α, T, D, loglik, converged, iter, nprojections)
end
