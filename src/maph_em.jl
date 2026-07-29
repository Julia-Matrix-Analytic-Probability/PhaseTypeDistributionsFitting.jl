# Approximate EM engine for MAPH distributions (competing-risks data), with
# support for independently right-censored records.
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
#     B̄ᵢ = αᵢ bᵢ / f          expected start-in-i indicator
#     Z̄ᵢ = C[i,i] / f          expected holding time in i
#     M̄ᵢⱼ = Tᵢⱼ C[i,j] / f    expected i→j transient jumps (i ≠ j)
#     Ēᵢ = aᵢ D[i,k] / f       expected absorbing-jump-from-i indicator
#     N̄ᵢ = Σⱼ M̄ᵢⱼ + Ēᵢ       expected exits from i
#
# A right-censored datum is a time c > 0 at which the subject is known only to
# be still alive: the observation is the event {τ > c} and the eventual cause is
# unobserved. The complete datum is still the whole path through its eventual
# absorption, so the E-step must both complete the path on either side of c and
# apportion the cause-resolved statistics over *all* causes. With
#
#     a(c)   = α' exp(Tc),      S(c) = a(c)·1        (survival, the likelihood)
#     g(c)   = a(c)(-T)⁻¹                            (occupation times after c)
#     R      = (-T)⁻¹D,   R[:,k] = Rₖ,   Σₖ Rₖ = 1   (eventual-cause matrix)
#     C(c;b) = ∫₀ᶜ exp(Tᵀu) (αᵀ bᵀ) exp(Tᵀ(c-u)) du
#
# the censored expectations are
#
#     B̄ᶜᵢₖ = αᵢ (exp(Tc) Rₖ)ᵢ / S
#     Z̄ᶜᵢ  = (C(c;1)[i,i] + gᵢ) / S
#     Ēᶜᵢₖ = gᵢ Dᵢₖ / S
#     M̄ᶜᵢⱼₖ = Tᵢⱼ (C(c;Rₖ)[i,j] + gᵢ ρⱼₖ) / S        (i ≠ j)
#     N̄ᶜᵢₖ = Σⱼ M̄ᶜᵢⱼₖ + Ēᶜᵢₖ
#
# and the record contributes log S(c) to the observed-data log-likelihood. Each
# censored record needs only *two* Van Loan exponentials, not one per cause:
# since Σₖ Rₖ = 1 and Σₖ ρⱼₖ = 1, the all-cause totals B̄ᶜᵢ, Z̄ᶜᵢ, N̄ᶜᵢ collapse
# onto C(c;1), while the reference-cause slices Mref/Nref need only C(c;R_ref).
# The cause-resolved B̄ᶜᵢₖ needs no block exponential at all — exp(Tc)R is one
# m×n product off the exp(Tᵀc) block the Van Loan call already returns. When
# n = 1 the two exponentials coincide and one suffices.
#
# Both kinds of record are then packaged into the same cause-resolved per-record
# contributions (bᵢₖ, zᵢ, mᵢⱼₖ, nᵢₖ) — an event puts its whole mass on its
# observed cause, a censored record spreads it fractionally over all causes —
# and the M-step is identical in the two cases. It maximizes the *relaxed*
# complete-data likelihood in the second parameterization (α, q, R, U) of
# maph_parameterization.jl — all closed form:
#
#     α̂ᵢ = Σℓ bᵢ / L                       ρ̂ᵢₖ = Σℓ bᵢₖ / Σℓ bᵢ
#     q̂ᵢ = Σℓ nᵢ / Σℓ zᵢ                  ûᵢⱼ = Σℓ mᵢⱼ,ref / Σℓ nᵢ,ref
#
# where `L` counts *all* records (events and censored) and `ref` is the
# reference cause (the most frequent observed event cause). The relaxed
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
is the observed-data log-likelihood Σℓ log f(tℓ, kℓ) + Σℓ log S(cℓ) at the
E-step parameters (the second sum over the right-censored records).

An exact-event record contributes to the single column `Bk[:, kℓ]` and, when
`kℓ` is the reference cause, to `Mref`/`Nref`. A censored record has no observed
cause, so the current model spreads it fractionally over every column of `Bk`
and over `Mref`/`Nref` in proportion to its posterior probability of eventually
being absorbed in the reference cause.
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
    _maph_estep(α, T, D, data, ref; censored = Float64[]) -> MAPHEMStats

One E-step: exact conditional expectations of the complete-data sufficient
statistics for MAPH(α, T, D), aggregated over the exact-event `data` (pairs
`(t, k)`) and the right-censored times `censored`.

Each exact-event record costs one (2m × 2m) Van Loan matrix exponential; each
censored record costs two — `C(c; 1)` for the all-cause totals and `C(c; R_ref)`
for the reference-cause slices — collapsing to one when `n == 1` (see the file
header for the formulas).
"""
function _maph_estep(α::Vector{Float64}, T::Matrix{Float64}, D::Matrix{Float64},
                     data::AbstractVector{<:Tuple{<:Real, <:Integer}}, ref::Int;
                     censored::AbstractVector{<:Real}=Float64[])
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

    isempty(censored) && return MAPHEMStats(B, Bk, Z, N, Mref, Nref, loglik)

    # ---- right-censored records ----
    # R and the factorization of (-T)ᵀ (used for g(c) = a(c)(-T)⁻¹, i.e.
    # (-T)ᵀ g = a) depend only on the parameters, so both are formed once.
    R = -T \ D
    negTt = lu(-Tt)
    ones_m = ones(m)
    dsum = vec(sum(D; dims=2))          # dsumᵢ = Σₖ Dᵢₖ, the total exit rate
    G1 = α * ones_m'                    # b = 1_m  → C(c; 1)
    Gref = n == 1 ? G1 : α * R[:, ref]' # b = R_ref → C(c; R_ref)

    for c in censored
        @views VL[1:m, m+1:2m] .= G1
        W1 = exp(VL .* c)
        expTt = @view W1[1:m, 1:m]      # exp(Tᵀc)
        C1 = @view W1[1:m, m+1:2m]      # C(c; 1)

        a = expTt * α                   # aᵢ = (α' exp(Tc))ᵢ
        S = sum(a)                      # S(c) = α' exp(Tc) 1
        S > 0 || continue               # skip numerically-degenerate points
        loglik += log(S)

        if n == 1
            Cref = C1
        else
            @views VL[1:m, m+1:2m] .= Gref
            Wref = exp(VL .* c)
            Cref = @view Wref[1:m, m+1:2m]   # C(c; R_ref)
        end

        g = negTt \ a                   # g(c) = a(c)(-T)⁻¹
        sv = expTt' * ones_m            # svᵢ = (exp(Tc) 1)ᵢ
        ER = expTt' * R                 # ER[i,k] = (exp(Tc) Rₖ)ᵢ

        invS = 1.0 / S
        @inbounds for i in 1:m
            B[i] += α[i] * sv[i] * invS
            for k in 1:n
                Bk[i, k] += α[i] * ER[i, k] * invS
            end
            Z[i] += (C1[i, i] + g[i]) * invS
            Mi = 0.0                    # Σⱼ Σₖ M̄ᶜᵢⱼₖ, all causes
            Miref = 0.0                 # Σⱼ M̄ᶜᵢⱼ,ref
            for j in 1:m
                i == j && continue
                Mi += T[i, j] * (C1[i, j] + g[i]) * invS
                Mijref = T[i, j] * (Cref[i, j] + g[i] * R[j, ref]) * invS
                Miref += Mijref
                Mref[i, j] += Mijref
            end
            N[i] += Mi + g[i] * dsum[i] * invS
            Nref[i] += Miref + g[i] * D[i, ref] * invS
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
    _maph_em(α0, T0, D0, data; ref, censored=Float64[], maxiter=500, tol=1e-7,
             verbose=false) -> MAPHEMResult

Run the approximate EM algorithm for MAPH(α, T, D) from the initial parameters,
on competing-risks `data` (exact-event pairs `(t, k)`) and the independently
right-censored times `censored`. `ref` is the reference absorbing cause used by
the second parameterization; it must be observed among the *events* and
reachable from every phase of the initial distribution, and is *not* relabeled
between iterations. Convergence is declared when the absolute change in the
observed-data log-likelihood drops below `tol` (the relaxed-plus-projected
M-step does not guarantee monotone increase).

At least one exact event is required — the cause count and the reference cause
are read off the events — and the M-step divides by the total record count
`length(data) + length(censored)`.
"""
function _maph_em(α0::AbstractVector{<:Real}, T0::AbstractMatrix{<:Real},
                  D0::AbstractMatrix{<:Real},
                  data::AbstractVector{<:Tuple{<:Real, <:Integer}};
                  ref::Int, censored::AbstractVector{<:Real}=Float64[],
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
    any(obs -> obs[2] == ref, data) ||
        throw(ArgumentError("reference cause $ref does not appear in the data"))
    cens = collect(Float64, censored)
    L = length(data) + length(cens)

    loglik = Float64[]
    converged = false
    nprojections = 0
    iter = 0
    prev_ll = -Inf
    smallsteps = 0          # consecutive iterations with |Δ loglik| < tol
    for outer iter in 1:maxiter
        stats = _maph_estep(α, T, D, data, ref; censored=cens)
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
