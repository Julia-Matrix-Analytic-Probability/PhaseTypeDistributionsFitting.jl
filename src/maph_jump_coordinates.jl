# Jump-chain coordinates of an MAPH distribution.
#
# An MAPH(α, T, D) is written equivalently as (α, q, Pλ, Pμ) where
#
#   q    m-vector of phase exit rates, qᵢ = -Tᵢᵢ
#   Pλ   m×m matrix of one-step transient jump probabilities, Pλᵢⱼ = Tᵢⱼ/qᵢ
#        (i ≠ j), zero diagonal
#   Pμ   m×n matrix of one-step absorption probabilities, Pμᵢₖ = Dᵢₖ/qᵢ
#
# so that each row of [Pλ | Pμ] is a probability vector. This is a change of
# variables, not a different model: T = Diag(q)(Pλ - I) and D = Diag(q)Pμ.
#
# The M-step of the MAPH EM (see maph_em.jl) is the exact maximizer of the
# expected complete-data log-likelihood in these coordinates and lands in the
# parameter set by construction, so no feasibility projection is required.
#
# The absorption-probability matrix R = -T⁻¹D, whose entry ρᵢₖ is the
# probability of eventual absorption in cause k starting from phase i, is a
# derived quantity: it is reported alongside a fit rather than estimated, and
# `absorption_matrix` computes it.

"""
    _jump_parameters(T, D) -> (q, Pλ, Pμ)

Extract the jump-chain coordinates `(q, Pλ, Pμ)` of MAPH(α, T, D). Requires
strictly negative diagonal entries of `T`.
"""
function _jump_parameters(T::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real})
    m, n = size(D)
    size(T) == (m, m) || throw(DimensionMismatch("T must be $m × $m"))
    q = -diag(T)
    all(q .> 0) || throw(ArgumentError("diagonal of T must be strictly negative"))
    Pl = zeros(m, m)
    Pm = Matrix{Float64}(undef, m, n)
    for i in 1:m
        for j in 1:m
            i == j && continue
            Pl[i, j] = T[i, j] / q[i]
        end
        for k in 1:n
            Pm[i, k] = D[i, k] / q[i]
        end
    end
    return q, Pl, Pm
end

"""
    _assert_absorption_certain(T, D, q; rtol=1e-12)

Check that absorption is certain, i.e. that `-T` is non-singular. The check is
structural: every phase must reach, through the support of the off-diagonal
entries of `T`, a phase `i` with exit rate `Σₖ Dᵢₖ > rtol·qᵢ` into the
absorbing states. Throws an informative error when the check fails.

For the EM this cannot fail — the fitted `Pμ` inherits the support of the
expected absorbing-jump counts, which come from records that were absorbed — so
the check is a numerical guard rather than a modelling restriction.
"""
function _assert_absorption_certain(T::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real},
                                    q::AbstractVector{<:Real}; rtol::Real=1e-12)
    m = size(T, 1)
    can_absorb = falses(m)
    stack = Int[]
    for i in 1:m
        if sum(view(D, i, :)) > rtol * q[i]
            can_absorb[i] = true
            push!(stack, i)
        end
    end
    while !isempty(stack)
        j = pop!(stack)
        for i in 1:m
            if !can_absorb[i] && i != j && T[i, j] > 0
                can_absorb[i] = true
                push!(stack, i)
            end
        end
    end
    trapped = findall(.!can_absorb)
    isempty(trapped) || error(
        "absorption is not certain: phases $trapped cannot reach the absorbing " *
        "states through the support of T, so -T is singular and (T, D) does not " *
        "define a valid MAPH distribution")
    return nothing
end

"""
    _generator_from_jump(α, q, Pλ, Pμ) -> (α, T, D)

Convert the jump-chain coordinates back to generator form, `Tᵢⱼ = qᵢ·Pλᵢⱼ`
(i ≠ j), `Dᵢₖ = qᵢ·Pμᵢₖ`, with the diagonal set to `Tᵢᵢ = -(Σ_{j≠i} Tᵢⱼ + Σₖ Dᵢₖ)`
so that the row sums are exactly zero whatever round-off the rows of
`[Pλ | Pμ]` carry. `α` is renormalized to sum to exactly 1. Throws if
absorption is not certain (see [`_assert_absorption_certain`](@ref)).
"""
function _generator_from_jump(α::AbstractVector{<:Real}, q::AbstractVector{<:Real},
                              Pl::AbstractMatrix{<:Real}, Pm::AbstractMatrix{<:Real})
    m, n = size(Pm)
    T = zeros(m, m)
    D = Matrix{Float64}(undef, m, n)
    for i in 1:m
        for j in 1:m
            i == j && continue
            T[i, j] = q[i] * max(Pl[i, j], 0.0)
        end
        for k in 1:n
            D[i, k] = q[i] * max(Pm[i, k], 0.0)
        end
    end
    for i in 1:m
        T[i, i] = -(sum(view(T, i, 1:m)) - T[i, i] + sum(view(D, i, :)))
    end
    _assert_absorption_certain(T, D, q)
    αout = collect(Float64, α)
    s = sum(αout)
    s > 0 || throw(ArgumentError("α summed to $s; cannot normalise"))
    αout ./= s
    return αout, T, D
end

"""
    absorption_matrix(T, D) -> R

The m×n matrix of eventual-absorption probabilities, `R = -T⁻¹D`, whose entry
`R[i, k]` is the probability that a trajectory started in phase `i` is
eventually absorbed in cause `k`. Rows sum to one. This is the competing-risks
summary of a fitted model; it is derived from `(T, D)`, not estimated.
"""
function absorption_matrix(T::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real})
    return -Matrix(T) \ Matrix(D)
end
