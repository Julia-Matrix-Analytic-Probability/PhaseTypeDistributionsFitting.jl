# Second (inference-oriented) parameterization of an MAPH distribution and the
# constraint-enforcement linear program.
#
# An MAPH(α, T, D) admits an equivalent parameterization (α, q, R, U) where
#
#   q   m-vector of phase exit rates, qᵢ = -Tᵢᵢ
#   R   m×n absorption-probability matrix, R = -T⁻¹D, rows sum to 1
#   U   m×m matrix of conditional one-step jump probabilities given absorption
#       in a fixed *reference* cause `ref`:  uᵢⱼ = p^λ_{ij|ref} = (Tᵢⱼ/qᵢ)·(ρ_{j,ref}/ρ_{i,ref})
#
# The M-step of the MAPH EM (see maph_em.jl) produces (α, q, R, U) in closed
# form by maximizing a *relaxed* likelihood, ignoring the linear constraints
#
#   Σⱼ (ρⱼₖ/ρⱼ,ref) uᵢⱼ ≤ ρᵢₖ/ρᵢ,ref      for all i = 1..m, k = 1..n,     (*)
#
# which are exactly the conditions making the recovered D = -T·R entrywise
# non-negative. When a row of U violates (*), it is returned to the feasible
# polytope by an ℓ1 projection — a small linear program that separates row by
# row (all rows share the constraint matrix A and differ only in the
# right-hand side), solved here with HiGHS.
#
# The paper uses cause 1 as the reference (assuming causes are ordered by
# empirical frequency). The functions below take `ref` explicitly so the data
# may keep its original cause labels; the EM picks `ref` as the most frequent
# observed cause.

# HiGHS uses values ≥ 1e30 as infinities.
const _HIGHS_INF = 1e30

# Default tolerance when testing the linear constraints (*) for violation.
const _U_FEAS_TOL = 1e-10

"""
    _second_parameterization(T, D; ref=1) -> (q, R, U)

Extract the `(q, R, U)` parameters of MAPH(α, T, D) with respect to the
reference absorbing cause `ref`. Requires `-T` non-singular (absorption certain)
and `R[:, ref] > 0` (the reference cause reachable from every phase).
"""
function _second_parameterization(T::AbstractMatrix{<:Real}, D::AbstractMatrix{<:Real}; ref::Int=1)
    m = size(T, 1)
    1 <= ref <= size(D, 2) || throw(ArgumentError("ref = $ref out of range 1..$(size(D, 2))"))
    q = -diag(T)
    all(q .> 0) || throw(ArgumentError("diagonal of T must be strictly negative"))
    R = -Matrix(T) \ Matrix(D)
    all(R[:, ref] .> 0) || throw(ArgumentError(
        "reference cause $ref must be reachable from every phase (R[:, $ref] > 0)"))
    U = zeros(m, m)
    for i in 1:m, j in 1:m
        i == j && continue
        U[i, j] = (T[i, j] / q[i]) * (R[j, ref] / R[i, ref])
    end
    return q, R, U
end

"""
    _uconstraints(R, ref) -> A

The `n × m` constraint matrix of (*), `A[k, j] = R[j, k] / R[j, ref]`, shared by
all row programs. The right-hand side for row `i` is `b[k] = R[i, k] / R[i, ref]`.
Row `ref` of `A` is all ones with right-hand side 1 — sub-stochasticity of U.
"""
function _uconstraints(R::AbstractMatrix{<:Real}, ref::Int)
    m, n = size(R)
    A = Matrix{Float64}(undef, n, m)
    for j in 1:m, k in 1:n
        A[k, j] = R[j, k] / R[j, ref]
    end
    return A
end

"""
    _l1_row_projection(A, b, uhat, i) -> Vector{Float64}

Solve the row program of the constraint-enforcement step: the ℓ1 projection of
the row `uhat` onto `{u ≥ 0 : A·u ≤ b, uᵢ = 0}`,

    min Σⱼ dⱼ   s.t.   -dⱼ ≤ uⱼ - ûⱼ ≤ dⱼ,   A·u ≤ b,   u ≥ 0,   uᵢ = 0,

as a linear program in the 2m variables (u, d), solved with HiGHS. The program
is always feasible (u = 0 works since b ≥ 0) and bounded, so an optimum exists.
"""
function _l1_row_projection(A::Matrix{Float64}, b::Vector{Float64},
                            uhat::AbstractVector{<:Real}, i::Int)
    n, m = size(A)
    length(b) == n || throw(DimensionMismatch("b must have length $n"))
    length(uhat) == m || throw(DimensionMismatch("uhat must have length $m"))

    ncol = 2m                     # variables: u₁..u_m, d₁..d_m
    nrow = 2m + n                 # envelope rows (2m) then feasibility rows (n)

    col_cost = vcat(zeros(m), ones(m))
    col_lower = zeros(ncol)
    col_upper = fill(_HIGHS_INF, ncol)
    col_upper[i] = 0.0            # uᵢᵢ = 0 (diagonal entry of U)

    # Row bounds: for j = 1..m,
    #   row j      :  uⱼ - dⱼ ∈ (-∞, ûⱼ]
    #   row m + j  :  uⱼ + dⱼ ∈ [ûⱼ, ∞)
    # and for k = 1..n,
    #   row 2m + k :  Σⱼ A[k,j]·uⱼ ∈ (-∞, bₖ]
    row_lower = vcat(fill(-_HIGHS_INF, m), Vector{Float64}(uhat), fill(-_HIGHS_INF, n))
    row_upper = vcat(Vector{Float64}(uhat), fill(_HIGHS_INF, m), b)

    # Constraint matrix in compressed sparse column form (0-based for HiGHS).
    # Column uⱼ hits rows j (+1), m+j (+1), and 2m+k (A[k,j]) for k = 1..n;
    # column dⱼ hits rows j (-1) and m+j (+1).
    nnz_total = m * (2 + n) + 2m
    a_start = Vector{HighsInt}(undef, ncol + 1)
    a_index = Vector{HighsInt}(undef, nnz_total)
    a_value = Vector{Float64}(undef, nnz_total)
    pos = 0
    for j in 1:m                  # u columns
        a_start[j] = pos
        a_index[pos + 1] = j - 1;          a_value[pos + 1] = 1.0
        a_index[pos + 2] = m + j - 1;      a_value[pos + 2] = 1.0
        pos += 2
        for k in 1:n
            a_index[pos + 1] = 2m + k - 1; a_value[pos + 1] = A[k, j]
            pos += 1
        end
    end
    for j in 1:m                  # d columns
        a_start[m + j] = pos
        a_index[pos + 1] = j - 1;          a_value[pos + 1] = -1.0
        a_index[pos + 2] = m + j - 1;      a_value[pos + 2] = 1.0
        pos += 2
    end
    a_start[ncol + 1] = pos

    highs = Highs_create()
    local u
    try
        Highs_setBoolOptionValue(highs, "output_flag", 0)
        ret = Highs_passLp(highs, HighsInt(ncol), HighsInt(nrow), HighsInt(nnz_total),
                           kHighsMatrixFormatColwise, kHighsObjSenseMinimize, 0.0,
                           col_cost, col_lower, col_upper, row_lower, row_upper,
                           a_start, a_index, a_value)
        ret == 0 || error("HiGHS rejected the projection LP (status $ret)")
        ret = Highs_run(highs)
        ret == 0 || error("HiGHS failed on the projection LP (status $ret)")
        Highs_getModelStatus(highs) == kHighsModelStatusOptimal ||
            error("projection LP did not reach optimality (model status " *
                  "$(Highs_getModelStatus(highs))); this should be impossible — " *
                  "the program is always feasible and bounded")
        col_value = Vector{Float64}(undef, ncol)
        Highs_getSolution(highs, col_value, C_NULL, C_NULL, C_NULL)
        u = max.(col_value[1:m], 0.0)   # clip solver round-off below zero
        u[i] = 0.0
    finally
        Highs_destroy(highs)
    end
    return u
end

"""
    _project_U!(U, R, ref; tol=_U_FEAS_TOL) -> Int

Enforce the linear constraints (*) on `U` (with `R` held fixed), row by row:
rows already feasible are left untouched; each violating row is replaced by its
ℓ1 projection onto the feasible polytope ([`_l1_row_projection`](@ref)). Returns
the number of rows that were projected, so callers can monitor how often the
constraint-enforcement step is active.
"""
function _project_U!(U::Matrix{Float64}, R::Matrix{Float64}, ref::Int;
                     tol::Real=_U_FEAS_TOL)
    m, n = size(R)
    A = _uconstraints(R, ref)
    nprojected = 0
    b = Vector{Float64}(undef, n)
    for i in 1:m
        for k in 1:n
            b[k] = R[i, k] / R[i, ref]
        end
        violated = false
        for k in 1:n
            s = 0.0
            for j in 1:m
                s += A[k, j] * U[i, j]
            end
            if s > b[k] + tol
                violated = true
                break
            end
        end
        violated || continue
        U[i, :] = _l1_row_projection(A, copy(b), view(U, i, :), i)
        nprojected += 1
    end
    return nprojected
end

"""
    _generator_from_second(α, q, R, U; ref=1) -> (α, T, D)

Convert `(α, q, R, U)` back to generator form. Off-diagonal rates are
`Tᵢⱼ = qᵢ·uᵢⱼ·ρᵢ,ref/ρⱼ,ref` and `D = -T·R` with `Tᵢᵢ = -qᵢ`. Solver and
floating-point round-off can leave tiny negative entries in `D` even after the
projection, so `D` is clamped at zero and the diagonal of `T` is recomputed as
`Tᵢᵢ = -(Σ_{j≠i} Tᵢⱼ + Σₖ Dᵢₖ)` — preserving exact zero row sums at the cost of
an O(round-off) shift in qᵢ. `α` is renormalized to sum to exactly 1.
"""
function _generator_from_second(α::AbstractVector{<:Real}, q::AbstractVector{<:Real},
                                R::AbstractMatrix{<:Real}, U::AbstractMatrix{<:Real};
                                ref::Int=1)
    m, n = size(R)
    T = zeros(m, m)
    for i in 1:m, j in 1:m
        i == j && continue
        T[i, j] = q[i] * U[i, j] * R[i, ref] / R[j, ref]
    end
    # D = -T·R with the diagonal -q in place; equivalently Dᵢₖ = qᵢ(ρᵢₖ - Σⱼ p^λᵢⱼ ρⱼₖ).
    D = Matrix{Float64}(undef, m, n)
    for i in 1:m, k in 1:n
        s = q[i] * R[i, k]
        for j in 1:m
            j == i && continue
            s -= T[i, j] * R[j, k]
        end
        D[i, k] = max(s, 0.0)
    end
    for i in 1:m
        T[i, i] = -(sum(view(T, i, 1:m)) - T[i, i] + sum(view(D, i, :)))
    end
    αout = collect(Float64, α)
    s = sum(αout)
    s > 0 || throw(ArgumentError("α summed to $s; cannot normalise"))
    αout ./= s
    return αout, T, D
end
