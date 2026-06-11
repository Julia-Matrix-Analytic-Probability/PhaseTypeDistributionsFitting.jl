module PhaseTypeDistributionsFitting

using LinearAlgebra
using Random
using Statistics
using Distributions
using PhaseTypeDistributions
using FixedSparsityMatrices: FixedSparsityVector, FixedSparsityMatrix, pattern
using HiGHS: HighsInt, Highs_create, Highs_destroy, Highs_setBoolOptionValue,
             Highs_passLp, Highs_run, Highs_getModelStatus, Highs_getSolution,
             kHighsMatrixFormatColwise, kHighsObjSenseMinimize, kHighsModelStatusOptimal

# Extend the Distributions.jl fitting interface rather than shadowing it.
import Distributions: fit, fit_mle

# Core EM algorithm (E-step via Van Loan block-matrix exponential, M-step,
# convergence loop) operating on (α, T) and preserving sparsity patterns.
include("em.jl")

# Initialization strategies consumed by the EM routines.
include("initialization.jl")

# MAPH (competing-risks) fitting: the second parameterization (α, q, R, U),
# the constraint-enforcement LP (HiGHS), the approximate EM engine, and the
# Appendix-C initialization heuristics.
include("maph_parameterization.jl")
include("maph_em.jl")
include("maph_initialization.jl")

# Public API: fit_mle (EM), fit_mm (stub), and the fit(...; method=) router,
# for PHDist, the structured PH families, and MAPHDist.
include("fit.jl")

# Public fitting API. `fit` and `fit_mle` are re-exported from Distributions.jl;
# `fit_mm` is new here.
export fit_mm

end
