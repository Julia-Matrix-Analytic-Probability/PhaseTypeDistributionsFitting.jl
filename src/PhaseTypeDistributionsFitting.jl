module PhaseTypeDistributionsFitting

using LinearAlgebra
using Random
using Statistics
using Distributions
using PhaseTypeDistributions
using FixedSparsityMatrices: FixedSparsityVector, FixedSparsityMatrix, pattern

# Extend the Distributions.jl fitting interface rather than shadowing it.
import Distributions: fit, fit_mle

# Core EM algorithm (E-step via Van Loan block-matrix exponential, M-step,
# convergence loop) operating on (α, T) and preserving sparsity patterns.
include("em.jl")

# Initialization strategies consumed by the EM routines.
include("initialization.jl")

# Public API: fit_mle (EM), fit_mm (stub), and the fit(...; method=) router,
# for PHDist and the structured PH families.
include("fit.jl")

# Public fitting API. `fit` and `fit_mle` are re-exported from Distributions.jl;
# `fit_mm` is new here.
export fit_mm

end
