using Test
using LinearAlgebra
using Statistics
using Random
using StableRNGs
using Distributions
using PhaseTypeDistributions
using PhaseTypeDistributionsFitting

const PTDF = PhaseTypeDistributionsFitting

@testset "PhaseTypeDistributionsFitting.jl" begin

    @testset "input validation" begin
        @test_throws ArgumentError fit_mle(PHDist, Float64[]; m=2)
        @test_throws ArgumentError fit_mle(PHDist, [1.0, -2.0]; m=2)
        @test_throws ArgumentError fit_mle(PHDist, [1.0, 2.0])         # no m, no init
        @test_throws ArgumentError fit_mle(ErlangPHDist, [1.0, 2.0])   # no shape
    end

    @testset "m=1 exponential recovery (exact)" begin
        rng = StableRNG(1)
        truth = Exponential(2.5)               # mean 2.5
        data = rand(rng, truth, 4000)
        fit = fit_mle(PHDist, data; m=1)
        # For a single phase the EM gives λ = 1/mean(data) exactly.
        @test mean(fit) ≈ mean(data) rtol = 1e-8
        @test isapprox(mean(fit), 2.5; rtol = 0.05)
        # And it really is an exponential in shape.
        @test distribution_isapprox(fit, PHDist(Exponential(mean(data))); atol = 1e-6)
    end

    @testset "Erlang closed-form MLE" begin
        rng = StableRNG(2)
        truth = ErlangPHDist(3, 1.2)
        data = rand(rng, truth, 3000)
        fit = fit_mle(ErlangPHDist, data; m=3)
        @test fit isa ErlangPHDist
        @test nphases(fit) == 3
        @test fit.rate ≈ 3 / mean(data)        # closed form
        @test isapprox(mean(fit), mean(data); rtol = 1e-12)
    end

    @testset "EM increases the log-likelihood monotonically" begin
        rng = StableRNG(3)
        truth = HyperExponentialDist([0.6, 0.4], [1.0, 0.25])
        data = rand(rng, truth, 2000)
        α0, T0 = PTDF.default_init(3, data; rng = StableRNG(99))
        res = PTDF._em(α0, T0, data; maxiter = 200, tol = 1e-10)
        ll = res.loglik
        @test length(ll) >= 2
        # Allow a tiny numerical slack on the non-decrease.
        @test all(diff(ll) .>= -1e-6)
        @test res.loglik[end] >= res.loglik[1]
    end

    @testset "structural zeros are preserved" begin
        rng = StableRNG(4)
        data = rand(rng, HyperExponentialDist([0.5, 0.5], [1.0, 0.3]), 1500)

        @testset "Coxian stays bidiagonal" begin
            fit = fit_mle(CoxianDist, data; m=3)
            @test fit isa CoxianDist
            T = subgenerator(fit)
            for i in 1:3, j in 1:3
                if j != i && j != i + 1
                    @test T[i, j] == 0.0      # exact zeros, not "approximately"
                end
            end
            @test initial_prob(fit) == [1.0, 0.0, 0.0]
        end

        @testset "Hyperexponential stays diagonal" begin
            fit = fit_mle(HyperExponentialDist, data; m=3)
            @test fit isa HyperExponentialDist
            T = subgenerator(fit)
            for i in 1:3, j in 1:3
                i != j && @test T[i, j] == 0.0
            end
        end

        @testset "Hypoexponential absorbs only from the last phase" begin
            fit = fit_mle(HypoExponentialDist, data; m=3)
            @test fit isa HypoExponentialDist
            t0 = exit_rates(fit)
            @test t0[1] == 0.0
            @test t0[2] == 0.0
            @test t0[3] > 0.0
        end

        @testset "explicit zeros in a PHDist init stay zero" begin
            # Upper-triangular T with a structural zero at (1,3); α with a zero.
            T_init = [-3.0  1.0  0.0;
                       0.0 -2.0  1.0;
                       0.0  0.0 -1.0]
            α_init = [1.0, 0.0, 0.0]
            init = PHDist(α_init, T_init)
            fit = fit_mle(PHDist, data; init = init)
            Tf = subgenerator(fit)
            @test Tf[2, 1] == 0.0
            @test Tf[3, 1] == 0.0
            @test Tf[3, 2] == 0.0
            @test Tf[1, 3] == 0.0
            @test initial_prob(fit)[2] == 0.0
            @test initial_prob(fit)[3] == 0.0
        end
    end

    @testset "fitted parameters are FixedSparsity with the right pattern" begin
        rng = StableRNG(40)
        data = rand(rng, HyperExponentialDist([0.5, 0.5], [1.0, 0.3]), 1500)

        @testset "EMResult carries FixedSparsity α and T" begin
            α0, T0 = PTDF.coxian_init(3, data; rng = StableRNG(41))
            res = PTDF._em(α0, T0, data; maxiter = 100, tol = 1e-9)
            @test res.α isa FixedSparsityVector
            @test res.T isa FixedSparsityMatrix
            # Coxian structure inferred from the (structured) init.
            @test Vector(pattern(res.α)) == Bool[1, 0, 0]
            @test Matrix(pattern(res.T)) == Bool[1 1 0; 0 1 1; 0 0 1]
        end

        @testset "structured PHDist fit locks the pattern at the type level" begin
            cox = fit_mle(CoxianDist, data; m = 3)
            phd = fit_mle(PHDist, data; init = cox)   # init from a Coxian
            T = subgenerator(phd)
            a = initial_prob(phd)
            @test T isa FixedSparsityMatrix
            @test a isa FixedSparsityVector
            @test Vector(pattern(a)) == Bool[1, 0, 0]
            @test Matrix(pattern(T)) == Bool[1 1 0; 0 1 1; 0 0 1]
            # Off-pattern writes throw; in-pattern writes are allowed.
            @test_throws ArgumentError a[2] = 0.3
            @test_throws ArgumentError T[3, 1] = 0.5
            @test (T[1, 2] = 0.7) == 0.7
        end

        @testset "a general (dense-init) PHDist fit has a full pattern" begin
            phd = fit_mle(PHDist, data; m = 3)
            @test all(pattern(subgenerator(phd)))     # every entry free
            @test all(pattern(initial_prob(phd)))
        end

        @testset "exit-vector zeros are enforced (hypoexponential init)" begin
            # Absorption only from the last phase: t⁰ = [0, 0, λ]. Fitting a
            # *general* PHDist from a hypo init must keep those exit zeros.
            phd = fit_mle(PHDist, data; init = HypoExponentialDist([1.0, 2.0, 3.0]))
            t0 = exit_rates(phd)
            @test t0[1] == 0.0
            @test t0[2] == 0.0
            @test t0[3] > 0.0
        end
    end

    @testset "recovery of a hyperexponential (moment-level)" begin
        rng = StableRNG(5)
        truth = HyperExponentialDist([0.7, 0.3], [1.0, 0.2])  # mean 2.2, SCV > 1
        data = rand(rng, truth, 6000)
        emp_mean = mean(data)
        emp_scv = var(data) / emp_mean^2

        fit = fit_mle(HyperExponentialDist, data; m=2,
                      init = HyperExponentialDist([0.5, 0.5], [2.0, 0.5]))
        @test isapprox(mean(fit), emp_mean; rtol = 0.05)
        @test isapprox(scv(fit), emp_scv; rtol = 0.20)

        # A general PHDist fit should match the data mean well too.
        gfit = fit_mle(PHDist, data; m=2, init = HyperExponentialDist([0.5, 0.5], [2.0, 0.5]))
        @test isapprox(mean(gfit), emp_mean; rtol = 0.05)
    end

    @testset "fit_mm is a stub" begin
        @test_throws ErrorException fit_mm(PHDist, [1.0, 2.0]; m=2)
        @test_throws ErrorException fit_mm(CoxianDist, [1.0, 2.0]; m=2)
    end

    @testset "fit router (method=)" begin
        rng = StableRNG(6)
        data = rand(rng, Exponential(1.5), 1000)
        f1 = fit(PHDist, data; method = :mle, m = 1)
        @test f1 isa PHDist
        @test isapprox(mean(f1), mean(data); rtol = 1e-6)
        @test_throws ErrorException fit(PHDist, data; method = :mm, m = 2)   # stub
        @test_throws ArgumentError fit(PHDist, data; method = :bogus, m = 2)
    end

    @testset "m / init consistency" begin
        rng = StableRNG(7)
        data = rand(rng, Exponential(1.0), 500)
        # m disagreeing with the init's size is an error.
        @test_throws ArgumentError fit_mle(PHDist, data;
                                           m = 2, init = CoxianDist([1.0, 2.0, 3.0], [0.3, 0.4]))
        # m agreeing with the init's size is fine (and m is then redundant).
        fit = fit_mle(PHDist, data; m = 3, init = CoxianDist([1.0, 2.0, 3.0], [0.3, 0.4]))
        @test nphases(fit) == 3
    end

    @testset "Erlang shape validation" begin
        @test_throws ArgumentError fit_mle(ErlangPHDist, [1.0, 2.0]; m = 0)
        @test_throws ArgumentError fit_mle(ErlangPHDist, [1.0, 2.0]; m = -1)
    end

    @testset "EM reports convergence on easy data" begin
        rng = StableRNG(8)
        data = rand(rng, Exponential(2.0), 3000)
        # A 1-phase fit converges in a couple of iterations, well within maxiter.
        res = PTDF._em([1.0], reshape([-1.0], 1, 1), data; maxiter = 100, tol = 1e-9)
        @test res.converged
        @test res.iterations < 100
        # The fitted single rate is 1 / mean(data).
        @test Matrix(res.T)[1, 1] ≈ -1 / mean(data) rtol = 1e-6
    end

end
