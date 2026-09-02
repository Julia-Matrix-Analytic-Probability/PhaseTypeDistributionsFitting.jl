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

    # =======================================================================
    # MAPH (competing-risks) fitting
    # =======================================================================
    @testset "MAPH fitting" begin

        # A well-connected ground truth used throughout: m = 3, n = 2.
        T_truth = [-3.0  1.0  0.5;
                    0.8 -2.5  0.7;
                    0.4  0.6 -2.0]
        D_truth = [1.0 0.5;
                   0.4 0.6;
                   0.3 0.7]
        α_truth = [0.5, 0.3, 0.2]
        truth = MAPHDist(α_truth, T_truth, D_truth)

        maph_loglik(d, data) = sum(log(pdf(d, t, k)) for (t, k) in data)

        @testset "input validation" begin
            @test_throws ArgumentError fit_mle(MAPHDist, Tuple{Float64, Int}[]; m = 2)
            @test_throws ArgumentError fit_mle(MAPHDist, [(1.0, 1), (-2.0, 2)]; m = 2)
            @test_throws ArgumentError fit_mle(MAPHDist, [(1.0, 1), (2.0, 0)]; m = 2)
            @test_throws ArgumentError fit_mle(MAPHDist, [(1.0, 1), (2.0, 2)])  # no m, no init
            @test_throws ArgumentError fit_mle(MAPHDist, [(1.0, 1)]; m = 2, init_method = :bogus)
            @test_throws ArgumentError fit_mle(MAPHDist, [(1.0, 1), (2.0, 2)]; m = 5, init = truth)
            @test_throws ArgumentError fit(MAPHDist, [(1.0, 1), (2.0, 2)]; method = :mm, m = 2)
        end

        @testset "jump-coordinate round trip" begin
            q, Pl, Pm = PTDF._jump_parameters(T_truth, D_truth)
            @test q ≈ -diag(T_truth)
            @test all(diag(Pl) .== 0.0)
            @test vec(sum(Pl; dims = 2)) .+ vec(sum(Pm; dims = 2)) ≈ ones(3)
            α2, T2, D2 = PTDF._generator_from_jump(α_truth, q, Pl, Pm)
            @test α2 ≈ α_truth atol = 1e-14
            @test T2 ≈ T_truth atol = 1e-12
            @test D2 ≈ D_truth atol = 1e-12
            R = absorption_matrix(T_truth, D_truth)
            @test vec(sum(R; dims = 2)) ≈ ones(3)
            @test all(Pl * R .<= R .+ 1e-12)
        end


        @testset "E-step identities and quadrature cross-check" begin
            t, k = 1.3, 1
            stats = PTDF._maph_estep(α_truth, T_truth, D_truth, [(t, k)])
            # Exactly one start, total holding time t, exactly one absorbing jump:
            @test sum(stats.B) ≈ 1.0 atol = 1e-12
            @test sum(stats.Z) ≈ t atol = 1e-10
            @test sum(stats.E) ≈ 1.0 atol = 1e-10
            # All absorbing mass sits on the observed cause, and the row flow
            # balance ΣM + ΣE = N holds.
            @test sum(stats.E[:, k]) ≈ sum(stats.E) atol = 1e-14
            @test vec(sum(stats.M; dims = 2)) .+ vec(sum(stats.E; dims = 2)) ≈ stats.N atol = 1e-10


            # Cross-check Z̄ and M̄ against direct numerical integration of
            # C[i,j] = ∫₀ᵗ (α'e^{Tu})_i (e^{T(t-u)}D[:,k])_j du.
            f = (α_truth' * exp(T_truth * t) * D_truth[:, k])
            npts = 4001
            us = range(0, t; length = npts)
            front = [α_truth' * exp(T_truth * u) for u in us]
            back = [exp(T_truth * (t - u)) * D_truth[:, k] for u in us]
            h = step(us)
            for i in 1:3, j in 1:3
                vals = [front[s][i] * back[s][j] for s in 1:npts]
                Cij = h * (sum(vals) - 0.5 * (vals[1] + vals[end]))
                if i == j
                    @test stats.Z[i] ≈ Cij / f rtol = 1e-6
                else
                    @test stats.M[i, j] ≈ T_truth[i, j] * Cij / f rtol = 1e-6
                end
            end
        end

        @testset "no feasibility step is needed" begin
            # The M-step output lies in the parameter set by construction: the
            # flow balance makes each row of [Pλ | Pμ] a probability vector, so
            # the recovered D is non-negative with no projection.
            rng = StableRNG(77)
            data = rand(rng, truth, 400)
            stats = PTDF._maph_estep(α_truth, T_truth, D_truth, data)
            qprev, Plprev, Pmprev = PTDF._jump_parameters(T_truth, D_truth)
            _, q, Pl, Pm = PTDF._maph_mstep(stats, length(data), qprev, Plprev, Pmprev)
            @test all(Pl .>= 0) && all(Pm .>= 0)
            @test vec(sum(Pl; dims = 2)) .+ vec(sum(Pm; dims = 2)) ≈ ones(3) atol = 1e-12
            _, T2, D2 = PTDF._generator_from_jump(α_truth, q, Pl, Pm)
            @test all(D2 .>= 0)
            @test maximum(abs.(vec(sum(T2; dims = 2)) .+ vec(sum(D2; dims = 2)))) < 1e-10
        end

        @testset "absorption-certain guard" begin
            # A stochastic jump matrix cycles forever: -T is singular, and the
            # conversion must refuse rather than return an invalid MAPH.
            q = [1.0, 2.0]
            α = [0.5, 0.5]
            @test_throws ErrorException PTDF._generator_from_jump(
                α, q, [0.0 1.0; 1.0 0.0], zeros(2, 2))

            # With an escape route to absorption the conversion goes through.
            α2, T2, D2 = PTDF._generator_from_jump(
                α, q, [0.0 1.0; 0.0 0.0], [0.0 0.0; 0.6 0.4])
            @test all(D2 .>= 0) && sum(D2[2, :]) > 0
            @test vec(sum(absorption_matrix(T2, D2); dims = 2)) ≈ ones(2) atol = 1e-12
        end


        @testset "simplified initialization matches π̂ and the overall mean" begin
            rng = StableRNG(31)
            data = rand(rng, truth, 1500)
            L = length(data)
            π̂ = [count(o -> o[2] == k, data) / L for k in 1:2]
            μ̄ = mean(first.(data))

            # jitter = 0 is the exact Appendix-C construction: τ ~ Exp(1/μ̄) ⊥ κ.
            d0 = PTDF.maph_simplified_init(4, data; beta = 0.5, jitter = 0.0)
            @test nphases(d0) == 4
            @test nabsorbing(d0) == 2
            @test marginal_absorption(d0) ≈ π̂ atol = 1e-12
            @test mean(PHDist(d0)) ≈ μ̄ rtol = 1e-10
            # Conditional times are exponential with mean μ̄ — SCV 1, mean μ̄.
            for k in 1:2
                cond = conditional_time(d0, k)
                @test mean(cond) ≈ μ̄ rtol = 1e-8
                @test scv(cond) ≈ 1.0 atol = 1e-8
            end
            # Strictly positive parameters: reachability and interior start.
            @test all(Vector(initial_prob(d0)) .> 0)
            @test all(Matrix(absorption_probs(d0)) .> 0)

            # Default jitter breaks phase exchangeability (so the EM can
            # differentiate phases) while keeping π̂ and the mean exact.
            dj = PTDF.maph_simplified_init(4, data)
            @test marginal_absorption(dj) ≈ π̂ atol = 1e-12
            @test mean(PHDist(dj)) ≈ μ̄ rtol = 1e-10
            qs = -diag(Matrix(subgenerator(dj)))
            @test length(unique(round.(qs; digits = 10))) == 4   # distinct sojourn rates

            # m = 1 forces beta = 0 and still matches π̂ and the mean.
            d1 = PTDF.maph_simplified_init(1, data)
            @test marginal_absorption(d1) ≈ π̂ atol = 1e-12
            @test mean(PHDist(d1)) ≈ μ̄ rtol = 1e-10
        end

        @testset "moment initialization matches per-cause moments" begin
            rng = StableRNG(32)
            data = rand(rng, truth, 4000)
            L = length(data)
            π̂ = [count(o -> o[2] == k, data) / L for k in 1:2]

            # Unregularized: matching is exact (Proposition in Appendix C).
            d0 = PTDF.maph_moment_init(8, data; epsilon = 0.0, theta = 0.0)
            @test nphases(d0) == 8
            @test marginal_absorption(d0) ≈ π̂ atol = 1e-10
            for k in 1:2
                times = [t for (t, kk) in data if kk == k]
                μ̂k = mean(times)
                ĉ²k = var(times) / μ̂k^2
                cond = conditional_time(d0, k)
                @test mean(cond) ≈ μ̂k rtol = 1e-8
                @test scv(cond) ≈ ĉ²k rtol = 1e-8
            end

            # Default regularization: strictly positive α, full reachability,
            # and the matched quantities only perturbed slightly.
            dreg = PTDF.maph_moment_init(8, data)
            @test all(Vector(initial_prob(dreg)) .> 0)
            @test all(Matrix(absorption_probs(dreg)) .> 0)
            @test marginal_absorption(dreg) ≈ π̂ rtol = 0.05
            for k in 1:2
                times = [t for (t, kk) in data if kk == k]
                cond = conditional_time(dreg, k)
                @test mean(cond) ≈ mean(times) rtol = 0.1
            end

            # Too few phases for any block: falls back to the simplified init.
            dsmall = PTDF.maph_moment_init(1, data)
            @test nphases(dsmall) == 1
            @test marginal_absorption(dsmall) ≈ π̂ atol = 1e-12
        end

        @testset "m=1 EM has the closed-form solution" begin
            rng = StableRNG(33)
            data = rand(rng, truth, 800)
            L = length(data)
            fitted = fit_mle(MAPHDist, data; m = 1, maxiter = 50)
            @test nphases(fitted) == 1
            # q̂ = 1/mean, ρ̂ₖ = empirical proportions — exact closed forms.
            @test Matrix(subgenerator(fitted))[1, 1] ≈ -1 / mean(first.(data)) rtol = 1e-8
            π̂ = [count(o -> o[2] == k, data) / L for k in 1:2]
            @test marginal_absorption(fitted) ≈ π̂ atol = 1e-8
        end

        @testset "end-to-end recovery on synthetic data" begin
            rng = StableRNG(34)
            data = rand(rng, truth, 2000)
            L = length(data)
            fitted = fit_mle(MAPHDist, data; m = 3, maxiter = 300)
            @test fitted isa MAPHDist
            @test nphases(fitted) == 3
            @test nabsorbing(fitted) == 2

            # Marginal absorption probabilities match the empirical frequencies.
            π̂ = [count(o -> o[2] == k, data) / L for k in 1:2]
            @test marginal_absorption(fitted) ≈ π̂ atol = 0.02

            # The moment-init route approaches the truth's explanatory power;
            # on this seed its trajectory rides the feasibility boundary (the
            # projection fires most iterations) and is still crawling at the
            # iteration cap, hence the looser slack.
            @test maph_loglik(fitted, data) >= maph_loglik(truth, data) - 6.0

            # The simplified-init route beats the truth outright on this data —
            # the decisive identifiability-free check.
            fitted2 = fit_mle(MAPHDist, data; m = 3, init_method = :simplified, maxiter = 300)
            @test maph_loglik(fitted2, data) >= maph_loglik(truth, data)
            @test marginal_absorption(fitted2) ≈ π̂ atol = 0.02

            # And its conditional means land within a few percent of the truth's.
            for k in 1:2
                mt = kth_joint_moment(truth, k, 1) / marginal_absorption(truth)[k]
                mf = kth_joint_moment(fitted2, k, 1) / marginal_absorption(fitted2)[k]
                @test isapprox(mf, mt; rtol = 0.10)
            end
        end

        @testset "EM internals: trace, monotone ascent, convergence" begin
            rng = StableRNG(35)
            data = rand(rng, truth, 600)
            d0 = PTDF.maph_moment_init(3, data)
            res = PTDF._maph_em(Vector(initial_prob(d0)), Matrix(subgenerator(d0)),
                                Matrix(exit_rate_matrix(d0)), data;
                                maxiter = 200, tol = 1e-8)
            @test all(isfinite, res.loglik)
            @test res.loglik[end] >= res.loglik[1]       # improves overall
            @test all(diff(res.loglik) .>= -1e-9)   # exact M-step: monotone
            @test res.iterations <= 200
            # The result is a valid MAPH parameterization.
            @test isapprox(sum(res.α), 1.0; atol = 1e-10)
            @test all(res.D .>= 0)
            @test maximum(abs.(vec(sum(res.T; dims = 2) .+ sum(res.D; dims = 2)))) < 1e-8
        end

        @testset "structural zeros of α and T are EM-invariant" begin
            rng = StableRNG(36)
            data = rand(rng, truth, 400)
            # Feed-forward T with T[2,1] = T[3,1] = T[3,2] = 0 and α₃ = 0; both
            # absorbing states reachable from every phase, so the EM can run.
            α0 = [0.7, 0.3, 0.0]
            T0 = [-3.0  1.0  0.5;
                   0.0 -2.5  0.7;
                   0.0  0.0 -2.0]
            D0 = [1.0 0.5;
                  0.9 0.9;
                  1.2 0.8]
            res = PTDF._maph_em(α0, T0, D0, data; maxiter = 20, tol = 0.0)
            @test res.α[3] == 0.0
            @test res.T[2, 1] == 0.0
            @test res.T[3, 1] == 0.0
            @test res.T[3, 2] == 0.0
        end

        @testset "n=1 MAPH agrees with the PH EM" begin
            rng = StableRNG(37)
            ph_truth = HyperExponentialDist([0.6, 0.4], [1.0, 0.25])
            times = rand(rng, ph_truth, 1500)
            data = [(t, 1) for t in times]

            mfit = fit_mle(MAPHDist, data; m = 2, maxiter = 300, tol = 1e-9)
            pfit = fit_mle(PHDist, times; m = 2, maxiter = 300, tol = 1e-9,
                           rng = StableRNG(38))
            ll_m = maph_loglik(mfit, data)
            ll_p = sum(log(pdf(pfit, t)) for t in times)
            @test isapprox(ll_m, ll_p; rtol = 1e-3)
        end

        @testset "unobserved cause keeps (near) zero probability" begin
            rng = StableRNG(39)
            data = [(t, 1) for t in rand(rng, Exponential(1.0), 500)]
            # init declares two absorbing states, but only cause 1 is observed.
            init = PTDF.maph_simplified_init(2, [(1.0, 1), (1.0, 2)])
            fitted = fit_mle(MAPHDist, data; init = init, maxiter = 100)
            @test marginal_absorption(fitted)[2] < 1e-6
        end

        @testset "right-censored observations" begin
            # All-cause survival S(u) = α' exp(Tu) 1, the likelihood of a record
            # censored at u. Computed here from the raw matrices so the tests do
            # not depend on an unreleased PhaseTypeDistributions.
            surv(u) = only(α_truth' * exp(T_truth * u) * ones(3))
            R_truth = -T_truth \ D_truth
            noevents = Tuple{Float64, Int}[]

            @testset "E-step against direct tail integration" begin
                c = 0.8
                sc = PTDF._maph_estep(α_truth, T_truth, D_truth, noevents;
                                      censored = [c])

                # The censored expectations are the exact-event ones averaged
                # over all event times t > c (Tonelli split of the paper's
                # proof), so integrating the existing E-step over the tail must
                # reproduce them.
                tmax, npts = c + 25.0, 8001
                ts = range(c, tmax; length = npts)
                h = step(ts)
                S = surv(c)
                B = zeros(3); Z = zeros(3); N = zeros(3)
                M = zeros(3, 3); E = zeros(3, 2)
                for (s, t) in enumerate(ts), k in 1:2
                    f = pdf(truth, t, k)
                    f > 0 || continue
                    st = PTDF._maph_estep(α_truth, T_truth, D_truth, [(t, k)])
                    w = (s == 1 || s == npts ? 0.5h : h) * f      # trapezoid
                    B .+= w .* st.B
                    Z .+= w .* st.Z
                    N .+= w .* st.N
                    M .+= w .* st.M
                    E .+= w .* st.E
                end

                @test sc.loglik ≈ log(S) atol = 1e-12
                @test sc.B ≈ B ./ S atol = 1e-5
                @test sc.Z ≈ Z ./ S atol = 1e-5
                @test sc.N ≈ N ./ S atol = 1e-5
                @test sc.M ≈ M ./ S atol = 1e-5
                @test sc.E ≈ E ./ S atol = 1e-5
            end

            @testset "structural identities" begin
                for c in [0.05, 0.8, 4.0]
                    sc = PTDF._maph_estep(α_truth, T_truth, D_truth, noevents;
                                          censored = [c])
                    # Every completed path starts somewhere and is absorbed once.
                    @test sum(sc.B) ≈ 1.0 atol = 1e-12
                    @test sum(sc.E) ≈ 1.0 atol = 1e-10
                    # Row flow balance ΣM + ΣE = N.
                    @test vec(sum(sc.M; dims = 2)) .+ vec(sum(sc.E; dims = 2)) ≈ sc.N atol = 1e-10
                    @test all(sc.Z .>= 0) && all(sc.M .>= 0) && all(sc.E .>= 0)
                    # A censored path is still alive at c, so it has occupied
                    # the transient phases for at least c.
                    @test sum(sc.Z) >= c - 1e-10
                end
            end

            @testset "c → 0 reduces to unconditional expectations" begin
                sc = PTDF._maph_estep(α_truth, T_truth, D_truth, noevents;
                                      censored = [1e-9])
                @test sc.B ≈ α_truth atol = 1e-8
                @test sc.Z ≈ vec((-T_truth)' \ α_truth) atol = 1e-8   # α(-T)⁻¹
                # Σᵢ E[Eᵢₖ] is the unconditional cause distribution α R.
                @test vec(sum(sc.E; dims = 1)) ≈ vec(α_truth' * R_truth) atol = 1e-8
                @test sc.loglik ≈ 0.0 atol = 1e-8                     # S(0) = 1
            end

            @testset "n = 1 collapses to the marginal PH survival" begin
                # With a single cause the statistics are the classical
                # right-censored PH ones.
                ph = MAPHDist(HyperExponentialDist([0.6, 0.4], [1.0, 0.25]))
                αp = Vector(initial_prob(ph)); Tp = Matrix(subgenerator(ph))
                Dp = Matrix(exit_rate_matrix(ph))
                c = 1.7
                sc = PTDF._maph_estep(αp, Tp, Dp, noevents; censored = [c])
                Sp = only(αp' * exp(Tp * c) * ones(2))
                @test sc.loglik ≈ log(Sp) atol = 1e-12
                @test sum(sc.E) ≈ 1.0 atol = 1e-12          # one absorbing jump
                a = vec(exp(Tp * c)' * αp)
                g = (-Tp)' \ a
                @test sc.E[:, 1] ≈ g .* vec(sum(Dp; dims = 2)) ./ Sp atol = 1e-12
            end


            @testset "an empty `censored` changes nothing" begin
                rng = StableRNG(41)
                data = rand(rng, truth, 300)
                s1 = PTDF._maph_estep(α_truth, T_truth, D_truth, data)
                s2 = PTDF._maph_estep(α_truth, T_truth, D_truth, data;
                                      censored = Float64[])
                @test s1.B == s2.B && s1.Z == s2.Z && s1.M == s2.M
                @test s1.N == s2.N && s1.E == s2.E
                @test s1.loglik == s2.loglik
                f1 = fit_mle(MAPHDist, data; m = 3, maxiter = 30)
                f2 = fit_mle(MAPHDist, data; m = 3, maxiter = 30, censored = Float64[])
                @test f1 ≈ f2
            end

            @testset "log-likelihood carries the survival terms" begin
                rng = StableRNG(42)
                data = rand(rng, truth, 200)
                cens = [0.3, 0.9, 1.4, 2.2]
                d0 = PTDF.maph_simplified_init(3, data; censored = cens)
                res = PTDF._maph_em(Vector(initial_prob(d0)), Matrix(subgenerator(d0)),
                                    Matrix(exit_rate_matrix(d0)), data;
                                    censored = cens, maxiter = 25, tol = 1e-9)
                expected = maph_loglik(d0, data) +
                           sum(log(only(Vector(initial_prob(d0))' *
                                        exp(Matrix(subgenerator(d0)) * c) * ones(3)))
                               for c in cens)
                @test res.loglik[1] ≈ expected rtol = 1e-10
                @test all(isfinite, res.loglik)
                @test res.loglik[end] >= res.loglik[1]
            end

            @testset "censoring-compatible initialization" begin
                rng = StableRNG(43)
                data = rand(rng, truth, 400)
                cens = [0.5, 1.0, 1.5, 2.0, 2.5]
                d = length(data)
                # π̂ is the observed-event proportion and μ̄ the exposure per
                # event — both summing censored exposure but not censored events.
                π̂ = [count(o -> o[2] == k, data) / d for k in 1:2]
                μ̄ = (sum(first.(data)) + sum(cens)) / d
                d0 = PTDF.maph_simplified_init(3, data; censored = cens, jitter = 0.0)
                @test marginal_absorption(d0) ≈ π̂ atol = 1e-12
                @test mean(PHDist(d0)) ≈ μ̄ rtol = 1e-10
                # Without censoring it is the plain sample mean, as before.
                d1 = PTDF.maph_simplified_init(3, data; jitter = 0.0)
                @test mean(PHDist(d1)) ≈ mean(first.(data)) rtol = 1e-10

                # The moment initialization refuses biased event-time targets ...
                @test_throws ArgumentError PTDF.maph_moment_init(4, data; censored = cens)
                @test_throws ArgumentError fit_mle(MAPHDist, data; m = 4,
                                                   censored = cens,
                                                   init_method = :moment)
                # ... but accepts externally adjusted ones.
                μk = [kth_joint_moment(truth, k, 1) / marginal_absorption(truth)[k]
                      for k in 1:2]
                c²k = [0.9, 1.2]
                d2 = PTDF.maph_moment_init(4, data; censored = cens,
                                           cond_means = μk, cond_scvs = c²k)
                @test d2 isa MAPHDist && nphases(d2) == 4
                @test_throws ArgumentError PTDF.maph_moment_init(4, data; cond_means = μk)
                @test_throws DimensionMismatch PTDF.maph_moment_init(
                    4, data; cond_means = [1.0], cond_scvs = [1.0])
            end

            @testset "end-to-end recovery under censoring" begin
                rng = StableRNG(44)
                L, chor = 2000, 0.5            # administrative horizon
                events = Tuple{Float64, Int}[]
                cens = Float64[]
                for _ in 1:L
                    τ, κ = rand(rng, truth)
                    τ <= chor ? push!(events, (τ, κ)) : push!(cens, chor)
                end
                @test 0.3 < length(cens) / L < 0.8      # a heavily censored sample

                fitted = fit_mle(MAPHDist, events; m = 3, censored = cens, maxiter = 200)
                dropped = fit_mle(MAPHDist, events; m = 3, init_method = :simplified,
                                  maxiter = 200)
                naive = fit_mle(MAPHDist, vcat(events, [(c, 1) for c in cens]);
                                m = 3, init_method = :simplified, maxiter = 200)

                πt = marginal_absorption(truth)
                μt = mean(PHDist(truth))
                condmean(f, k) = kth_joint_moment(f, k, 1) / marginal_absorption(f)[k]

                # The censored-aware fit recovers the truth's scale and cause
                # split, neither of which is identified by the events alone.
                @test marginal_absorption(fitted) ≈ πt atol = 0.05
                @test mean(PHDist(fitted)) ≈ μt rtol = 0.10
                for k in 1:2
                    @test condmean(fitted, k) ≈ condmean(truth, k) rtol = 0.10
                end

                # Both ways of ignoring the censoring mechanism are badly biased:
                # dropping the censored records keeps only the short times, and
                # recording a censoring time as an event does the same while also
                # attributing the record to a cause it may never have reached.
                @test mean(PHDist(dropped)) < 0.5 * μt
                @test abs(mean(PHDist(fitted)) - μt) < abs(mean(PHDist(naive)) - μt)
                @test maximum(abs.(marginal_absorption(fitted) .- πt)) <
                      maximum(abs.(marginal_absorption(naive) .- πt))
            end

            @testset "input validation" begin
                data = [(1.0, 1), (2.0, 2)]
                @test_throws ArgumentError fit_mle(MAPHDist, data; m = 2, censored = [0.0])
                @test_throws ArgumentError fit_mle(MAPHDist, data; m = 2, censored = [-1.0])
                @test_throws ArgumentError fit_mle(MAPHDist, Tuple{Float64, Int}[];
                                                   m = 2, censored = [1.0, 2.0])
                @test_throws ArgumentError PTDF._maph_em(α_truth, T_truth, D_truth, data;
                                                          censored = [-0.5])
                @test_throws ArgumentError PTDF.maph_simplified_init(2, data;
                                                                     censored = [0.0])
            end
        end

    end

end
