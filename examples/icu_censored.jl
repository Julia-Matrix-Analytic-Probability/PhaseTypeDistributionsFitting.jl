# Fitting a MAPH distribution to right-censored competing-risks data.
#
# The intensive-care length-of-stay data of Beyersmann, Allignol and Schumacher
# (2012), Ch. 1: 650 patients admitted without pneumonia, each leaving the unit
# through one of two competing causes -- alive discharge or hospital death --
# except for six whose stay is right censored. Those six records are exactly
# what `fit_mle(MAPHDist, events; censored = ...)` exists to use: dropping them
# throws away information, and recording a censoring time as an event both
# compresses the fitted time scale and attributes the record to a cause the
# patient may never have reached.
#
# Run from the package root with the example environment:
#
#     julia --project=examples examples/icu_censored.jl
#
# Everything below is on the original day scale.

using PhaseTypeDistributionsFitting, PhaseTypeDistributions
using Distributions, Statistics, LinearAlgebra, Printf

# ---- data ----------------------------------------------------------------

function readcsv(path)
    lines = readlines(path)
    hdr = [strip(h, ['"']) for h in split(strip(lines[1]), ",")]
    rows = [[strip(c, ['"']) for c in split(l, ",")]
            for l in lines[2:end] if !isempty(strip(l))]
    return hdr, rows
end
col(hdr, rows, name) = [r[findfirst(==(name), hdr)] for r in rows]

hdr, rows = readcsv(joinpath(@__DIR__, "sir_adm.csv"))
pneu = parse.(Int, col(hdr, rows, "pneu"))
stat = parse.(Int, col(hdr, rows, "status"))     # 0 censored, 1 discharge, 2 death
time = parse.(Float64, col(hdr, rows, "time"))

keep = findall(==(0), pneu)
events = [(time[i], stat[i]) for i in keep if stat[i] != 0]
censored = [time[i] for i in keep if stat[i] == 0]

@printf("%d patients: %d discharged, %d died, %d right censored at %s days\n",
        length(keep), count(e -> e[2] == 1, events), count(e -> e[2] == 2, events),
        length(censored), string(Int.(sort(censored))))

# ---- fit -----------------------------------------------------------------
# `init_method` defaults to :auto, which picks the censoring-compatible
# simplified initialization as soon as any record is censored: the per-cause
# event-time moments the moment-based start needs are biased once the long
# stays are censored.

fitted = fit_mle(MAPHDist, events; m = 3, censored = censored, maxiter = 800)

# One run from the default start, for brevity. The relaxed-and-projected M-step
# is sensitive to its starting point, so a single run can settle well below the
# best fit available at this order -- on these data, tens of nats below. In
# earnest, fit from several starts and keep the best log-likelihood; see the
# `init` keyword and the discussion of boundary-riding in the README.

# The observed-data log-likelihood: a density term per event, a survival term
# per censored record.
survival(d, u) = only(Vector(initial_prob(d))' * exp(Matrix(subgenerator(d)) * u) *
                      ones(nphases(d)))
loglik(d) = sum(log(pdf(d, t, k)) for (t, k) in events) +
            sum(log(survival(d, c)) for c in censored; init = 0.0)

println("\nfitted MAPH_{3,2}, all 650 records")
@printf("  log-likelihood        %.2f\n", loglik(fitted))
@printf("  P(alive discharge)    %.4f\n", marginal_absorption(fitted)[1])
@printf("  P(hospital death)     %.4f\n", marginal_absorption(fitted)[2])
@printf("  E[length of stay]     %.2f days\n", mean(PHDist(fitted)))
for k in 1:2
    cause = k == 1 ? "discharge" : "death"
    μ = kth_joint_moment(fitted, k, 1) / marginal_absorption(fitted)[k]
    @printf("  E[stay | %-9s]  %.2f days\n", cause, μ)
end

# ---- what the censored records buy ---------------------------------------
# Refit from the 644 events alone and score both fits on all 650 records.

dropped = fit_mle(MAPHDist, events; m = 3, init_method = :simplified, maxiter = 800)

println("\nsame order, censored records discarded")
@printf("  log-likelihood on all 650   %.2f   (censored fit %.2f)\n",
        loglik(dropped), loglik(fitted))
@printf("  E[length of stay]           %.2f days   (censored fit %.2f)\n",
        mean(PHDist(dropped)), mean(PHDist(fitted)))

# With only 6 of 650 records censored the two analyses agree closely, as they
# should. The difference is decisive when censoring is heavy -- see the
# simulated check below, and the package's test suite.

# ---- the same comparison under heavy censoring ---------------------------

using Random
rng = MersenneTwister(2024)
truth = MAPHDist([0.5, 0.3, 0.2],
                 [-3.0 1.0 0.5; 0.8 -2.5 0.7; 0.4 0.6 -2.0],
                 [1.0 0.5; 0.4 0.6; 0.3 0.7])
sim = rand_censored(rng, truth, 2000, 0.5)          # administrative horizon

f_cens = fit_mle(MAPHDist, sim.events; m = 3, censored = sim.censored, maxiter = 300)
f_drop = fit_mle(MAPHDist, sim.events; m = 3, init_method = :simplified, maxiter = 300)

@printf("\nsimulated MAPH_{3,2}, %.0f%% of 2000 subjects censored at u = 0.5\n",
        100 * length(sim.censored) / 2000)
@printf("  true       E[tau] = %.3f\n", mean(PHDist(truth)))
@printf("  censored   E[tau] = %.3f  (%+.1f%%)\n", mean(PHDist(f_cens)),
        100 * (mean(PHDist(f_cens)) / mean(PHDist(truth)) - 1))
@printf("  discarded  E[tau] = %.3f  (%+.1f%%)\n", mean(PHDist(f_drop)),
        100 * (mean(PHDist(f_drop)) / mean(PHDist(truth)) - 1))
