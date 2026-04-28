# `legacy/` — pre-redesign MAPH fitting code

This directory is a verbatim copy of the older MAPH EM-fitting code that lived
in `src/maph/` and `test/maph/` of
[PhaseTypeDistributions.jl](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl)
prior to the redesign. The redesign commit
[`f8cdd77`](https://github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributions.jl/commit/f8cdd77)
narrowed that package to distribution types only; the fitting code was left in
place but disconnected from the module.

It is preserved here as a **reference** — not loaded, not registered, not
expected to compile against the current `PhaseTypeDistributions.jl` API — so
that the prior work informs the rewrite happening in `src/`.

## Files

| File | Purpose (legacy) |
| --- | --- |
| `src/MAPHDist.jl` | Old MAPH struct (superseded by the version in PhaseTypeDistributions.jl). |
| `src/MAPH_initial.jl` | Initialization (moment-matched starts). |
| `src/MAPHStatistics.jl` | Sufficient statistics + CTMC scaffolding. |
| `src/MAPHfit.jl` | EM (E-step / M-step), `JuMP`/`HiGHS` U-projection. |
| `src/moment_function.jl` | Moment-related helpers. |
| `test/*.jl` | Original tests for the above. |

## Why preserved

The new module will rebuild fitting on top of the redesigned types
(`AbstractPHDist`, `MAPHDist(α, T, D)`, `MAPHDist(π, μ, σ²)`, etc.) with a
clean separation between moment-matching (`fit_mm`) and EM (`fit_mle`). The
legacy code remains useful as:

- a check on algorithmic choices that were already debugged,
- a reference for the U-projection LP and sufficient-statistic formulas,
- a place to look up earlier initialization strategies.

Once the new implementation reaches parity, this directory should be deleted.

## Provenance

Full development history lives in PhaseTypeDistributions.jl. To trace it,
clone that repo and run:

```sh
git log --follow -- src/MAPHfit.jl
git log --follow -- src/maph/MAPHfit.jl
```

Most of the EM development is by Zhihao Qiao; initialization and integration
work is shared with Yoni Nazarathy.
