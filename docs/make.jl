using Documenter
using PhaseTypeDistributions
using PhaseTypeDistributionsFitting

DocMeta.setdocmeta!(
    PhaseTypeDistributionsFitting,
    :DocTestSetup,
    :(using PhaseTypeDistributions, PhaseTypeDistributionsFitting, Distributions, Random, Statistics);
    recursive = true,
)

makedocs(;
    modules = [PhaseTypeDistributionsFitting],
    authors = "Zhihao Qiao, Yoni Nazarathy, and contributors",
    sitename = "PhaseTypeDistributionsFitting.jl",
    format = Documenter.HTML(;
        canonical = "https://julia-matrix-analytic-probability.github.io/PhaseTypeDistributionsFitting.jl",
        edit_link = "main",
        assets = String[],
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Fitting PH distributions" => "fitting.md",
        "The EM algorithm" => "em.md",
        "API reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(;
    repo = "github.com/Julia-Matrix-Analytic-Probability/PhaseTypeDistributionsFitting.jl",
    devbranch = "main",
    push_preview = true,
)
