# Examples

| Script | What it shows |
|---|---|
| `icu_censored.jl` | Fitting a `MAPHDist` to right-censored competing-risks data, on real intensive-care length-of-stay records, and what the censored records buy over discarding them |

Run from the package root:

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/icu_censored.jl
```

## Data

`sir_adm.csv` is the `sir.adm` dataset: 747 intensive-care patients from the
SIR-3 cohort, distributed with the R package
[`mvna`](https://cran.r-project.org/package=mvna) and analysed throughout
Beyersmann, Allignol and Schumacher (2012), *Competing Risks and Multistate
Models with R*, Ch. 1. Columns are `id`, `pneu` (pneumonia at admission),
`status` (0 censored, 1 discharged alive, 2 hospital death), `time` (days),
`age`, `sex`. The example uses the 650 pneumonia-free patients, of whom six are
right censored.

`mvna` is distributed under the GPL, so the dataset carries that licence rather
than this package's MIT terms.
