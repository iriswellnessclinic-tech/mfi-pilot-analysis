# Migraine Freedom Index (MFI) — pilot analysis code

Analysis code and documentation for the study:

> **Beyond headache days: Initial development and exploratory psychometric evaluation of the Migraine Freedom Index using linked daily digital-diary data**
> (cross-sectional digital-health pilot, frozen dataset n = 116, data cut 18 July 2026)

This repository is provided for **transparency and independent re-analysis**. The
application operator is also the first author, data custodian, and analyst; to
mitigate this conflict of interest, the analysis plan was prespecified and dated,
and the analysis code and the frozen-dataset definition are shared here.

## What is (and is not) here

- **Here:** the analysis scripts, the dated analysis plan, the data dictionary,
  exact package versions, and the definition of the frozen dataset.
- **Not here:** participant-level data. De-identified participant-level data are
  not publicly posted because consent and platform-governance conditions restrict
  unrestricted sharing. A de-identified dataset sufficient to reproduce the
  reported analyses may be made available to qualified researchers on reasonable
  request, subject to ethics approval, a data-use agreement, and applicable law.

## Reproduce

```bash
# 1) Build the analysis matrices from the pseudonymized study database.
#    Requires environment variables for database access (not included).
#    The data cut is frozen by the CUTOFF constant inside the script.
npx tsx extraction/mfi_full_analysis.ts
#    -> writes mfi_matrix.csv (one row per completed respondent)
#       and    mfi_funnel.csv (one row per invited user; feasibility funnel)

# 2) Run the analyses (from the directory holding the two CSVs).
#    statsmodels is intentionally not used; OLS etc. are implemented directly.
python3 analysis/mfi_rev_efa.py     # parallel analysis, omega, bootstrap loadings, domain-corr
python3 analysis/mfi_rev_reg.py     # continuous regression, incremental value, density sensitivity
python3 analysis/mfi_rev_desc.py    # descriptives, validity table, funnel, 2x2 discordance
python3 analysis/mfi_rev_tables.py  # Tables 1-4 (HTML)
FIG_DPI=300 python3 analysis/mfi_rev_figs.py  # Figures 1-3 (300 dpi)
```

See [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), [`data_dictionary.md`](data_dictionary.md),
and [`requirements.txt`](requirements.txt).

## Ethics

Approved by the Institutional Ethics Committee of the Saitama Neuropsychiatric
Institute (approval number **SNI26-006**, approved **9 July 2026**). All participants
gave electronic informed consent. Recruitment window **10–18 July 2026**.

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). This license covers
the code only, not the data or the questionnaire content.
