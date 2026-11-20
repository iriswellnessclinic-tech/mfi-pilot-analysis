# Migraine Freedom Index (MFI) — pilot analysis code

Analysis code and documentation for the study:

> **Item generation and preliminary psychometric evaluation of the Migraine Freedom Index: an exploratory study linked to a headache-diary application**
> (first submitted as "Development of the Migraine Freedom Index, a patient-reported measure of migraine freedom in daily life: an exploratory linked-diary study"; cross-sectional digital-health pilot, frozen dataset n = 116, data cut 18 July 2026, 115 after one account deletion)

This repository is provided for **transparency and to facilitate future independent
verification**. The application operator is also the first author, data custodian,
and analyst; to mitigate this conflict of interest, the analysis plan was dated,
and the analysis code and the frozen-dataset definition are shared here. This study
is exploratory and hypothesis-generating: it does not validate a finished
patient-reported outcome measure, and the two dimensions are reported as a
provisional profile rather than established scales.

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
python3 analysis/mfi_rev_reg.py     # continuous regression, exploratory wellbeing model, density sensitivity
python3 analysis/mfi_rev_desc.py    # descriptives, validity table, funnel, 2x2 discordance
python3 analysis/rev9_analysis.py   # funnel reconciliation, missing-item rule, recording-density controls,
                                    #   partial/adjusted models for the second profile, Figure 3 diagnostics
python3 analysis/r2_reviewer_reanalysis.py  # item-level analysis, alternative scores, recording-density models,
                                    #   redundancy and domain unidimensionality (added after peer review, October 2026)
python3 analysis/mfi_rev_tables.py  # Tables 1-4 (HTML)
python3 analysis/build_supp_v2.py && python3 analysis/build_supp_v3.py  # Supplementary Tables S1-S4 (HTML)
python3 analysis/build_supp_v4_jhp.py  # Additional file 2 as submitted to The Journal of Headache and Pain
                                    #   (mfi_supp_jhp.html; labels and wording aligned with the manuscript)
FIG_DPI=300 FIG_TITLES=0 python3 analysis/mfi_rev_figs.py  # Figures 1-3 + Supplementary Figure S1 (300 dpi;
                                    #   titles are in the manuscript, as in figures/)

# 3) Revision 1 for The Journal of Headache and Pain (November 2026). Recomputes every reported
#    statistic and writes jhp_r1_numbers.json, mfi_supp_jhp_r1.html (Additional file 2) and the
#    revised Figures 1-3 (figures/fig1_flow_r1.png, fig2_loadings.png, fig3_scatter.png).
FIG_DPI=300 python3 analysis/jhp_r1_build.py
```

**Changes in revision 1.** The scripts of the first submission used factor_analyzer's
`method='principal'`, which extracts principal components; the manuscript had called this
principal-axis factoring. `jhp_r1_build.py` uses minimum-residual extraction (`method='minres'`)
and keeps the component solution for comparison. The sex covariate is now female versus not
female (the earlier scripts entered the questionnaire code). The split at the sample medians
(former Table 4, Table S3l and Figure S1) was removed, as Reviewer 1 suggested. The dated analysis plan
(`ANALYSIS_PLAN.md`, 19 July 2026) is unchanged; the changes from it are listed in the manuscript.

**Changes in revision 2.** No analysis or number changed. The note of Table S1 in Additional
file 2 now names the licence of the items (see License below).

`analysis/r2_reviewer_reanalysis.py` also checks the item wording against the in-app
survey definition (`MFI_SURVEY_JSON`), which is not included here; without it, that
self-check is skipped and all other results are unchanged.

See [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), [`data_dictionary.md`](data_dictionary.md),
and [`requirements.txt`](requirements.txt).

## Ethics

Approved by the Institutional Ethics Committee of the Saitama Neuropsychiatric
Institute (approval number **SNI26-006**, approved **9 July 2026**). All participants
gave electronic informed consent. Recruitment opened on **10 July 2026** and closed on
**18 July 2026** (00:00 Japan Standard Time).

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). This license covers
the code only, not the data or the questionnaire content.

The 82 MFI items and the scoring rule (Additional files 1 and 2 of the article) are licensed
under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International licence
([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)). In addition, they may be
used free of charge in clinical practice. Commercial use, translation, or modification requires
permission from the corresponding authors.
