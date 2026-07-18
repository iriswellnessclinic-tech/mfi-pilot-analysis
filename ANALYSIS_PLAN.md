# Analysis plan (frozen)

**Status:** specified before the frozen n = 116 re-analysis, after preliminary
examination of an earlier dataset (n ≈ 102). This document records the plan that
governed the reported n = 116 analyses; it is not a prospective registration
(the study was not prospectively registered).

**Frozen dataset:** all completed MFI responses available at the data cut of
**18 July 2026 (00:00 JST)**, after exclusion of internal test accounts, yielding
116 valid responses. The cut is enforced reproducibly by the `CUTOFF` constant in
`extraction/mfi_full_analysis.ts`. Responses accruing after the cut are not
included.

## Primary and secondary analyses

1. **Feasibility.** Completion funnel (available → consented/started → completed →
   discontinued), position at which discontinuers stopped, completion time,
   item-level missingness; completers vs non-completers on age and sex.

2. **Internal consistency.** Cronbach alpha and McDonald omega for each higher-order
   factor and for the total scored set (interpreted as item-score consistency).

3. **Higher-order structure (primary structural analysis).** Exploratory factor
   analysis of the **11 prespecified domain scores** (not the 82 items, which the
   sample cannot support), principal-axis extraction with oblimin rotation. Number
   of factors by **Horn parallel analysis** (observed eigenvalues vs 95th percentile
   of 1,000 random datasets), supported by scree and 1/2/3-factor comparison. KMO,
   Bartlett, communalities, full pattern matrix, and **1,000-resample bootstrap**
   loading stability (percentile 95% CIs, sign/Procrustes alignment).

4. **Convergent and diary-linked validity.** Spearman correlations (skewed
   outcomes), with analytic n, Fisher-z 95% CIs, two-sided p, organized around
   prespecified directional hypotheses. No multiplicity adjustment (exploratory).

5. **Frequency vs freedom (primary substantive analysis).** Each factor modelled as
   a continuous function of the diary-derived headache-day proportion within a fixed
   **28-day** window preceding completion (Spearman + OLS with prediction interval
   and residual diagnostics). Primary linked sample: **≥ 7 recorded diary days**
   (n = 81). Density sensitivity at ≥ 1, ≥ 7, ≥ 14, ≥ 21 days and ≥ 80% recording.

6. **Median split (secondary, descriptive).** Headache-day proportion and Life
   Restoration dichotomized at sample medians to illustrate discordance. Groups are
   sample-relative and are **not** diagnostic thresholds.

7. **Exploratory incremental explanatory value.** DV = headache-free-day wellbeing;
   base model (headache-day proportion + MIDAS) vs +Life Restoration +Migraine
   Agency. Report ΔR², F-change and its p, adjusted R², standardized coefficients,
   VIF, Cook's distance, residual normality, and a **bootstrap 95% CI for ΔR²**
   (2,000 resamples, percentile method). Labelled exploratory, not established
   incremental validity.

8. **Candidate short form.** Within-domain corrected item-total selection with
   1,000-bootstrap selection stability; presented as proof that shortening may be
   possible, not as a ready instrument.

## Migraine case definitions (for sensitivity, not eligibility)

Study entry required only adult age (≥ 18); migraine status was **not** an entry
criterion. Migraine was characterized post hoc. The **broad composite** definition
required at least one of: use of a migraine-specific prescription medication;
self-identification as having migraine; or a **diary-derived migraine-symptom
proxy** (on headache days, documentation of ≥ 2 of nausea, photophobia/phonophobia,
and headache-related disability — conceptually related to, but not, the ID Migraine
instrument). Higher-specificity subgroups: migraine-specific medication use, and
diary-derived proxy positive. Main analysis uses all 116.

## Reproducibility

Random seed 20260718 where resampling is used. Exact package versions in
`requirements.txt`. `statsmodels` is deliberately not a dependency; OLS, VIF,
Cook's distance, and bootstrap procedures are implemented directly in NumPy/SciPy.
