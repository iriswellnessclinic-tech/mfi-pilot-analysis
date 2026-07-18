# Data dictionary

Two matrices are produced by `extraction/mfi_full_analysis.ts`. Participant-level
values are not distributed; this dictionary documents the schema so the analysis
scripts can be read and audited.

## `mfi_matrix.csv` — one row per completed respondent (n = 116)

| Column | Meaning |
|---|---|
| `sid` | pseudonymized respondent id (opaque; not linkable outside the study db) |
| `age`, `sex` | age (years), sex |
| `mig_days` | self-reported migraine days, preceding 4 weeks |
| `ha_days` | self-reported headache days, preceding 4 weeks |
| `aura`, `prev_use`, `cgrp_use`, `employ` | aura; any preventive use; CGRP-pathway use; employment |
| `acute_days` | self-reported acute-medication days |
| `case_def` | meets broad composite migraine definition (1/0) |
| `id_mig` | diary-derived migraine-symptom proxy positive (1/0/na) |
| `drug_specific` | uses migraine-specific prescription medication (1/0) |
| `mfi_total` | total scored MFI (0–100; auxiliary summary only) |
| `f1`, `f2` | factor scores: Life Restoration (F1), Migraine Agency (F2), 0–100 |
| `midas`, `mibs4` | MIDAS total; MIBS-4 total |
| `diary_ha_rate`, `diary_mibs`, `diary_fulfill`, `diary_logged` | all-time diary summaries |
| `w28_days` | recorded diary days within the fixed 28-day pre-completion window |
| `w28_ha`, `w28_haprop` | headache days / headache-day proportion in the window |
| `w28_free` | headache-free days in the window |
| `w28_fulfill`, `w28_fulfilln` | mean headache-free-day wellbeing; n of such days |
| `w28_mibs`, `w28_mibsn` | mean daily interictal-burden score; n of days |
| `dom1`…`dom11` | the 11 prespecified domain scores (0–100), inputs to the domain-level EFA |
| `q0_18`…`q<d>_<k>` | individual scored item responses (0–6), grouped by domain index |

Domain index order (dom1…dom11): premonitory symptom awareness; trigger
recognition; pre-emptive action; attack control; prevention of medication overuse;
freedom of activity; hope and treatment continuity; life restoration; social
freedom; impact on/understanding by close others; global freedom evaluation.
(The exact item→domain map and English translations are in Supplementary Table S1.)

## `mfi_funnel.csv` — one row per invited user (feasibility funnel, n = 1,190)

| Column | Meaning |
|---|---|
| `sid` | pseudonymized id |
| `status` | `pending` (never started) / `in_progress` (started, not completed) / `completed` |
| `consented` | consent recorded (1/0) |
| `n_answered` | number of items answered |
| `last_order_index` | questionnaire position of last answered item (discontinuers) |
| `last_section` | section of last answered item |
| `age`, `sex` | for completer vs non-completer comparison |
