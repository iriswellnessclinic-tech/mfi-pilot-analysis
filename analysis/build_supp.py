# -*- coding: utf-8 -*-
import json, html
items = json.load(open('/tmp/s1_full.json'))
# order: F2 domains then F1 in the dump; present grouped by factor for readability
order_dom = ["Brain-state transition","Trigger recognition","Pre-emptive action","MOH prevention",
             "Attack control","Activity freedom","Hope and treatment continuation","Life restoration",
             "Social freedom","Impact on/understanding by close others","Global freedom evaluation"]
def domkey(d):
    return order_dom.index(d) if d in order_dom else 99
items.sort(key=lambda r:(0 if r['factor']=='F1' else 1, domkey(r['domain_en']), r['order_index']))

def esc(s): return html.escape(str(s)) if s is not None else ""

DISPLAY = {
 "Brain-state transition":"Premonitory-symptom awareness",
 "MOH prevention":"Prevention of medication overuse",
 "Activity freedom":"Freedom of activity",
 "Hope and treatment continuation":"Hope and treatment continuity",
 "Life restoration":"Reclaiming your life",
 "Impact on/understanding by close others":"Impact on close others",
}

css = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,'Hiragino Kaku Gothic ProN',sans-serif;color:#111;line-height:1.5}
h1{font-size:20px;border-bottom:2px solid #444;padding-bottom:4px;margin-top:28px}
h2{font-size:16px;color:#0a4;margin-top:22px}
p.note{font-size:12px;color:#555}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 18px}
th,td{border:1px solid #bbb;padding:4px 6px;vertical-align:top;text-align:left}
th{background:#f0f4f8}
td.num,th.num{text-align:right;white-space:nowrap}
caption{text-align:left;font-weight:bold;margin:6px 0}
</style>"""

h=[css]
h.append("<h1>Supplementary material — Migraine Freedom Index (MFI) pilot (frozen n=116, data cut 18 July 2026)</h1>")
h.append("<p class='note'>English item translations are provisional, prepared by the study team with a structured back-translation check; formal independent forward-and-back translation and cognitive debriefing are planned and were not completed here. All scored items use a common 7-point agreement scale (0 = strongly disagree … 6 = strongly agree) and are positively keyed at the item level, with no reverse-scored items. <b>Scoring direction differs by factor.</b> For the candidate <b>Life Recovery</b> factor, higher scores indicate greater endorsement of the freedom a patient has regained. For the <b>Awareness and Management</b> factor, higher scores indicate greater endorsement of migraine-related awareness and management behaviours and should <b>not</b> currently be interpreted as unequivocally better outcomes. Domain and factor scores are the mean of available item responses rescaled to 0–100; a domain score required ≥75% of its items answered and a factor score ≥80%, and all 116 completers met these thresholds (minimum per-domain completion 87.5%, minimum overall 98.8%), so none were excluded. Item-level missingness was low.</p>")

# ---------------- S1 ----------------
h.append("<h1>Supplementary Table S1. Scored MFI item bank (82 items): domain, English translation, Japanese original, and floor/ceiling</h1>")
h.append("<p class='note'>Floor = percentage answering 0 (strongly disagree); Ceiling = percentage answering 6 (strongly agree), among respondents to each item (n≈102–116). Across items, floor was a median 6.0% (max 36.2%) and ceiling a median 10.8% (max 31.9%); 8 items exceeded 15% floor and 28 exceeded 15% ceiling.</p>")
h.append("<table><tr><th>#</th><th>Factor</th><th>Domain</th><th>Item (English, provisional)</th><th>Japanese original</th><th>Scoring</th><th class='num'>Floor %</th><th class='num'>Ceiling %</th></tr>")
for i,r in enumerate(items,1):
    fac = "F1 Life Recovery" if r['factor']=='F1' else "F2 Awareness and Management"
    score = "↑ = more freedom" if r['factor']=='F1' else "↑ = more awareness/management (not necessarily better)"
    h.append(f"<tr><td class='num'>{i}</td><td>{esc(fac)}</td><td>{esc(DISPLAY.get(r['domain_en'], r['domain_en']))}</td><td>{esc(r['en'])}</td><td>{esc(r['ja'])}</td><td>{esc(score)}</td><td class='num'>{r['floor_pct']}</td><td class='num'>{r['ceiling_pct']}</td></tr>")
h.append("</table>")
h.append("<p class='note'>Migraine-specific medications counted for subgroup (iii): triptans, lasmiditan, and rimegepant (acute); CGRP-pathway agents and atogepant (preventive). Other preventives available in the app (for example, lomerizine, valproate, amitriptyline, propranolol) are also used for non-migraine indications and were not counted as migraine-specific.</p>")

# ---------------- S2 ----------------
h.append("<h1>Supplementary Table S2. Domain-level exploratory factor analysis and factor-score distributions</h1>")
h.append("<h2>S2a. Two-factor pattern matrix (principal axis, oblimin), communalities, and 1,000-resample bootstrap loading stability</h2>")
h.append("<p class='note'>KMO=0.78; Bartlett χ²=621.1, p&lt;0.001. Parallel analysis retained two factors (observed eigenvalues 4.34, 1.99 exceeded random 95th percentiles 1.67, 1.48; third eigenvalue 1.05 &lt; 1.33). Two-factor solution explained 57.5% of variance (39.4% + 18.1%). Inter-factor correlation (oblimin) 0.00; factor-score correlation −0.05.</p>")
efa=[
 # domain, F1, F2, h2, bF1, bF2, expected
 ("Premonitory-symptom awareness",-0.24,0.59,0.41,"−0.24 [−0.50, −0.02]","0.59 [0.32, 0.74]","F2"),
 ("Trigger recognition",-0.18,0.69,0.51,"−0.18 [−0.43, 0.02]","0.69 [0.49, 0.80]","F2"),
 ("Pre-emptive action",0.03,0.72,0.53,"0.02 [−0.13, 0.20]","0.73 [0.56, 0.83]","F2"),
 ("Prevention of medication overuse",0.22,0.74,0.60,"0.21 [0.08, 0.38]","0.75 [0.57, 0.85]","F2"),
 ("Attack control",0.61,0.15,0.40,"0.61 [0.43, 0.74]","0.14 [−0.09, 0.38]","F1"),
 ("Freedom of activity",0.83,0.01,0.69,"0.83 [0.76, 0.88]","0.02 [−0.11, 0.17]","F1"),
 ("Hope and treatment continuity",0.85,0.16,0.75,"0.85 [0.79, 0.89]","0.17 [0.06, 0.27]","F1"),
 ("Reclaiming your life",0.92,-0.12,0.86,"0.92 [0.89, 0.95]","−0.12 [−0.19, −0.04]","F1"),
 ("Social freedom",0.58,-0.10,0.34,"0.58 [0.42, 0.71]","−0.09 [−0.37, 0.19]","F1"),
 ("Impact on close others",0.70,0.03,0.49,"0.70 [0.56, 0.81]","0.04 [−0.21, 0.25]","F1"),
 ("Global freedom evaluation",0.86,-0.06,0.75,"0.87 [0.82, 0.90]","−0.05 [−0.18, 0.06]","F1"),
]
h.append("<table><tr><th>Domain</th><th class='num'>F1 (Life Recovery)</th><th class='num'>F2 (Awareness &amp; Management)</th><th class='num'>Communality h²</th><th>F1 bootstrap median [95% CI]</th><th>F2 bootstrap median [95% CI]</th><th>Expected</th></tr>")
for d,f1,f2,h2,b1,b2,ex in efa:
    h.append(f"<tr><td>{esc(d)}</td><td class='num'>{f1:+.2f}</td><td class='num'>{f2:+.2f}</td><td class='num'>{h2:.2f}</td><td>{esc(b1)}</td><td>{esc(b2)}</td><td>{ex}</td></tr>")
h.append("</table>")
h.append("<p class='note'>Internal consistency: F1 Life Recovery (50 items) Cronbach α=0.95, McDonald ω=0.96; F2 Awareness and Management (32 items) α=0.85, ω=0.88; total scored set α=0.91 (auxiliary only, because the two factors were uncorrelated). The very high F1 α (0.95) may indicate item redundancy rather than reliability alone. Item-level EFA of all 82 items was underpowered (KMO=0.64; participant-to-item ratio 1.4) and is not used to define structure.</p>")

h.append("<h2>S2b. Inter-domain correlation matrix (Pearson)</h2>")
labels=["D1 Premonitory-symptom awareness","D2 Trigger recognition","D3 Pre-emptive action","D4 Attack control","D5 Prevention of medication overuse","D6 Freedom of activity","D7 Hope and treatment continuity","D8 Reclaiming your life","D9 Social freedom","D10 Impact on close others","D11 Global freedom evaluation"]
M=[
[1.00,0.46,0.11,-0.10,0.24,-0.13,-0.07,-0.27,-0.07,-0.10,-0.24],
[0.46,1.00,0.29,0.03,0.21,-0.10,-0.01,-0.19,-0.13,-0.06,-0.24],
[0.11,0.29,1.00,0.05,0.55,0.04,0.10,-0.08,-0.05,0.00,-0.02],
[-0.10,0.03,0.05,1.00,0.16,0.46,0.50,0.45,0.16,0.35,0.48],
[0.24,0.21,0.55,0.16,1.00,0.10,0.24,0.09,0.04,0.17,0.21],
[-0.13,-0.10,0.04,0.46,0.10,1.00,0.73,0.79,0.36,0.45,0.64],
[-0.07,-0.01,0.10,0.50,0.24,0.73,1.00,0.78,0.36,0.46,0.70],
[-0.27,-0.19,-0.08,0.45,0.09,0.79,0.78,1.00,0.52,0.54,0.81],
[-0.07,-0.13,-0.05,0.16,0.04,0.36,0.36,0.52,1.00,0.57,0.36],
[-0.10,-0.06,0.00,0.35,0.17,0.45,0.46,0.54,0.57,1.00,0.51],
[-0.24,-0.24,-0.02,0.48,0.21,0.64,0.70,0.81,0.36,0.51,1.00],
]
h.append("<table><tr><th>&nbsp;</th>"+"".join(f"<th class='num'>D{j+1}</th>" for j in range(11))+"</tr>")
for i,row in enumerate(M):
    h.append(f"<tr><td>{esc(labels[i])}</td>"+"".join(f"<td class='num'>{v:+.2f}</td>" for v in row)+"</tr>")
h.append("</table>")
h.append("<h2>S2c. Factor-score distributions (0–100)</h2>")
h.append("<table><tr><th>Factor</th><th class='num'>Mean (SD)</th><th class='num'>Median (IQR)</th><th class='num'>Min–Max</th><th class='num'>Skewness</th><th class='num'>Floor / Ceiling</th></tr>"
 "<tr><td>F1 Life Recovery</td><td class='num'>59.5 (14.4)</td><td class='num'>61.5 (51.6–68.2)</td><td class='num'>9.0–86.3</td><td class='num'>−0.62</td><td class='num'>none</td></tr>"
 "<tr><td>F2 Awareness and Management</td><td class='num'>56.3 (11.4)</td><td class='num'>57.8 (49.2–64.1)</td><td class='num'>22.4–79.7</td><td class='num'>−0.52</td><td class='num'>none</td></tr>"
 "<tr><td>Total (auxiliary)</td><td class='num'>58.3 (9.7)</td><td class='num'>59.0 (52.0–64.7)</td><td class='num'>17.3–77.6</td><td class='num'>−0.70</td><td class='num'>none</td></tr>"
 "</table>")
h.append("<p class='note'>Factor and total scores were mildly negatively skewed with no floor or ceiling clustering (no participant at the scale minimum or maximum). A candidate short form is not reported in this version; item reduction is deferred to a larger, independent sample after content-validity revision.</p>")

# ---------------- S3 ----------------
h.append("<h1>Supplementary Table S3. Feasibility funnel, correlation hypotheses, recording-density sensitivity, and sample-median-split discordance</h1>")
h.append("<h2>S3a. Completion funnel and drop-off</h2>")
h.append("<table><tr><th>Stage</th><th class='num'>n</th><th>Note</th></tr>"
 "<tr><td>Survey made available in-app</td><td class='num'>1,190</td><td>a send record was created; access logs do not confirm how many users viewed the invitation, so no uptake/response rate is computed against this denominator</td></tr>"
 "<tr><td>Entered consent flow and began</td><td class='num'>181</td><td>—</td></tr>"
 "<tr><td>Completed (analysed)</td><td class='num'>116</td><td>64% of those who began</td></tr>"
 "<tr><td>Discontinued</td><td class='num'>65</td><td>29 answered no items (consent only); 36 answered ≥1 item then stopped</td></tr>"
 "</table>")
h.append("<p class='note'>Among the 36 who answered at least one item before stopping, the last-answered questionnaire position had a median of 57 (IQR 33–99; range 2–116), distributed across sections rather than concentrated at any single item. Completers vs non-completers did not differ appreciably in age (43.9 vs 41.9 years; Mann–Whitney p=0.24) or sex (90% vs 97% women). Median completion time was 15.6 minutes (IQR 12.0–24.6) among 107 same-session completers; item-level missingness was low.</p>")

h.append("<h2>S3b. Correlation hypotheses and observed associations (Spearman ρ [95% CI], p)</h2>")
h.append("<p class='note'>Inverse associations of <b>Life Recovery</b> with disability and MMD were prespecified in the dated plan. Directional hypotheses for the <b>second factor</b> were not firmly prespecified; its associations are reported as exploratory. Because the two factors were uncorrelated (factor-score r=−0.05), the total is an auxiliary summary only and is shown last.</p>")
rows=[
 ("MIDAS (disability)","−0.41 [−0.55, −0.25], &lt;0.001","−0.53 [−0.65, −0.39], &lt;0.001","+0.18 [−0.01, +0.35], 0.059"),
 ("MIBS-4, 4 weeks (interictal burden)","−0.38 [−0.52, −0.21], &lt;0.001","−0.56 [−0.68, −0.42], &lt;0.001","+0.31 [+0.13, +0.46], 0.001"),
 ("Self-reported MHD","−0.14 [−0.31, +0.05], 0.139","−0.24 [−0.41, −0.06], 0.008","+0.23 [+0.05, +0.40], 0.013"),
 ("Diary headache-day proportion, 28-day ≥7","−0.31 [−0.50, −0.10], 0.004","−0.27 [−0.47, −0.06], 0.013","−0.09 [−0.31, +0.13], 0.402"),
 ("Diary fulfillment on headache-free days, 28-day","+0.23 [+0.01, +0.44], 0.043","+0.21 [−0.02, +0.42], 0.070","+0.10 [−0.13, +0.32], 0.384"),
]
h.append("<table><tr><th>External measure</th><th>F1 Life Recovery (prespecified)</th><th>Second factor (exploratory)</th><th>Total (auxiliary)</th></tr>")
for a,b,c,d in rows:
    h.append(f"<tr><td>{a}</td><td class='num'>{c}</td><td class='num'>{d}</td><td class='num'>{b}</td></tr>")
h.append("</table>")

h.append("<h2>S3c. Recording-density sensitivity: factor vs diary headache-day proportion (Spearman ρ [95% CI])</h2>")
h.append("<p class='note'>Primary linked analysis used ≥7 recorded days in the 28-day window (n=81). Associations were directionally consistent at ≥1, ≥7, and ≥14 days; at ≥21 days and ≥80% recording the sample contracted and confidence intervals widened to include zero. A ≥7-day minimum was chosen a priori to balance measurement precision against sample size; a ≥14-day minimum yielded materially similar estimates.</p>")
dens=[("≥1 day",108,"−0.22 [−0.40, −0.04]","−0.23 [−0.40, −0.04]"),
      ("≥7 days (primary)",81,"−0.27 [−0.47, −0.06]","−0.31 [−0.50, −0.10]"),
      ("≥14 days",65,"−0.35 [−0.55, −0.11]","−0.31 [−0.52, −0.07]"),
      ("≥21 days",52,"−0.20 [−0.45, +0.08]","−0.15 [−0.41, +0.12]"),
      ("≥80% (≥22.4 days)",45,"−0.28 [−0.53, +0.02]","−0.15 [−0.43, +0.15]")]
h.append("<table><tr><th>Minimum recorded days</th><th class='num'>n</th><th>F1 Life Recovery ↔ headache-day proportion</th><th>Total ↔ headache-day proportion</th></tr>")
for lab,n,f1,tot in dens:
    h.append(f"<tr><td>{lab}</td><td class='num'>{n}</td><td class='num'>{f1}</td><td class='num'>{tot}</td></tr>")
h.append("</table>")
h.append("<p class='note'>Exploratory additional explanatory value (DV = fulfillment on headache-free days, n=75): base model (headache-day proportion + MIDAS) R²=0.01; adding the two factors R²=0.09; the nested-model comparison did not reach significance (F-change(2,70)=2.99, p=0.057). Among the added terms, Life Recovery was the only coefficient reaching nominal significance (standardized β=+0.31, p=0.048); VIF 1.0–1.8; residuals approximately normal (Shapiro p=0.86); maximum Cook's distance 0.16. A percentile bootstrap CI for ΔR² is not reported, because in nested ordinary least squares the in-sample R² cannot decrease and such an interval is biased away from the null.</p>")

h.append("<h2>S3d. Sample-median-split discordance (secondary, descriptive; linked n=81)</h2>")
h.append("<p class='note'>Split at the sample medians of diary headache-day proportion (0.52) and Life Recovery (62.0). Groups are sample-relative and are not clinical thresholds.</p>")
h.append("<table><tr><th>Group</th><th class='num'>n</th><th class='num'>MIDAS median (IQR)</th></tr>"
 "<tr><td>Lower proportion / higher restoration (concordant)</td><td class='num'>25</td><td class='num'>5 (3–18)</td></tr>"
 "<tr><td>Higher proportion / lower restoration (concordant)</td><td class='num'>25</td><td class='num'>53 (16–112)</td></tr>"
 "<tr><td>Lower proportion / lower restoration (discordant)</td><td class='num'>16</td><td class='num'>24 (14–36)</td></tr>"
 "<tr><td>Higher proportion / higher restoration (discordant)</td><td class='num'>15</td><td class='num'>14 (4–21)</td></tr>"
 "</table>")
h.append("<p class='note'>Discordant total = 31/81 (38%). MIDAS was descriptively higher in the lower-proportion/lower-restoration group than in the higher-proportion/higher-restoration group; no formal between-group test was prespecified. This 2×2 is displayed in Supplementary Figure S1.</p>")

# ---------------- S3e (recording-density controls & adjusted models) ----------------
h.append("<h2>S3e. Recording behaviour as an alternative explanation for the second factor and for the headache-day–Life Recovery association</h2>")
h.append("<p class='note'>To test whether either factor is an artifact of how actively participants recorded, we related the number of recorded diary days (28-day window) to each factor, computed partial Spearman correlations of each factor with headache-day proportion controlling for recorded days, fitted multivariable models adjusting each factor for headache-day proportion, recorded days, self-reported MMD, and preventive-medication use, and ran a recording-day-weighted sensitivity fit. Coefficients in the multivariable models are standardized (β).</p>")
h.append("<table><tr><th>Analysis</th><th>Life Recovery (F1)</th><th>Second factor (F2)</th></tr>"
 "<tr><td>Recorded diary days ↔ factor (Spearman ρ, n=108)</td><td class='num'>+0.04 (p=0.673)</td><td class='num'>+0.10 (p=0.281)</td></tr>"
 "<tr><td>Recorded diary days ↔ headache-day proportion (Spearman ρ, n=108)</td><td class='num' colspan='2' style='text-align:center'>−0.50 (p&lt;0.001)</td></tr>"
 "<tr><td>Profile ↔ headache-day proportion, zero-order (ρ, n=81)</td><td class='num'>−0.27 (p=0.013)</td><td class='num'>−0.09 (p=0.402)</td></tr>"
 "<tr><td>Profile ↔ headache-day proportion, partial | recorded days (ρ, n=81)</td><td class='num'>−0.30 (p=0.007)</td><td class='num'>+0.01 (p=0.929)</td></tr>"
 "<tr><td>Adjusted model: headache-day proportion (β, n=80)</td><td class='num'>−0.39 (p=0.003)</td><td class='num'>−0.17 (p=0.179)</td></tr>"
 "<tr><td>Adjusted model: recorded days (β)</td><td class='num'>−0.19 (p=0.150)</td><td class='num'>+0.02 (p=0.903)</td></tr>"
 "<tr><td>Adjusted model: self-reported MMD (β)</td><td class='num'>+0.05 (p=0.622)</td><td class='num'>+0.09 (p=0.377)</td></tr>"
 "<tr><td>Adjusted model: preventive-medication use (β)</td><td class='num'>−0.01 (p=0.940)</td><td class='num'>+0.38 (p=0.001)</td></tr>"
 "<tr><td>Recording-day-weighted OLS: headache-day proportion (standardized slope, n=81)</td><td class='num'>−0.35 (p=0.003)</td><td class='num'>—</td></tr>"
 "</table>")
h.append("<p class='note'>Number of recorded days was unrelated to either factor, so neither factor is explained by recording zeal. The Life Recovery–headache-day association persisted after partialling out recorded days and after adjustment for recorded days, self-reported MMD, and preventive use, and under recording-day weighting. The second factor was independently associated only with preventive-medication use, consistent with its reflecting engagement with migraine management rather than the headache-day proportion.</p>")

open('/tmp/mfi_supp.html','w').write("\n".join(h))
print("written /tmp/mfi_supp.html  bytes:", len("\n".join(h)))
print("S1 rows:", len(items))
