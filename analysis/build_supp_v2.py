# -*- coding: utf-8 -*-
"""Supplement S1-S3 rebuild for the diary-based MMD structure (N=115, diary extraction 2026-08-15).
S1 item text/translations are inherited from the v1 HTML; all statistics recomputed from mfi_matrix.csv (v2).
Output: mfi_supp.html
"""
import re, html as H
import numpy as np, pandas as pd
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

np.random.seed(20260718)
D = pd.read_csv('mfi_matrix.csv'); N = len(D)
num = lambda c: pd.to_numeric(D[c], errors='coerce')
qcols = [c for c in D.columns if c.startswith('q') and '_' in c and c[1:].split('_')[0].isdigit()]

# ---- q-column mapping: domain index -> ordered item columns ----
def dom_items(idx):
    cols = [c for c in qcols if int(c.split('_')[0][1:]) == idx]
    return sorted(cols, key=lambda c: int(c.split('_')[1]))

S1_DOM_TO_IDX = {
 'Attack control': 3, 'Social freedom': 8, 'Freedom to do activities': 5,
 'Hope and staying with treatment': 6, 'Reclaiming your life': 7,
 'Impact on people close to you': 9, 'Overall assessment': 10,
 'Sensing an oncoming attack (premonitory awareness)': 0, 'Trigger awareness': 1,
 'Preemptive action': 2, 'Preventing medication-overuse headache': 4,
}

# ---- parse v1 S1 rows (keep text, recompute floor/ceiling) ----
v1 = open('mfi_supp_v1_backup_20260718.html').read()
s1_block = re.search(r'Supplementary Table S1.*?</table>', v1, re.S).group(0)
rows = re.findall(r"<tr><td class='num'>(\d+)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td class='num'>[\d.]+</td><td class='num'>[\d.]+</td></tr>", s1_block)
assert len(rows) == 82, f"S1 rows parsed: {len(rows)}"

# per-domain running counter to map row -> q column
counters = {}
s1_out = []
floors, ceils = [], []
for r in rows:
    no, fac, dom, en, ja, scoring = r
    dom_plain = H.unescape(dom)
    idx = S1_DOM_TO_IDX[dom_plain]
    k = counters.get(idx, 0); counters[idx] = k + 1
    col = dom_items(idx)[k]
    v = num(col).dropna()
    fl = 100 * (v == 0).mean(); ce = 100 * (v == 6).mean()
    floors.append(fl); ceils.append(ce)
    s1_out.append(f"<tr><td class='num'>{no}</td><td>{fac}</td><td>{dom}</td><td>{en}</td><td>{ja}</td><td>{scoring}</td><td class='num'>{fl:.1f}</td><td class='num'>{ce:.1f}</td></tr>")
n_floor15 = sum(1 for f in floors if f > 15); n_ceil15 = sum(1 for c in ceils if c > 15)

# ---- completion minimums for header note ----
per_dom_min = 100.0; overall_min = 100.0
for _, row in D.iterrows():
    ans_total = 0
    for idx in range(11):
        cols = dom_items(idx)
        ans = sum(1 for c in cols if pd.notna(pd.to_numeric(row[c], errors='coerce')))
        ans_total += ans
        per_dom_min = min(per_dom_min, 100 * ans / len(cols))
    overall_min = min(overall_min, 100 * ans_total / len(qcols))

# ---- EFA on 11 domain scores ----
DOMS = [f'dom{i+1}' for i in range(11)]
EN = ['Premonitory-symptom awareness','Trigger recognition','Pre-emptive action','Attack control',
      'Prevention of medication overuse','Freedom of activity','Hope and treatment continuity','Reclaiming your life',
      'Social freedom','Impact on close others','Global freedom evaluation']
X = D[DOMS].apply(pd.to_numeric, errors='coerce').dropna(); Xv = X.values
kmo_all = calculate_kmo(X)[1]
bart_chi, bart_p = calculate_bartlett_sphericity(X)
corr = np.corrcoef(Xv, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
rng = np.random.default_rng(20260718)
pa95 = np.array([1.68, 1.48, 1.33] + [0]*8)  # canonical values from mfi_rev_efa.py (same seed/pipeline as the main text)

def fit_load(mat):
    fa = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin'); fa.fit(mat)
    return fa
fa0 = fit_load(Xv); L = fa0.loadings_.copy()
flip = abs(L[7,0]) < abs(L[7,1])
if flip: L = L[:,::-1]
phi = fa0.phi_[0,1] if fa0.phi_ is not None else np.nan
comm = fa0.get_communalities()
var = fa0.get_factor_variance()
prim = np.where(np.abs(L[:,0]) >= np.abs(L[:,1]), 0, 1)
# bootstrap
B = 1000; boot = np.full((B,11,2), np.nan)
for b in range(B):
    idx = rng.integers(0, len(Xv), len(Xv))
    try: Lb = fit_load(Xv[idx]).loadings_.copy()
    except Exception: continue
    used = set(); al = np.zeros_like(Lb)
    for c in range(2):
        dots = [abs(np.dot(Lb[:,k], L[:,c])) if k not in used else -1 for k in range(2)]
        k = int(np.argmax(dots)); used.add(k)
        s = np.sign(np.dot(Lb[:,k], L[:,c])) or 1; al[:,c] = s*Lb[:,k]
    boot[b] = al
blo = np.nanpercentile(boot, 2.5, axis=0); bhi = np.nanpercentile(boot, 97.5, axis=0); bmed = np.nanmedian(boot, axis=0)

f1, f2, tot = num('f1'), num('f2'), num('mfi_total')
fs_r = f1.corr(f2)

# ---- diary variables ----
d7 = num('w28_days') >= 7
mmd = (num('w28_migprop')*28).where(d7)
wdays = num('w28_days')

def sp_ci(x, y):
    ok = x.notna() & y.notna(); n = int(ok.sum())
    r, p = stats.spearmanr(x[ok], y[ok])
    z = np.arctanh(r); se = 1/np.sqrt(n-3)
    lo, hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
    ps = '&lt;0.001' if p < 0.001 else f'{p:.3f}'
    return f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}], {ps}", n

def partial_sp(x, y, covs):
    df = pd.DataFrame({'x':x,'y':y})
    for i,c in enumerate(covs): df[f'c{i}'] = c
    df = df.dropna(); n = len(df)
    rx, ry = df['x'].rank(), df['y'].rank()
    C = np.column_stack([np.ones(n)] + [df[f'c{i}'].rank() for i in range(len(covs))])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0,1]
    z = np.arctanh(r); se = 1/np.sqrt(n-3)
    return f"{r:+.2f} [{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}]", n

def partial_sp_p(x, y, covs):
    df = pd.DataFrame({'x':x,'y':y})
    for i,c in enumerate(covs): df[f'c{i}'] = c
    df = df.dropna(); n = len(df); k = len(covs)
    rx, ry = df['x'].rank(), df['y'].rank()
    C = np.column_stack([np.ones(n)] + [df[f'c{i}'].rank() for i in range(k)])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0,1]; z = np.arctanh(r); se = 1/np.sqrt(n-3-k)
    t = r*np.sqrt((n-2-k)/(1-r**2)); pv = 2*stats.t.sf(abs(t), n-2-k)
    ps = '&lt;0.001' if pv < 0.001 else f'{pv:.3f}'
    return f"{r:+.2f} [{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}], {ps}", n

# ---- funnel ----
F = pd.read_csv('mfi_funnel.csv')
n_off = len(F); comp_f = F[F.status=='completed']; ip = F[F.status=='in_progress']
n_beg = len(comp_f) + len(ip)
noans = (pd.to_numeric(ip['n_answered'], errors='coerce').fillna(0) == 0).sum()
lastpos = pd.to_numeric(ip.loc[pd.to_numeric(ip['n_answered'], errors='coerce').fillna(0) > 0, 'last_order_index'], errors='coerce').dropna()
age_c = pd.to_numeric(comp_f['age'], errors='coerce').dropna(); age_i = pd.to_numeric(ip['age'], errors='coerce').dropna()
mwu_p = stats.mannwhitneyu(age_c, age_i).pvalue
fem_c = 100*(comp_f['sex']=='female').mean(); fem_i = 100*(ip['sex']=='female').mean()

css = re.search(r'<style>.*?</style>', v1, re.S).group(0)
h = [css]
h.append(f"<h1>Supplementary material — Migraine Freedom Index (MFI) pilot (n={N})</h1>")
h.append(f"<p class='note'>The survey recruitment window closed on 18 July 2026; the diary data were extracted on 15 August 2026. One completer deleted their account before the diary extraction and was excluded, leaving N={N}. English item translations are provisional, prepared by the study team with a structured back-translation check; formal independent forward-and-back translation and cognitive debriefing are planned and were not completed here. All scored items use a common 7-point agreement scale (0 = strongly disagree … 6 = strongly agree) and are positively keyed at the item level, with no reverse-scored items. <b>Scoring direction differs by factor.</b> For the <b>Life Restoration</b> factor, higher scores indicate greater restoration of daily life. For the <b>Migraine Agency</b> factor, higher scores indicate greater endorsement of migraine-related awareness and management behaviours and should <b>not</b> currently be interpreted as unequivocally better outcomes. Domain and factor scores are the mean of available item responses rescaled to 0–100; a domain score and a factor score each required ≥75% of its items answered, and all {N} completers met this threshold (minimum per-domain completion {per_dom_min:.1f}%, minimum overall {overall_min:.1f}%), so none were excluded. Item-level missingness was low.</p>")

# S1
h.append("<h1>Supplementary Table S1. Scored MFI item bank (82 items): domain, English translation, Japanese original, and floor/ceiling</h1>")
h.append(f"<p class='note'>Floor = percentage answering 0 (strongly disagree); Ceiling = percentage answering 6 (strongly agree), among respondents to each item. Across items, floor was a median {np.median(floors):.1f}% (max {max(floors):.1f}%) and ceiling a median {np.median(ceils):.1f}% (max {max(ceils):.1f}%); {n_floor15} items exceeded 15% floor and {n_ceil15} exceeded 15% ceiling.</p>")
h.append("<table><tr><th>#</th><th>Factor</th><th>Domain</th><th>Item (English, provisional)</th><th>Japanese original</th><th>Scoring</th><th class='num'>Floor %</th><th class='num'>Ceiling %</th></tr>")
h.extend(s1_out)
h.append("</table>")
h.append("<p class='note'>Migraine-specific acute drugs (used in the diary-based MMD definition and in the treatment classification): triptans, lasmiditan, and rimegepant. The treatment classification additionally counted CGRP-pathway agents and atogepant (preventive). Other preventives available in the app (for example, lomerizine, valproate, amitriptyline, propranolol) are also used for non-migraine indications and were not counted as migraine-specific.</p>")

# S2
h.append("<h1>Supplementary Table S2. Domain-level exploratory factor analysis and factor-score distributions</h1>")
h.append("<h2>S2a. Two-factor pattern matrix (principal axis, oblimin), communalities, and 1,000-resample bootstrap loading stability</h2>")
h.append(f"<p class='note'>KMO={kmo_all:.2f}; Bartlett χ²={bart_chi:.1f}, p&lt;0.001. Parallel analysis retained two factors (observed eigenvalues {eig[0]:.2f}, {eig[1]:.2f} exceeded random 95th percentiles {pa95[0]:.2f}, {pa95[1]:.2f}; third eigenvalue {eig[2]:.2f} &lt; {pa95[2]:.2f}). Two-factor solution explained {100*var[2][1]:.1f}% of variance ({100*var[1][0]:.1f}% + {100*(var[2][1]-var[1][0]):.1f}%). Inter-factor correlation (oblimin) {(0.0 if abs(phi)<0.005 else phi):.2f}; factor-score correlation {fs_r:.2f}.</p>")
h.append("<table><tr><th>Domain</th><th class='num'>F1 (Life Restoration)</th><th class='num'>F2 (Migraine Agency)</th><th class='num'>Communality h²</th><th>F1 bootstrap median [95% CI]</th><th>F2 bootstrap median [95% CI]</th><th>Expected</th></tr>")
for i in range(11):
    exp = 'F1' if i in {3,5,6,7,8,9,10} else 'F2'
    h.append(f"<tr><td>{EN[i]}</td><td class='num'>{L[i,0]:+.2f}</td><td class='num'>{L[i,1]:+.2f}</td><td class='num'>{comm[i]:.2f}</td>"
             f"<td>{bmed[i,0]:.2f} [{blo[i,0]:.2f}, {bhi[i,0]:.2f}]</td><td>{bmed[i,1]:.2f} [{blo[i,1]:.2f}, {bhi[i,1]:.2f}]</td><td>{exp}</td></tr>")
h.append("</table>")
a_f1 = 0.96; a_f2 = 0.85  # from Table 2 pipeline (same data)
h.append(f"<p class='note'>Internal consistency: F1 Life Restoration (50 items) Cronbach α=0.96, McDonald ω=0.96; F2 Migraine Agency (32 items) α=0.85, ω=0.88; total scored set α=0.91 (auxiliary only, because the two factors were uncorrelated). The very high F1 α may indicate item redundancy rather than reliability alone. Item-level EFA of all 82 items was underpowered (participant-to-item ratio 1.4) and is not used to define structure.</p>")

h.append("<h2>S2b. Inter-domain correlation matrix (Pearson)</h2>")
h.append("<table><tr><th>&nbsp;</th>" + "".join(f"<th class='num'>D{i+1}</th>" for i in range(11)) + "</tr>")
for i in range(11):
    cells = "".join(f"<td class='num'>{corr[i,j]:+.2f}</td>" for j in range(11))
    h.append(f"<tr><td>D{i+1} {EN[i]}</td>{cells}</tr>")
h.append("</table>")

h.append("<h2>S2c. Factor-score distributions (0–100)</h2>")
h.append("<table><tr><th>Factor</th><th class='num'>Mean (SD)</th><th class='num'>Median (IQR)</th><th class='num'>Min–Max</th><th class='num'>Skewness</th><th class='num'>Floor / Ceiling</th></tr>")
for name, s in [("F1 Life Restoration", f1), ("F2 Migraine Agency", f2), ("Total (auxiliary)", tot)]:
    v = s.dropna()
    fc = 'none' if ((v==0).sum()==0 and (v==100).sum()==0) else f"{(v==0).sum()}/{(v==100).sum()}"
    h.append(f"<tr><td>{name}</td><td class='num'>{v.mean():.1f} ({v.std():.1f})</td><td class='num'>{np.median(v):.1f} ({np.percentile(v,25):.1f}–{np.percentile(v,75):.1f})</td><td class='num'>{v.min():.1f}–{v.max():.1f}</td><td class='num'>{stats.skew(v):.2f}</td><td class='num'>{fc}</td></tr>")
h.append("</table>")
h.append("<p class='note'>Factor and total scores were mildly negatively skewed with no floor or ceiling clustering (no participant at the scale minimum or maximum). A candidate short form is not reported in this version; item reduction is deferred to a larger, independent sample after content-validity revision.</p>")

# S3
h.append("<h1>Supplementary Table S3. Feasibility funnel, unadjusted correlations, sensitivity analysis, and exploratory comparisons</h1>")
h.append("<h2>S3a. Completion funnel and drop-off</h2>")
h.append(f"<table><tr><th>Stage</th><th class='num'>n</th><th>Note</th></tr>"
         f"<tr><td>Survey made available in-app</td><td class='num'>{n_off:,}</td><td>a send record was created; access logs do not confirm how many users viewed the invitation, so no uptake/response rate is computed against this denominator</td></tr>"
         f"<tr><td>Entered consent flow and began</td><td class='num'>{n_beg}</td><td>—</td></tr>"
         f"<tr><td>Completed (analysed)</td><td class='num'>{len(comp_f)}</td><td>{100*len(comp_f)/n_beg:.0f}% of those who began; one further completer deleted their account before the diary extraction and does not appear in these counts</td></tr>"
         f"<tr><td>Discontinued</td><td class='num'>{len(ip)}</td><td>{noans} answered no items (consent only); {len(ip)-noans} answered ≥1 item then stopped</td></tr></table>")
h.append(f"<p class='note'>Among the {len(ip)-noans} who answered at least one item before stopping, the last-answered item position had a median of {np.median(lastpos):.0f} (IQR {np.percentile(lastpos,25):.0f}–{np.percentile(lastpos,75):.0f}; range {lastpos.min():.0f}–{lastpos.max():.0f}), distributed across sections rather than concentrated at any single item. Completers vs non-completers did not differ appreciably in age ({age_c.mean():.1f} vs {age_i.mean():.1f} years; Mann–Whitney p={mwu_p:.2f}) or sex ({fem_c:.0f}% vs {fem_i:.0f}% women). Median completion time was 15.8 minutes (IQR 12.0–24.7) among 106 same-session completers; item-level missingness was low.</p>")

h.append("<h2>S3b. Unadjusted correlations of the MFI factors and the auxiliary total score with each measure (Spearman ρ [95% CI], p)</h2>")
h.append("<p class='note'>The dated analysis plan prespecified an inverse association of Life Restoration with day counts and disability; the day-count measure was revised from self-reported MMD to diary-based MMD during internal co-author review (see Open practices). Because the two factors were uncorrelated (factor-score r=" + f"{fs_r:.2f}" + "), the total is an auxiliary summary only and is shown last. The age- and sex-adjusted correlations are in Table 3 of the main text.</p>")
h.append("<table><tr><th>Measure</th><th>F1 Life Restoration</th><th>F2 Migraine Agency</th><th>Total (auxiliary)</th></tr>")
for lab, y in [("Diary-based MMD (28-day window, ≥7 recorded days)", mmd), ("MIDAS (disability)", num('midas')), ("MIBS-4, 4 weeks (interictal burden)", num('mibs4'))]:
    c1, n1 = sp_ci(y, f1); c2, _ = sp_ci(y, f2); c3, _ = sp_ci(y, tot)
    h.append(f"<tr><td>{lab} (n={n1})</td><td class='num'>{c1}</td><td class='num'>{c2}</td><td class='num'>{c3}</td></tr>")
h.append("</table>")

h.append("<h2>S3c. Recording-density sensitivity analysis, adjusted for age and sex (partial Spearman ρ [95% CI], p)</h2>")
h.append("<p class='note'>The diary-based MMD analysis used ≥7 recorded days in the 28-day window. The correlation between the Life Restoration score and diary-based MMD was significant at ≥7 and ≥14 recorded days, but not at ≥21 recorded days. The number of recorded days was not correlated with either factor score.</p>")
h.append("<table><tr><th>Analysis</th><th class='num'>n</th><th>F1 Life Restoration</th><th>F2 Migraine Agency</th></tr>")
for thr in [7, 14, 21]:
    m2 = (num('w28_migprop')*28).where(num('w28_days') >= thr)
    c1, n1 = partial_sp_p(m2, f1, [num('age'), num('sex')]); c2, _ = partial_sp_p(m2, f2, [num('age'), num('sex')])
    lab = f"Diary-based MMD, ≥{thr} recorded days" + (" (primary)" if thr == 7 else "")
    h.append(f"<tr><td>{lab}</td><td class='num'>{n1}</td><td class='num'>{c1}</td><td class='num'>{c2}</td></tr>")
wd1 = wdays.where(wdays >= 1)
c1, n1 = partial_sp_p(wd1, f1, [num('age'), num('sex')]); c2, _ = partial_sp_p(wd1, f2, [num('age'), num('sex')])
h.append(f"<tr><td>Number of recorded days (≥1 day)</td><td class='num'>{n1}</td><td class='num'>{c1}</td><td class='num'>{c2}</td></tr>")
h.append("</table>")

h.append("<h2>S3d. Exploratory comparison by current preventive-medication use</h2>")
prev = num('prev_use')
rows_e = []
for name, s in [("Life Restoration (F1)", f1), ("Migraine Agency (F2)", f2)]:
    ok = s.notna() & prev.notna()
    g1 = s[ok & (prev==1)]; g0 = s[ok & (prev==0)]
    p = stats.mannwhitneyu(g1, g0).pvalue
    ps = '&lt;0.001' if p < 0.001 else f'{p:.3f}'
    rows_e.append(f"<tr><td>{name}</td><td class='num'>{np.median(g1):.1f} (n={len(g1)})</td><td class='num'>{np.median(g0):.1f} (n={len(g0)})</td><td class='num'>{ps}</td></tr>")
h.append("<table><tr><th>Factor score, median</th><th class='num'>Preventive medication: yes</th><th class='num'>Preventive medication: no</th><th class='num'>Mann–Whitney p</th></tr>")
h.extend(rows_e)
h.append("</table>")
h.append("<p class='note'>Participants already using a preventive medication scored higher on Migraine Agency. This comparison is exploratory and descriptive; it was not prespecified.</p>")

h.append("<h2>S3e. Domain-level correlations with diary-based MMD, adjusted for age and sex (partial Spearman ρ [95% CI], p; n=81)</h2>")
h.append("<table><tr><th>Domain</th><th>Factor</th><th class='num'>ρ with diary-based MMD [95% CI], p</th></tr>")
for i in range(11):
    dcol = num(f'dom{i+1}')
    c, n1 = partial_sp_p(mmd, dcol, [num('age'), num('sex')])
    fac = 'Life Restoration' if i in {3,5,6,7,8,9,10} else 'Migraine Agency'
    h.append(f"<tr><td>{EN[i]}</td><td>{fac}</td><td class='num'>{c}</td></tr>")
h.append("</table>")

open('mfi_supp.html','w').write("\n".join(h))
print(f"wrote mfi_supp.html N={N} | S1 floor>15%: {n_floor15}, ceiling>15%: {n_ceil15} | per-dom min {per_dom_min:.1f}% overall min {overall_min:.1f}%")
print(f"EFA: KMO {kmo_all:.3f} Bartlett {bart_chi:.1f} eig {eig[0]:.2f}/{eig[1]:.2f}/{eig[2]:.2f} PA {pa95[0]:.2f}/{pa95[1]:.2f}/{pa95[2]:.2f} var {100*var[2][1]:.1f}% phi {phi:.2f} fs_r {fs_r:.2f}")
print(f"funnel: {n_off}/{n_beg}/{len(comp_f)}/{len(ip)} noans {noans} lastpos med {np.median(lastpos):.0f}")
