# -*- coding: utf-8 -*-
"""Supplement v3: post-processes mfi_supp.html (built by build_supp_v2.py) and adds the material requested after
peer review of the first version (Cephalalgia, October 2026):
  S3c  extra rows: all 28 days recorded; adjustment for recorded days; raw migraine-day count; exclusion of the
       three participants in whom neither classification identified migraine; Poisson regression with offset;
       recorded days vs headache-day proportion
  S3d  comparison by CGRP-related drug, plus diary-based MMD by group
  S3f  alternative scores (domain mean, loading-weighted mean, regression-based factor score)
  S4   item-level loadings, corrected item-rest correlations, floor/ceiling, alpha-if-deleted; domain unidimensionality
Run from the repo root after build_supp_v2.py:  python3 analysis/build_supp_v3.py  -> mfi_supp.html (overwritten)
"""
import re, html as H
import numpy as np, pandas as pd
from scipy import stats
from factor_analyzer import FactorAnalyzer

np.random.seed(20260718)
D = pd.read_csv('mfi_matrix.csv'); N = len(D)
num = lambda c: pd.to_numeric(D[c], errors='coerce')
qcols = [c for c in D.columns if re.match(r'^q\d+_\d+$', c)]
dom_items = {i: sorted([c for c in qcols if int(c.split('_')[0][1:]) == i], key=lambda c: int(c.split('_')[1])) for i in range(11)}
M = D[qcols].apply(pd.to_numeric, errors='coerce')
EN = ['Premonitory-symptom awareness', 'Trigger recognition', 'Pre-emptive action', 'Attack control',
      'Prevention of medication overuse', 'Freedom of activity', 'Hope and treatment continuity', 'Reclaiming your life',
      'Social freedom', 'Impact on close others', 'Global freedom evaluation']
F1_IDX = [3, 5, 6, 7, 8, 9, 10]; F2_IDX = [0, 1, 2, 4]
S1_ORDER = [3, 8, 5, 6, 7, 9, 10, 0, 1, 2, 4]
s1_cols = [c for i in S1_ORDER for c in dom_items[i]]
item_no = {c: k + 1 for k, c in enumerate(s1_cols)}
f1_cols = [c for i in F1_IDX for c in dom_items[i]]; f2_cols = [c for i in F2_IDX for c in dom_items[i]]
age, sex = num('age'), num('sex'); female = (sex == 1).astype(float)
wdays = num('w28_days'); d7 = wdays >= 7
mmd = (num('w28_migprop') * 28).where(d7); mig_raw = num('w28_mig').where(d7)
midas, mibs = num('midas'), num('mibs4'); f1, f2 = num('f1'), num('f2')
supp = open('mfi_supp.html', encoding='utf-8').read()
# S1 English text per item number (for S4)
s1_text = {}
for r in re.findall(r"<tr><td class='num'>(\d+)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td>", supp):
    s1_text[int(r[0])] = r[3]

def partial_sp_p(x, y, covs):
    df = pd.DataFrame({'x': x, 'y': y})
    for i, c in enumerate(covs): df[f'c{i}'] = c
    df = df.dropna(); n = len(df); k = len(covs)
    rx, ry = df['x'].rank(), df['y'].rank()
    C = np.column_stack([np.ones(n)] + [df[f'c{i}'].rank() for i in range(k)])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]; ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; z = np.arctanh(r); se = 1 / np.sqrt(n - 3 - k)
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2)); pv = 2 * stats.t.sf(abs(t), n - 2 - k)
    ps = '&lt;0.001' if pv < 0.001 else f'{pv:.3f}'
    return f"{r:+.2f} [{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}], {ps}", n
def sp_p(x, y):
    df = pd.DataFrame({'x': x, 'y': y}).dropna(); n = len(df)
    r, pv = stats.spearmanr(df['x'], df['y']); z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    ps = '&lt;0.001' if pv < 0.001 else f'{pv:.3f}'
    return f"{r:+.2f} [{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}], {ps}", n
def cronbach(X):
    X = X.dropna(); k = X.shape[1]
    return k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
def poisson_glm(y, X, offset):
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean()) - offset.mean()
    for _ in range(50):
        eta = X @ b + offset; mu = np.exp(eta); z = eta - offset + (y - mu) / mu
        XtW = X.T * mu; bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-9: b = bn; break
        b = bn
    mu = np.exp(X @ b + offset); Ib = np.linalg.inv((X.T * mu) @ X); S = (X.T * ((y - mu) ** 2)) @ X; V = Ib @ S @ Ib
    return b, np.sqrt(np.diag(V)), np.sum((y - mu) ** 2 / mu) / (len(y) - X.shape[1])
COV = [age, sex]

# ---------------- S3c: extra rows ----------------
rows = []
m28 = (num('w28_migprop') * 28).where(wdays >= 28)
c1, n1 = partial_sp_p(m28, f1, COV); c2, _ = partial_sp_p(m28, f2, COV)
rows.append(f"<tr><td>Diary-based MMD, all 28 days recorded</td><td class='num'>{n1}</td><td class='num'>{c1}</td><td class='num'>{c2}</td></tr>")
c1, n1 = partial_sp_p(mmd, f1, COV + [wdays]); c2, _ = partial_sp_p(mmd, f2, COV + [wdays])
rows.append(f"<tr><td>Diary-based MMD, ≥7 recorded days, additionally adjusted for the number of recorded days</td><td class='num'>{n1}</td><td class='num'>{c1}</td><td class='num'>{c2}</td></tr>")
c1, n1 = partial_sp_p(mig_raw, f1, COV); c2, _ = partial_sp_p(mig_raw, f2, COV)
rows.append(f"<tr><td>Raw number of migraine days (no extrapolation), ≥7 recorded days</td><td class='num'>{n1}</td><td class='num'>{c1}</td><td class='num'>{c2}</td></tr>")
cd = num('case_def') == 1
c1, n1 = partial_sp_p(mmd[cd], f1[cd], [age[cd], sex[cd]]); c2, _ = partial_sp_p(mmd[cd], f2[cd], [age[cd], sex[cd]])
rows.append(f"<tr><td>Diary-based MMD, ≥7 recorded days, excluding the {int((~cd).sum())} participants in whom neither classification identified migraine</td><td class='num'>{n1}</td><td class='num'>{c1}</td><td class='num'>{c2}</td></tr>")
c1, n1 = sp_p(wdays[wdays >= 1], num('w28_haprop')[wdays >= 1])
rows.append(f"<tr><td>Number of recorded days vs proportion of headache days (≥1 day; unadjusted Spearman)</td><td class='num'>{n1}</td><td class='num' colspan='2'>{c1}</td></tr>")
supp = supp.replace("<tr><td>Number of recorded days (≥1 day)</td>", "".join(rows) + "<tr><td>Number of recorded days (≥1 day)</td>", 1)
# Poisson table after S3c
g = pd.DataFrame({'y': num('w28_mig'), 'wd': wdays, 'f1': f1, 'f2': f2, 'age': age, 'female': female})
pois = ["<h2>S3c-2. Poisson regression of the number of migraine days on each factor score, with the logarithm of the number of recorded days as an offset (robust standard errors)</h2>",
        "<p class='note'>Each model: number of migraine days in the 28-day window ~ factor score (per 10 points) + age + sex, offset = log(recorded days). The dispersion ratio indicates overdispersion, so robust (sandwich) standard errors are reported. IRR = incidence rate ratio.</p>",
        "<table><tr><th>Factor</th><th>Recorded days</th><th class='num'>n</th><th class='num'>IRR per 10 points [95% CI]</th><th class='num'>p</th><th class='num'>Dispersion ratio</th></tr>"]
for fac, lab in [('f1', 'Life Restoration'), ('f2', 'Migraine Agency')]:
    for thr in [7, 14, 21]:
        gg = g[(g['wd'] >= thr)].dropna(); X = np.column_stack([np.ones(len(gg)), gg[fac] / 10, gg['age'], gg['female']])
        b, se, disp = poisson_glm(gg['y'].values.astype(float), X, np.log(gg['wd'].values))
        pv = 2 * stats.norm.sf(abs(b[1] / se[1])); ps = '&lt;0.001' if pv < 0.001 else f'{pv:.3f}'
        pois.append(f"<tr><td>{lab}</td><td>≥{thr}</td><td class='num'>{len(gg)}</td><td class='num'>{np.exp(b[1]):.2f} [{np.exp(b[1]-1.96*se[1]):.2f}, {np.exp(b[1]+1.96*se[1]):.2f}]</td><td class='num'>{ps}</td><td class='num'>{disp:.2f}</td></tr>")
pois.append("</table>")
supp = supp.replace("<h2>S3d. Exploratory comparison by current preventive-medication use</h2>", "\n".join(pois) + "\n<h2>S3d. Exploratory comparison by current preventive-medication use</h2>", 1)
# S3c note update
supp = supp.replace("The correlation between the Life Restoration score and diary-based MMD was significant at ≥7 and ≥14 recorded days, but not at ≥21 recorded days. The number of recorded days was not correlated with either factor score.",
    "The correlation between the Life Restoration score and diary-based MMD was significant at ≥7 and ≥14 recorded days, but not at ≥21 recorded days or in participants with all 28 days recorded. The correlation was similar with additional adjustment for the number of recorded days, with the raw number of migraine days, and after excluding the participants in whom neither classification identified migraine. The number of recorded days was not correlated with either factor score, but participants who recorded fewer days had a higher proportion of headache days.", 1)

# ---------------- S3d: add CGRP-related drug ----------------
cg = num('cgrp_use'); prev = num('prev_use')
def grp_rows(flag, lab_yes, lab_no):
    out = []
    for name, s in [("Life Restoration (F1)", f1), ("Migraine Agency (F2)", f2), ("Diary-based MMD (≥7 recorded days)", mmd)]:
        ok = s.notna() & flag.notna(); g1 = s[ok & (flag == 1)]; g0 = s[ok & (flag == 0)]
        p = stats.mannwhitneyu(g1, g0).pvalue; ps = '&lt;0.001' if p < 0.001 else f'{p:.3f}'
        q = lambda v: f"{np.median(v):.1f} ({np.percentile(v,25):.1f}–{np.percentile(v,75):.1f})"
        out.append(f"<tr><td>{name}</td><td class='num'>{q(g1)} (n={len(g1)})</td><td class='num'>{q(g0)} (n={len(g0)})</td><td class='num'>{ps}</td></tr>")
    return out
cgrp_tab = ["<h2>S3d-2. Exploratory comparison by current use of a CGRP-related drug (median (IQR))</h2>",
            "<table><tr><th>Score</th><th class='num'>CGRP-related drug: yes</th><th class='num'>CGRP-related drug: no</th><th class='num'>Mann–Whitney p</th></tr>"] + grp_rows(cg, 'yes', 'no') + ["</table>",
            "<p class='note'>CGRP-related drugs: fremanezumab, galcanezumab, erenumab, atogepant, and rimegepant used for prevention. This comparison is exploratory and descriptive; it was not prespecified.</p>"]
supp = supp.replace("<h2>S3e. Domain-level correlations", "\n".join(cgrp_tab) + "\n<h2>S3e. Domain-level correlations", 1)

# ---------------- S3f: alternative scores ----------------
Xd = D[[f'dom{i+1}' for i in range(11)]].apply(pd.to_numeric, errors='coerce'); ok = Xd.notna().all(axis=1)
fad = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin'); fad.fit(Xd[ok])
Ld = fad.loadings_.copy()
if abs(Ld[7, 0]) < abs(Ld[7, 1]): Ld = Ld[:, ::-1]
FS = pd.DataFrame(fad.transform(Xd[ok]), index=Xd[ok].index, columns=['A', 'B'])
if np.corrcoef(FS['A'], f1[ok])[0, 1] < np.corrcoef(FS['B'], f1[ok])[0, 1]: FS = FS.rename(columns={'A': 'B', 'B': 'A'})
s3f = ["<h2>S3f. Alternative scoring of the two factors: correlations with the unit-weighted score and with each measure (partial Spearman ρ [95% CI], p, adjusted for age and sex)</h2>",
       "<p class='note'>Unit-weighted = mean of all items in the factor's domains (main text). Domain mean = mean of the domain scores (each domain weighted equally). Loading-weighted = mean of the domain scores weighted by the domain's loading on the factor. Regression-based = factor score from the domain-level exploratory factor analysis. ρ vs unit-weighted is the unadjusted Spearman correlation between the alternative score and the unit-weighted score.</p>",
       "<table><tr><th>Factor</th><th>Score</th><th class='num'>ρ vs unit-weighted</th><th class='num'>Diary-based MMD (n=81)</th><th class='num'>MIDAS (n=115)</th><th class='num'>MIBS-4 (n=115)</th></tr>"]
for fac, idx, unit, col in [('Life Restoration', F1_IDX, f1, 0), ('Migraine Agency', F2_IDX, f2, 1)]:
    doms = [f'dom{i+1}' for i in idx]; w = np.clip(Ld[idx, col], 0, None)
    alts = [('Unit-weighted (main text)', unit), ('Domain mean', Xd[doms].mean(axis=1)),
            ('Loading-weighted domain mean', (Xd[doms] * w).sum(axis=1) / w.sum()), ('Regression-based factor score', FS['A' if col == 0 else 'B'].reindex(D.index))]
    for name, s in alts:
        rr = stats.spearmanr(pd.DataFrame({'a': unit, 'b': s}).dropna().values)[0]
        cells = "".join(f"<td class='num'>{partial_sp_p(x, s, COV)[0]}</td>" for x in [mmd, midas, mibs])
        s3f.append(f"<tr><td>{fac}</td><td>{name}</td><td class='num'>{rr:.3f}</td>{cells}</tr>")
s3f.append("</table>")
supp = supp.rstrip() + "\n" + "\n".join(s3f)

# ---------------- S4: item-level ----------------
miss = M.isna().mean(); drop_cols = list(miss[miss > 0.05].index)
Mi = M.drop(columns=drop_cols).dropna()
fa = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin'); fa.fit(Mi)
L = pd.DataFrame(fa.loadings_, index=Mi.columns, columns=['L1', 'L2'])
if L.loc[[c for c in f1_cols if c in L.index], 'L1'].mean() < L.loc[[c for c in f1_cols if c in L.index], 'L2'].mean(): L = L.rename(columns={'L1': 'L2', 'L2': 'L1'})
a_LR, a_MA = cronbach(M[f1_cols]), cronbach(M[f2_cols])
s4 = ["<h1>Supplementary Table S4. Item-level analysis (exploratory) and unidimensionality of each domain</h1>",
      "<h2>S4a. Item loadings from an exploratory two-factor analysis of the items (principal axis, oblimin), corrected item-rest correlations, alpha if the item is deleted, and floor/ceiling</h2>",
      f"<p class='note'>The item-level factor analysis used {len(Mi)} participants with complete responses to {Mi.shape[1]} items; item #{', #'.join(str(item_no[c]) for c in drop_cols)} (menstruation-related, shown only to respondents reporting female sex) was omitted because of missing responses ({', '.join(f'{100*miss[c]:.0f}%' for c in drop_cols)}). The participant-to-item ratio is low (1.4), so these results are exploratory. Item-rest (domain) = correlation of the item with the mean of the other items in its domain; item-rest (factor) = with the mean of the other items in its factor. Alpha if deleted = Cronbach α of the factor without the item (Life Restoration α={a_LR:.3f}; Migraine Agency α={a_MA:.3f}). Loadings below 0.40 on the item's own factor are marked with an asterisk.</p>",
      "<table><tr><th>#</th><th>Factor</th><th>Domain</th><th>Item (English, provisional)</th><th class='num'>Loading F1</th><th class='num'>Loading F2</th><th class='num'>Item-rest (domain)</th><th class='num'>Item-rest (factor)</th><th class='num'>α if deleted</th><th class='num'>Floor %</th><th class='num'>Ceiling %</th></tr>"]
n_weak = n_low = 0
for c in s1_cols:
    d = int(c.split('_')[0][1:]); fac_cols = f1_cols if c in f1_cols else f2_cols; fac = 'LR' if c in f1_cols else 'MA'
    rest_dom = M[[x for x in dom_items[d] if x != c]].mean(axis=1); rest_fac = M[[x for x in fac_cols if x != c]].mean(axis=1)
    r_dom = pd.concat([M[c], rest_dom], axis=1).dropna().corr().iloc[0, 1]; r_fac = pd.concat([M[c], rest_fac], axis=1).dropna().corr().iloc[0, 1]
    a_del = cronbach(M[fac_cols].drop(columns=[c])); v = M[c].dropna()
    l1 = L.loc[c, 'L1'] if c in L.index else np.nan; l2 = L.loc[c, 'L2'] if c in L.index else np.nan
    own = l1 if fac == 'LR' else l2; star = '*' if (not np.isnan(own) and own < 0.40) else ''
    n_weak += bool(star); n_low += r_fac < 0.30
    fmt = lambda x: 'n.a.' if np.isnan(x) else f"{x:+.2f}"
    s4.append(f"<tr><td class='num'>{item_no[c]}</td><td>{'F1 Life Restoration' if fac=='LR' else 'F2 Migraine Agency'}</td><td>{EN[d]}</td><td>{s1_text.get(item_no[c], '')}</td>"
              f"<td class='num'>{fmt(l1)}{star if fac=='LR' else ''}</td><td class='num'>{fmt(l2)}{star if fac=='MA' else ''}</td><td class='num'>{r_dom:.2f}</td><td class='num'>{r_fac:.2f}</td><td class='num'>{a_del:.3f}</td>"
              f"<td class='num'>{100*(v==0).mean():.1f}</td><td class='num'>{100*(v==6).mean():.1f}</td></tr>")
s4.append("</table>")
# redundancy
R = M[f1_cols].corr(); pairs = [(item_no[a], item_no[b], R.loc[a, b]) for i, a in enumerate(f1_cols) for b in f1_cols[i+1:]]
hi = sorted([p for p in pairs if p[2] >= 0.80], key=lambda t: -t[2])
Rm = M[f2_cols].corr(); pm = sorted([(item_no[a], item_no[b], Rm.loc[a, b]) for i, a in enumerate(f2_cols) for b in f2_cols[i+1:] if Rm.loc[a, b] >= 0.80], key=lambda t: -t[2])
s4.append(f"<p class='note'>{n_weak} items loaded below 0.40 on their factor, and the item-rest correlation with the factor was below 0.30 for {n_low} items.</p>")
s4.append("<h2>S4b. Item pairs with a correlation of 0.80 or more within a factor (Pearson)</h2>")
s4.append("<table><tr><th>Factor</th><th>Item pair</th><th class='num'>r</th><th>Items</th></tr>")
for a, b, r in hi: s4.append(f"<tr><td>Life Restoration</td><td>#{a} × #{b}</td><td class='num'>{r:.2f}</td><td>{s1_text.get(a,'')} / {s1_text.get(b,'')}</td></tr>")
for a, b, r in pm: s4.append(f"<tr><td>Migraine Agency</td><td>#{a} × #{b}</td><td class='num'>{r:.2f}</td><td>{s1_text.get(a,'')} / {s1_text.get(b,'')}</td></tr>")
s4.append("</table>")
s4.append(f"<p class='note'>Life Restoration: {len(hi)} of {len(pairs)} item pairs correlated at 0.80 or more ({sum(1 for a,b,r in hi if s1_cols[a-1].split('_')[0]==s1_cols[b-1].split('_')[0])} within the same domain). Migraine Agency: {len(pm)} pairs.</p>")
# unidimensionality
s4.append("<h2>S4c. Unidimensionality of each domain: mean inter-item correlation, Cronbach α, first two eigenvalues, parallel analysis (500 random datasets, 95th percentile), and the smallest single-factor loading</h2>")
s4.append("<table><tr><th>Domain</th><th>Factor</th><th class='num'>Items</th><th class='num'>n</th><th class='num'>Mean inter-item r</th><th class='num'>α</th><th class='num'>λ1</th><th class='num'>λ2</th><th class='num'>λ1/λ2</th><th class='num'>λ2 random 95th</th><th>Second factor retained</th><th class='num'>Smallest loading (item)</th></tr>")
for i in S1_ORDER:
    cols = dom_items[i]; X = M[cols].dropna(); k = len(cols); fac = 'Life Restoration' if i in F1_IDX else 'Migraine Agency'
    if k < 3:
        s4.append(f"<tr><td>{EN[i]}</td><td>{fac}</td><td class='num'>{k}</td><td class='num'>{len(X)}</td><td class='num'>{X.corr().iloc[0,1]:.2f}</td><td class='num'>{cronbach(X):.2f}</td><td class='num' colspan='7'>not testable with two items</td></tr>"); continue
    Rd = X.corr().values; iu = np.triu_indices(k, 1); ev = np.sort(np.linalg.eigvalsh(Rd))[::-1]
    re_ = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(np.random.normal(size=(len(X), k)), rowvar=False)))[::-1] for _ in range(500)])
    pa = np.percentile(re_, 95, axis=0); keep2 = ev[1] > pa[1]
    fa1 = FactorAnalyzer(n_factors=1, method='principal', rotation=None); fa1.fit(X); l1 = np.abs(fa1.loadings_[:, 0])
    s4.append(f"<tr><td>{EN[i]}</td><td>{fac}</td><td class='num'>{k}</td><td class='num'>{len(X)}</td><td class='num'>{Rd[iu].mean():.2f}</td><td class='num'>{cronbach(X):.2f}</td><td class='num'>{ev[0]:.2f}</td><td class='num'>{ev[1]:.2f}</td><td class='num'>{ev[0]/ev[1]:.1f}</td><td class='num'>{pa[1]:.2f}</td><td>{'yes' if keep2 else 'no'}</td><td class='num'>{l1.min():.2f} (#{item_no[cols[int(np.argmin(l1))]]})</td></tr>")
s4.append("</table>")
s4.append("<p class='note'>A second factor was retained within three domains (prevention of medication overuse, hope and treatment continuity, and impact on close others), so these domains are not unidimensional in this sample.</p>")
supp = supp + "\n" + "\n".join(s4) + "\n"
# header: S1–S4
supp = supp.replace("Supplementary material — Migraine Freedom Index (MFI) pilot (n=", "Supplementary material — Migraine Freedom Index (MFI) pilot (n=", 1)
open('mfi_supp.html', 'w', encoding='utf-8').write(supp)
print(f"wrote mfi_supp.html (v3): S3c +{len(rows)} rows, Poisson table, S3d-2 CGRP, S3f alternatives, S4 ({len(s1_cols)} items; weak {n_weak}, low item-rest {n_low}; LR pairs≥0.80 {len(hi)}, MA {len(pm)})")
