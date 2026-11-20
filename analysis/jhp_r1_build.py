#!/usr/bin/env python3
"""Revision 1 for The Journal of Headache and Pain (November 2026).

Recomputes every reported statistic from the frozen matrix (mfi_matrix.csv, n = 115) and writes
  - jhp_r1_numbers.json      numbers quoted in the manuscript and the response letter
  - mfi_supp_jhp_r1.html     Additional file 2 (Supplementary Tables S1 to S4)
  - fig1_flow_r1.png, fig2_loadings.png, fig3_scatter.png   (FIG_DPI=300 for submission)

Changes from the first submission, all made in response to peer review:
  1. Factor extraction. The first submission called factor_analyzer's method='principal'
     "principal-axis"; that routine extracts principal components. The factor analyses now use a
     common-factor method (minimum residual, method='minres'), and the component solution is kept
     in Table S2b for comparison. McDonald's omega uses the same extraction.
  2. Sex covariate. Sex was entered as the questionnaire code (0 male, 1 female, 2 other,
     3 no answer). It is now entered as female versus not female.
  3. Added: correlations among diary-based MMD, self-reported MMD, MIDAS and MIBS-4 in the same
     participants; self-reported MMD; included versus excluded participants; MMD strata;
     additional adjustment; item subsets; domain omega. Figure 1 and Table S3a give the counts at the
     close of recruitment (18 July 2026) and the account deletions before the diary extraction.
  4. The split at the sample medians was removed, as Reviewer 1 suggested (no Table S3l, no Figure S1).
     It is still computed below (median_split in jhp_r1_numbers.json) but not reported.
Run from the repository root:  python3 analysis/jhp_r1_build.py
"""
import html, json, os, re, warnings
import numpy as np, pandas as pd
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

SEED = 20260718
np.random.seed(SEED)
D = pd.read_csv('mfi_matrix.csv'); N = len(D)
num = lambda c: pd.to_numeric(D[c], errors='coerce')
NUM = {}

EN = ['Premonitory-symptom awareness', 'Trigger recognition', 'Pre-emptive action', 'Attack control',
      'Prevention of medication overuse', 'Freedom of activity', 'Hope and treatment continuity', 'Reclaiming your life',
      'Social freedom', 'Impact on close others', 'Global freedom evaluation']
F1_IDX = [3, 5, 6, 7, 8, 9, 10]; F2_IDX = [0, 1, 2, 4]
S1_ORDER = [3, 8, 5, 6, 7, 9, 10, 0, 1, 2, 4]
item_cols = [c for c in D.columns if re.match(r'^q\d+_\d+$', c)]
dom_items = {i: [c for c in item_cols if c.startswith(f'q{i}_')] for i in range(11)}
M = D[item_cols].apply(pd.to_numeric, errors='coerce')
s1_cols = [c for i in S1_ORDER for c in dom_items[i]]
item_no = {c: k + 1 for k, c in enumerate(s1_cols)}; col_of = {v: k for k, v in item_no.items()}
f1_cols = [c for i in F1_IDX for c in dom_items[i]]; f2_cols = [c for i in F2_IDX for c in dom_items[i]]

# item wording (English, Japanese) from the first-submission Additional file 2
_src = open('mfi_supp_jhp.html', encoding='utf-8').read()
CSS = re.search(r'<style>.*?</style>', _src, re.S).group(0)
_body = re.sub(r'<style.*?</style>', '', _src, flags=re.S)
S1 = {}
for r in re.findall(r'<tr[^>]*>(.*?)</tr>', _body, flags=re.S):
    cells = [re.sub(r'<[^>]+>', '', c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, flags=re.S)]
    if len(cells) >= 5 and re.fullmatch(r'\d{1,2}', cells[0]) and int(cells[0]) not in S1: S1[int(cells[0])] = cells
assert len(S1) == 82

# ---------------------------------------------------------------- variables
age = num('age'); sexcode = num('sex'); female = (sexcode == 1).astype(float)
COV = [age, female]
wdays = num('w28_days'); d7 = wdays >= 7
mmd = (num('w28_migprop') * 28).where(d7); mig_raw = num('w28_mig').where(d7)
srmmd = num('mig_days').where(num('mig_days') <= 28)
midas, mibs = num('midas'), num('mibs4'); f1, f2 = num('f1'), num('f2')
prev, cg = num('prev_use'), num('cgrp_use')
assert (M[f1_cols].mean(axis=1) / 6 * 100 - f1).abs().max() < 1e-6 and (M[f2_cols].mean(axis=1) / 6 * 100 - f2).abs().max() < 1e-6

# ---------------------------------------------------------------- helpers
def pcor(x, y, covs=COV):
    """Partial Spearman: rank, residualise on ranked covariates; Fisher z with SE 1/sqrt(n-3-k)."""
    df = pd.DataFrame({'x': x, 'y': y})
    for i, c in enumerate(covs): df[f'c{i}'] = c
    df = df.dropna(); n = len(df); k = len(covs)
    rx, ry = df['x'].rank(), df['y'].rank()
    C = np.column_stack([np.ones(n)] + [df[f'c{i}'].rank() for i in range(k)])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]; ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; z = np.arctanh(r); se = 1 / np.sqrt(n - 3 - k)
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2)); p = 2 * stats.t.sf(abs(t), n - 2 - k)
    return dict(r=float(r), lo=float(np.tanh(z - 1.96 * se)), hi=float(np.tanh(z + 1.96 * se)), p=float(p), n=n)
def scor(x, y): return pcor(x, y, [])
def sg(x):
    t = f"{x:+.2f}"; return '0.00' if t in ('+0.00', '-0.00') else t
def ci(d, sep=', '): return f"{sg(d['r'])} [{sg(d['lo'])}{sep}{sg(d['hi'])}]"
def cell(d): return f"{ci(d)}"
def miqr(s): s = s.dropna(); return f"{s.median():.1f} ({s.quantile(.25):.1f} to {s.quantile(.75):.1f})"
def miqr0(s): s = s.dropna(); return f"{s.median():.0f} ({s.quantile(.25):.0f} to {s.quantile(.75):.0f})"
def pfmt(p): return '&lt;0.001' if p < 0.001 else f'{p:.3f}'
def alpha(X): X = X.dropna(); k = X.shape[1]; return k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
def omega(X):
    X = X.dropna(); f = FactorAnalyzer(n_factors=1, method='minres', rotation=None); f.fit(X); l = f.loadings_[:, 0]
    return l.sum() ** 2 / (l.sum() ** 2 + (1 - l ** 2).sum())
def poisson(y, X, off):
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean()) - off.mean()
    for _ in range(200):
        eta = X @ b + off; mu = np.exp(eta); z = eta - off + (y - mu) / mu; XtW = X.T * mu; bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-10: b = bn; break
        b = bn
    mu = np.exp(X @ b + off); Ib = np.linalg.inv((X.T * mu) @ X); S = (X.T * ((y - mu) ** 2)) @ X; V = Ib @ S @ Ib
    return b, np.sqrt(np.diag(V)), float(np.sum((y - mu) ** 2 / mu) / (len(y) - X.shape[1]))
def fa2(X, method='minres'):
    fa = FactorAnalyzer(n_factors=2, method=method, rotation='oblimin'); fa.fit(X); return fa
H = []
def h(s): H.append(s)
def table(head, rows, numcols=()):
    cls = lambda i: " class='num'" if i in numcols else ""
    out = ["<table><tr>" + "".join("<th%s>%s</th>" % (cls(i), c) for i, c in enumerate(head)) + "</tr>"]
    for r in rows: out.append("<tr>" + "".join("<td%s>%s</td>" % (cls(i), c) for i, c in enumerate(r)) + "</tr>")
    out.append("</table>"); h("\n".join(out))

# ================================================================ sample
NUM['sample'] = dict(N=N, women=int((sexcode == 1).sum()), men=int((sexcode == 0).sum()), other=int((sexcode == 2).sum()), no_answer=int((sexcode == 3).sum()),
                     prev_yes=int((prev == 1).sum()), prev_n=int(prev.notna().sum()), cgrp_yes=int((cg == 1).sum()), cgrp_n=int(cg.notna().sum()),
                     item65_missing=int(M[col_of[65]].isna().sum()), srmmd_valid=int(srmmd.notna().sum()), srmmd_median=miqr0(srmmd),
                     n81=int(d7.sum()), n_ge1=int((wdays >= 1).sum()))
F = pd.read_csv('mfi_funnel.csv'); comp_f = F[F.status == 'completed']; ip = F[F.status == 'in_progress']
noans = int((pd.to_numeric(ip['n_answered'], errors='coerce').fillna(0) == 0).sum())
# Counts at the close of recruitment (18 July 2026), recorded by extraction/mfi_full_analysis.ts on that day
# (analysis-results, section R1). mfi_funnel.csv was extracted on 15 August 2026, after 10 account deletions.
OFFERED0, ENTERED0, COMPLETED0 = 1190, 181, 116
del_all = OFFERED0 - len(F); del_entered = ENTERED0 - (len(comp_f) + len(ip)); del_completed = COMPLETED0 - len(comp_f)
assert (del_all, del_entered, del_completed) == (10, 1, 1) and len(ip) == ENTERED0 - COMPLETED0, (del_all, del_entered, del_completed)
NUM['funnel'] = dict(offered=OFFERED0, entered=ENTERED0, completed=COMPLETED0, discontinued=len(ip), no_items=noans, some_items=len(ip) - noans,
                     deleted=del_all, deleted_completed=del_completed, analysed=len(comp_f),
                     pct_of_entered=round(100 * COMPLETED0 / ENTERED0, 1), pct_of_offered=round(100 * COMPLETED0 / OFFERED0, 1))

# ================================================================ domain-level factor analysis
Xd = D[[f'dom{i+1}' for i in range(11)]].apply(pd.to_numeric, errors='coerce'); Xd.columns = EN
assert Xd.notna().all().all()
R = np.corrcoef(Xd.values, rowvar=False); n, p = Xd.shape
eig = np.sort(np.linalg.eigvalsh(R))[::-1]
def red_eigs(Rm):
    Rr = Rm.copy(); np.fill_diagonal(Rr, 1 - 1 / np.diag(np.linalg.inv(Rm))); return np.sort(np.linalg.eigvalsh(Rr))[::-1]
obs_red = red_eigs(R); rng = np.random.default_rng(SEED); rc = np.zeros((1000, p)); rf = np.zeros((1000, p))
for b in range(1000):
    Rz = np.corrcoef(rng.normal(size=(n, p)), rowvar=False); rc[b] = np.sort(np.linalg.eigvalsh(Rz))[::-1]; rf[b] = red_eigs(Rz)
pa_c, pa_f = np.percentile(rc, 95, 0), np.percentile(rf, 95, 0)
def n_keep(obs, ref):
    k = 0
    while k < len(obs) and obs[k] > ref[k]: k += 1
    return k
kmo = calculate_kmo(Xd)[1]; bart_chi, bart_p = calculate_bartlett_sphericity(Xd)
def oriented(fa):
    L = fa.loadings_.copy(); phi = fa.phi_.copy(); order = [0, 1]
    if abs(L[7, 0]) < abs(L[7, 1]): L = L[:, ::-1]; phi = phi[::-1, ::-1]; order = [1, 0]
    sgn = [1 if L[F1_IDX, 0].sum() >= 0 else -1, 1 if L[F2_IDX, 1].sum() >= 0 else -1]
    L = L * np.array(sgn); phi = phi * np.outer(sgn, sgn)
    return L, phi, order, sgn
fa_m = fa2(Xd); Lm, phim, order_m, sg_m = oriented(fa_m); hm = fa_m.get_communalities()
fa_c = fa2(Xd, 'principal'); Lc, phic, _, _ = oriented(fa_c); hc = fa_c.get_communalities()
var_m = [float((Lm[:, k] ** 2).sum() / 11) for k in range(2)]
def boot_load(method):
    Lref = Lm if method == 'minres' else Lc; out = []
    r = np.random.default_rng(SEED)
    for b in range(1000):
        idx = r.integers(0, n, n)
        try: Lb = fa2(pd.DataFrame(Xd.values[idx], columns=EN), method).loadings_.copy()
        except Exception: continue
        best = None
        for perm in ([0, 1], [1, 0]):
            for s1_ in (1, -1):
                for s2_ in (1, -1):
                    C = Lb[:, perm] * np.array([s1_, s2_]); sc = float(np.sum(C * Lref))
                    if best is None or sc > best[0]: best = (sc, C)
        out.append(best[1])
    out = np.array(out); return np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0), np.median(out, 0), len(out)
blo, bhi, bmed, nboot = boot_load('minres')
prim = [abs(Lm[i, 0]) if i in F1_IDX else abs(Lm[i, 1]) for i in range(11)]
sec = [abs(Lm[i, 1]) if i in F1_IDX else abs(Lm[i, 0]) for i in range(11)]
assert all((abs(Lm[i, 0]) > abs(Lm[i, 1])) == (i in F1_IDX) for i in range(11))
fa3 = FactorAnalyzer(n_factors=3, method='minres', rotation='oblimin'); fa3.fit(Xd); L3 = fa3.loadings_
NUM['efa'] = dict(kmo=round(float(kmo), 2), bartlett_chi=round(float(bart_chi), 1), eig=[round(float(x), 2) for x in eig[:4]], pa_component=[round(float(x), 2) for x in pa_c[:4]],
                  keep_component=n_keep(eig, pa_c), eig_reduced=[round(float(x), 2) for x in obs_red[:5]], pa_factor=[round(float(x), 2) for x in pa_f[:5]], keep_factor=n_keep(obs_red, pa_f),
                  var_total=round(100 * float(hm.sum() / 11), 1), var_each=[round(100 * v, 1) for v in var_m], phi=round(float(phim[0, 1]), 2),
                  score_r=round(float(f1.corr(f2)), 2), score_rho=scor(f1, f2),
                  prim_LR=[round(min(prim[i] for i in F1_IDX), 2), round(max(prim[i] for i in F1_IDX), 2)], prim_MA=[round(min(prim[i] for i in F2_IDX), 2), round(max(prim[i] for i in F2_IDX), 2)],
                  max_secondary=round(max(sec), 2), h2=[round(float(x), 2) for x in hm], loadings=[[round(float(a), 2), round(float(b), 2)] for a, b in Lm],
                  component=dict(var_total=round(100 * float(hc.sum() / 11), 1), phi=round(float(phic[0, 1]), 2)),
                  three_factor=[[round(float(v), 2) for v in row] for row in L3])

# ================================================================ reliability
REL = {}
for i in range(11):
    X = M[dom_items[i]]
    REL[EN[i]] = dict(k=X.shape[1], alpha=round(float(alpha(X)), 2), omega=(round(float(omega(X)), 2) if X.shape[1] >= 3 else None), r=(round(float(X.corr().iloc[0, 1]), 2) if X.shape[1] == 2 else None))
REL['LR'] = dict(k=len(f1_cols), alpha=round(float(alpha(M[f1_cols])), 2), omega=round(float(omega(M[f1_cols])), 2))
REL['MA'] = dict(k=len(f2_cols), alpha=round(float(alpha(M[f2_cols])), 2), omega=round(float(omega(M[f2_cols])), 2))
NUM['reliability'] = REL

# ================================================================ correlations
V = {'LR': f1, 'MA': f2, 'MMD': mmd, 'srMMD': srmmd, 'MIDAS': midas, 'MIBS4': mibs}
COR = {}; CORU = {}; COR81 = {}
keys = list(V)
for a in range(len(keys)):
    for b in range(a + 1, len(keys)):
        COR[f'{keys[a]}~{keys[b]}'] = pcor(V[keys[a]], V[keys[b]]); CORU[f'{keys[a]}~{keys[b]}'] = scor(V[keys[a]], V[keys[b]])
        COR81[f'{keys[a]}~{keys[b]}'] = pcor(V[keys[a]].where(d7), V[keys[b]].where(d7))
NUM['cor_adj'] = COR; NUM['cor_unadj'] = CORU; NUM['cor_adj_81'] = COR81

SENS = {}
for thr in [7, 14, 21, 28]:
    m2 = (num('w28_migprop') * 28).where(wdays >= thr); SENS[f'ge{thr}'] = dict(LR=pcor(m2, f1), MA=pcor(m2, f2))
SENS['adj_recorded'] = dict(LR=pcor(mmd, f1, COV + [wdays]), MA=pcor(mmd, f2, COV + [wdays]))
SENS['raw_days'] = dict(LR=pcor(mig_raw, f1), MA=pcor(mig_raw, f2))
cd = num('case_def') == 1
SENS['excl_no_migraine'] = dict(LR=pcor(mmd.where(cd), f1), MA=pcor(mmd.where(cd), f2), n_excl=int((~cd).sum()))
hap = num('w28_haprop').where(d7)
SENS['haprop'] = dict(LR=pcor(hap, f1), MA=pcor(hap, f2))
low = f1.idxmin(); notlow = pd.Series(D.index != low, index=D.index)
SENS['excl_lowest_LR'] = dict(LR=pcor(mmd.where(notlow), f1), MA=pcor(mmd.where(notlow), f2), lr=round(float(f1[low]), 1), mmd=round(float(mmd[low]), 1))
SENS['rec_vs_LR'] = pcor(wdays.where(wdays >= 1), f1); SENS['rec_vs_MA'] = pcor(wdays.where(wdays >= 1), f2)
SENS['rec_vs_haprop'] = scor(wdays.where(wdays >= 1), num('w28_haprop')); SENS['rec_vs_migprop'] = scor(wdays.where(wdays >= 1), num('w28_migprop'))
POIS = {}
g = pd.DataFrame({'y': num('w28_mig'), 'wd': wdays, 'f1': f1, 'f2': f2, 'age': age, 'female': female, 'midas': midas, 'mibs': mibs})
for fac, sc in [('f1', 10), ('f2', 10), ('midas', 10), ('mibs', 1)]:
    for thr in [7, 14, 21]:
        gg = g[g.wd >= thr].dropna(); X = np.column_stack([np.ones(len(gg)), gg[fac] / sc, gg.age, gg.female])
        b, se, disp = poisson(gg.y.values.astype(float), X, np.log(gg.wd.values))
        POIS[f'{fac}_ge{thr}'] = dict(irr=float(np.exp(b[1])), lo=float(np.exp(b[1] - 1.96 * se[1])), hi=float(np.exp(b[1] + 1.96 * se[1])),
                                      p=float(2 * stats.norm.sf(abs(b[1] / se[1]))), disp=round(disp, 2), n=len(gg))
NUM['sens'] = SENS; NUM['poisson'] = POIS

# additional adjustment and joint explained variance
def r2(y, Xs):
    df = pd.concat([y] + Xs, axis=1).dropna(); Y = df.iloc[:, 0].rank().values
    X = np.column_stack([np.ones(len(df))] + [df.iloc[:, i].rank().values for i in range(1, df.shape[1])])
    res = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]; return round(float(1 - res.var() / Y.var()), 2), len(df)
ADD = dict(LR_MMD_adj_midas_mibs=pcor(mmd, f1, COV + [midas, mibs]), LR_MIDAS_adj_mmd_mibs=pcor(midas.where(d7), f1, COV + [mmd, mibs]),
           LR_MIBS_adj_mmd_midas=pcor(mibs.where(d7), f1, COV + [mmd, midas]),
           LR_MMD_adj_tx=pcor(mmd, f1, COV + [prev, cg]), LR_MIDAS_adj_tx=pcor(midas, f1, COV + [prev, cg]), LR_MIBS_adj_tx=pcor(mibs, f1, COV + [prev, cg]),
           MA_MMD_adj_tx=pcor(mmd, f2, COV + [prev, cg]), MA_MIDAS_adj_tx=pcor(midas, f2, COV + [prev, cg]), MA_MIBS_adj_tx=pcor(mibs, f2, COV + [prev, cg]),
           R2_LR_midas_mibs_115=r2(f1, [midas, mibs]), R2_LR_mmd_81=r2(f1, [mmd]), R2_LR_midas_mibs_81=r2(f1, [midas.where(d7), mibs.where(d7)]), R2_LR_all_81=r2(f1, [mmd, midas, mibs]),
           R2_MA_midas_mibs_115=r2(f2, [midas, mibs]), R2_MA_all_81=r2(f2, [mmd, midas, mibs]))
NUM['additional'] = ADD

# item subsets
gen = [k for k in range(1, 51) if not re.search(r'migraine|attack', S1[k][3], flags=re.I)]; spe = [k for k in range(1, 51) if k not in gen]
lr_gen = M[[col_of[k] for k in gen]].mean(axis=1) / 6 * 100; lr_spe = M[[col_of[k] for k in spe]].mean(axis=1) / 6 * 100
TC = [72, 74, 79, 80]; ma28 = M[[c for c in f2_cols if item_no[c] not in TC]].mean(axis=1) / 6 * 100
lr48 = M[[c for c in f1_cols if item_no[c] not in (8, 29)]].mean(axis=1) / 6 * 100
SUBS = dict(generic_items=gen, n_generic=len(gen), n_specific=len(spe), r_gen_spe=scor(lr_gen, lr_spe),
            specific={k: pcor(x, lr_spe) for k, x in [('MMD', mmd), ('MIDAS', midas), ('MIBS4', mibs)]},
            generic={k: pcor(x, lr_gen) for k, x in [('MMD', mmd), ('MIDAS', midas), ('MIBS4', mibs)]},
            ma28={k: pcor(x, ma28) for k, x in [('MMD', mmd), ('MIDAS', midas), ('MIBS4', mibs)]}, r_ma28=scor(ma28, f2),
            lr48={k: pcor(x, lr48) for k, x in [('MMD', mmd), ('MIDAS', midas), ('MIBS4', mibs)]}, r_lr48=scor(lr48, f1))
NUM['subsets'] = SUBS


# groups
def mwu(s, gflag):
    a, b = s[gflag == 1].dropna(), s[gflag == 0].dropna()
    return dict(yes=miqr(a), n_yes=len(a), no=miqr(b), n_no=len(b), med_yes=round(float(a.median()), 1), med_no=round(float(b.median()), 1), p=float(stats.mannwhitneyu(a, b).pvalue))
GRP = {f'{gn}_{sn}': mwu(s, gf) for gn, gf in [('prev', prev), ('cgrp', cg)] for sn, s in [('LR', f1), ('MA', f2), ('MMD', mmd), ('MA28', ma28)]}
NUM['groups'] = GRP

# included vs excluded
inc = d7
def cmpc(s):
    a, b = s[inc].dropna(), s[~inc].dropna(); return dict(inc=miqr(a) if s is not midas and s is not mibs and s is not age and s is not srmmd else miqr0(a), n_inc=len(a),
                                                         exc=miqr(b) if s is not midas and s is not mibs and s is not age and s is not srmmd else miqr0(b), n_exc=len(b), p=float(stats.mannwhitneyu(a, b).pvalue))
def cmpb(gf):
    v = gf.notna(); a = gf[inc & v]; b = gf[~inc & v]; t = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
    return dict(inc=f"{t[0][0]}/{len(a)} ({100*a.mean():.0f}%)", exc=f"{t[1][0]}/{len(b)} ({100*b.mean():.0f}%)", p=float(stats.fisher_exact(t)[1]))
INC = dict(age=cmpc(age), women=cmpb(female), srmmd=cmpc(srmmd), midas=cmpc(midas), mibs=cmpc(mibs), LR=cmpc(f1), MA=cmpc(f2),
           prev=cmpb(prev), cgrp=cmpb(cg), drug=cmpb(num('drug_specific')), aura=cmpb(num('aura')))
NUM['included_vs_excluded'] = INC

# MMD strata
mr = mmd.round()
STR = dict(zero=int((mmd == 0).sum()), below1=int((mmd < 1).sum()), le3=int((mr <= 3).sum()), cap28=int((mmd >= 27.999).sum()), strata=[])
for lab, lo_, hi_ in [('0 to 3', -1, 3), ('4 to 7', 4, 7), ('8 to 14', 8, 14), ('15 or more', 15, 99)]:
    gsel = (mr >= lo_) & (mr <= hi_)
    STR['strata'].append(dict(label=lab, n=int(gsel.sum()), LR=miqr(f1[gsel]), LR_range=f"{f1[gsel].min():.0f} to {f1[gsel].max():.0f}", MA=miqr(f2[gsel]), MIDAS=miqr0(midas[gsel]), MIBS4=miqr0(mibs[gsel])))
NUM['mmd_strata'] = STR

# domain correlations with MMD
NUM['domain_mmd'] = {EN[i]: pcor(mmd, num(f'dom{i+1}')) for i in range(11)}

# alternative scores
FS = pd.DataFrame(fa_m.transform(Xd), index=Xd.index)[order_m] * np.array(sg_m); FS.columns = ['A', 'B']
ALT = {}
for fac, idx, unit, col, k in [('LR', F1_IDX, f1, 'A', 0), ('MA', F2_IDX, f2, 'B', 1)]:
    doms = [EN[i] for i in idx]; w = pd.Series(Lm[idx, k], index=doms).clip(lower=0)
    sc = {'Unit-weighted mean of the items (main analysis)': unit, 'Mean of the domain scores': Xd[doms].mean(axis=1),
          'Loading-weighted mean of the domain scores': (Xd[doms] * w).sum(axis=1) / w.sum(), 'Regression-based factor score': FS[col]}
    ALT[fac] = {nm: dict(r=round(scor(unit, s)['r'], 3), MMD=pcor(mmd, s), MIDAS=pcor(midas, s), MIBS4=pcor(mibs, s)) for nm, s in sc.items()}
NUM['alt_scores'] = ALT

# median split (descriptive)
def split(score):
    sub = pd.DataFrame({'m': mmd, 's': score, 'midas': midas, 'lr': f1, 'ma': f2}).dropna(subset=['m', 's'])
    hmv, smv = sub.m.median(), sub.s.median(); sub['hf'] = sub.m > hmv; sub['hs'] = sub.s > smv
    rows = []
    for hf, hs in [(False, True), (True, False), (False, False), (True, True)]:
        x = sub[(sub.hf == hf) & (sub.hs == hs)]; rows.append(dict(hf=hf, hs=hs, n=len(x), midas=miqr0(x.midas), lr=miqr0(x.lr), ma=miqr0(x.ma)))
    return dict(mmd_median=float(hmv), score_median=round(float(smv), 1), rows=rows, ties_mmd=int((sub.m == hmv).sum()), ties_score=int((sub.s == smv).sum()), n=len(sub))
SPL = dict(LR=split(f1), MA=split(f2)); ru = CORU['LR~MMD']['r']
SPL['expected_same_side'] = round(100 * float(0.5 + np.arcsin(ru) / np.pi), 1)
NUM['median_split'] = SPL

# ================================================================ item level
miss = M.isna().mean(); drop_cols = list(miss[miss > 0.05].index); Mi = M.drop(columns=drop_cols).dropna()
fai = fa2(Mi); Li = pd.DataFrame(fai.loadings_, index=Mi.columns, columns=['L1', 'L2'])
if Li.loc[[c for c in f1_cols if c in Li.index], 'L1'].mean() < Li.loc[[c for c in f1_cols if c in Li.index], 'L2'].mean(): Li = Li.rename(columns={'L1': 'L2', 'L2': 'L1'})
for c_ in ['L1', 'L2']:
    own = [c for c in (f1_cols if c_ == 'L1' else f2_cols) if c in Li.index]
    if Li.loc[own, c_].sum() < 0: Li[c_] *= -1
IT = []
for c in s1_cols:
    d = int(c.split('_')[0][1:]); fac_cols = f1_cols if c in f1_cols else f2_cols; fac = 'LR' if c in f1_cols else 'MA'
    rest_dom = M[[x for x in dom_items[d] if x != c]].mean(axis=1); rest_fac = M[[x for x in fac_cols if x != c]].mean(axis=1)
    l1 = float(Li.loc[c, 'L1']) if c in Li.index else np.nan; l2 = float(Li.loc[c, 'L2']) if c in Li.index else np.nan
    v = M[c].dropna()
    IT.append(dict(no=item_no[c], fac=fac, dom=EN[d], l1=l1, l2=l2, own=(l1 if fac == 'LR' else l2), other=(l2 if fac == 'LR' else l1),
                   r_dom=float(pd.concat([M[c], rest_dom], axis=1).dropna().corr().iloc[0, 1]), r_fac=float(pd.concat([M[c], rest_fac], axis=1).dropna().corr().iloc[0, 1]),
                   a_del=float(alpha(M[fac_cols].drop(columns=[c]))), floor=float(100 * (v == 0).mean()), ceil=float(100 * (v == 6).mean())))
ITd = pd.DataFrame(IT); own = ITd.own.dropna()
Rl = M[f1_cols].corr(); pairs_lr = sorted([(item_no[a], item_no[b], float(Rl.loc[a, b])) for i, a in enumerate(f1_cols) for b in f1_cols[i + 1:] if Rl.loc[a, b] >= 0.80], key=lambda t: -t[2])
Rm_ = M[f2_cols].corr(); pairs_ma = sorted([(item_no[a], item_no[b], float(Rm_.loc[a, b])) for i, a in enumerate(f2_cols) for b in f2_cols[i + 1:] if Rm_.loc[a, b] >= 0.80], key=lambda t: -t[2])
UNI = []
r_ = np.random.default_rng(SEED)
for i in S1_ORDER:
    cols = dom_items[i]; X = M[cols].dropna(); k = len(cols)
    if k < 3: UNI.append(dict(dom=EN[i], k=k, n=len(X), r=float(X.corr().iloc[0, 1]))); continue
    Rd = X.corr().values; ev = np.sort(np.linalg.eigvalsh(Rd))[::-1]
    re_ = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(r_.normal(size=(len(X), k)), rowvar=False)))[::-1] for _ in range(500)]); pa = np.percentile(re_, 95, axis=0)
    f1_ = FactorAnalyzer(n_factors=1, method='minres', rotation=None); f1_.fit(X); l1 = np.abs(f1_.loadings_[:, 0])
    UNI.append(dict(dom=EN[i], k=k, n=len(X), mean_r=float(Rd[np.triu_indices(k, 1)].mean()), alpha=float(alpha(X)), omega=float(omega(X)), ev1=float(ev[0]), ev2=float(ev[1]), pa2=float(pa[1]),
                    second=bool(ev[1] > pa[1]), min_load=float(l1.min()), min_item=item_no[cols[int(np.argmin(l1))]]))
NUM['items'] = dict(n=len(Mi), k=Mi.shape[1], kmo=round(float(calculate_kmo(Mi)[1]), 2), median_own=round(float(own.median()), 2), min_own=round(float(own.min()), 2), max_own=round(float(own.max()), 2),
                    n_below_040=int((own < 0.40).sum()), below_040=[int(x) for x in ITd[ITd.own < 0.40].no], higher_on_other=[int(x) for x in ITd[ITd.other.abs() > ITd.own.abs()].no],
                    n_itemrest_below_030=int((ITd.r_fac < 0.30).sum()), itemrest_below_030=[int(x) for x in ITd[ITd.r_fac < 0.30].no],
                    floor15=int((ITd.floor > 15).sum()), ceil15=int((ITd.ceil > 15).sum()), pairs_lr=len(pairs_lr), pairs_lr_same_domain=sum(1 for a, b, r in pairs_lr if s1_cols[a - 1].split('_')[0] == s1_cols[b - 1].split('_')[0]),
                    pairs_ma=len(pairs_ma), not_unidimensional=[u['dom'] for u in UNI if u.get('second')], item9_own=round(float(ITd[ITd.no == 9].own.iloc[0]), 2))

json.dump(NUM, open('jhp_r1_numbers.json', 'w'), indent=1, ensure_ascii=False)

# ================================================================ Additional file 2 (HTML)
PRINT = ("<style>@page{size:A4 landscape;margin:12mm}body{color:#000;font-family:'Helvetica Neue',Arial,'Hiragino Sans','Hiragino Kaku Gothic ProN',sans-serif}"
         "h2{color:#000}p.note{color:#222}th{background:none}tr{page-break-inside:avoid}h1,h2{page-break-after:avoid}</style>")
h(CSS + PRINT)
h("<h1>Additional file 2. Supplementary Tables S1 to S4</h1><p>Item generation and preliminary psychometric evaluation of the Migraine Freedom Index: an exploratory study linked to a headache-diary application (n=115)</p>")
h(f"<p class='note'>The questionnaire was developed and administered in Japanese. The English wording of the items is a provisional translation by the study team; a formal forward and back translation and cognitive debriefing have not been done. The items did not state a recall period. All items used a 7-point agreement scale (0 = strongly disagree, 6 = strongly agree) and were phrased positively, with no reverse-scored items. For Life Restoration, a higher score indicates that more of daily life is restored. For Migraine Agency, a higher score indicates more awareness and management of migraine and does not necessarily indicate a better state. Each domain score and each factor score is the mean of the answered items rescaled to a 0-100 scale, and required at least 75% of its items to be answered. All {N} participants met this rule. All partial Spearman correlations are adjusted for age and sex (female or not), with 95% confidence intervals from the Fisher z transformation (standard error 1/√(n − 3 − k), where k is the number of covariates).</p>")

# ---- S1
fl = [100 * (M[col_of[k]].dropna() == 0).mean() for k in range(1, 83)]; ce = [100 * (M[col_of[k]].dropna() == 6).mean() for k in range(1, 83)]
h("<h1>Supplementary Table S1. The 82 MFI items: domain, English wording, Japanese original, and floor and ceiling</h1>")
h(f"<p class='note'>Floor = percentage of respondents answering 0. Ceiling = percentage answering 6. The median floor was {np.median(fl):.1f}% (maximum {max(fl):.1f}%) and the median ceiling was {np.median(ce):.1f}% (maximum {max(ce):.1f}%). {sum(1 for x in fl if x > 15)} items exceeded 15% at the floor and {sum(1 for x in ce if x > 15)} exceeded 15% at the ceiling. Item 65 was shown only to participants who reported female sex ({NUM['sample']['item65_missing']} participants did not receive it); the trigger recognition domain score of these participants is the mean of the other seven items. The items and the scoring rule (Additional file 1: SM4) are licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International licence (https://creativecommons.org/licenses/by-nc-nd/4.0/). In addition, they may be used free of charge in clinical practice. Commercial use, translation, or modification requires permission from the corresponding authors.</p>")
table(["#", "Factor", "Domain", "Item (English, provisional)", "Japanese original", "Floor %", "Ceiling %"],
      [[k, 'Life Restoration' if k <= 50 else 'Migraine Agency', S1[k][2], S1[k][3], S1[k][4], f"{fl[k-1]:.1f}", f"{ce[k-1]:.1f}"] for k in range(1, 83)], numcols=(0, 5, 6))
h("<p class='note'>Migraine-specific acute drugs in the definition of a diary-based migraine day: triptans, lasmiditan, and rimegepant. The treatment classification also counted monoclonal antibodies against CGRP or its receptor and atogepant. Other preventive drugs (for example lomerizine, valproate, amitriptyline, and propranolol) are also used for other conditions and were not counted as migraine-specific.</p>")

# ---- S2
E = NUM['efa']
h("<h1>Supplementary Table S2. Exploratory factor analysis of the 11 domain scores</h1>")
h("<h2>S2a. Two-factor pattern matrix (minimum-residual extraction, oblimin rotation), communalities, and bootstrap intervals</h2>")
h(f"<p class='note'>Kaiser-Meyer-Olkin measure {E['kmo']:.2f}; Bartlett's test χ²={E['bartlett_chi']:.1f}, p&lt;0.001. Parallel analysis of the eigenvalues of the correlation matrix (1,000 random datasets, 95th percentile) retained {E['keep_component']} factors: observed eigenvalues {E['eig'][0]:.2f}, {E['eig'][1]:.2f}, and {E['eig'][2]:.2f} against random values {E['pa_component'][0]:.2f}, {E['pa_component'][1]:.2f}, and {E['pa_component'][2]:.2f}. Parallel analysis of the eigenvalues of the reduced correlation matrix (squared multiple correlations on the diagonal) retained {E['keep_factor']} factors: observed eigenvalues {', '.join(f'{x:.2f}' for x in E['eig_reduced'][:5])} against random values {', '.join(f'{x:.2f}' for x in E['pa_factor'][:5])}; the third and fourth observed eigenvalues were close to the random values, and in a three-factor solution the third factor was defined by the social freedom domain alone (loading {abs(L3[8]).max():.2f}). The two-factor solution explained {E['var_total']:.1f}% of the variance ({E['var_each'][0]:.1f}% and {E['var_each'][1]:.1f}%). The correlation between the two factors was {E['phi']:+.2f}, and the correlation between the two unit-weighted factor scores was {E['score_r']:+.2f}. Bootstrap intervals are the 2.5th to 97.5th percentiles of {nboot} resamples.</p>")
table(["Domain", "Life Restoration", "Migraine Agency", "Communality", "Life Restoration, bootstrap 95% interval", "Migraine Agency, bootstrap 95% interval", "Assigned factor"],
      [[EN[i], f"{Lm[i,0]:+.2f}", f"{Lm[i,1]:+.2f}", f"{hm[i]:.2f}", f"{blo[i,0]:+.2f} to {bhi[i,0]:+.2f}", f"{blo[i,1]:+.2f} to {bhi[i,1]:+.2f}", 'Life Restoration' if i in F1_IDX else 'Migraine Agency'] for i in S1_ORDER], numcols=(1, 2, 3))
h("<h2>S2b. Two-factor pattern matrix with principal-component extraction (oblimin rotation), as reported in the first submission</h2>")
h(f"<p class='note'>The first submission described this extraction as principal-axis factoring. It was a principal-component extraction, which explained {E['component']['var_total']:.1f}% of the variance (correlation between the two components {E['component']['phi']:+.2f}). It is shown for comparison with Table S2a.</p>")
table(["Domain", "Component 1", "Component 2", "Communality"], [[EN[i], f"{Lc[i,0]:+.2f}", f"{Lc[i,1]:+.2f}", f"{hc[i]:.2f}"] for i in S1_ORDER], numcols=(1, 2, 3))
h("<h2>S2c. Correlations between the domain scores (Pearson)</h2>")
table(["&nbsp;"] + [f"D{k+1}" for k in range(11)], [[f"D{k+1} {EN[i]}"] + [f"{R[i,j]:+.2f}" for j in S1_ORDER] for k, i in enumerate(S1_ORDER)], numcols=tuple(range(1, 12)))
h("<h2>S2d. Distribution of the factor scores (0-100)</h2>")
table(["Factor score", "Mean (SD)", "Median (IQR)", "Minimum to maximum", "Skewness", "Participants at 0 or 100"],
      [[nm, f"{s.mean():.1f} ({s.std():.1f})", miqr(s), f"{s.min():.1f} to {s.max():.1f}", f"{stats.skew(s):.2f}", int(((s == 0) | (s == 100)).sum())] for nm, s in [("Life Restoration", f1), ("Migraine Agency", f2)]], numcols=(1, 2, 3, 4, 5))

# ---- S3
Fn = NUM['funnel']
h("<h1>Supplementary Table S3. Participant flow, correlations, sensitivity analyses, and exploratory analyses</h1>")
h("<h2>S3a. Participant flow</h2>")
table(["Stage", "n", "Note"], [
    ["MFI questionnaire offered in the app", f"{Fn['offered']:,}", "At the close of recruitment (18 July 2026). The app did not log whether a user saw the offer."],
    ["Entered the consent flow", Fn['entered'], f"{100*Fn['entered']/Fn['offered']:.0f}% of those offered"],
    ["Completed the MFI questionnaire", Fn['completed'], f"{Fn['pct_of_entered']:.0f}% of those who entered the consent flow; {Fn['pct_of_offered']:.0f}% of those offered"],
    ["Discontinued", Fn['discontinued'], f"{Fn['no_items']} answered no items; {Fn['some_items']} answered at least one item"],
    ["Deleted the account before the diary extraction (15 August 2026)", Fn['deleted'], f"{Fn['deleted_completed']} participant who had completed the questionnaire and {Fn['deleted'] - Fn['deleted_completed']} users who had not entered the consent flow"],
    ["Analysed", Fn['analysed'], ""]], numcols=(1,))
lastpos = pd.to_numeric(ip.loc[pd.to_numeric(ip['n_answered'], errors='coerce').fillna(0) > 0, 'last_order_index'], errors='coerce').dropna()
age_c = pd.to_numeric(comp_f['age'], errors='coerce').dropna(); age_i = pd.to_numeric(ip['age'], errors='coerce').dropna()
h(f"<p class='note'>The counts of users offered, users who entered the consent flow, and participants who completed the questionnaire were recorded at the close of recruitment. The other analyses in this table used the extraction of 15 August 2026. Among the {Fn['some_items']} participants who answered at least one item and then stopped, the median position of the last answered item was {np.median(lastpos):.0f} (IQR {np.percentile(lastpos,25):.0f} to {np.percentile(lastpos,75):.0f}). The {len(comp_f)} analysed participants and the {len(ip)} participants who discontinued had a similar age (mean {age_c.mean():.1f} versus {age_i.mean():.1f} years; Mann-Whitney p={stats.mannwhitneyu(age_c, age_i).pvalue:.2f}). In the account profiles, {100*(comp_f['sex']=='female').mean():.0f}% and {100*(ip['sex']=='female').mean():.0f}% were women; the sex reported in the questionnaire was used for Table 1 and for all adjusted analyses.</p>")

LAB = {'LR': 'Life Restoration score', 'MA': 'Migraine Agency score', 'MMD': 'Diary-based MMD', 'srMMD': 'Self-reported MMD', 'MIDAS': 'MIDAS score', 'MIBS4': 'MIBS-4 score'}
def mat(Cd, title, note):
    h(f"<h2>{title}</h2>"); h(f"<p class='note'>{note}</p>")
    rows = []
    for a in keys:
        row = [LAB[a]]
        for b in keys:
            if a == b: row.append('—'); continue
            d = Cd.get(f'{a}~{b}') or Cd.get(f'{b}~{a}'); row.append(f"{ci(d)}<br>n={d['n']}")
        rows.append(row)
    table(["&nbsp;"] + [LAB[k] for k in keys], rows, numcols=tuple(range(1, 7)))
mat(COR, "S3b. Partial Spearman correlations among the MFI factor scores and the other measures, adjusted for age and sex (ρ [95% CI])",
    f"Diary-based MMD was available in {NUM['sample']['n81']} participants with at least seven recorded days. Self-reported MMD was the number of migraine days in the previous 4 weeks reported in the questionnaire ({NUM['sample']['srmmd_valid']} participants; two answers above 28 were excluded).")
mat(CORU, "S3c. Unadjusted Spearman correlations among the MFI factor scores and the other measures (ρ [95% CI])", "The same participants as in Table S3b.")

h("<h2>S3d. Sensitivity analyses of the correlations with diary-based MMD (partial Spearman ρ [95% CI], adjusted for age and sex)</h2>")
S = SENS
rows = [[("Diary-based MMD, at least 7 recorded days (main analysis)" if t == 7 else f"Diary-based MMD, at least {t} recorded days" if t < 28 else "Diary-based MMD, all 28 days recorded"), S[f'ge{t}']['LR']['n'], ci(S[f'ge{t}']['LR']), ci(S[f'ge{t}']['MA'])] for t in [7, 14, 21, 28]]
rows += [["At least 7 recorded days, with additional adjustment for the number of recorded days", S['adj_recorded']['LR']['n'], ci(S['adj_recorded']['LR']), ci(S['adj_recorded']['MA'])],
         ["Raw number of migraine days instead of diary-based MMD, at least 7 recorded days", S['raw_days']['LR']['n'], ci(S['raw_days']['LR']), ci(S['raw_days']['MA'])],
         [f"At least 7 recorded days, excluding the {S['ge7']['LR']['n'] - S['excl_no_migraine']['LR']['n']} participants in whom neither classification identified migraine", S['excl_no_migraine']['LR']['n'], ci(S['excl_no_migraine']['LR']), ci(S['excl_no_migraine']['MA'])],
         [f"At least 7 recorded days, excluding the participant with the lowest Life Restoration score (score {S['excl_lowest_LR']['lr']:.0f}, diary-based MMD {S['excl_lowest_LR']['mmd']:.0f})", S['excl_lowest_LR']['LR']['n'], ci(S['excl_lowest_LR']['LR']), ci(S['excl_lowest_LR']['MA'])],
         ["Diary-based proportion of headache days (the day-count measure in the dated analysis plan), at least 7 recorded days", S['haprop']['LR']['n'], ci(S['haprop']['LR']), ci(S['haprop']['MA'])],
         ["Self-reported MMD (questionnaire)", COR['LR~srMMD']['n'], ci(COR['LR~srMMD']), ci(COR['MA~srMMD'])],
         ["Number of recorded days (at least 1 recorded day) with each factor score", S['rec_vs_LR']['n'], ci(S['rec_vs_LR']), ci(S['rec_vs_MA'])]]
table(["Analysis", "n", "Life Restoration score", "Migraine Agency score"], rows, numcols=(1, 2, 3))
h(f"<p class='note'>The number of recorded days correlated with the proportion of headache days at {ci(S['rec_vs_haprop'])} and with the proportion of migraine days at {ci(S['rec_vs_migprop'])} (unadjusted Spearman, n={S['rec_vs_haprop']['n']}).</p>")
h("<h2>S3e. Poisson regression of the number of migraine days, with the logarithm of the number of recorded days as an offset</h2>")
h("<p class='note'>Each model: number of migraine days in the 28-day window ~ one measure + age + sex, offset = log(number of recorded days), robust standard errors. IRR = incidence rate ratio.</p>")
PL = {'f1': 'Life Restoration score (per 10 points)', 'f2': 'Migraine Agency score (per 10 points)', 'midas': 'MIDAS score (per 10 points)', 'mibs': 'MIBS-4 score (per 1 point)'}
table(["Measure", "Recorded days", "n", "IRR [95% CI]", "Dispersion ratio"],
      [[PL[fac], f"at least {t}", POIS[f'{fac}_ge{t}']['n'], f"{POIS[f'{fac}_ge{t}']['irr']:.2f} [{POIS[f'{fac}_ge{t}']['lo']:.2f}, {POIS[f'{fac}_ge{t}']['hi']:.2f}]", f"{POIS[f'{fac}_ge{t}']['disp']:.2f}"] for fac in ['f1', 'f2', 'midas', 'mibs'] for t in [7, 14, 21]], numcols=(2, 3, 4))

h("<h2>S3f. Participants included in and excluded from the diary-based MMD analyses</h2>")
h("<p class='note'>Included = at least seven recorded days in the 28-day window. Values are median (IQR) with the Mann-Whitney U test, or n/N (%) with Fisher's exact test.</p>")
IL = [('age', 'Age, years'), ('women', 'Women'), ('srmmd', 'Self-reported MMD'), ('midas', 'MIDAS score'), ('mibs', 'MIBS-4 score'), ('LR', 'Life Restoration score'), ('MA', 'Migraine Agency score'),
      ('prev', 'Current preventive medication'), ('cgrp', 'Current CGRP-related drug'), ('drug', 'Migraine-specific drug'), ('aura', 'Aura')]
table(["Characteristic", f"Included (n={int(inc.sum())})", f"Excluded (n={int((~inc).sum())})", "p"], [[lab, INC[k]['inc'], INC[k]['exc'], pfmt(INC[k]['p'])] for k, lab in IL], numcols=(1, 2, 3))

h("<h2>S3g. MFI factor scores by diary-based MMD</h2>")
h(f"<p class='note'>Diary-based MMD was rounded to the nearest day. Diary-based MMD was 0 in {STR['zero']} participants and below 1 in {STR['below1']} participants. Values are median (IQR).</p>")
table(["Diary-based MMD", "n", "Life Restoration score", "Range of the Life Restoration score", "Migraine Agency score", "MIDAS score", "MIBS-4 score"],
      [[s_['label'], s_['n'], s_['LR'], s_['LR_range'], s_['MA'], s_['MIDAS'], s_['MIBS4']] for s_ in STR['strata']], numcols=(1, 2, 3, 4, 5, 6))

h("<h2>S3h. Exploratory comparisons by current preventive medication and by current CGRP-related drug (median (IQR))</h2>")
h(f"<p class='note'>{N - NUM['sample']['prev_n']} participants did not report whether they used a preventive medication, and {N - NUM['sample']['cgrp_n']} did not report whether they used a CGRP-related drug. Diary-based MMD was available in participants with at least seven recorded days. Migraine Agency (28 items) omits items 72, 74, 79, and 80, which refer to preventive treatment or to consultation with a doctor.</p>")
rows = []
for gn, glab in [('prev', 'Preventive medication'), ('cgrp', 'CGRP-related drug')]:
    for sn, slab in [('LR', 'Life Restoration score'), ('MA', 'Migraine Agency score'), ('MA28', 'Migraine Agency score (28 items)'), ('MMD', 'Diary-based MMD')]:
        g_ = GRP[f'{gn}_{sn}']; rows.append([glab, slab, f"{g_['yes']} (n={g_['n_yes']})", f"{g_['no']} (n={g_['n_no']})", pfmt(g_['p'])])
table(["Group", "Measure", "Yes", "No", "Mann-Whitney p"], rows, numcols=(2, 3, 4))

h("<h2>S3i. Correlations of the 11 domain scores with diary-based MMD (partial Spearman ρ [95% CI], adjusted for age and sex; n=81)</h2>")
table(["Domain", "Factor", "ρ [95% CI]"], [[EN[i], 'Life Restoration' if i in F1_IDX else 'Migraine Agency', ci(NUM['domain_mmd'][EN[i]])] for i in S1_ORDER], numcols=(2,))

h("<h2>S3j. Alternative scores and item subsets (partial Spearman ρ [95% CI], adjusted for age and sex)</h2>")
h(f"<p class='note'>ρ with the main score = Spearman correlation with the unit-weighted score of the same factor. Life Restoration (48 items) omits items 8 and 29. The Life Restoration items were also divided by wording into the {SUBS['n_specific']} items that contain the word migraine or attack and the {SUBS['n_generic']} items that do not (items {', '.join(str(x) for x in gen)}); these two scores correlated at {SUBS['r_gen_spe']['r']:+.2f}.</p>")
rows = []
for fac, flab in [('LR', 'Life Restoration'), ('MA', 'Migraine Agency')]:
    for nm, a_ in ALT[fac].items(): rows.append([flab, nm, f"{a_['r']:.3f}", ci(a_['MMD']), ci(a_['MIDAS']), ci(a_['MIBS4'])])
    if fac == 'LR':
        rows.append([flab, "48 items (without items 8 and 29)", f"{SUBS['r_lr48']['r']:.3f}"] + [ci(SUBS['lr48'][k]) for k in ['MMD', 'MIDAS', 'MIBS4']])
        rows.append([flab, f"{SUBS['n_specific']} items that contain the word migraine or attack", f"{scor(lr_spe, f1)['r']:.3f}"] + [ci(SUBS['specific'][k]) for k in ['MMD', 'MIDAS', 'MIBS4']])
        rows.append([flab, f"{SUBS['n_generic']} items that do not contain these words", f"{scor(lr_gen, f1)['r']:.3f}"] + [ci(SUBS['generic'][k]) for k in ['MMD', 'MIDAS', 'MIBS4']])
    else:
        rows.append([flab, "28 items (without items 72, 74, 79, and 80)", f"{SUBS['r_ma28']['r']:.3f}"] + [ci(SUBS['ma28'][k]) for k in ['MMD', 'MIDAS', 'MIBS4']])
table(["Factor", "Score", "ρ with the main score", "Diary-based MMD (n=81)", "MIDAS score (n=115)", "MIBS-4 score (n=115)"], rows, numcols=(2, 3, 4, 5))

h("<h2>S3k. Additional adjustment (exploratory)</h2>")
A = ADD
h(f"<p class='note'>Partial Spearman correlations with additional covariates. In a regression of the ranks, the MIDAS score and the MIBS-4 score together explained {100*A['R2_LR_midas_mibs_115'][0]:.0f}% of the variance of the Life Restoration score (n={A['R2_LR_midas_mibs_115'][1]}) and {100*A['R2_MA_midas_mibs_115'][0]:.0f}% of the variance of the Migraine Agency score. In the {A['R2_LR_all_81'][1]} participants with diary-based MMD, diary-based MMD alone explained {100*A['R2_LR_mmd_81'][0]:.0f}% of the variance of the Life Restoration score, the MIDAS score and the MIBS-4 score together explained {100*A['R2_LR_midas_mibs_81'][0]:.0f}%, and the three measures together explained {100*A['R2_LR_all_81'][0]:.0f}%.</p>")
rows = [["Life Restoration score and diary-based MMD", "age, sex, MIDAS score, MIBS-4 score", A['LR_MMD_adj_midas_mibs']['n'], ci(A['LR_MMD_adj_midas_mibs'])],
        ["Life Restoration score and MIDAS score", "age, sex, diary-based MMD, MIBS-4 score", A['LR_MIDAS_adj_mmd_mibs']['n'], ci(A['LR_MIDAS_adj_mmd_mibs'])],
        ["Life Restoration score and MIBS-4 score", "age, sex, diary-based MMD, MIDAS score", A['LR_MIBS_adj_mmd_midas']['n'], ci(A['LR_MIBS_adj_mmd_midas'])]]
for fl_, lab in [('LR', 'Life Restoration score'), ('MA', 'Migraine Agency score')]:
    for k_, ml in [('MMD', 'diary-based MMD'), ('MIDAS', 'MIDAS score'), ('MIBS', 'MIBS-4 score')]:
        d_ = A[f'{fl_}_{k_}_adj_tx']; rows.append([f"{lab} and {ml}", "age, sex, preventive medication, CGRP-related drug", d_['n'], ci(d_)])
table(["Correlation", "Adjusted for", "n", "ρ [95% CI]"], rows, numcols=(2, 3))

# ---- S4
I = NUM['items']
h("<h1>Supplementary Table S4. Item-level analysis (exploratory) and unidimensionality of each domain</h1>")
h("<h2>S4a. Item loadings from a two-factor analysis of the items (minimum-residual extraction, oblimin rotation), corrected item-rest correlations, and floor and ceiling</h2>")
h(f"<p class='note'>The item-level factor analysis used {I['n']} participants and {I['k']} items; item 65 was omitted because it was shown only to participants who reported female sex. The participant-to-item ratio was {I['n']/I['k']:.1f}, so this analysis is unstable and is used only to describe the items (Kaiser-Meyer-Olkin measure {I['kmo']:.2f}). Item-rest (domain) = correlation of the item with the mean of the other items of its domain. Item-rest (factor) = correlation with the mean of the other items of its factor. Alpha if deleted = Cronbach's alpha of the factor without the item (Life Restoration {alpha(M[f1_cols]):.3f}; Migraine Agency {alpha(M[f2_cols]):.3f}). A loading below 0.40 on the item's own factor is marked with an asterisk.</p>")
rows = []
for r_ in IT:
    fm = lambda x: 'n.a.' if np.isnan(x) else f"{x:+.2f}"
    star = '*' if (not np.isnan(r_['own']) and r_['own'] < 0.40) else ''
    rows.append([r_['no'], 'Life Restoration' if r_['fac'] == 'LR' else 'Migraine Agency', r_['dom'], S1[r_['no']][3], fm(r_['l1']) + (star if r_['fac'] == 'LR' else ''), fm(r_['l2']) + (star if r_['fac'] == 'MA' else ''),
                 f"{r_['r_dom']:.2f}", f"{r_['r_fac']:.2f}", f"{r_['a_del']:.3f}", f"{r_['floor']:.1f}", f"{r_['ceil']:.1f}"])
table(["#", "Factor", "Domain", "Item (English, provisional)", "Loading, Life Restoration", "Loading, Migraine Agency", "Item-rest (domain)", "Item-rest (factor)", "Alpha if deleted", "Floor %", "Ceiling %"], rows, numcols=(0, 4, 5, 6, 7, 8, 9, 10))
h(f"<p class='note'>The median loading of an item on its own factor was {I['median_own']:.2f} (range {I['min_own']:.2f} to {I['max_own']:.2f}). {I['n_below_040']} items loaded below 0.40 on their own factor, and {len(I['higher_on_other'])} items (items {', '.join(str(x) for x in I['higher_on_other'])}) loaded more strongly on the other factor. The item-rest correlation with the factor was below 0.30 for {I['n_itemrest_below_030']} items.</p>")
h("<h2>S4b. Item pairs within a factor with a correlation of 0.80 or more (Pearson)</h2>")
table(["Factor", "Item pair", "r", "Items"], [["Life Restoration", f"{a} and {b}", f"{r:.2f}", f"{S1[a][3]} / {S1[b][3]}"] for a, b, r in pairs_lr] + [["Migraine Agency", f"{a} and {b}", f"{r:.2f}", f"{S1[a][3]} / {S1[b][3]}"] for a, b, r in pairs_ma], numcols=(2,))
h("<h2>S4c. Each domain: mean inter-item correlation, Cronbach's alpha, McDonald's omega, first two eigenvalues, parallel analysis, and the smallest single-factor loading</h2>")
h("<p class='note'>Parallel analysis used 500 random datasets and the 95th percentile. A second factor is retained when the second observed eigenvalue exceeds the random value. McDonald's omega and the single-factor loadings are from a one-factor solution with minimum-residual extraction. The global freedom evaluation domain has two items, so only the correlation between them is reported.</p>")
rows = []
for u in UNI:
    if u['k'] < 3: rows.append([u['dom'], u['k'], u['n'], f"{u['r']:.2f}", f"{REL[u['dom']]['alpha']:.2f}", 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.']); continue
    rows.append([u['dom'], u['k'], u['n'], f"{u['mean_r']:.2f}", f"{u['alpha']:.2f}", f"{u['omega']:.2f}", f"{u['ev1']:.2f}, {u['ev2']:.2f}", f"{u['pa2']:.2f}", 'retained' if u['second'] else 'not retained', f"{u['min_load']:.2f} (item {u['min_item']})"])
table(["Domain", "Items", "n", "Mean inter-item r", "Cronbach's alpha", "McDonald's omega", "Eigenvalues 1 and 2", "Random value for eigenvalue 2", "Second factor", "Smallest loading"], rows, numcols=(1, 2, 3, 4, 5, 6, 7))
open('mfi_supp_jhp_r1.html', 'w', encoding='utf-8').write(re.sub(r'(?<![\d.])[+-]0\.00(?!\d)', '0.00', "\n".join(H)))

# ================================================================ figures
DPI = int(os.environ.get("FIG_DPI", "150"))
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.8, "savefig.dpi": DPI, "figure.dpi": 110})
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#666666"
# Figure 1
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(6.8, 7.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 11)
GREEN = "#009E73"
def box(x, y, w, hgt, txt, fc="#EAF2F8", ec=BLUE, fs=8.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - hgt / 2), w, hgt, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.05, color=GRAY))
Fn = NUM['funnel']
box(3.2, 10.3, 4.8, 0.95, f"MFI questionnaire offered in the app\nN = {Fn['offered']:,}\n(views of the offer not logged)")
box(3.2, 8.7, 4.8, 0.7, f"Entered the consent flow\nn = {Fn['entered']}")
box(3.2, 7.1, 4.8, 0.7, f"Completed the MFI questionnaire\nn = {Fn['completed']}")
box(3.2, 5.5, 4.8, 0.7, f"Analysed\nn = {Fn['analysed']}")
box(8.05, 7.9, 3.5, 1.15, f"Discontinued n = {Fn['discontinued']}\n- answered no items: {Fn['no_items']}\n- answered ≥1 item: {Fn['some_items']}", fc="#FDF2E9", ec=ORANGE, fs=8.0)
box(8.05, 6.3, 3.5, 0.95, f"Deleted the account before\nthe diary extraction\nn = {Fn['deleted_completed']}", fc="#FDF2E9", ec=ORANGE, fs=8.0)
arrow(3.2, 9.82, 3.2, 9.05); arrow(3.2, 8.35, 3.2, 7.45); arrow(3.2, 6.75, 3.2, 5.85)
arrow(3.2, 7.9, 6.3, 7.9); arrow(3.2, 6.3, 6.3, 6.3)
box(3.2, 4.0, 4.4, 0.7, f"≥1 recorded diary day\nn = {NUM['sample']['n_ge1']}", fc="#E8F6F0", ec=GREEN, fs=8.0)
box(3.2, 2.6, 4.4, 0.7, f"≥7 recorded days (diary-based MMD analysis)\nn = {NUM['sample']['n81']}", fc="#E8F6F0", ec=GREEN, fs=8.0)
arrow(3.2, 5.15, 3.2, 4.35); arrow(3.2, 3.65, 3.2, 2.95)
ax.text(8.05, 3.3, "Recruitment closed 18 July 2026\nDiary extracted 15 August 2026", ha="center", fontsize=7.6, style="italic", color=GRAY)
plt.tight_layout(); plt.savefig("fig1_flow_r1.png", bbox_inches="tight"); plt.close()

# Figure 2
order = sorted(range(11), key=lambda i: (0 if i in F1_IDX else 1, -(Lm[i, 0] if i in F1_IDX else Lm[i, 1])))
fig, ax = plt.subplots(figsize=(7.8, 6.4)); yy = np.arange(11)[::-1]; bw = 0.38
for k, i in enumerate(order):
    y = yy[k]
    for f_, col, off in [(0, BLUE, +bw / 2), (1, ORANGE, -bw / 2)]:
        v = float(Lm[i, f_]); ax.barh(y + off, v, height=bw, color=col, edgecolor="white", zorder=2)
        ax.errorbar(v, y + off, xerr=[[max(v - blo[i, f_], 0)], [max(bhi[i, f_] - v, 0)]], fmt='none', ecolor="#333333", elinewidth=0.8, capsize=2, zorder=3)
        lx, ha = (max(bhi[i, f_], v) + 0.03, 'left') if v >= 0 else (min(blo[i, f_], v) - 0.03, 'right')
        ax.text(lx, y + off, f"{v:+.2f}", va="center", ha=ha, fontsize=6.6, color="#333333")
ax.set_yticks(yy); ax.set_yticklabels([EN[i] for i in order], fontsize=8.4); ax.axvline(0, color="k", lw=0.9, zorder=1)
ax.axhline(yy[len(F1_IDX) - 1] - 0.5, color=GRAY, lw=0.8, ls=":"); ax.set_xlim(-0.75, 1.15); ax.set_ylim(-0.7, 10.7)
ax.set_xlabel("Pattern loading on each factor, with bootstrap 95% interval")
ax.legend(handles=[Patch(color=BLUE, label="Factor 1: Life Restoration"), Patch(color=ORANGE, label="Factor 2: Migraine Agency")], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False); plt.tight_layout(); plt.savefig("fig2_loadings.png", bbox_inches="tight"); plt.close()

def lowess(x, y, frac=0.75, grid=None):
    grid = np.linspace(x.min(), x.max(), 120) if grid is None else grid; k = int(np.ceil(frac * len(x))); out = []
    for g0 in grid:
        d = np.abs(x - g0); hh = np.sort(d)[k - 1]; w = np.clip(1 - (d / max(hh, 1e-9)) ** 3, 0, None) ** 3
        Xw = np.column_stack([np.ones(len(x)), x - g0]); W = Xw.T * w; beta = np.linalg.solve(W @ Xw + 1e-9 * np.eye(2), W @ y); out.append(beta[0])
    return grid, np.array(out)
def panel3(ax, s, label, letter, key):
    m = mmd.notna() & s.notna(); x = mmd[m].values.astype(float); y = s[m].values.astype(float)
    ax.scatter(x, y, s=27, color=GRAY, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=2)
    gx, gy = lowess(x, y); ax.plot(gx, gy, color=BLUE, lw=1.9, zorder=3, label="Locally weighted smoothing (LOWESS)")
    d = COR[key]; ax.text(0.03, 0.05, f"Partial Spearman ρ = {sg(d['r'])} [{sg(d['lo'])} to {sg(d['hi'])}]\n(adjusted for age and sex; n = {d['n']})", transform=ax.transAxes, fontsize=7.8, bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))
    ax.set_xlabel("Diary-based MMD (days)"); ax.set_ylabel(f"{label} score (0-100)"); ax.set_xlim(-0.8, 29); ax.set_ylim(0, 100)
    ax.set_title(f"{letter}  {label} score", fontsize=9.2, loc="left", weight="bold"); ax.spines[['top', 'right']].set_visible(False)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.5)); panel3(axA, f1, 'Life Restoration', 'A', 'LR~MMD'); panel3(axB, f2, 'Migraine Agency', 'B', 'MA~MMD')
axA.legend(loc="upper right", fontsize=8, frameon=False); plt.tight_layout(); plt.savefig("fig3_scatter.png", bbox_inches="tight"); plt.close()

print("wrote jhp_r1_numbers.json, mfi_supp_jhp_r1.html, fig1_flow_r1.png, fig2_loadings.png, fig3_scatter.png")
