#!/usr/bin/env python3
"""MFI 査読対応 Table 1-4 → HTML（Google Docs で罫線付きテーブルに変換される）。
n=116 凍結。N が変われば再実行で作り直せる。出力: mfi_tables.html
"""
import numpy as np, pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy import stats
np.random.seed(20260718)
D = pd.read_csv('mfi_matrix.csv'); N = len(D)
num = lambda c: pd.to_numeric(D[c], errors='coerce')
DOMS = ['脳状態遷移','トリガー認識','先制行動','発作コントロール','MOH予防','活動の自由','希望と治療継続','人生回復','社会的自由','身近な人への影響','総合評価']
DOM_EN = ['Premonitory symptom awareness','Trigger recognition','Pre-emptive action','Attack control','Prevention of medication overuse','Freedom of activity','Hope and treatment continuity','Life restoration','Social freedom','Impact on close others','Global freedom evaluation']
FAC = ['Agency','Agency','Agency','Restoration','Agency','Restoration','Restoration','Restoration','Restoration','Restoration','Restoration']
qcols = [c for c in D.columns if c.startswith('q') and '_' in c and c[1:].split('_')[0].isdigit()]

def alpha(items):
    df = D[items].apply(pd.to_numeric, errors='coerce').dropna(); k = df.shape[1]
    iv = df.var(axis=0, ddof=1).sum(); tv = df.sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1-iv/tv)
def omega(items):
    df = D[items].apply(pd.to_numeric, errors='coerce').dropna()
    fa = FactorAnalyzer(n_factors=1, method='principal', rotation=None); fa.fit(df)
    lam = fa.loadings_[:,0]; sl = lam.sum()
    return (sl**2)/(sl**2+(1-lam**2).sum())
def sp(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]; n=len(x)
    r,p=stats.spearmanr(x,y); z=np.arctanh(r); se=1/np.sqrt(n-3)
    ps='&lt;0.001' if p<0.001 else f'{p:.3f}'
    return f"{r:+.2f} [{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}], {ps} (n={n})"
def miqr(c):
    x=num(c).dropna().values; return np.median(x),np.percentile(x,25),np.percentile(x,75)
def pct(cond, base):
    return f"{int(cond.sum())} ({100*cond.sum()/base:.0f}%)"

f1i=[c for c in qcols if int(c.split('_')[0][1:]) in {3,5,6,7,8,9,10}]
f2i=[c for c in qcols if int(c.split('_')[0][1:]) in {0,1,2,4}]

def tbl(headers, rows):
    h="".join(f"<th>{c}</th>" for c in headers)
    b="".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)
    return f'<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'

# ---- Table 1 ----
age=num('age'); sex=num('sex'); mg=miqr('mig_days'); mid=miqr('midas'); acq=miqr('acute_days'); mbq=miqr('mibs4')
grade=pd.cut(num('midas').dropna(),[-1,5,10,20,1e9],labels=['I','II','III','IV'])
freq=pd.cut(num('mig_days'),[-1,7.5,14.5,1e9],labels=['<8','8-14','>=15'])
EMP=['常勤 full-time','非常勤 part-time','自営業 self-employed','学生 student','主婦・主夫 homemaker','休職中 on leave','無職 unemployed','退職 retired','その他 other']
emp=num('employ'); empstr=" / ".join(f"{EMP[int(i)]} {int((emp==i).sum())}" for i in range(9) if (emp==i).sum()>0)
t1=tbl(["Characteristic","Value"],[
 ["Age, years, mean (SD)", f"{age.mean():.1f} ({age.std():.1f})"],
 ["Women, n (%)", pct(sex==1, sex.notna().sum())],
 ["Self-reported migraine days / 4 weeks, median (IQR)", f"{mg[0]:.0f} ({mg[1]:.0f}–{mg[2]:.0f})"],
 ["Frequency group: &lt;8 / 8–14 / &ge;15 days, n", f"{int((freq=='<8').sum())} / {int((freq=='8-14').sum())} / {int((freq=='>=15').sum())}"],
 ["Chronic-migraine-equivalent (&ge;15 headache days/4wk), n (%)", pct(num('ha_days')>=15, num('ha_days').notna().sum())],
 ["Aura (any attacks with aura), n (%)", pct(num('aura')==1, num('aura').notna().sum())],
 ["Acute-medication days / 4 weeks, median (IQR)", f"{acq[0]:.0f} ({acq[1]:.0f}–{acq[2]:.0f})"],
 ["Acute-medication overuse risk (&ge;10 days/4wk), n (%)", pct(num('acute_days')>=10, num('acute_days').notna().sum())],
 ["Current preventive medication, n (%)", pct(num('prev_use')==1, num('prev_use').notna().sum())],
 ["CGRP-pathway treatment, n (%)", pct(num('cgrp_use')==1, num('cgrp_use').notna().sum())],
 ["MIDAS, median (IQR)", f"{mid[0]:.0f} ({mid[1]:.0f}–{mid[2]:.0f})"],
 ["MIDAS grade: I(0–5)/II(6–10)/III(11–20)/IV(21+), n", f"{int((grade=='I').sum())} / {int((grade=='II').sum())} / {int((grade=='III').sum())} / {int((grade=='IV').sum())}"],
 ["MIBS-4 (4-week), median (IQR)", f"{mbq[0]:.0f} ({mbq[1]:.0f}–{mbq[2]:.0f})"],
 ["Migraine-specific prescription medication, n", f"{int((num('drug_specific')==1).sum())}"],
 ["Diary-derived migraine-symptom proxy positive, n", f"{int((num('id_mig')==1).sum())}"],
 ["Meeting broad composite migraine definition, n (%)", pct(num('case_def')==1, N)],
 ["Employment status, n", empstr],
])

# ---- Table 2 ----
t2a=tbl(["Factor","Items","Cronbach α","McDonald ω"],[
 ["Life Restoration", len(f1i), f"{alpha(f1i):.2f}", f"{omega(f1i):.2f}"],
 ["Migraine Agency", len(f2i), f"{alpha(f2i):.2f}", f"{omega(f2i):.2f}"],
 ["Total (auxiliary)", len(qcols), f"{alpha(qcols):.2f}", "—"]])
domrows=[]
for i in range(11):
    items=[c for c in qcols if int(c.split('_')[0][1:])==i]
    domrows.append([f"{DOM_EN[i]}", FAC[i], len(items), f"{alpha(items):.2f}"])
t2b=tbl(["Domain","Factor","Items","Cronbach α"], domrows)

# ---- Table 3 ----
f1,f2,tot=num('f1'),num('f2'),num('mfi_total')
d7=num('w28_days')>=7
hap=num('w28_haprop').where(d7); ful=num('w28_fulfill').where(d7)
ext=[('MIDAS (disability)','&minus;', num('midas')),
     ('MIBS-4 (interictal burden)','&minus; (F1) / + (F2)', num('mibs4')),
     ('Self-reported migraine days','weak &minus; / + (F2)', num('mig_days')),
     ('Diary headache-day proportion (28d, &ge;7d)','&minus;', hap),
     ('Diary headache-free-day wellbeing (28d)','+', ful)]
t3=tbl(["External measure","Prespecified direction","Total (auxiliary)","Life Restoration","Migraine Agency"],
       [[lab,dirn,sp(tot,y),sp(f1,y),sp(f2,y)] for lab,dirn,y in ext])

# ---- Table 4 ----
sub=pd.DataFrame({'haprop':num('w28_haprop'),'f1':f1,'f2':f2,'midas':num('midas')})[d7].dropna(subset=['haprop','f1'])
hmed,fmed=sub['haprop'].median(),sub['f1'].median()
sub['hf']=sub['haprop']>hmed; sub['hr']=sub['f1']>fmed
def g(hf,hr):
    x=sub[(sub['hf']==hf)&(sub['hr']==hr)]
    def mi(col): v=x[col].dropna().values; return f"{np.median(v):.0f} ({np.percentile(v,25):.0f}–{np.percentile(v,75):.0f})"
    return len(x),mi('midas'),mi('f2')
L={(False,True):'Lower proportion / higher restoration (concordant)',(True,False):'Higher proportion / lower restoration (concordant)',
   (False,False):'Lower proportion / lower restoration (DISCORDANT)',(True,True):'Higher proportion / higher restoration (DISCORDANT)'}
rows4=[];disc=0
for k in [(False,True),(True,False),(False,False),(True,True)]:
    n_,md,f2m=g(*k)
    if k[0]==k[1]: disc+=n_
    rows4.append([L[k],n_,md,f2m])
t4=tbl(["Group","n","MIDAS median (IQR)","F2 Agency median (IQR)"], rows4)

html=f"""<h2>MFI (Cephalalgia Rev3) — Tables 1–4</h2>
<p>Frozen pilot sample, N={N}. Data cut: 18 July 2026 (00:00 JST). Generated from the frozen response matrix; re-runnable if N changes.</p>
<h3>Table 1. Participant characteristics (N={N})</h3>{t1}
<h3>Table 2. Internal consistency (N={N})</h3><p>By higher-order factor (primary):</p>{t2a}<p>By prespecified domain:</p>{t2b}
<h3>Table 3. Convergent and diary-linked validity — Spearman ρ [95% CI], p</h3>{t3}
<h3>Table 4. Secondary sample-median-split discordance (linked n={len(sub)}; splits: headache-day proportion {hmed:.2f}, Life Restoration {fmed:.0f})</h3>{t4}
<p>Discordant total = {disc}/{len(sub)} ({100*disc/len(sub):.0f}%). Sample-relative and descriptive; splits are not clinical thresholds.</p>"""
open('mfi_tables.html','w').write(html)
print("wrote mfi_tables.html (", len(html), "bytes )")
print("Table1 aura+", int((num('aura')==1).sum()), "CMequiv", int((num('ha_days')>=15).sum()), "prev", int((num('prev_use')==1).sum()), "cgrp", int((num('cgrp_use')==1).sum()), "MOHrisk", int((num('acute_days')>=10).sum()))
