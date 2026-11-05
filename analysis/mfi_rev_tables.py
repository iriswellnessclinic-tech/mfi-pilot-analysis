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
DOM_EN = ['Premonitory-symptom awareness','Trigger recognition','Pre-emptive action','Attack control','Prevention of medication overuse','Freedom of activity','Hope and treatment continuity','Reclaiming your life','Social freedom','Impact on close others','Global freedom evaluation']
FAC = ['Agency','Agency','Agency','Restoration','Agency','Restoration','Restoration','Restoration','Restoration','Restoration','Restoration']
FAC_DISP = {'Restoration':'Life Restoration','Agency':'Migraine Agency'}
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

def psp(x,y,covs):
    df=pd.DataFrame({'x':x,'y':y})
    for i,c in enumerate(covs): df[f'c{i}']=c
    df=df.dropna(); n=len(df); k=len(covs)
    rx,ry=df['x'].rank(),df['y'].rank()
    C=np.column_stack([np.ones(n)]+[df[f'c{i}'].rank() for i in range(k)])
    ex=rx-C@np.linalg.lstsq(C,rx,rcond=None)[0]; ey=ry-C@np.linalg.lstsq(C,ry,rcond=None)[0]
    r=np.corrcoef(ex,ey)[0,1]; z=np.arctanh(r); se=1/np.sqrt(n-3-k)
    t=r*np.sqrt((n-2-k)/(1-r**2)); pv=2*stats.t.sf(abs(t),n-2-k)
    lo,hi=np.tanh(z-1.96*se),np.tanh(z+1.96*se)
    def f3(v): return f"{v:+.3f}" if abs(round(v,2))<0.005 else f"{v:+.2f}"
    ps='&lt;0.001' if pv<0.001 else f'{pv:.3f}'
    return f"{f3(r)} [{f3(lo)}, {f3(hi)}], {ps} (n={n})"
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
age=num('age'); sex=num('sex'); mid=miqr('midas'); acq=miqr('acute_days'); mbq=miqr('mibs4')
grade=pd.cut(num('midas').dropna(),[-1,5,10,20,1e9],labels=['I','II','III','IV'])
d7_t1=num('w28_days')>=7
mmd_t1=(num('w28_migprop')*28).where(d7_t1)
wdays_t1=num('w28_days').where(d7_t1)
n81=int(d7_t1.sum())
def miqr_s(s):
    x=s.dropna().values; return np.median(x),np.percentile(x,25),np.percentile(x,75)
mmq=miqr_s(mmd_t1); wdq=miqr_s(wdays_t1)
freq=pd.cut(mmd_t1,[-1,7.5,14.5,1e9],labels=['<8','8-14','>=15'])
EMP=['常勤 full-time','非常勤 part-time','自営業 self-employed','学生 student','主婦・主夫 homemaker','休職中 on leave','無職 unemployed','退職 retired','その他 other']
emp=num('employ'); empstr=" / ".join(f"{EMP[int(i)]} {int((emp==i).sum())}" for i in range(9) if (emp==i).sum()>0)
t1=tbl(["Characteristic","Value"],[
 ["Age, years, mean (95% CI)", f"{age.mean():.1f} ({age.mean()-1.96*age.std()/np.sqrt(age.notna().sum()):.1f} to {age.mean()+1.96*age.std()/np.sqrt(age.notna().sum()):.1f})"],
 ["Women, n (%)", pct(sex==1, sex.notna().sum())],
 [f"Diary-based MMD, 28-day window (&ge;7 recorded days; n={n81}), median (IQR)", f"{mmq[0]:.0f} ({mmq[1]:.0f}–{mmq[2]:.0f})"],
 [f"Frequency group by diary-based MMD: &lt;8 / 8–14 / &ge;15 days, n", f"{int((freq=='<8').sum())} / {int((freq=='8-14').sum())} / {int((freq=='>=15').sum())}"],
 [f"Recorded diary days in the 28-day window (&ge;7 recorded days; n={n81}), median (IQR)", f"{wdq[0]:.0f} ({wdq[1]:.0f}–{wdq[2]:.0f})"],
 ["Aura (any attacks with aura), n (%)", pct(num('aura')==1, num('aura').notna().sum())],
 ["Acute-medication days / 4 weeks, median (IQR)", f"{acq[0]:.0f} ({acq[1]:.0f}–{acq[2]:.0f})"],
 ["Acute-medication use on &ge;10 days during the preceding 4 weeks, n (%)", pct(num('acute_days')>=10, num('acute_days').notna().sum())],
 ["Current preventive medication, n (%)", pct(num('prev_use')==1, N)],
 ["CGRP-related preventive treatment, n (%)", pct(num('cgrp_use')==1, N)],
 ["MIDAS score, median (IQR)", f"{mid[0]:.0f} ({mid[1]:.0f}–{mid[2]:.0f})"],
 ["MIDAS grade: I(0–5)/II(6–10)/III(11–20)/IV(21+), n", f"{int((grade=='I').sum())} / {int((grade=='II').sum())} / {int((grade=='III').sum())} / {int((grade=='IV').sum())}"],
 ["MIBS-4 score (4-week), median (IQR)", f"{mbq[0]:.0f} ({mbq[1]:.0f}–{mbq[2]:.0f})"],
 ["Migraine-specific prescription medication, n", f"{int((num('drug_specific')==1).sum())}"],
 ["Self-reported migraine (history or &ge;1 migraine day), n (%)", pct(num('case_def')==1, N)],
 ["Employment status, n", empstr],
])

# ---- Table 2 ----
t2a=tbl(["Factor","Items","Cronbach α","McDonald ω"],[
 ["Life Restoration", len(f1i), f"{alpha(f1i):.2f}", f"{omega(f1i):.2f}"],
 ["Migraine Agency", len(f2i), f"{alpha(f2i):.2f}", f"{omega(f2i):.2f}"]])
domrows=[]
for i in range(11):
    items=[c for c in qcols if int(c.split('_')[0][1:])==i]
    domrows.append([f"{DOM_EN[i]}", FAC_DISP[FAC[i]], len(items), f"{alpha(items):.2f}"])
t2b=tbl(["Domain","Factor","Items","Cronbach α"], domrows)

# ---- Table 3 ----
f1,f2,tot=num('f1'),num('f2'),num('mfi_total')
d7=num('w28_days')>=7
mmd=(num('w28_migprop')*28).where(d7)
age_c, sex_c = num('age'), num('sex')
ext=[('Diary-based MMD (28-day window, &ge;7 recorded days)','&minus;', mmd),
     ('MIDAS score (disability)','&minus;', num('midas')),
     ('MIBS-4 score (interictal burden)','&minus; (F1) / + (F2)', num('mibs4'))]
t3=tbl(["External measure","Expected direction","Life Restoration score","Migraine Agency score"],
       [[lab,dirn,psp(y,f1,[age_c,sex_c]),psp(y,f2,[age_c,sex_c])] for lab,dirn,y in ext])

# ---- Table 4 (both factors) ----
def quad_rows(score, other, score_lab, other_lab, labelled):
    sub=pd.DataFrame({'mmd':mmd,'s':score,'midas':num('midas'),'o':other}).dropna(subset=['mmd','s'])
    hm,sm=sub['mmd'].median(),sub['s'].median()
    sub['hf']=sub['mmd']>hm; sub['hs']=sub['s']>sm
    def mi(x,col): v=x[col].dropna().values; return f"{np.median(v):.0f} ({np.percentile(v,25):.0f}–{np.percentile(v,75):.0f})"
    lab={(False,True):f'Lower MMD / higher {score_lab}',(True,False):f'Higher MMD / lower {score_lab}',
         (False,False):f'Lower MMD / lower {score_lab}',(True,True):f'Higher MMD / higher {score_lab}'}
    rows=[]; disc=0
    for k in [(False,True),(True,False),(False,False),(True,True)]:
        x=sub[(sub['hf']==k[0])&(sub['hs']==k[1])]
        tag=('' if not labelled else (' (DISCORDANT)' if k[0]==k[1] else ' (concordant)'))
        if k[0]==k[1]: disc+=len(x)
        rows.append([lab[k]+tag,len(x),mi(x,'midas'),mi(x,'o')])
    return rows,disc,hm,sm,len(sub)
rows4a,disc,hmed,fmed,n4=quad_rows(f1,f2,'Life Restoration score','Migraine Agency score',True)
rows4b,_,_,amed,_=quad_rows(f2,f1,'Migraine Agency score','Life Restoration score',False)
sub=pd.DataFrame({'mmd':mmd,'f1':f1}).dropna()
t4=(tbl(["Group","n","MIDAS score, median (IQR)","Migraine Agency score, median (IQR)"], rows4a)
    + "<p>B. Split by the Migraine Agency score</p>"
    + tbl(["Group","n","MIDAS score, median (IQR)","Life Restoration score, median (IQR)"], rows4b))

html=f"""<h2>MFI (Cephalalgia Rev22) — Tables 1–4</h2>
<p>Pilot sample, N={N} (one participant deleted their account after the initial cut and was excluded). Survey recruitment window closed 18 July 2026; diary extraction 15 August 2026. Generated from the response matrix; re-runnable if N changes.</p>
<h3>Table 1. Participant characteristics (N={N})</h3>{t1}
<h3>Table 2. Internal consistency (N={N})</h3><p>By higher-order factor (primary):</p>{t2a}<p>By prespecified domain:</p>{t2b}
<h3>Table 3. Correlations of the MFI factor scores with diary-based MMD, the MIDAS score, and the MIBS-4 score, adjusted for age and sex — partial Spearman ρ [95% CI], p</h3><p>Unadjusted correlations are in Supplementary Table S3.</p>{t3}
<h3>Table 4. Split at the clinical thresholds for each MFI factor score (n={n4}; thresholds = sample medians: diary-based MMD {hmed:.0f} days, Life Restoration score {fmed:.0f}, Migraine Agency score {amed:.0f})</h3><p>A. Split by the Life Restoration score</p>{t4}
<p>For the Life Restoration score, an inverse association with diary-based MMD was expected, so participants below or above both thresholds are labelled discordant (total {disc}/{n4}, {100*disc/n4:.0f}%). No direction was expected for the Migraine Agency score, so its groups are not labelled. The clinical thresholds are the sample medians of this cohort, defined in the Methods; they are not established cut-offs.</p>"""
open('mfi_tables.html','w').write(html)
print("wrote mfi_tables.html (", len(html), "bytes )")
print("Table1 aura+", int((num('aura')==1).sum()), "CMequiv", int((num('ha_days')>=15).sum()), "prev", int((num('prev_use')==1).sum()), "cgrp", int((num('cgrp_use')==1).sum()), "MOHrisk", int((num('acute_days')>=10).sum()))
