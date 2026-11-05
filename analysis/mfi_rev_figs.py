#!/usr/bin/env python3
"""MFI Figures 1-3 + Supplementary Figure S1 (n=116 frozen; Okabe-Ito CVD-safe).
Rev9 (Shimazu Major Revision): Fig1 funnel 29/36 + nested subsets; Fig2 Awareness&Management,
r=0.00, pattern loadings + bootstrap 95% CI error bars; Fig3 = Panel A only (OLS + 95% CI of mean);
the 2x2 median-split display moves to Supplementary Figure S1."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from factor_analyzer import FactorAnalyzer
from scipy import stats
import os
np.random.seed(20260718)
DPI=int(os.environ.get("FIG_DPI","150"))  # 投稿最終は FIG_DPI=300 で再実行
TITLES=os.environ.get("FIG_TITLES","1")=="1"  # J Headache Pain: titles go in the manuscript, not in the graphic -> FIG_TITLES=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.linewidth":0.8,"savefig.dpi":DPI,"figure.dpi":110})
BLUE,ORANGE,GREEN,VERM,GRAY="#0072B2","#E69F00","#009E73","#D55E00","#666666"
D=pd.read_csv('mfi_matrix.csv'); num=lambda c: pd.to_numeric(D[c],errors='coerce')

def partial_spearman(x, y, covs):
    """Partial Spearman: rank-transform, residualise on ranked covariates, Fisher-z CI."""
    df = pd.DataFrame({'x': x, 'y': y})
    for i, c in enumerate(covs): df[f'c{i}'] = c
    df = df.dropna(); n = len(df); k = len(covs)
    rx, ry = df['x'].rank(), df['y'].rank()
    C = np.column_stack([np.ones(n)] + [df[f'c{i}'].rank() for i in range(k)])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; z = np.arctanh(r); se = 1/np.sqrt(n-3-k)
    return r, np.tanh(z-1.96*se), np.tanh(z+1.96*se), n


# ============ Figure 1: study flow (funnel 29/36 + nested subsets) ============
fig,ax=plt.subplots(figsize=(6.8,6.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,txt,fc="#EAF2F8",ec=BLUE,fs=8.3):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec=ec,lw=1.2))
    ax.text(x,y,txt,ha="center",va="center",fontsize=fs)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,lw=1.05,color=GRAY))
box(3.2,9.3,4.8,1.0,"Survey made available in-app\nN = 1180\n(invitation views not logged)")
box(3.2,7.5,4.8,0.9,"Entered consent flow and began\nn = 180")
box(3.2,5.7,4.8,1.0,"Completed (analytic sample)\nn = 115")
arrow(3.2,8.8,3.2,7.95); arrow(3.2,7.05,3.2,6.2)
box(8.05,6.6,3.5,1.3,"Discontinued n = 65\n• answered no items: 28\n• answered ≥1 item: 37",fc="#FDF2E9",ec=ORANGE,fs=8.0)
arrow(3.2,6.6,6.3,6.6)
# nested linked subsets (each a subset of the one above)
box(3.2,3.9,4.4,0.8,"≥1 recorded diary day\nn = 107",fc="#E8F6F0",ec=GREEN,fs=8.0)
box(3.2,2.5,4.4,0.8,"≥7 recorded days (diary-based MMD analysis)\nn = 81",fc="#E8F6F0",ec=GREEN,fs=8.0)
arrow(3.2,5.2,3.2,4.3); arrow(3.2,3.5,3.2,2.9)
ax.text(8.05,3.0,"Survey window closed 18 July 2026\nDiary extracted 15 August 2026",ha="center",fontsize=7.6,style="italic",color=GRAY)
if TITLES: ax.set_title("Figure 1. Study flow and nested linked-data subsets",fontsize=10,loc="left",weight="bold")
plt.tight_layout(); plt.savefig("fig1_flow.png",bbox_inches="tight"); plt.close()

# ============ Figure 2: FULL 2-factor loading pattern (both factors per domain, signed) + bootstrap 95% CI ============
DOMS=[f'dom{i+1}' for i in range(11)]
EN=['Premonitory-symptom awareness','Trigger recognition','Pre-emptive action','Attack control',
    'Prevention of medication overuse','Freedom of activity','Hope and treatment continuity','Reclaiming your life',
    'Social freedom','Impact on close others','Global freedom evaluation']
X=D[DOMS].apply(pd.to_numeric,errors='coerce').dropna()
Xv=X.values
def fit_load(mat):
    fa=FactorAnalyzer(n_factors=2,method='principal',rotation='oblimin'); fa.fit(mat)
    return fa.loadings_.copy()
L=fit_load(Xv)
if abs(L[7,0])<abs(L[7,1]): L=L[:,::-1]   # F1 = Life Restoration (idx7 loads highest)
prim=np.where(np.abs(L[:,0])>=np.abs(L[:,1]),0,1)
# bootstrap: aligned signed loadings for BOTH factors (1,000 resamples)
rng=np.random.default_rng(20260718); B=1000; boot=np.full((B,11,2),np.nan)
for b in range(B):
    idx=rng.integers(0,len(Xv),len(Xv))
    try: Lb=fit_load(Xv[idx])
    except Exception: continue
    used=set(); al=np.zeros_like(Lb)
    for c in range(2):
        dots=[abs(np.dot(Lb[:,k],L[:,c])) if k not in used else -1 for k in range(2)]
        k=int(np.argmax(dots)); used.add(k)
        s=np.sign(np.dot(Lb[:,k],L[:,c])) or 1; al[:,c]=s*Lb[:,k]
    boot[b]=al
ci_lo=np.nanpercentile(boot,2.5,axis=0)   # 11x2 signed
ci_hi=np.nanpercentile(boot,97.5,axis=0)  # 11x2 signed
# order: Factor-1-primary domains first (by F1 loading desc), then Factor-2-primary (by F2 loading desc)
order=sorted(range(11),key=lambda i:(prim[i], -L[i,prim[i]]))
nF1=int(np.sum(prim==0))
fig,ax=plt.subplots(figsize=(7.8,6.4))
yy=np.arange(11)[::-1]; bw=0.38
for k,i in enumerate(order):
    y=yy[k]
    for f,col,off in [(0,BLUE,+bw/2),(1,ORANGE,-bw/2)]:
        v=float(L[i,f])
        ax.barh(y+off,v,height=bw,color=col,edgecolor="white",zorder=2)
        err=[[v-ci_lo[i,f]],[ci_hi[i,f]-v]]
        ax.errorbar(v,y+off,xerr=err,fmt='none',ecolor="#333333",elinewidth=0.8,capsize=2,zorder=3)
        if v>=0: lx,ha=ci_hi[i,f]+0.03,'left'
        else:    lx,ha=ci_lo[i,f]-0.03,'right'
        ax.text(lx,y+off,f"{v:+.2f}",va="center",ha=ha,fontsize=6.6,color="#333333")
ax.set_yticks(yy); ax.set_yticklabels([EN[i] for i in order],fontsize=8.4)
ax.axvline(0,color="k",lw=0.9,zorder=1)
ax.axhline(yy[nF1-1]-0.5,color=GRAY,lw=0.8,ls=":")   # separator between the two factor blocks
ax.set_xlim(-0.65,1.05); ax.set_ylim(-0.7,10.7)
ax.set_xlabel("Pattern loading on each factor (principal-axis, oblimin) with bootstrap 95% CI")
ax.legend(handles=[Patch(color=BLUE,label="Factor 1 — Life Restoration"),
                   Patch(color=ORANGE,label="Factor 2 — Migraine Agency")],
          loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=8.5,frameon=False)
if TITLES: ax.set_title("Figure 2. Two-factor loading pattern of the 11 domain scores (both factors shown)",fontsize=9.6,loc="left",weight="bold")
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig2_loadings.png",bbox_inches="tight"); plt.close()

# ============ Figure 3: two panels — A Life Restoration, B Migraine Agency (OLS + 95% CI of the mean) ============
d7=num('w28_days')>=7
xmmd=(num('w28_migprop')*28).where(d7)
def fig3_panel(ax, fcol, label, letter):
    y_=num(fcol); m=xmmd.notna()&y_.notna(); x=xmmd[m].values.astype(float); y=y_[m].values.astype(float); n=len(x)
    b1,b0=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),100); yh=b0+b1*xs
    resid=y-(b0+b1*x); s2=(resid@resid)/(n-2); xbar=x.mean(); Sxx=((x-xbar)**2).sum()
    tval=stats.t.ppf(0.975,n-2)
    ci_mean=tval*np.sqrt(s2*(1/n+(xs-xbar)**2/Sxx))         # 95% CI of the mean fit (not prediction interval)
    seb=np.sqrt(s2/Sxx); slo,shi=b1-tval*seb,b1+tval*seb
    rho_u,_=stats.spearmanr(x,y)
    prho,plo,phi,pn=partial_spearman(xmmd,y_,[num('age'),num('sex')])
    ax.fill_between(xs,yh-ci_mean,yh+ci_mean,color=BLUE,alpha=0.15,label="95% CI of the mean")
    ax.plot(xs,yh,color=BLUE,lw=1.9,label="OLS fit")
    ax.scatter(x,y,s=27,color=GRAY,alpha=0.75,edgecolor="white",linewidth=0.4)
    mx,my=np.median(x),np.median(y)
    ax.axvline(mx,color=GRAY,ls=":",lw=0.9); ax.axhline(my,color=GRAY,ls=":",lw=0.9)
    _lb=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.85)
    ax.text(mx+0.3,2,f"threshold {mx:.0f} (median)",fontsize=7,color=GRAY,bbox=_lb)
    ax.text(0.3,my+1.8,f"threshold {my:.0f} (median)",fontsize=7,color=GRAY,bbox=_lb)
    ax.set_xlabel("Diary-based MMD (28-day window)"); ax.set_ylabel(f"{label} score (0-100)")
    ax.set_xlim(-0.5,28.5); ax.set_ylim(0,100)
    ax.text(1,10,f"OLS slope = {b1:.2f}  (95% CI {slo:.2f} to {shi:.2f})\nPartial Spearman ρ (adjusted for age and sex) = {prho:+.2f}\n[{plo:+.2f} to {phi:+.2f}];  n = {pn}",
            fontsize=7.6,bbox=dict(boxstyle="round",fc="white",ec=GRAY,alpha=0.9))
    ax.set_title(f"{letter}  {label} score",fontsize=9.2,loc="left",weight="bold")
    ax.spines[['top','right']].set_visible(False)
    return dict(b1=b1,slo=slo,shi=shi,rho=rho_u,prho=prho,plo=plo,phi=phi,n=pn,mx=mx,my=my)
fig,(axA,axB)=plt.subplots(1,2,figsize=(10.4,4.5))
resA=fig3_panel(axA,'f1','Life Restoration','A'); resB=fig3_panel(axB,'f2','Migraine Agency','B')
axA.legend(loc="upper right",fontsize=8,frameon=False)
if TITLES: fig.suptitle("Figure 3. Diary-based MMD and the MFI factor scores (n = 81)",fontsize=9.8,x=0.01,ha="left",weight="bold")
plt.tight_layout(); plt.savefig("fig3_proportion_restoration.png",bbox_inches="tight"); plt.close()
b1,slope_lo,slope_hi,rho,prho,plo,phi,pn=resA['b1'],resA['slo'],resA['shi'],resA['rho'],resA['prho'],resA['plo'],resA['phi'],resA['n']
print(f"Fig3 B (Migraine Agency): slope={resB['b1']:.2f} [{resB['slo']:.2f},{resB['shi']:.2f}] rho_adj={resB['prho']:+.2f} [{resB['plo']:+.2f},{resB['phi']:+.2f}] median score={resB['my']:.0f}")

# ============ Supplementary Figure S1: split at the clinical thresholds, both factors (diary-based MMD, n=81) ============
def s1_panel(ax, fcol, label, letter, labelled):
    y_=num(fcol); mm2=xmmd.notna()&y_.notna()
    xm=xmmd[mm2].values.astype(float); ym=y_[mm2].values.astype(float); mid2=num('midas')[mm2].values
    hmed=np.median(xm); fmed=np.median(ym); xmax=np.nanmax(xm)
    for hf in [False,True]:
        for hr in [False,True]:
            sel=((xm>hmed)==hf)&((ym>fmed)==hr)
            disc = (hf==hr) if labelled else False
            col = ORANGE if disc else GRAY
            ax.scatter(xm[sel],ym[sel],s=32,color=col,alpha=0.85,edgecolor="white",linewidth=0.4)
            if sel.sum()>0:
                mm=np.median(mid2[sel][np.isfinite(mid2[sel])])
                cx=(hmed*0.5 if not hf else hmed+(xmax-hmed)*0.5); cy=(20 if not hr else 90)
                ax.text(cx,cy,f"n={sel.sum()}\nMIDAS {mm:.0f}",ha="center",fontsize=8.0,
                        color=(VERM if disc else "black"),
                        bbox=dict(boxstyle="round",fc="white",ec=(ORANGE if disc else GRAY),alpha=0.9))
    ax.axvline(hmed,color="k",ls="--",lw=0.8); ax.axhline(fmed,color="k",ls="--",lw=0.8)
    ax.text(hmed+0.4,2,f"threshold = {hmed:.0f} (median)",fontsize=7.2,color=GRAY)
    ax.text(0.3,fmed+1,f"threshold = {fmed:.0f} (median)",fontsize=7.2,color=GRAY)
    ax.set_xlabel("Diary-based MMD (days)"); ax.set_ylabel(f"{label} score")
    ax.set_xlim(-0.5,xmax+2); ax.set_ylim(0,100)
    ax.set_title(f"{letter}  {label} score",fontsize=9.2,loc="left",weight="bold")
    ax.spines[['top','right']].set_visible(False)
    return hmed,fmed
figs,(s1a,s1b)=plt.subplots(1,2,figsize=(10.8,4.7))
hmed,fmed=s1_panel(s1a,'f1','Life Restoration','A',True); s1_panel(s1b,'f2','Migraine Agency','B',False)
s1b.text(0.97,0.085,"No expected direction with diary-based MMD;\nquadrants not classified as concordant or discordant",
         transform=s1b.transAxes,ha="right",va="bottom",fontsize=7.2,color="#444444",
         bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#999999",alpha=0.9))
s1a.legend(handles=[Patch(color=ORANGE,label="Discordant (threshold-based)"),Patch(color=GRAY,label="Concordant")],
           loc="lower right",fontsize=7.8,frameon=False)
figs.suptitle("Supplementary Figure S1. Split at the clinical thresholds (sample medians), diary-based MMD (n = 81)",fontsize=8.6,x=0.01,ha="left",weight="bold")
plt.tight_layout(); plt.savefig("figS1_discordance.png",bbox_inches="tight"); plt.close()

print("wrote fig1_flow.png, fig2_loadings.png, fig3_proportion_restoration.png, figS1_discordance.png")
print(f"Fig2 primary loadings + bootstrap CI computed (B={B})")
print(f"Fig3: OLS slope={b1:.1f} [{slope_lo:.1f},{slope_hi:.1f}] rho_unadj={rho:+.2f} rho_adj={prho:+.2f} [{plo:+.2f},{phi:+.2f}] n={pn}; medians hap={hmed:.2f} f1={fmed:.0f}")
