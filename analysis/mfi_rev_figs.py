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
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.linewidth":0.8,"savefig.dpi":DPI,"figure.dpi":110})
BLUE,ORANGE,GREEN,VERM,GRAY="#0072B2","#E69F00","#009E73","#D55E00","#666666"
D=pd.read_csv('mfi_matrix.csv'); num=lambda c: pd.to_numeric(D[c],errors='coerce')

# ============ Figure 1: study flow (funnel 29/36 + nested subsets) ============
fig,ax=plt.subplots(figsize=(6.8,6.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,txt,fc="#EAF2F8",ec=BLUE,fs=8.3):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec=ec,lw=1.2))
    ax.text(x,y,txt,ha="center",va="center",fontsize=fs)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,lw=1.05,color=GRAY))
box(3.2,9.3,4.8,1.0,"Survey made available in-app\nN = 1190\n(invitation views not logged)")
box(3.2,7.5,4.8,0.9,"Entered consent flow and began\nn = 181")
box(3.2,5.7,4.8,1.0,"Completed (analytic sample)\nn = 116")
arrow(3.2,8.8,3.2,7.95); arrow(3.2,7.05,3.2,6.2)
box(8.05,6.6,3.5,1.3,"Discontinued n = 65\n• answered no items: 29\n• answered ≥1 item: 36",fc="#FDF2E9",ec=ORANGE,fs=8.0)
arrow(3.2,6.6,6.3,6.6)
# nested linked subsets (each a subset of the one above)
box(3.2,3.9,4.4,0.8,"≥1 recorded diary day\nn = 108",fc="#E8F6F0",ec=GREEN,fs=8.0)
box(3.2,2.5,4.4,0.8,"≥7 diary days (primary linked)\nn = 81",fc="#E8F6F0",ec=GREEN,fs=8.0)
box(3.2,1.1,4.4,0.8,"Exploratory fulfillment-model subset\nn = 75",fc="#E8F6F0",ec=GREEN,fs=8.0)
arrow(3.2,5.2,3.2,4.3); arrow(3.2,3.5,3.2,2.9); arrow(3.2,2.1,3.2,1.5)
ax.text(8.05,3.0,"Frozen data cut:\n18 July 2026 (00:00 JST)",ha="center",fontsize=7.6,style="italic",color=GRAY)
ax.set_title("Figure 1. Study flow and nested linked-data subsets",fontsize=10,loc="left",weight="bold")
plt.tight_layout(); plt.savefig("fig1_flow.png",bbox_inches="tight"); plt.close()

# ============ Figure 2: provisional 2-factor loadings + bootstrap 95% CI ============
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
if abs(L[7,0])<abs(L[7,1]): L=L[:,::-1]   # F1 = Life Recovery (idx7 loads highest)
prim=np.where(np.abs(L[:,0])>=np.abs(L[:,1]),0,1)
# bootstrap 95% CI of each domain's primary-factor loading (1,000 resamples, aligned to full-sample)
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
lo=np.zeros(11); hi=np.zeros(11)
for i in range(11):
    pf=prim[i]; vals=boot[:,i,pf]; vals=vals[np.isfinite(vals)]
    lo[i],hi[i]=np.nanpercentile(vals,[2.5,97.5])
order=sorted(range(11),key=lambda i:(prim[i], -abs(L[i,prim[i]])))
fig,ax=plt.subplots(figsize=(7.2,4.7))
yy=np.arange(11)[::-1]
for k,i in enumerate(order):
    pf=prim[i]; col=BLUE if pf==0 else ORANGE; v=abs(L[i,pf])
    err=[[v-abs(lo[i])],[abs(hi[i])-v]]
    ax.barh(yy[k],v,color=col,height=0.6,edgecolor="white")
    ax.errorbar(v,yy[k],xerr=err,fmt='none',ecolor="#222222",elinewidth=0.9,capsize=2.5)
    ax.text(abs(hi[i])+0.03,yy[k],f"{v:.2f}",va="center",fontsize=7.8)
ax.set_yticks(yy); ax.set_yticklabels([EN[i] for i in order],fontsize=8.2)
ax.set_xlim(0,1.05); ax.set_xlabel("Primary pattern loading (principal-axis, oblimin) with bootstrap 95% CI")
ax.axvline(0,color="k",lw=0.8)
ax.legend(handles=[Patch(color=BLUE,label="Factor 1 — Life Recovery"),
                   Patch(color=ORANGE,label="Factor 2 — Awareness and Management")],
          loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=2,fontsize=8.3,frameon=False)
ax.set_title("Figure 2. Provisional two-factor summary of 11 prespecified domain scores",fontsize=9.8,loc="left",weight="bold")
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig2_loadings.png",bbox_inches="tight"); plt.close()

# ============ Figure 3: Panel A only — OLS + 95% CI of the mean ============
d7=num('w28_days')>=7
x=num('w28_haprop').where(d7); y1=num('f1')
m=x.notna()&y1.notna(); x=x[m].values; y=y1[m].values; n=len(x)
b1,b0=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),100); yh=b0+b1*xs
resid=y-(b0+b1*x); s2=(resid@resid)/(n-2); xbar=x.mean(); Sxx=((x-xbar)**2).sum()
tval=stats.t.ppf(0.975,n-2)
ci_mean=tval*np.sqrt(s2*(1/n+(xs-xbar)**2/Sxx))         # 95% CI of the mean fit (not prediction interval)
seb=np.sqrt(s2/Sxx); slope_lo,slope_hi=b1-tval*seb,b1+tval*seb
rho,pval=stats.spearmanr(x,y)
fig,ax=plt.subplots(figsize=(5.6,4.5))
ax.fill_between(xs,yh-ci_mean,yh+ci_mean,color=BLUE,alpha=0.15,label="95% CI of the mean")
ax.plot(xs,yh,color=BLUE,lw=1.9,label="OLS fit")
ax.scatter(x,y,s=27,color=GRAY,alpha=0.75,edgecolor="white",linewidth=0.4)
ax.set_xlabel("Diary headache-day proportion (28-day window)")
ax.set_ylabel("Life Recovery score (0-100)")
ax.set_xlim(-0.02,1.02); ax.set_ylim(0,100)
ax.text(0.03,10,f"OLS slope = {b1:.1f}  (95% CI {slope_lo:.1f} to {slope_hi:.1f})\nSpearman ρ = {rho:+.2f};  n = {n}",
        fontsize=8.2,bbox=dict(boxstyle="round",fc="white",ec=GRAY,alpha=0.9))
ax.legend(loc="upper right",fontsize=8,frameon=False)
ax.set_title("Figure 3. Diary headache-day proportion vs Life Recovery",fontsize=9.6,loc="left",weight="bold")
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig3_freq_restoration.png",bbox_inches="tight"); plt.close()

# ============ Supplementary Figure S1: 2x2 sample-median-split discordance ============
hmed=np.median(x); fmed=np.median(y)
mid=num('midas').where(d7).reindex(D.index)[m].values
figs,axs=plt.subplots(figsize=(5.8,4.7))
for hf in [False,True]:
    for hr in [False,True]:
        sel=((x>hmed)==hf)&((y>fmed)==hr)
        disc = hf==hr
        col = ORANGE if disc else GRAY
        axs.scatter(x[sel],y[sel],s=32,color=col,alpha=0.85,edgecolor="white",linewidth=0.4)
        if sel.sum()>0:
            mm=np.median(mid[sel][np.isfinite(mid[sel])])
            cx=(0.22 if not hf else 0.80); cy=(28 if not hr else 84)
            axs.text(cx,cy,f"n={sel.sum()}\nMIDAS {mm:.0f}",ha="center",fontsize=8.0,
                     color=(VERM if disc else "black"),
                     bbox=dict(boxstyle="round",fc="white",ec=(ORANGE if disc else GRAY),alpha=0.9))
axs.axvline(hmed,color="k",ls="--",lw=0.8); axs.axhline(fmed,color="k",ls="--",lw=0.8)
axs.text(hmed+0.01,2,f"sample median = {hmed:.2f}",fontsize=7.2,color=GRAY)
axs.text(0.01,fmed+1,f"sample median = {fmed:.1f}",fontsize=7.2,color=GRAY)
axs.set_xlabel("Diary headache-day proportion"); axs.set_ylabel("Life Recovery score")
axs.set_xlim(-0.02,1.02); axs.set_ylim(0,100)
axs.legend(handles=[Patch(color=ORANGE,label="Discordant (sample-relative)"),Patch(color=GRAY,label="Concordant")],
           loc="lower right",fontsize=7.8,frameon=False)
axs.set_title("Supplementary Figure S1. Sample-median-split discordance (secondary, descriptive)",fontsize=9.0,loc="left",weight="bold")
axs.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("figS1_discordance.png",bbox_inches="tight"); plt.close()

print("wrote fig1_flow.png, fig2_loadings.png, fig3_freq_restoration.png, figS1_discordance.png")
print(f"Fig2 primary loadings + bootstrap CI computed (B={B})")
print(f"Fig3: OLS slope={b1:.1f} [{slope_lo:.1f},{slope_hi:.1f}] rho={rho:+.2f} n={n}; medians hap={hmed:.2f} f1={fmed:.0f}")
