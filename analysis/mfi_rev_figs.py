#!/usr/bin/env python3
"""MFI 査読対応 Figure 1-3（n=116 凍結・出版用 300dpi・色覚バリアフリー Okabe-Ito）。"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from factor_analyzer import FactorAnalyzer
from scipy import stats
np.random.seed(20260718)
import os
DPI=int(os.environ.get("FIG_DPI","150"))  # 150=表示/Drive用・軽量／投稿最終は FIG_DPI=300 で再実行
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","axes.linewidth":0.8,"savefig.dpi":DPI,"figure.dpi":110})
BLUE,ORANGE,GREEN,VERM,GRAY="#0072B2","#E69F00","#009E73","#D55E00","#666666"
D=pd.read_csv('mfi_matrix.csv'); num=lambda c: pd.to_numeric(D[c],errors='coerce')

# ============ Figure 1: study flow ============
fig,ax=plt.subplots(figsize=(6.6,5.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,txt,fc="#EAF2F8",ec=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec=ec,lw=1.2))
    ax.text(x,y,txt,ha="center",va="center",fontsize=8.5)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,lw=1.1,color=GRAY))
box(3.3,9.2,4.6,1.0,"App users for whom the survey\nwas available\nN = 1190")
box(3.3,7.2,4.6,1.0,"Consented and began\nn = 181")
box(3.3,5.2,4.6,1.0,"Completed the questionnaire\n(analytic sample)\nn = 116")
arrow(3.3,8.7,3.3,7.7); arrow(3.3,6.7,3.3,5.7)
box(8.0,6.2,3.4,1.5,"Discontinued n = 65\n• at consent/first item: 30\n• after ≥1 item: 35\n(spread across sections)",fc="#FDF2E9",ec=ORANGE)
arrow(3.3,6.2,6.3,6.2)  # Consented→Completed の縦フローから横に枝分かれ（CONSORT作法）
box(2.0,3.0,3.4,1.2,"Linked diary analysis\n(≥7 days in 28-d window)\nn = 81",fc="#E8F6F0",ec=GREEN)
box(6.4,3.0,3.4,1.2,"Incremental-validity\nsubset (wellbeing)\nn = 75",fc="#E8F6F0",ec=GREEN)
arrow(2.6,4.7,2.0,3.6); arrow(4.0,4.7,6.4,3.6)
ax.text(5,0.8,"Frozen data cut: 18 July 2026 (00:00 JST)",ha="center",fontsize=8,style="italic",color=GRAY)
ax.set_title("Figure 1. Study flow and linked-data sample",fontsize=10,loc="left",weight="bold")
plt.tight_layout(); plt.savefig("fig1_flow.png",bbox_inches="tight"); plt.close()

# ============ Figure 2: provisional 2-factor loadings ============
DOMS=['脳状態遷移','トリガー認識','先制行動','発作コントロール','MOH予防','活動の自由','希望と治療継続','人生回復','社会的自由','身近な人への影響','総合評価']
EN=['Premonitory symptom awareness','Trigger recognition','Pre-emptive action','Attack control','Prevention of medication overuse','Freedom of activity','Hope & treatment continuity','Life restoration','Social freedom','Impact on close others','Global freedom evaluation']
X=D[[f'dom{i+1}' for i in range(11)]].apply(pd.to_numeric,errors='coerce'); X.columns=DOMS; X=X.dropna()
fa=FactorAnalyzer(n_factors=2,method='principal',rotation='oblimin'); fa.fit(X)
L=fa.loadings_.copy()
if abs(L[7,0])<abs(L[7,1]): L=L[:,::-1]  # F1=Life Restoration (人生回復=idx7 が最も乗る側)
prim=np.where(np.abs(L[:,0])>=np.abs(L[:,1]),0,1)  # 各ドメインの主因子
order=sorted(range(11),key=lambda i:(prim[i], -abs(L[i,prim[i]])))
fig,ax=plt.subplots(figsize=(7.0,4.6))
y=np.arange(11)[::-1]
for k,i in enumerate(order):
    pf=prim[i]; col=BLUE if pf==0 else ORANGE
    ax.barh(y[k],L[i,pf],color=col,height=0.62,edgecolor="white")
    ax.text(L[i,pf]+0.02,y[k],f"{L[i,pf]:.2f}",va="center",fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([EN[i] for i in order],fontsize=8.3)
ax.set_xlim(0,1.0); ax.set_xlabel("Primary factor loading (principal-axis, oblimin)")
ax.axvline(0,color="k",lw=0.8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="Factor 1 — Life Restoration"),Patch(color=ORANGE,label="Factor 2 — Migraine Agency")],
          loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=2,fontsize=8.5,frameon=False)
ax.set_title("Figure 2. Provisional higher-order structure (11 prespecified domains)\ninter-factor r = -0.00; parallel analysis supported 2 factors; 57.5% variance",fontsize=9.5,loc="left")
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig2_loadings.png",bbox_inches="tight"); plt.close()

# ============ Figure 3: frequency vs restoration ============
d7=num('w28_days')>=7
x=num('w28_haprop').where(d7); y1=num('f1')
m=x.notna()&y1.notna(); x=x[m].values; y=y1[m].values; n=len(x)
b1,b0=np.polyfit(x,y,1); xs=np.linspace(0,1,100); yh=b0+b1*xs
resid=y-(b0+b1*x); s2=(resid@resid)/(n-2); xbar=x.mean(); Sxx=((x-xbar)**2).sum()
tval=stats.t.ppf(0.975,n-2)
pi=tval*np.sqrt(s2*(1+1/n+(xs-xbar)**2/Sxx))
rho,pval=stats.spearmanr(x,y)
fig,axes=plt.subplots(1,2,figsize=(9.2,4.3))
# Panel A: continuous
ax=axes[0]
ax.fill_between(xs,yh-pi,yh+pi,color=BLUE,alpha=0.12,label="95% prediction interval")
ax.plot(xs,yh,color=BLUE,lw=1.8,label="OLS fit")
ax.scatter(x,y,s=26,color=GRAY,alpha=0.75,edgecolor="white",linewidth=0.4)
ax.set_xlabel("Diary headache-day proportion (28-day window)"); ax.set_ylabel("Life Restoration score (0-100)")
ax.set_xlim(-0.02,1.02); ax.set_ylim(0,100)
ax.text(0.03,8,f"Spearman ρ = {rho:+.2f}, p = {pval:.3f}\nn = {n}",fontsize=8.5,
        bbox=dict(boxstyle="round",fc="white",ec=GRAY,alpha=0.9))
ax.legend(loc="upper right",fontsize=8,frameon=False)
ax.set_title("A. Continuous (primary)",fontsize=9.5,loc="left"); ax.spines[['top','right']].set_visible(False)
# Panel B: 2x2 discordance
ax=axes[1]
hmed=np.median(x); fmed=np.median(y)
mid=num('midas').where(d7).reindex(D.index)[m].values
def q(hf,hr): return (x>hmed)==hf if False else None
for hf in [False,True]:
    for hr in [False,True]:
        sel=((x>hmed)==hf)&((y>fmed)==hr)
        disc = hf==hr
        col = VERM if disc else GRAY
        ax.scatter(x[sel],y[sel],s=30,color=col,alpha=0.8,edgecolor="white",linewidth=0.4,
                   label=None)
        if sel.sum()>0:
            mm=np.median(mid[sel][np.isfinite(mid[sel])])
            cx=(0.25 if not hf else 0.78); cy=(30 if not hr else 82)
            ax.text(cx,cy,f"n={sel.sum()}\nMIDAS {mm:.0f}"+("\n(discordant)" if disc else ""),
                    ha="center",fontsize=7.8,color=(VERM if disc else "black"),
                    bbox=dict(boxstyle="round",fc="white",ec=(VERM if disc else GRAY),alpha=0.85))
ax.axvline(hmed,color="k",ls="--",lw=0.8); ax.axhline(fmed,color="k",ls="--",lw=0.8)
ax.set_xlabel("Diary headache-day proportion (median split)"); ax.set_ylabel("Life Restoration (median split)")
ax.set_xlim(-0.02,1.02); ax.set_ylim(0,100)
ax.set_title("B. Sample-relative discordance (secondary)",fontsize=9.5,loc="left"); ax.spines[['top','right']].set_visible(False)
fig.suptitle("Figure 3. Headache-day frequency and Life Restoration are non-equivalent",fontsize=10,x=0.02,ha="left",weight="bold")
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("fig3_freq_restoration.png",bbox_inches="tight"); plt.close()
print("wrote fig1_flow.png, fig2_loadings.png, fig3_freq_restoration.png")
print(f"Fig3: rho={rho:+.2f} p={pval:.3f} n={n}; medians hap={hmed:.2f} f1={fmed:.0f}")
