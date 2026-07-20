import numpy as np, pandas as pd
from scipy import stats

D = pd.read_csv('mfi_matrix.csv')
F = pd.read_csv('mfi_funnel.csv')
num = lambda c,src=D: pd.to_numeric(src[c], errors='coerce')

print("="*70)
print("COMMENT 5 — FUNNEL RECONCILIATION (from mfi_funnel.csv, 1,190 rows)")
print("="*70)
print("total rows in funnel:", len(F))
print("status counts:\n", F['status'].value_counts(dropna=False))
comp = F['status']=='completed'
print("\ncompleted:", comp.sum())
noncomp = F[~comp]
print("non-completers:", len(noncomp))
# scored items start at order_index 18 (q*_18..q*_99 = 82 scored items). 0-17 = consent/demographics
na = pd.to_numeric(noncomp['n_answered'], errors='coerce').fillna(0)
loi = pd.to_numeric(noncomp['last_order_index'], errors='coerce')
print("\n-- non-completers split by n_answered --")
print("  n_answered == 0 :", (na==0).sum())
print("  n_answered >= 1 :", (na>=1).sum())
print("\n-- non-completers split by last_order_index >= 18 (reached scored items) --")
print("  reached scored (loi>=18):", (loi>=18).sum())
print("  did NOT reach scored     :", (~(loi>=18)).sum(), " (NaN loi:", loi.isna().sum(),")")
# began/consent flow
print("\nconsented flag counts (all rows):\n", F['consented'].value_counts(dropna=False))
began = F[(F['consented']==True) | (pd.to_numeric(F['n_answered'],errors='coerce').fillna(0)>0)]
print("entered consent flow OR answered >=1 (proxy for 'began'):", len(began))

print("\n"+"="*70)
print("COMMENT 7 — MISSING-ITEM COMPLETENESS among 116 completers")
print("="*70)
dom_items = {1:'q0_',2:'q1_',3:'q2_',4:'q3_',5:'q4_',6:'q5_',7:'q6_',8:'q7_',9:'q8_',10:'q9_',11:'q10_'}
item_cols = [c for c in D.columns if any(c.startswith(p) for p in dom_items.values())]
print("total scored item columns:", len(item_cols))
# F1 = domains mapped; from supplement F1=50 items (dom with q0..? ) — use factor score presence.
# per-item non-missing fraction per participant
M = D[item_cols].apply(pd.to_numeric, errors='coerce')
frac_overall = M.notna().mean(axis=1)
print("overall item-answered fraction: min=%.3f  median=%.3f  (all>=0.75: %d/116)" % (
    frac_overall.min(), frac_overall.median(), (frac_overall>=0.75).sum()))
# per-domain fraction
dom_cols = {d:[c for c in item_cols if c.startswith(p)] for d,p in dom_items.items()}
min_dom_frac = pd.DataFrame({d: M[cols].notna().mean(axis=1) for d,cols in dom_cols.items()}).min(axis=1)
print("min per-domain answered fraction across 11 domains: min over ppl=%.3f (all domains>=0.75 for %d/116)" % (
    min_dom_frac.min(), (min_dom_frac>=0.75).sum()))
print("participants with ANY missing item:", (frac_overall<1.0).sum())

print("\n"+"="*70)
print("COMMENT 6 — RECORDING DENSITY & SECOND-FACTOR ADJUSTED ANALYSES")
print("="*70)
D['f1']=num('f1'); D['f2']=num('f2'); D['hap']=num('w28_haprop')
D['wd']=num('w28_days'); D['migd']=num('mig_days'); D['prev']=num('prev_use')
def sp(a,b,mask):
    s=D[mask].dropna(subset=[a,b]); r,p=stats.spearmanr(s[a],s[b]); return r,p,len(s)
link1 = D['wd']>=1
print("\n-- recording days (w28_days) vs factors, linked >=1 day --")
for f in ['f1','f2']:
    r,p,n=sp('wd',f,link1); print(f"  w28_days x {f}: rho={r:+.2f} p={p:.3f} n={n}")
r,p,n=sp('wd','hap',link1); print(f"  w28_days x headache-day proportion: rho={r:+.2f} p={p:.3f} n={n}")

def partial_spearman(y,x,controls,mask):
    s=D[mask].dropna(subset=[y,x]+controls).copy()
    R={c:stats.rankdata(s[c]) for c in [y,x]+controls}
    Z=np.column_stack([np.ones(len(s))]+[R[c] for c in controls])
    def resid(v):
        b=np.linalg.lstsq(Z,R[v],rcond=None)[0]; return R[v]-Z@b
    ry,rx=resid(y),resid(x)
    r=np.corrcoef(ry,rx)[0,1]; n=len(s); k=len(controls)
    t=r*np.sqrt((n-2-k)/(1-r**2)); p=2*stats.t.sf(abs(t),n-2-k)
    return r,p,n
link7 = D['wd']>=7
print("\n-- partial Spearman: factor x headache-day proportion, controlling recording days (>=7d) --")
for f in ['f1','f2']:
    r0,p0,n0=sp(f,'hap',link7)
    r,p,n=partial_spearman(f,'hap',['wd'],link7)
    print(f"  {f}: zero-order rho={r0:+.2f} (p={p0:.3f}) -> partial(|wd) rho={r:+.2f} (p={p:.3f}) n={n}")

def ols(y,Xcols,mask,std=True,w=None):
    s=D[mask].dropna(subset=[y]+Xcols+([w] if w else [])).copy(); n=len(s)
    X=s[Xcols].values.astype(float); yv=s[y].values.astype(float)
    if std:
        X=(X-X.mean(0))/X.std(0); yv=(yv-yv.mean())/yv.std()
    Xd=np.column_stack([np.ones(n),X])
    if w:
        wt=s[w].values.astype(float); W=np.diag(wt)
        beta=np.linalg.solve(Xd.T@W@Xd, Xd.T@W@yv); res=yv-Xd@beta
        sigma2=(res@ (wt*res))/(n-Xd.shape[1]); cov=sigma2*np.linalg.inv(Xd.T@W@Xd)
    else:
        beta=np.linalg.lstsq(Xd,yv,rcond=None)[0]; res=yv-Xd@beta
        sigma2=(res@res)/(n-Xd.shape[1]); cov=sigma2*np.linalg.inv(Xd.T@Xd)
    se=np.sqrt(np.diag(cov)); t=beta/se; p=2*stats.t.sf(np.abs(t),n-Xd.shape[1])
    return beta,se,p,n,Xcols
print("\n-- adjusted regression: f2 ~ headache-day prop + recording days + migraine freq + preventive use (standardized, >=7d) --")
b,se,p,n,cols=ols('f2',['hap','wd','migd','prev'],link7)
print(f"  n={n}")
for i,c in enumerate(['intercept']+cols):
    print(f"    {c:<8} beta={b[i]:+.2f} SE={se[i]:.2f} p={p[i]:.3f}")
print("\n-- same for f1 (candidate Life Restoration) --")
b,se,p,n,cols=ols('f1',['hap','wd','migd','prev'],link7)
print(f"  n={n}")
for i,c in enumerate(['intercept']+cols):
    print(f"    {c:<8} beta={b[i]:+.2f} SE={se[i]:.2f} p={p[i]:.3f}")

print("\n-- record-weighted OLS: f1 ~ headache-day prop, weights=recording days (>=7d) --")
b,se,p,n,cols=ols('f1',['hap'],link7,std=True,w='wd')
print(f"  n={n} standardized slope={b[1]:+.2f} SE={se[1]:.2f} p={p[1]:.3f}")

print("\n"+"="*70)
print("FIGURE 3 PANEL A — OLS f1 ~ headache-day proportion (raw units, >=7d)")
print("="*70)
s=D[link7].dropna(subset=['f1','hap']).copy(); n=len(s)
x=s['hap'].values; y=s['f1'].values
X=np.column_stack([np.ones(n),x])
beta=np.linalg.lstsq(X,y,rcond=None)[0]; res=y-X@beta
dof=n-2; sigma2=(res@res)/dof; cov=sigma2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov))
tcrit=stats.t.ppf(0.975,dof)
print(f"n={n}  slope={beta[1]:.1f} [95%CI {beta[1]-tcrit*se[1]:.1f}, {beta[1]+tcrit*se[1]:.1f}]  intercept={beta[0]:.1f}")
R2=1-(res@res)/(((y-y.mean())**2).sum()); print(f"R2={R2:.3f}")
rho,pr=stats.spearmanr(x,y); print(f"Spearman rho={rho:+.2f} p={pr:.3f}")
# Cook's distance
h=np.diag(X@np.linalg.inv(X.T@X)@X.T); mse=(res@res)/dof
cook=(res**2/(2*mse))*(h/(1-h)**2)
print(f"max Cook's distance={cook.max():.3f} (idx of max, f1={y[cook.argmax()]:.1f}, hap={x[cook.argmax()]:.2f})")
print(f"min f1 in sample={y.min():.1f} at hap={x[y.argmin()]:.2f}")
# sensitivity excluding min-f1 point
keep=np.arange(n)!=y.argmin()
xr,yr=x[keep],y[keep]; Xr=np.column_stack([np.ones(len(xr)),xr])
br=np.linalg.lstsq(Xr,yr,rcond=None)[0]; rr=yr-Xr@br
ser=np.sqrt(np.diag(((rr@rr)/(len(xr)-2))*np.linalg.inv(Xr.T@Xr)))
print(f"excluding min-f1 point: slope={br[1]:.1f} [95%CI {br[1]-tcrit*ser[1]:.1f}, {br[1]+tcrit*ser[1]:.1f}] n={len(xr)}")
# Huber robust regression (simple IRLS)
def huber(x,y,c=1.345,it=50):
    X=np.column_stack([np.ones(len(x)),x]); b=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(it):
        r=y-X@b; s=1.4826*np.median(np.abs(r-np.median(r))) or 1
        u=r/s; w=np.where(np.abs(u)<=c,1,c/np.abs(u))
        W=np.diag(w); b=np.linalg.solve(X.T@W@X,X.T@W@y)
    return b
bh=huber(x,y); print(f"Huber robust slope={bh[1]:.1f}")
print(f"residual normality Shapiro p={stats.shapiro(res).pvalue:.2f}")
