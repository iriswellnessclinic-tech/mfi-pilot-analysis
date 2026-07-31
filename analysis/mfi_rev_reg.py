#!/usr/bin/env python3
"""MFI 査読対応・回帰パッケージ（n=116 凍結データ）。
査読コメント⑦⑧対応：
 - median-split を降格し、主解析＝連続回帰（F1/F2/総合 ↔ 28日窓の頭痛日割合）: 傾き・95%CI・予測区間・残差
 - incremental validity（探索的）: DV=28日窓の無頭痛日充実度、M1=頭痛日割合+MIDAS、M2=+F1+F2
   標準化β・VIF・Cook's D・調整R²・F-change p・ΔR²のブートストラップCI・残差正規性(Shapiro)
 - 日誌28日窓の記録日数しきい値（≥7/≥14/≥21/≥80%）で感度解析
出力: コンソール。
"""
import numpy as np, pandas as pd
from scipy import stats
np.random.seed(20260718)

D = pd.read_csv('mfi_matrix.csv')
num = lambda c: pd.to_numeric(D[c], errors='coerce')
d = pd.DataFrame({'f1': num('f1'), 'f2': num('f2'), 'total': num('mfi_total'), 'midas': num('midas'),
                  'haprop': num('w28_haprop'), 'wdays': num('w28_days'), 'fulfill': num('w28_fulfill'),
                  'ha_all': num('diary_ha_rate')})

def ols(y, Xcols, data):
    """OLS（切片自動付与）→ 係数・SE・t・p・R²・調整R²・F・標準化β・VIF・Cook's D・残差。"""
    sub = data[[*Xcols, y]].dropna()
    n = len(sub); k = len(Xcols)
    X = np.column_stack([np.ones(n)] + [sub[c].values for c in Xcols])
    yv = sub[y].values
    XtXinv = np.linalg.inv(X.T @ X)
    beta = XtXinv @ X.T @ yv
    resid = yv - X @ beta
    dof = n - k - 1
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtXinv))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    sst = np.sum((yv - yv.mean())**2); ssr = resid @ resid
    r2 = 1 - ssr/sst; adj = 1 - (1-r2)*(n-1)/dof
    F = (r2/k) / ((1-r2)/dof); Fp = stats.f.sf(F, k, dof)
    # 標準化β
    sy = yv.std(ddof=1); std_beta = [beta[j+1]*sub[Xcols[j]].std(ddof=1)/sy for j in range(k)]
    # VIF
    vif = []
    for j in range(k):
        others = [c for c in Xcols if c != Xcols[j]]
        if not others: vif.append(1.0); continue
        Xj = np.column_stack([np.ones(n)] + [sub[c].values for c in others])
        bj = np.linalg.lstsq(Xj, sub[Xcols[j]].values, rcond=None)[0]
        rj = sub[Xcols[j]].values - Xj @ bj
        r2j = 1 - (rj@rj)/np.sum((sub[Xcols[j]].values-sub[Xcols[j]].mean())**2)
        vif.append(1/(1-r2j) if r2j < 1 else np.inf)
    # Cook's D
    H = X @ XtXinv @ X.T; h = np.diag(H)
    mse = ssr/(n-k-1)
    cook = (resid**2 / ((k+1)*mse)) * (h/(1-h)**2)
    return dict(n=n, k=k, beta=beta, se=se, t=t, p=pval, r2=r2, adj=adj, F=F, Fp=Fp,
                std_beta=std_beta, vif=vif, resid=resid, cook=cook, Xcols=Xcols, sub=sub)

def spearman_ci(x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    r, p = stats.spearmanr(x, y)
    z = np.arctanh(r); se = 1/np.sqrt(n-3)
    lo, hi = np.tanh(z-1.96*se), np.tanh(z+1.96*se)
    return r, lo, hi, p, n

print("===== MFI 回帰パッケージ（査読対応・n=116）=====\n")

# ---------- ① 連続回帰（主解析）: 各スコア ↔ 28日窓 頭痛日割合 ----------
print("【① 連続回帰（主解析）: MFIスコア ↔ 28日窓の頭痛日割合】(記録≥7日に限定)")
base = d[d['wdays'] >= 7].copy()
for lab, col in [('F1 生活回復', 'f1'), ('F2 気づき対処', 'f2'), ('総合', 'total')]:
    r, lo, hi, p, nn = spearman_ci(base[col].values, base['haprop'].values)
    m = ols(col, ['haprop'], base)
    b, seb = m['beta'][1], m['se'][1]
    print(f"  {lab:<10} Spearman ρ={r:+.2f} [95%CI {lo:+.2f},{hi:+.2f}] p={p:.3f} (n={nn}) / OLS傾き={b:+.1f}±{seb:.1f} (100%頭痛あたり点)")
# 予測区間（F1~haprop 例）
m = ols('f1', ['haprop'], base); sub = m['sub']
xg = np.linspace(0, 1, 5); Xn = len(sub)
s2 = (m['resid']@m['resid'])/(Xn-2)
xbar = sub['haprop'].mean(); sxx = np.sum((sub['haprop']-xbar)**2)
print("  F1予測（回帰線±95%予測区間）:")
for xv in xg:
    yhat = m['beta'][0] + m['beta'][1]*xv
    pi = 1.96*np.sqrt(s2*(1 + 1/Xn + (xv-xbar)**2/sxx))
    print(f"    頭痛日割合 {xv:.0%}: F1予測 {yhat:.1f} [{yhat-pi:.1f}, {yhat+pi:.1f}]")
sh = stats.shapiro(m['resid'])
print(f"  残差正規性(F1モデル) Shapiro W={sh.statistic:.3f} p={sh.pvalue:.3f}")

# ---------- ② incremental validity（探索的）DV=無頭痛日充実度 ----------
print("\n【② incremental validity（探索的）DV=28日窓 無頭痛日の充実度】")
reg = d.dropna(subset=['fulfill', 'haprop', 'midas', 'f1', 'f2']).copy()
reg = reg[reg['wdays'] >= 7]
print(f"  解析対象 n={len(reg)}（28日窓≥7日 かつ 充実度・MIDAS・F1・F2 完備）")
m1 = ols('fulfill', ['haprop', 'midas'], reg)
m2 = ols('fulfill', ['haprop', 'midas', 'f1', 'f2'], reg)
dR2 = m2['r2'] - m1['r2']
df1 = m2['k'] - m1['k']; df2 = m2['n'] - m2['k'] - 1
Fchg = (dR2/df1) / ((1-m2['r2'])/df2); Fp = stats.f.sf(Fchg, df1, df2)
print(f"  M1(頭痛日割合+MIDAS): R²={m1['r2']:.3f} 調整R²={m1['adj']:.3f}")
print(f"  M2(+F1+F2):          R²={m2['r2']:.3f} 調整R²={m2['adj']:.3f}")
print(f"  ΔR²={dR2:.3f}  F-change({df1},{df2})={Fchg:.2f}  p={Fp:.3f}  {'（有意）' if Fp<0.05 else '（有意水準に未達＝探索的）'}")
# M2 係数（標準化β・p・VIF）
print("  M2 係数: " + "  ".join([f"{c}: β*={m2['std_beta'][j]:+.2f}(p={m2['p'][j+1]:.3f},VIF={m2['vif'][j]:.1f})" for j, c in enumerate(m2['Xcols'])]))
sh2 = stats.shapiro(m2['resid'])
infl = np.sum(m2['cook'] > 4/m2['n'])
print(f"  残差正規性 Shapiro W={sh2.statistic:.3f} p={sh2.pvalue:.3f} / 影響観測(Cook's D>4/n): {infl}件 (最大 {m2['cook'].max():.3f})")
# ΔR² のブートストラップ95%CI
B = 2000; dr = []
rv = reg.reset_index(drop=True)
for b in range(B):
    idx = np.random.randint(0, len(rv), len(rv)); bs = rv.iloc[idx]
    try:
        a1 = ols('fulfill', ['haprop', 'midas'], bs); a2 = ols('fulfill', ['haprop', 'midas', 'f1', 'f2'], bs)
        dr.append(a2['r2'] - a1['r2'])
    except Exception: pass
print(f"  ΔR² ブートストラップ95%CI [{np.percentile(dr,2.5):.3f}, {np.percentile(dr,97.5):.3f}]（0を含む={'はい' if np.percentile(dr,2.5)<=0 else 'いいえ'}）")

# ---------- ③ 日誌28日窓 記録日数しきい値の感度解析 ----------
print("\n【③ 感度解析: 28日窓の記録日数しきい値別（F1↔頭痛日割合 / 総合↔頭痛日割合）】")
print(f"  {'しきい値':<12}{'n':>4}{'F1↔haprop ρ[95%CI]':>26}{'総合↔haprop ρ[95%CI]':>26}")
for lab, th in [('≥1日', 1), ('≥7日', 7), ('≥14日', 14), ('≥21日', 21), ('≥80%(22.4)', 22.4)]:
    s = d[d['wdays'] >= th]
    r1, l1, h1, _, n1 = spearman_ci(s['f1'].values, s['haprop'].values)
    r2, l2, h2, _, _ = spearman_ci(s['total'].values, s['haprop'].values)
    print(f"  {lab:<12}{n1:>4}{f'{r1:+.2f}[{l1:+.2f},{h1:+.2f}]':>26}{f'{r2:+.2f}[{l2:+.2f},{h2:+.2f}]':>26}")
print("  ※ 高頻度患者ほど無頭痛日が少なく充実度の観測日が減る（頻度条件づき選択バイアス）→Limitationに明記。")

# ---------- ④ 記録行動アーティファクト検証 + 年齢・性別調整（Supplementary S3e）----------
print("\n【④ 記録アーティファクト検証・調整モデル・年齢性別調整（S3e）】")
from scipy.stats import rankdata
def pspear(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]
e = pd.DataFrame({'f1': num('f1'), 'f2': num('f2'), 'haprop': num('w28_haprop'),
                  'wdays': num('w28_days'), 'mmd': num('mig_days'), 'prev': num('prev_use'), 'age': num('age')})
e['female'] = (num('sex') == 1).astype(float)
rec = e[e['wdays'] >= 1].copy()
for lab, col in [('F1', 'f1'), ('F2', 'f2')]:
    s = rec.dropna(subset=[col, 'wdays']); r, lo, hi, p, nn = spearman_ci(s['wdays'].values, s[col].values)
    print(f"  記録日数↔{lab}: rho={r:+.2f} p={p:.3f} (n={nn})")
s = rec.dropna(subset=['wdays', 'haprop']); r, lo, hi, p, nn = spearman_ci(s['wdays'].values, s['haprop'].values)
print(f"  記録日数↔haprop: rho={r:+.2f} p={p:.3f} (n={nn})")
b7 = e[e['wdays'] >= 7].dropna(subset=['haprop']).copy()
print(f"  F1↔haprop 偏|記録日数: rho={pspear(b7['f1'], b7['haprop'], b7['wdays']):+.2f} (n={len(b7)})")
print(f"  F2↔haprop 偏|記録日数: rho={pspear(b7['f2'], b7['haprop'], b7['wdays']):+.2f}")
for lab, y in [('F1', 'f1'), ('F2', 'f2')]:
    m = ols(y, ['haprop', 'wdays', 'mmd', 'prev'], b7)
    print(f"  [{lab} 調整(記録日数/MMD/予防薬)] " + " ".join(f"{c} b*={m['std_beta'][j]:+.2f}(p={m['p'][j+1]:.3f})" for j, c in enumerate(m['Xcols'])) + f" n={m['n']}")
    m2 = ols(y, ['haprop', 'wdays', 'mmd', 'prev', 'age', 'female'], b7)
    print(f"  [{lab} +年齢+性別]          " + " ".join(f"{c} b*={m2['std_beta'][j]:+.2f}(p={m2['p'][j+1]:.3f})" for j, c in enumerate(m2['Xcols'])) + f" n={m2['n']}")

# ---------- ⑤ 薬剤特異的サブグループの収束的妥当性（Life Restoration）----------
print("\n【⑤ 薬剤特異的サブグループ(n=62) Life Restoration 収束的妥当性】")
sub_mask = (num('drug_specific') == 1)
for lab, x in [('MIDAS', 'midas'), ('MIBS-4', 'mibs4')]:
    s = pd.DataFrame({'f1': num('f1'), 'x': num(x)})[sub_mask].dropna()
    r, lo, hi, p, nn = spearman_ci(s['f1'].values, s['x'].values)
    print(f"  Life Restoration↔{lab}: rho={r:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f} (n={nn})")
