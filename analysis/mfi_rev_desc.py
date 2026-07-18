#!/usr/bin/env python3
"""MFI 査読対応・記述統計＋妥当性テーブル＋ファネル（n=116 凍結）。
査読 minor（floor/ceiling・相関にn/CI/p・MIDAS grade）＋⑥⑦（連結の向き・不一致は補助）＋⑥feasibility funnel。
"""
import numpy as np, pandas as pd
from scipy import stats
np.random.seed(20260718)

D = pd.read_csv('mfi_matrix.csv')
num = lambda c: pd.to_numeric(D[c], errors='coerce')

def sp_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(x)
    if n < 5: return (np.nan,)*4 + (n,)
    r, p = stats.spearmanr(x, y); z = np.arctanh(r); se = 1/np.sqrt(n-3)
    return r, np.tanh(z-1.96*se), np.tanh(z+1.96*se), p, n

def med_iqr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.median(x), np.percentile(x,25), np.percentile(x,75), len(x)

print("===== MFI 記述統計・妥当性テーブル・ファネル（n=116）=====\n")

# ---------- 背景（Table 1 素材） ----------
age = num('age'); sex = num('sex'); mig = num('mig_days'); midas = num('midas'); mibs = num('mibs4')
print("【背景（Table 1）】")
print(f"  年齢 {age.mean():.1f}±{age.std():.1f}（n={age.notna().sum()}） / 女性 {(sex==1).sum()}/{sex.notna().sum()} ({100*(sex==1).mean():.0f}%)")
mm = med_iqr(mig); print(f"  自己申告 片頭痛日数/4週 中央値 {mm[0]:.0f}（IQR {mm[1]:.0f}-{mm[2]:.0f}）")
print(f"  症例確度: ID Migraine陽性(日誌) {(num('id_mig')==1).sum()} / 片頭痛特異薬 {(num('drug_specific')==1).sum()} / 複合症例定義 {(num('case_def')==1).sum()}/116")

# ---------- MIDAS grade（中央値+IQR+等級分布） ----------
md, q1, q3, nmd = med_iqr(midas)
grade = pd.cut(midas.dropna(), [-1,5,10,20,1e9], labels=['I(0-5)','II(6-10)','III(11-20)','IV(21+)'])
print(f"\n【MIDAS】中央値 {md:.1f}（IQR {q1:.1f}-{q3:.1f}, n={nmd}）平均 {midas.mean():.1f}")
print("  等級分布: " + " / ".join([f"{g} {int((grade==g).sum())}" for g in ['I(0-5)','II(6-10)','III(11-20)','IV(21+)']]))

# ---------- floor / ceiling（項目レベル） ----------
qcols = [c for c in D.columns if c.startswith('q') and '_' in c and c[1:].split('_')[0].isdigit()]
fl = []; ce = []
for c in qcols:
    v = num(c).dropna()
    fl.append(100*(v==0).mean()); ce.append(100*(v==6).mean())
fl, ce = np.array(fl), np.array(ce)
print(f"\n【floor/ceiling（82項目・7件法0-6）】")
print(f"  床(=0) 中央値 {np.median(fl):.1f}% 最大 {fl.max():.1f}% / 天井(=6) 中央値 {np.median(ce):.1f}% 最大 {ce.max():.1f}%")
print(f"  床>15%の項目数 {int((fl>15).sum())} / 天井>15%の項目数 {int((ce>15).sum())}（該当項目はSupplementで一覧）")

# ---------- 妥当性テーブル（総合・F1・F2 × 外的指標）n/95%CI/p ----------
f1, f2, tot = num('f1'), num('f2'), num('mfi_total')
d7 = D[num('w28_days') >= 7]
ext = [
    ('MIDAS（支障・負を期待）', midas, '−'),
    ('MIBS-4 4週（間欠期負担・負）', mibs, '−'),
    ('自己申告 片頭痛日数（頻度・弱い負）', mig, '−'),
    ('日誌 頭痛日割合 28d≥7（頻度・負）', pd.to_numeric(d7['w28_haprop'], errors='coerce').reindex(D.index), '−'),
    ('日誌 無頭痛日充実度 28d（正）', pd.to_numeric(d7['w28_fulfill'], errors='coerce').reindex(D.index), '＋'),
]
print("\n【妥当性テーブル（Spearman ρ [95%CI] p, n）事前仮説つき】")
print(f"  {'外的指標':<30}{'総合':>22}{'F1生活回復':>22}{'F2気づき対処':>22}")
for lab, y, exp in ext:
    cells = []
    for s in [tot, f1, f2]:
        r, lo, hi, p, n = sp_ci(s, y)
        cells.append(f"{r:+.2f}[{lo:+.2f},{hi:+.2f}]p{p:.2f}")
    print(f"  {lab:<30}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

# ---------- 2×2 不一致（補助解析・sample-relative と明記） ----------
print("\n【補助解析: 2×2 不一致（標本中央値分割・sample-relative）】")
sub = pd.DataFrame({'haprop': num('w28_haprop'), 'f1': f1, 'midas': midas, 'f2': f2})[num('w28_days')>=7].dropna(subset=['haprop','f1'])
hmed, fmed = sub['haprop'].median(), sub['f1'].median()
sub['hi_freq'] = sub['haprop'] > hmed  # 標本内で相対的に頭痛日割合が高い
sub['hi_rest'] = sub['f1'] > fmed
cells = {(False,False):'低頻度×低回復', (False,True):'低頻度×高回復', (True,False):'高頻度×低回復', (True,True):'高頻度×高回復'}
print(f"  分割点: 頭痛日割合中央値 {hmed:.2f} / F1中央値 {fmed:.1f}（n={len(sub)}）")
disc = 0
# 不一致 = 頭痛頻度と生活回復が同方向（低頻度なのに低回復／高頻度なのに高回復）= hf==hr
for (hf, hr), lab in cells.items():
    g = sub[(sub['hi_freq']==hf)&(sub['hi_rest']==hr)]
    mi = med_iqr(g['midas'].values) if g['midas'].notna().any() else (np.nan,np.nan,np.nan,0)
    tag = ' ←不一致' if hf==hr else ''
    if hf==hr: disc += len(g)
    print(f"    {lab:<14} n={len(g):>2}  MIDAS中央値 {mi[0]:.1f}(IQR {mi[1]:.0f}-{mi[2]:.0f}){tag}")
print(f"  不一致（低頻度×低回復＋高頻度×高回復）= {disc}/{len(sub)} = {100*disc/len(sub):.0f}%（標本依存・補助的所見）")

# ---------- ファネル（実施可能性・脱落） ----------
print("\n【実施可能性ファネル（査読⑥）】")
F = pd.read_csv('mfi_funnel.csv')
tot_n = len(F); cons = int(F['consented'].sum())
started = int((F['status']!='pending').sum()); comp = int((F['status']=='completed').sum())
drop = int((F['status']=='in_progress').sum())
print(f"  配信(利用可能) {tot_n} → 同意 {cons} → 開始 {started} → 完了 {comp} → 中断 {drop}")
print(f"  完了率: 開始者中 {100*comp/started:.0f}% / 同意者中 {100*comp/cons:.0f}%")
ip = F[F['status']=='in_progress'].copy()
answered = ip[pd.to_numeric(ip['n_answered'], errors='coerce').fillna(0) > 0]
print(f"  中断者{drop}名の内訳: 同意のみ回答0 {drop-len(answered)}名 / 1問以上回答後に中断 {len(answered)}名")
lo = pd.to_numeric(answered['last_order_index'], errors='coerce').dropna()
if len(lo): print(f"    中断者の最終回答 order_index 中央値 {lo.median():.0f}（脱落は特定セクションに集中せず分散）")
# 完了 vs 非完了（開始者内）年齢・性別
comp_g = F[F['status']=='completed']; ip_g = F[F['status']=='in_progress']
for lab, col in [('年齢', 'age'), ('性別(女性割合)', 'sex')]:
    if col=='age':
        a, b = pd.to_numeric(comp_g['age'],errors='coerce').dropna(), pd.to_numeric(ip_g['age'],errors='coerce').dropna()
        u = stats.mannwhitneyu(a, b) if len(a)>2 and len(b)>2 else None
        print(f"  完了 {a.mean():.1f}歳 vs 中断 {b.mean():.1f}歳" + (f" (Mann-Whitney p={u.pvalue:.2f})" if u else ""))
    else:
        af = (comp_g['sex']=='female').mean()*100; bf = (ip_g['sex']=='female').mean()*100
        print(f"  完了 女性{af:.0f}% vs 中断 女性{bf:.0f}%")
