#!/usr/bin/env python3
"""MFI 査読対応・因子分析パッケージ（n=116 凍結データ）。
査読コメント②③対応：
 - ドメインレベル11変数EFA（主因子法+oblimin）を主解析（provisional higher-order structure）
 - 平行分析・スクリー・初期固有値・共通性・全ローディング（クロス含む）・1/2/3因子比較・ブートストラップ安定性
 - 2因子別 alpha + McDonald's omega・ドメイン間相関行列
 - 項目レベル82項目EFAは探索的（n/項目比が低い）として KMO・平行分析・第1固有値のみ
出力: コンソール（analysis-results.md へ転記）。
"""
import numpy as np, pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

np.random.seed(20260718)  # 再現性（乱数はブートストラップ・平行分析のみ）
D = pd.read_csv('mfi_matrix.csv')
DOMS = ['脳状態遷移','トリガー認識','先制行動','発作コントロール','MOH予防','活動の自由','希望と治療継続','人生回復','社会的自由','身近な人への影響','総合評価']
DOM_EN = {'脳状態遷移':'Premonitory awareness','トリガー認識':'Trigger recognition','先制行動':'Preemptive action','発作コントロール':'Attack control','MOH予防':'MOH prevention','活動の自由':'Activity freedom','希望と治療継続':'Hope/treatment continuation','人生回復':'Life restoration','社会的自由':'Social freedom','身近な人への影響':'Impact on close others','総合評価':'Overall'}
F1 = ['発作コントロール','活動の自由','希望と治療継続','人生回復','社会的自由','身近な人への影響','総合評価']   # Life Restoration
F2 = ['脳状態遷移','トリガー認識','先制行動','MOH予防']                                              # Migraine Agency
domcol = {DOMS[i]: f'dom{i+1}' for i in range(11)}

X = D[[domcol[d] for d in DOMS]].apply(pd.to_numeric, errors='coerce').dropna()
X.columns = DOMS
n, p = X.shape
print(f"===== MFI 因子分析（査読対応・n={n} ドメインレベル {p}変数）=====\n")

# --- KMO / Bartlett ---
kmo_all, kmo_model = calculate_kmo(X)
chi2, bart_p = calculate_bartlett_sphericity(X)
print(f"KMO(全体)={kmo_model:.3f}  Bartlett χ²={chi2:.1f}, p={bart_p:.2e}")

# --- 初期固有値・スクリー・平行分析 ---
R = np.corrcoef(X.values, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(R))[::-1]
# Horn 平行分析（ランダム正規データの固有値95%タイル）
B = 1000
rand_eigs = np.zeros((B, p))
for b in range(B):
    Z = np.random.normal(size=(n, p))
    rand_eigs[b] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
pa95 = np.percentile(rand_eigs, 95, axis=0)
n_retain = int(np.sum(eig > pa95))
print("\n【初期固有値・平行分析】(retain: 実固有値 > ランダム95%タイル)")
for i in range(min(6, p)):
    print(f"  因子{i+1}: 実固有値 {eig[i]:.2f}  / PA95 {pa95[i]:.2f}  {'← 保持' if eig[i]>pa95[i] else ''}")
print(f"  → 平行分析が支持する因子数 = {n_retain}")

# --- 1/2/3因子モデル比較（累積寄与率） ---
print("\n【因子数モデル比較（主因子法・寄与率）】")
for k in [1, 2, 3]:
    fa = FactorAnalyzer(n_factors=k, method='principal', rotation='oblimin' if k > 1 else None)
    fa.fit(X)
    ev = fa.get_factor_variance()  # (variance, proportional, cumulative)
    print(f"  {k}因子: 各寄与 {[round(v,3) for v in ev[1]]}  累積 {ev[2][-1]:.3f}")

# --- 2因子解（主解析） ---
fa2 = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin')
fa2.fit(X)
load = pd.DataFrame(fa2.loadings_, index=DOMS, columns=['Factor1', 'Factor2'])
comm = fa2.get_communalities()
phi = fa2.phi_  # 因子間相関
# 因子の向きを Life Restoration が Factor1 に来るよう整える（人生回復が最も高く乗る方をF1）
if abs(load.loc['人生回復', 'Factor1']) < abs(load.loc['人生回復', 'Factor2']):
    load = load[['Factor2', 'Factor1']]; load.columns = ['Factor1', 'Factor2']; phi = phi[::-1, ::-1]
print("\n【2因子解（主因子法+oblimin）パターン行列・共通性】")
print(f"{'ドメイン':<16}{'F1(生活回復)':>12}{'F2(気づき対処)':>14}{'共通性h²':>10}   期待因子")
for i, d in enumerate(DOMS):
    exp = 'F1' if d in F1 else 'F2'
    l1, l2 = load.iloc[i, 0], load.iloc[i, 1]
    cross = '  ★クロス' if min(abs(l1), abs(l2)) >= 0.32 else ''
    print(f"{d:<16}{l1:>12.2f}{l2:>14.2f}{comm[i]:>10.2f}   {exp}{cross}")
print(f"  因子間相関 r(F1,F2) = {phi[0,1]:+.3f}   （≈0 なら2軸は独立）")

# --- 2因子別 alpha + McDonald's omega（項目レベル） ---
qcols = [c for c in D.columns if c.startswith('q') and '_' in c and c[1:].split('_')[0].isdigit()]
f1idx, f2idx = {3,5,6,7,8,9,10}, {0,1,2,4}
f1items = [c for c in qcols if int(c.split('_')[0][1:]) in f1idx]
f2items = [c for c in qcols if int(c.split('_')[0][1:]) in f2idx]

def cronbach_alpha(df):
    df = df.dropna(); k = df.shape[1]
    iv = df.var(axis=0, ddof=1).sum(); tv = df.sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1 - iv/tv), df.shape[0]

def omega_total(df):
    df = df.dropna()
    fa = FactorAnalyzer(n_factors=1, method='principal', rotation=None); fa.fit(df)
    lam = fa.loadings_[:, 0]; sl = lam.sum()
    uniq = 1 - lam**2
    return (sl**2) / (sl**2 + uniq.sum()), df.shape[0]

print("\n【2因子別 信頼性（項目レベル）】")
for name, items in [('F1 生活回復', f1items), ('F2 気づき対処', f2items)]:
    sub = D[items].apply(pd.to_numeric, errors='coerce')
    a, na = cronbach_alpha(sub); om, no = omega_total(sub)
    print(f"  {name}: {len(items)}項目  α={a:.2f}  ω={om:.2f}  (listwise n={na})")
suball = D[qcols].apply(pd.to_numeric, errors='coerce')
a, na = cronbach_alpha(suball)
print(f"  （参考）総合 {len(qcols)}項目: α={a:.2f} (listwise n={na}) ※2因子が独立なため総合点は補助指標に降格")

# --- ドメイン間相関行列 ---
print("\n【ドメイン間相関行列（Pearson）】")
Rd = pd.DataFrame(R, index=[DOM_EN[d] for d in DOMS], columns=[f'D{i+1}' for i in range(p)])
print(Rd.round(2).to_string())

# --- ブートストラップ・ローディング安定性 ---
print("\n【ブートストラップ ローディング安定性（B=1000, 2因子oblimin）】")
ref = load.values.copy()
Bl = 1000; boot = np.full((Bl, p, 2), np.nan)
for b in range(Bl):
    idx = np.random.randint(0, n, n)
    Xb = X.values[idx]
    try:
        fb = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin'); fb.fit(pd.DataFrame(Xb, columns=DOMS))
        L = fb.loadings_.copy()
        # 参照解へ列マッチ＆符号合わせ（Tucker congruence 最大化）
        best = None
        for perm in ([0,1],[1,0]):
            for s1 in (1,-1):
                for s2 in (1,-1):
                    C = L[:, perm] * np.array([s1, s2])
                    score = abs(np.sum(C*ref))
                    if best is None or score > best[0]: best = (score, C)
        boot[b] = best[1]
    except Exception:
        pass
print(f"{'ドメイン':<16}{'F1 median[2.5,97.5]':>26}{'F2 median[2.5,97.5]':>26}")
for i, d in enumerate(DOMS):
    f1b = boot[:, i, 0][~np.isnan(boot[:, i, 0])]; f2b = boot[:, i, 1][~np.isnan(boot[:, i, 1])]
    print(f"{d:<16}{np.median(f1b):>8.2f}[{np.percentile(f1b,2.5):>5.2f},{np.percentile(f1b,97.5):>5.2f}]"
          f"{np.median(f2b):>10.2f}[{np.percentile(f2b,2.5):>5.2f},{np.percentile(f2b,97.5):>5.2f}]")

# --- 項目レベル82項目EFA（探索的・n/項目比が低い） ---
print("\n【項目レベル 82項目EFA（探索的）】")
It = D[qcols].apply(pd.to_numeric, errors='coerce')
It = It.fillna(It.median())
kmo_i_all, kmo_i = calculate_kmo(It)
Ri = np.corrcoef(It.values, rowvar=False)
eig_i = np.sort(np.linalg.eigvalsh(Ri))[::-1]
rand_i = np.zeros((200, len(qcols)))
for b in range(200):
    Z = np.random.normal(size=It.shape); rand_i[b] = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
pa_i = np.percentile(rand_i, 95, axis=0)
print(f"  KMO={kmo_i:.2f}  第1固有値 {eig_i[0]:.1f}（強い一般因子）  第2 {eig_i[1]:.1f} 第3 {eig_i[2]:.1f}")
print(f"  平行分析 保持因子数 ≈ {int(np.sum(eig_i > pa_i))}（n/項目比 {n/len(qcols):.2f} と低く不安定→項目レベルは探索的位置づけ）")
print("\n注: 主解析はドメインレベル2因子（provisional higher-order structure）。項目レベルは大規模Nで確認予定。")
