#!/usr/bin/env python3
"""Cephalalgia 査読（2026-10-26・不採択）への追加解析。凍結データ mfi_matrix.csv（n=115）を使う。
Referee 2 の主要点に対応:
 A. 項目→S1番号の対応づけと自己点検（dom スコア = 項目平均 の再現）
 B. 項目レベル: 2因子 EFA の項目ローディング・項目-残り相関（自ドメイン内／自因子内）・alpha-if-deleted
 C. 重み付けスコア: 単純平均（現行）vs ドメイン平均（ドメイン等重み）vs ローディング重み vs 回帰法因子得点 → 外的指標との相関を比較
 D. 記録密度: Poisson GLM（offset=log 記録日数・頑健SE）・記録日数で調整した偏相関・外挿なしの実数・記録密度と頭痛率の関係（MAR の確認）・記録日数しきい値
 E. 冗長性: 項目間相関 ≥0.80 の組・ドメイン別 平均項目間相関・alpha-if-deleted・ドメイン内一次元性（固有値比・平行分析・単因子ローディング）
 F. 片頭痛非該当 3 名を除いた感度解析
 G. MIBS-4 の採点と分布
 H. 治療の種類別（CGRP 関連薬・予防薬・片頭痛特異薬）の因子得点と MMD
 I. 自己効力感に近い 2 項目（#8・#29）を外した Life Restoration
 J. Migraine Agency: 予兆・MOH ドメインの Life Restoration への交差負荷、症状を前提とする項目の床効果
実行: cd <repo root> && python3 analysis/r2_reviewer_reanalysis.py
"""
import json, os, re, html, sys
import numpy as np, pandas as pd
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo

np.random.seed(20260718)
D = pd.read_csv('mfi_matrix.csv')
num = lambda c: pd.to_numeric(D[c], errors='coerce')
N = len(D)
print(f"===== Cephalalgia 査読対応・追加解析（n={N}）=====")

DOMS_JA = ['脳状態遷移', 'トリガー認識', '先制行動', '発作コントロール', 'MOH予防', '活動の自由', '希望と治療継続', '人生回復', '社会的自由', '身近な人への影響', '総合評価']
DOM_EN = ['Premonitory-symptom awareness', 'Trigger recognition', 'Pre-emptive action', 'Attack control', 'Prevention of medication overuse',
          'Freedom of activity', 'Hope and treatment continuity', 'Reclaiming your life', 'Social freedom', 'Impact on close others', 'Global freedom evaluation']
F1_IDX = [3, 5, 6, 7, 8, 9, 10]   # Life Restoration
F2_IDX = [0, 1, 2, 4]             # Migraine Agency
item_cols = [c for c in D.columns if re.match(r'^q\d+_\d+$', c)]
dom_items = {i: [c for c in item_cols if c.startswith(f'q{i}_')] for i in range(11)}
M = D[item_cols].apply(pd.to_numeric, errors='coerce')

# ---------------- A. 項目番号の対応（S1 の #1–82 → 列）と自己点検 ----------------
# The in-app survey definition is used only for this text self-check and is not part of the
# repository (the item wording is in Supplementary Table S1). Without it, the check is skipped.
SURVEY_JSON = os.environ.get('MFI_SURVEY_JSON', 'mfi_survey.json')
sv = json.load(open(SURVEY_JSON, encoding='utf-8')) if os.path.exists(SURVEY_JSON) else {'questions': []}
qs = sv['questions'] if isinstance(sv, dict) and 'questions' in sv else sv
text_by_col = {}
for i in range(11):
    sec = [q for q in qs if q.get('section_title') == DOMS_JA[i]]
    sec = sorted(sec, key=lambda q: q.get('order_index', 0))
    for c, q in zip(dom_items[i], sec):
        text_by_col[c] = q['question_text']
# S1 順（F1: attack control, social, activity, hope, reclaiming, close others, global / F2: premonitory, trigger, pre-emptive, MOH）
S1_ORDER = [3, 8, 5, 6, 7, 9, 10, 0, 1, 2, 4]
s1_cols = [c for i in S1_ORDER for c in dom_items[i]]
item_no = {c: k + 1 for k, c in enumerate(s1_cols)}   # 列 → S1 番号
col_of = {v: k for k, v in item_no.items()}
# S1 の日本語文と突き合わせ
supp = open('mfi_supp.html', encoding='utf-8').read()
supp = re.sub(r'<style.*?</style>', '', supp, flags=re.S)
s1 = {}
for r in re.findall(r'<tr[^>]*>(.*?)</tr>', supp, flags=re.S):
    cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, flags=re.S)]
    # first occurrence only: Table S4 (added later) also has rows that start with the item number
    if len(cells) >= 5 and re.fullmatch(r'\d{1,2}', cells[0]) and int(cells[0]) not in s1: s1[int(cells[0])] = cells
mism = [k for k in range(1, 83) if k in s1 and text_by_col.get(col_of[k], '') != s1[k][4]] if text_by_col else []
if not text_by_col: print(f"\n[A] text self-check skipped ({SURVEY_JSON} not found)")
print(f"\n[A] 項目列 {len(item_cols)}（ドメイン別 {[len(dom_items[i]) for i in range(11)]}）／S1 行 {len(s1)}／S1 日本語文と不一致 {len(mism)} 件 {mism[:10]}")
for k in mism[:10]: print(f"   #{k}: S1='{s1[k][4][:30]}' / 列={col_of[k]} '{text_by_col.get(col_of[k],'')[:30]}'")
# dom スコアの再現
for i in range(11):
    rec = M[dom_items[i]].mean(axis=1) / 6 * 100
    diff = (rec - num(f'dom{i+1}')).abs().max()
    if diff > 0.5: print(f"   !! dom{i+1} 再現ずれ max {diff:.2f}")
f1_cols = [c for i in F1_IDX for c in dom_items[i]]; f2_cols = [c for i in F2_IDX for c in dom_items[i]]
f1_rec = M[f1_cols].mean(axis=1) / 6 * 100; f2_rec = M[f2_cols].mean(axis=1) / 6 * 100
print(f"   F1 再現ずれ max {(f1_rec-num('f1')).abs().max():.3f} / F2 {(f2_rec-num('f2')).abs().max():.3f}（単純平均＝全項目の平均であることの確認）")

# ---------------- 共通ヘルパ ----------------
def partial_spearman(x, y, covs):
    df = pd.DataFrame({'x': x, 'y': y})
    for i, c in enumerate(covs): df[f'c{i}'] = c
    df = df.dropna(); n = len(df); k = len(covs)
    rx, ry = df['x'].rank(), df['y'].rank()
    C = np.column_stack([np.ones(n)] + [df[f'c{i}'].rank() for i in range(k)])
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; z = np.arctanh(r); se = 1 / np.sqrt(n - 3 - k)
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2)); p = 2 * stats.t.sf(abs(t), n - 2 - k)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), p, n
def fmt(res): r, lo, hi, p, n = res; return f"{r:+.2f} [{lo:+.2f},{hi:+.2f}] p={'<0.001' if p<0.001 else f'{p:.3f}'} n={n}"
def sp_ci(x, y):
    df = pd.DataFrame({'x': x, 'y': y}).dropna(); n = len(df)
    r, p = stats.spearmanr(df['x'], df['y']); z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), p, n
def cronbach(X):
    X = X.dropna(); k = X.shape[1]
    return k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
age, sex = num('age'), num('sex'); female = (sex == 1).astype(float)
COV = [age, sex]
wdays = num('w28_days'); d7 = wdays >= 7
mmd = (num('w28_migprop') * 28).where(d7)            # 論文の diary-based MMD（外挿）
mig_raw = num('w28_mig').where(d7)                    # 外挿なしの実数
midas, mibs = num('midas'), num('mibs4')
f1, f2 = num('f1'), num('f2')
EXT = [('diary MMD(≥7d)', mmd), ('MIDAS', midas), ('MIBS-4', mibs)]

print("\n[再現] 論文 Table 3（年齢・性別調整 偏 Spearman）")
for lab, x in EXT:
    print(f"   LR↔{lab}: {fmt(partial_spearman(x, f1, COV))} / MA↔{lab}: {fmt(partial_spearman(x, f2, COV))}")

# ---------------- B. 項目レベル ----------------
print("\n[B] 項目レベル（82 項目・2因子 EFA 主因子法 oblimin・探索的）")
miss = M.isna().mean()
drop_cols = list(miss[miss > 0.05].index)
Mi = M.drop(columns=drop_cols).dropna()
print(f"   欠測 >5% で除いた項目: {[f'#{item_no[c]}' for c in drop_cols]}（{[f'{miss[c]*100:.0f}%' for c in drop_cols]}）→ 完全ケース n={len(Mi)} × {Mi.shape[1]} 項目")
kmo_all, kmo_model = calculate_kmo(Mi)
fa = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin'); fa.fit(Mi)
L = pd.DataFrame(fa.loadings_, index=Mi.columns, columns=['L1', 'L2'])
# 因子の向き合わせ（LR 項目の平均ローディングが大きい方を F1 とする）
if L.loc[[c for c in f1_cols if c in L.index], 'L1'].mean() < L.loc[[c for c in f1_cols if c in L.index], 'L2'].mean(): L = L.rename(columns={'L1': 'L2', 'L2': 'L1'})
L['item'] = [item_no[c] for c in L.index]; L['dom'] = [DOM_EN[int(c.split('_')[0][1:])] for c in L.index]
L['assigned'] = ['LR' if c in f1_cols else 'MA' for c in L.index]
L['primary_on_assigned'] = np.where(L['assigned'] == 'LR', L['L1'], L['L2']); L['other'] = np.where(L['assigned'] == 'LR', L['L2'], L['L1'])
print(f"   KMO={kmo_model:.2f}  因子間相関 r={np.corrcoef(fa.transform(Mi).T)[0,1]:+.2f}  寄与率 {fa.get_factor_variance()[2][-1]:.1%}")
print(f"   自因子への主負荷: 中央値 {L['primary_on_assigned'].median():.2f}（範囲 {L['primary_on_assigned'].min():.2f}–{L['primary_on_assigned'].max():.2f}）")
weak = L[L['primary_on_assigned'] < 0.40].sort_values('primary_on_assigned')
cross = L[(L['other'].abs() >= 0.30)]
wrong = L[L['other'].abs() > L['primary_on_assigned'].abs()]
print(f"   自因子負荷 <0.40 の項目 {len(weak)}: " + ", ".join(f"#{int(r.item)}({r.dom[:12]} {r.primary_on_assigned:+.2f})" for r in weak.itertuples()))
print(f"   交差負荷 |他因子| ≥0.30 の項目 {len(cross)}: " + ", ".join(f"#{int(r.item)}({r.dom[:12]} 自{r.primary_on_assigned:+.2f}/他{r.other:+.2f})" for r in cross.itertuples()))
print(f"   他因子の方が大きい項目 {len(wrong)}: " + ", ".join(f"#{int(r.item)}({r.dom[:12]})" for r in wrong.itertuples()))
# 項目-残り相関（自ドメイン内・自因子内）と alpha-if-deleted（自因子）
rows = []
for c in item_cols:
    d = int(c.split('_')[0][1:]); fac_cols = f1_cols if c in f1_cols else f2_cols
    rest_dom = M[[x for x in dom_items[d] if x != c]].mean(axis=1); rest_fac = M[[x for x in fac_cols if x != c]].mean(axis=1)
    r_dom = pd.concat([M[c], rest_dom], axis=1).dropna().corr().iloc[0, 1] if len(dom_items[d]) > 1 else np.nan
    r_fac = pd.concat([M[c], rest_fac], axis=1).dropna().corr().iloc[0, 1]
    Xf = M[fac_cols].drop(columns=[c]); a_del = cronbach(Xf)
    rows.append(dict(item=item_no[c], col=c, dom=DOM_EN[d], fac='LR' if c in f1_cols else 'MA', r_rest_dom=r_dom, r_rest_fac=r_fac, alpha_if_deleted=a_del,
                     floor=(M[c] == 0).mean(), ceiling=(M[c] == 6).mean(), mean=M[c].mean()))
IT = pd.DataFrame(rows).set_index('item').sort_index()
IT = IT.join(L.set_index('item')[['L1', 'L2']])
a_LR, a_MA = cronbach(M[f1_cols]), cronbach(M[f2_cols])
print(f"   alpha LR={a_LR:.3f} MA={a_MA:.3f}；項目-残り相関（自因子内）LR 中央値 {IT[IT.fac=='LR'].r_rest_fac.median():.2f}（{IT[IT.fac=='LR'].r_rest_fac.min():.2f}–{IT[IT.fac=='LR'].r_rest_fac.max():.2f}）/ MA 中央値 {IT[IT.fac=='MA'].r_rest_fac.median():.2f}（{IT[IT.fac=='MA'].r_rest_fac.min():.2f}–{IT[IT.fac=='MA'].r_rest_fac.max():.2f}）")
lowfac = IT[IT.r_rest_fac < 0.30]
print(f"   自因子内 項目-残り相関 <0.30: {len(lowfac)} 項目: " + ", ".join(f"#{i}({r.dom[:12]} {r.r_rest_fac:.2f})" for i, r in lowfac.iterrows()))
print(f"   alpha-if-deleted: LR {IT[IT.fac=='LR'].alpha_if_deleted.min():.3f}–{IT[IT.fac=='LR'].alpha_if_deleted.max():.3f} / MA {IT[IT.fac=='MA'].alpha_if_deleted.min():.3f}–{IT[IT.fac=='MA'].alpha_if_deleted.max():.3f}（削除で alpha が上がる項目: LR {(IT[IT.fac=='LR'].alpha_if_deleted > a_LR).sum()} / MA {(IT[IT.fac=='MA'].alpha_if_deleted > a_MA).sum()}）")
IT.to_csv('r2_item_table.csv', float_format='%.3f')
print("   → 項目表 r2_item_table.csv（S1 に列を足す材料）")

# ---------------- C. 重み付けスコアの比較 ----------------
print("\n[C] 重み付けスコア（Life Restoration / Migraine Agency）")
Xd = D[[f'dom{i+1}' for i in range(11)]].apply(pd.to_numeric, errors='coerce'); Xd.columns = DOMS_JA
ok = Xd.notna().all(axis=1)
fad = FactorAnalyzer(n_factors=2, method='principal', rotation='oblimin'); fad.fit(Xd[ok])
Ld = pd.DataFrame(fad.loadings_, index=DOMS_JA, columns=['A', 'B'])
if Ld.loc[[DOMS_JA[i] for i in F1_IDX], 'A'].mean() < Ld.loc[[DOMS_JA[i] for i in F1_IDX], 'B'].mean(): Ld = Ld.rename(columns={'A': 'B', 'B': 'A'})
print("   ドメイン負荷（再現）: " + "; ".join(f"{DOM_EN[i][:14]} {Ld.iloc[i,0]:+.2f}/{Ld.iloc[i,1]:+.2f}" for i in range(11)))
# 回帰法因子得点
FS = pd.DataFrame(fad.transform(Xd[ok]), index=Xd[ok].index, columns=['A', 'B'])
if np.corrcoef(FS['A'], f1[ok])[0, 1] < np.corrcoef(FS['B'], f1[ok])[0, 1]: FS = FS.rename(columns={'A': 'B', 'B': 'A'})
scores = {}
for fac, idx, unit in [('LR', F1_IDX, f1), ('MA', F2_IDX, f2)]:
    doms = [DOMS_JA[i] for i in idx]; col = 'A' if fac == 'LR' else 'B'
    w = Ld.loc[doms, col].clip(lower=0)
    scores[fac] = {
        'unit (item mean; 現行)': unit,
        'domain mean (ドメイン等重み)': Xd[doms].mean(axis=1),
        'loading-weighted (ドメイン負荷で重み)': (Xd[doms] * w).sum(axis=1) / w.sum(),
        'regression factor score': FS[col].reindex(D.index),
    }
for fac in ['LR', 'MA']:
    print(f"   -- {fac} --")
    base = scores[fac]['unit (item mean; 現行)']
    for name, s in scores[fac].items():
        rr = sp_ci(base, s)[0]
        print(f"   {name:40s} ρ(vs 現行)={rr:+.3f} | " + " | ".join(f"{lab} {fmt(partial_spearman(x, s, COV))}" for lab, x in EXT))
# 総合評価ドメイン（2項目）の寄与
print(f"   参考: Global freedom evaluation は LR 50 項目中 2 項目（4%）。ドメイン等重みでは 1/7（14%）")

# ---------------- D. 記録密度 ----------------
print("\n[D] 記録密度と diary-based MMD")
for lab, x in [('MMD 外挿(現行)', mmd), ('片頭痛日 実数（外挿なし）', mig_raw)]:
    print(f"   LR↔{lab}: {fmt(partial_spearman(x, f1, COV))} / 記録日数も調整: {fmt(partial_spearman(x, f1, COV+[wdays]))}")
print(f"   MA↔MMD 記録日数も調整: {fmt(partial_spearman(mmd, f2, COV+[wdays]))}")
d1 = wdays >= 1
print(f"   記録日数↔頭痛日割合(≥1d): {fmt(sp_ci(wdays[d1], num('w28_haprop')[d1]))} / 記録日数↔片頭痛日割合: {fmt(sp_ci(wdays[d1], num('w28_migprop')[d1]))}  ← 記録が少ない人ほど頭痛率が高いか（MAR の確認）")
print(f"   記録日数↔LR: {fmt(sp_ci(wdays[d1], f1[d1]))} / 記録日数↔MA: {fmt(sp_ci(wdays[d1], f2[d1]))}")
for thr in [1, 7, 14, 21, 28]:
    m = wdays >= thr; x = (num('w28_migprop') * 28).where(m)
    print(f"   ≥{thr:2d}日: LR↔MMD {fmt(partial_spearman(x, f1, COV))} / MA↔MMD {fmt(partial_spearman(x, f2, COV))} / 記録日数 中央値 {wdays[m].median():.0f}")
# Poisson GLM（IRLS）with offset, robust SE
def poisson_glm(y, X, offset):
    b = np.zeros(X.shape[1]); b[0] = np.log(y.mean()) - offset.mean()
    for _ in range(50):
        eta = X @ b + offset; mu = np.exp(eta); W = mu
        z = eta - offset + (y - mu) / mu
        XtW = X.T * W; bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-9: b = bn; break
        b = bn
    mu = np.exp(X @ b + offset); Ib = np.linalg.inv((X.T * mu) @ X)
    S = (X.T * ((y - mu) ** 2)) @ X; V = Ib @ S @ Ib     # sandwich
    disp = np.sum((y - mu) ** 2 / mu) / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(V)), disp
g = pd.DataFrame({'y': num('w28_mig'), 'wd': wdays, 'f1': f1, 'f2': f2, 'age': age, 'female': female})[d7].dropna()
for fac in ['f1', 'f2']:
    X = np.column_stack([np.ones(len(g)), g[fac] / 10, g['age'], g['female']])
    b, se, disp = poisson_glm(g['y'].values.astype(float), X, np.log(g['wd'].values))
    irr = np.exp(b[1]); lo, hi = np.exp(b[1] - 1.96 * se[1]), np.exp(b[1] + 1.96 * se[1]); p = 2 * stats.norm.sf(abs(b[1] / se[1]))
    print(f"   Poisson GLM 片頭痛日数 ~ {fac.upper()}(10点あたり)+年齢+性別, offset=log(記録日数), 頑健SE: IRR={irr:.3f} [{lo:.3f},{hi:.3f}] p={p:.3f} 分散比={disp:.2f} n={len(g)}")
    for thr in [14, 21]:
        gg = g[g['wd'] >= thr]; X = np.column_stack([np.ones(len(gg)), gg[fac] / 10, gg['age'], gg['female']])
        b, se, disp = poisson_glm(gg['y'].values.astype(float), X, np.log(gg['wd'].values))
        print(f"      ≥{thr}日: IRR={np.exp(b[1]):.3f} [{np.exp(b[1]-1.96*se[1]):.3f},{np.exp(b[1]+1.96*se[1]):.3f}] p={2*stats.norm.sf(abs(b[1]/se[1])):.3f} n={len(gg)}")

# ---------------- E. 冗長性・ドメイン内一次元性 ----------------
print("\n[E] 冗長性")
R = M[f1_cols].corr()
pairs = [(item_no[a], item_no[b], R.loc[a, b]) for i, a in enumerate(f1_cols) for b in f1_cols[i+1:]]
hi = [p for p in pairs if p[2] >= 0.80]
same_dom = sum(1 for a, b, r in hi if (col_of[a].split('_')[0] == col_of[b].split('_')[0]))
print(f"   LR 50 項目 {len(pairs)} 組のうち r≥0.80: {len(hi)} 組（同一ドメイン内 {same_dom}・ドメイン間 {len(hi)-same_dom}）；r≥0.70: {sum(1 for p in pairs if p[2]>=0.70)} 組")
for a, b, r in sorted(hi, key=lambda t: -t[2])[:12]: print(f"      #{a}×#{b} r={r:.2f}  {s1.get(a,['','','','',''])[3][:45]} / {s1.get(b,['','','','',''])[3][:45]}")
Rm = M[f2_cols].corr(); pm = [(item_no[a], item_no[b], Rm.loc[a, b]) for i, a in enumerate(f2_cols) for b in f2_cols[i+1:]]
print(f"   MA 32 項目: r≥0.80 {sum(1 for p in pm if p[2]>=0.80)} 組 / r≥0.70 {sum(1 for p in pm if p[2]>=0.70)} 組")
print("   ドメイン別: 平均項目間相関 / alpha / λ1,λ2 / λ1÷λ2 / λ1寄与 / 平行分析で2因子目保持? / 単因子ローディング最小")
for i in range(11):
    cols = dom_items[i]; X = M[cols].dropna(); k = len(cols)
    if k < 3: print(f"   {DOM_EN[i][:28]:28s} 項目{k}: r={X.corr().iloc[0,1]:.2f}（項目2つのため一次元性の検定は不可）"); continue
    Rd = X.corr().values; iu = np.triu_indices(k, 1); mean_r = Rd[iu].mean()
    ev = np.sort(np.linalg.eigvalsh(Rd))[::-1]
    re_ = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(np.random.normal(size=(len(X), k)), rowvar=False)))[::-1] for _ in range(500)])
    pa = np.percentile(re_, 95, axis=0); keep2 = ev[1] > pa[1]
    fa1 = FactorAnalyzer(n_factors=1, method='principal', rotation=None); fa1.fit(X); l1 = fa1.loadings_[:, 0]
    print(f"   {DOM_EN[i][:28]:28s} 項目{k} n={len(X)}: r̄={mean_r:.2f} α={cronbach(X):.2f} λ={ev[0]:.2f},{ev[1]:.2f} 比={ev[0]/ev[1]:.1f} 寄与={ev[0]/k:.0%} 2因子目={'保持' if keep2 else '棄却'}(λ2 {ev[1]:.2f} vs PA95 {pa[1]:.2f}) 最小負荷={np.abs(l1).min():.2f}(#{item_no[cols[int(np.argmin(np.abs(l1)))]]})")

# ---------------- F. 片頭痛非該当 3 名の除外 ----------------
print("\n[F] 片頭痛非該当（自己申告でも治療でも該当せず）3 名を除く感度解析")
cd = num('case_def') == 1
print(f"   除外後 n={cd.sum()}")
for lab, x in EXT:
    print(f"   LR↔{lab}: {fmt(partial_spearman(x[cd], f1[cd], [age[cd], sex[cd]]))} / MA↔{lab}: {fmt(partial_spearman(x[cd], f2[cd], [age[cd], sex[cd]]))}")

# ---------------- G. MIBS-4 ----------------
print("\n[G] MIBS-4（4 項目・各 0–3・合計 0–12）の分布")
vc = mibs.value_counts().sort_index()
print("   " + ", ".join(f"{int(k)}:{v}" for k, v in vc.items()))
cat = pd.cut(mibs, [-1, 0, 2, 4, 12], labels=['0 none', '1–2 mild', '3–4 moderate', '≥5 severe']).value_counts().sort_index()
print("   " + ", ".join(f"{k} {v} ({v/N:.0%})" for k, v in cat.items()))
print(f"   中央値 {mibs.median():.0f}（IQR {mibs.quantile(.25):.0f}–{mibs.quantile(.75):.0f}）平均 {mibs.mean():.2f}；MIDAS 中央値 {midas.median():.0f}（IQR {midas.quantile(.25):.0f}–{midas.quantile(.75):.0f}）；MMD 中央値 {mmd.median():.0f}")
print(f"   MIBS-4↔MIDAS: {fmt(sp_ci(mibs, midas))} / MIBS-4↔MMD: {fmt(sp_ci(mibs, mmd))}")

# ---------------- H. 治療の種類別 ----------------
print("\n[H] 治療の種類別（探索的）")
for lab, g1 in [('CGRP 関連薬 使用', num('cgrp_use') == 1), ('予防薬 使用', num('prev_use') == 1), ('片頭痛特異薬 使用', num('drug_specific') == 1)]:
    g0 = ~g1 & num(lab and 'cgrp_use' if False else 'age').notna()
    for fac, s in [('LR', f1), ('MA', f2), ('MMD', mmd)]:
        a, b = s[g1].dropna(), s[~g1].dropna()
        if len(a) < 3 or len(b) < 3: continue
        p = stats.mannwhitneyu(a, b).pvalue
        print(f"   {lab:14s} {fac:3s}: あり 中央値 {a.median():.1f} (IQR {a.quantile(.25):.1f}–{a.quantile(.75):.1f}, n={len(a)}) / なし {b.median():.1f} ({b.quantile(.25):.1f}–{b.quantile(.75):.1f}, n={len(b)}) MWU p={p:.3f}")
    print(f"   {lab:14s} LR↔MMD あり: {fmt(partial_spearman(mmd[g1], f1[g1], [age[g1], sex[g1]]))} / なし: {fmt(partial_spearman(mmd[~g1], f1[~g1], [age[~g1], sex[~g1]]))}")
    print(f"   {lab:14s} LR↔MIDAS あり: {fmt(partial_spearman(midas[g1], f1[g1], [age[g1], sex[g1]]))} / なし: {fmt(partial_spearman(midas[~g1], f1[~g1], [age[~g1], sex[~g1]]))}")

# ---------------- I. 自己効力感に近い 2 項目を外す ----------------
print("\n[I] #8・#29（自己効力感に近い）を外した Life Restoration（48 項目）")
f1x = M[[c for c in f1_cols if item_no[c] not in (8, 29)]].mean(axis=1) / 6 * 100
print(f"   ρ(LR48 vs LR50)={sp_ci(f1x, f1)[0]:+.3f}；LR48↔MA: {fmt(sp_ci(f1x, f2))}（LR50↔MA: {fmt(sp_ci(f1, f2))}）")
for lab, x in EXT: print(f"   LR48↔{lab}: {fmt(partial_spearman(x, f1x, COV))}")
print(f"   #8 自因子内 項目-残り相関 {IT.loc[8,'r_rest_fac']:.2f} / #29 {IT.loc[29,'r_rest_fac']:.2f}；#8↔MA {sp_ci(M[col_of[8]], f2)[0]:+.2f} / #29↔MA {sp_ci(M[col_of[29]], f2)[0]:+.2f}")

# ---------------- J. Migraine Agency ----------------
print("\n[J] Migraine Agency の扱い")
print(f"   ドメイン負荷（LR 因子への交差負荷）: 予兆 {Ld.loc['脳状態遷移','A']:+.2f} / MOH 予防 {Ld.loc['MOH予防','A']:+.2f} / トリガー {Ld.loc['トリガー認識','A']:+.2f} / 先制行動 {Ld.loc['先制行動','A']:+.2f}")
for i in [0, 4]:
    print(f"   {DOM_EN[i]} ↔ LR: {fmt(sp_ci(Xd[DOMS_JA[i]], f1))}")
print("   症状・誘因を前提とする項目の床効果（0 点の割合）と平均:")
for k in [51, 52, 53, 59, 60, 65]:
    c = col_of[k]; print(f"      #{k} {s1[k][3][:60]:60s} 床 {IT.loc[k,'floor']:.0%} 平均 {IT.loc[k,'mean']:.2f}")
print(f"   参考: 全 82 項目の床 中央値 {IT['floor'].median():.0%}、MA 32 項目 {IT[IT.fac=='MA']['floor'].median():.0%}、LR 50 項目 {IT[IT.fac=='LR']['floor'].median():.0%}")
print("\n完了")
