/**
 * MFI 本解析（N≥100）。予備解析に加えて:
 *   - 背景記述（年齢・性別・片頭痛歴・自己申告片頭痛日数・前兆・治療薬）
 *   - 併存的妥当性（アンケート内 MIDAS/MIBS-4 + 自己申告片頭痛日数/急性薬日数）
 *   - DB連結妥当性（MFI ↔ 日誌の頭痛率・日次MIBS-4・充実度 day_fulfillment）
 *   - 既知グループ（片頭痛頻度 低/高頻度EM/CM・性別・年代）
 *   - 症例定義＆感度解析（片頭痛特異薬 使用者サブグループ）
 *   - EFA/ネットワーク用に回答者×項目行列を CSV 出力（scratchpad）
 * テストユーザー（名前に「テスト」）は全解析から除外。
 * 標本は source='app'（アプリ内回答）に限定する。公開ページ /freedom からの Web 回答は
 * パイロットとは別標本であり、日次記録とも連結できないため、ここでは混ぜない。
 * 実行: npx tsx scripts/research/mfi_full_analysis.ts
 */
import { writeFileSync } from "fs";
import { admin, mean, sd, median, quantile, MWU } from "./lib";

const OUT_CSV = "/private/tmp/claude-501/-Users-hm-Developer-zutsu-note/a7ad6114-4541-4cb1-b50e-83f54fc48692/scratchpad/mfi_matrix.csv";

function variance(a: number[]) { const m = mean(a); return a.length < 2 ? NaN : a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1); }
function pearson(x: number[], y: number[]) {
  const n = x.length; if (n < 3) return { r: NaN, n };
  const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return { r: sxx && syy ? sxy / Math.sqrt(sxx * syy) : NaN, n };
}
function rank(a: number[]): number[] {
  const idx = a.map((v, i) => ({ v, i })).sort((p, q) => p.v - q.v);
  const r = new Array(a.length).fill(0); let i = 0;
  while (i < idx.length) { let j = i; while (j + 1 < idx.length && idx[j + 1].v === idx[i].v) j++; const rr = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k].i] = rr; i = j + 1; }
  return r;
}
function spearman(x: number[], y: number[]) { if (x.length < 3) return { r: NaN, n: x.length }; return { r: pearson(rank(x), rank(y)).r, n: x.length }; }
function cronbach(rows: number[][]) {
  const n = rows.length; if (!n) return { alpha: NaN, k: 0, n: 0 };
  const k = rows[0].length;
  const itemVars = Array.from({ length: k }, (_, j) => variance(rows.map(r => r[j])));
  const totVar = variance(rows.map(r => r.reduce((a, b) => a + b, 0)));
  const alpha = totVar > 0 ? (k / (k - 1)) * (1 - itemVars.reduce((a, b) => a + b, 0) / totVar) : NaN;
  return { alpha, k, n };
}
const rf = (r: number) => isNaN(r) ? "n.a." : (r >= 0 ? "+" : "") + r.toFixed(2);

async function fetchAll<T>(q: (from: number, to: number) => PromiseLike<{ data: T[] | null; error: unknown }>): Promise<T[]> {
  const out: T[] = [];
  for (let p = 0; p < 200; p++) { const { data, error } = await q(p * 1000, p * 1000 + 999); if (error) throw error; if (!data || !data.length) break; out.push(...data); if (data.length < 1000) break; }
  return out;
}
async function fetchByIds<T>(table: string, select: string, col: string, ids: string[], extra?: (q: any) => any): Promise<T[]> {
  const out: T[] = [];
  for (let i = 0; i < ids.length; i += 300) {
    const chunk = ids.slice(i, i + 300);
    let q: any = admin.from(table).select(select).in(col, chunk);
    if (extra) q = extra(q);
    const { data, error } = await q; if (error) throw error; if (data) out.push(...(data as T[]));
  }
  return out;
}

const SCALE7 = (opts: string[] | null) => Array.isArray(opts) && opts.length === 7 && opts[0].startsWith("0 ");
const FEEDBACK_SEC = "このアンケートについて（任意・採点対象外）";

async function main() {
  const { data: tmpl } = await admin.from("survey_templates").select("id, title, reward_points").eq("is_admin", true).order("created_at", { ascending: false }).limit(1).single();
  if (!tmpl) { console.log("テンプレなし"); return; }
  console.log(`\n===== MFI 本解析 =====\nアンケート: ${tmpl.title}（完了報酬 ${tmpl.reward_points ?? "?"}pt＝アプリ内育成ポイントのみ・換金不可）\n`);

  const { data: questions } = await admin.from("survey_questions").select("id, section_title, order_index, question_type, question_text, options").eq("template_id", tmpl.id).order("order_index");
  const Q = questions ?? [];
  const qByText = (kw: string) => Q.find(q => q.question_text?.includes(kw));

  const sends = await fetchAll<{ id: string; status: string; consented_at: string | null; responded_at: string | null; patient_id: string }>(
    (f, t) => admin.from("survey_sends").select("id, status, consented_at, responded_at, patient_id").eq("template_id", tmpl.id).eq("source", "app").range(f, t));

  // テストユーザー除外
  const pids = [...new Set(sends.map(s => s.patient_id))];
  const patients = await fetchByIds<{ id: string; name: string; gender: string | null; birthdate: string | null }>("patients", "id, name, gender, birthdate", "id", pids);
  const patById = new Map(patients.map(p => [p.id, p]));
  const isTest = (pid: string) => (patById.get(pid)?.name ?? "").includes("テスト");

  // 凍結データカット（査読対応：再現可能な frozen dataset）。2026-07-18 00:00 JST 時点＝完了116名。
  // これ以降の完了は本論文の解析に含めない（大規模フェーズで別途）。
  const CUTOFF = "2026-07-17T15:00:00.000Z"; // = 2026-07-18T00:00:00+09:00
  const completedAll = sends.filter(s => s.status === "completed" && s.responded_at && s.responded_at < CUTOFF);
  const completed = completedAll.filter(s => !isTest(s.patient_id));
  const inProg = sends.filter(s => s.status === "in_progress" && !isTest(s.patient_id) && (!s.consented_at || s.consented_at < CUTOFF));
  console.log(`【実施可能性】配信${sends.filter(s=>!isTest(s.patient_id)).length} / 完了${completed.length}（テスト除外前${completedAll.length}） / 回答中${inProg.length}`);
  const elapsed = completed.map(s => s.consented_at && s.responded_at ? (new Date(s.responded_at).getTime() - new Date(s.consented_at).getTime()) / 60000 : NaN).filter(v => !isNaN(v) && v >= 0 && v <= 180);
  if (elapsed.length) console.log(`  所要(同一セッション${elapsed.length}名) 中央値 ${median(elapsed).toFixed(1)}分 (IQR ${quantile(elapsed, .25).toFixed(1)}-${quantile(elapsed, .75).toFixed(1)})`);

  if (completed.length < 30) { console.log(`完了${completed.length}名（<30）。中止。`); return; }

  // 回答マトリクス
  const responses = await fetchByIds<{ send_id: string; question_id: string; answer_options: number[] | null; answer_value: number | null }>(
    "survey_responses", "send_id, question_id, answer_options, answer_value", "send_id", completed.map(s => s.id));
  const resp = new Map<string, Map<string, { opts: number[] | null; val: number | null }>>();
  for (const r of responses) { if (!resp.has(r.send_id)) resp.set(r.send_id, new Map()); resp.get(r.send_id)!.set(r.question_id, { opts: r.answer_options, val: r.answer_value }); }
  const scaleVal = (sid: string, qid: string): number | null => { const a = resp.get(sid)?.get(qid); return a?.opts && a.opts.length ? a.opts[0] : null; };
  const numVal = (sid: string, qid: string): number | null => { const a = resp.get(sid)?.get(qid); return a?.val ?? null; };
  const optsVal = (sid: string, qid: string): number[] => resp.get(sid)?.get(qid)?.opts ?? [];

  const comp = completed.map(s => ({ sid: s.id, pid: s.patient_id, respondedAt: s.responded_at }));

  // 採点項目・ドメイン
  const scored = Q.filter(q => q.question_type === "single_choice" && SCALE7(q.options as string[] | null) && q.section_title !== FEEDBACK_SEC);
  const domains = [...new Set(scored.map(q => q.section_title))];
  // 2因子（ドメインレベルEFAで確定した所属。査読対応で F1/F2 を主指標に）
  const F1D = ["発作コントロール", "活動の自由", "希望と治療継続", "人生回復", "社会的自由", "身近な人への影響", "総合評価"];
  const F2D = ["脳状態遷移", "トリガー認識", "先制行動", "MOH予防"];
  const factorScore = (sid: string, doms: string[]) => { const vs = scored.filter(q => doms.includes(q.section_title!)).map(q => scaleVal(sid, q.id)).filter((v): v is number => v != null); return vs.length ? mean(vs) / 6 * 100 : null; };

  // 各回答者の総合スコア(0-100)とドメインスコア
  const totalScore = new Map<string, number>();
  const domScore: Record<string, Map<string, number>> = {};
  for (const dom of domains) domScore[dom] = new Map();
  for (const { sid } of comp) {
    const all = scored.map(q => scaleVal(sid, q.id)).filter((v): v is number => v != null);
    if (all.length) totalScore.set(sid, mean(all) / 6 * 100);
    for (const dom of domains) {
      const vs = scored.filter(q => q.section_title === dom).map(q => scaleVal(sid, q.id)).filter((v): v is number => v != null);
      if (vs.length) domScore[dom].set(sid, mean(vs) / 6 * 100);
    }
  }

  // ---- 信頼性 ----
  console.log(`\n【内的整合性】（採点${scored.length}問 / ${domains.length}ドメイン）`);
  for (const dom of domains) {
    const items = scored.filter(q => q.section_title === dom);
    const rows: number[][] = [];
    for (const { sid } of comp) { const vs = items.map(q => scaleVal(sid, q.id)); if (vs.every(v => v != null)) rows.push(vs as number[]); }
    const a = cronbach(rows);
    const sc = [...domScore[dom].values()];
    console.log(`  ${dom}: 項目${items.length} α=${isNaN(a.alpha) ? "n.a." : a.alpha.toFixed(2)}(n=${a.n}) スコア${mean(sc).toFixed(1)}±${sd(sc).toFixed(1)}`);
  }
  const totRows: number[][] = [];
  for (const { sid } of comp) { const vs = scored.map(q => scaleVal(sid, q.id)); if (vs.every(v => v != null)) totRows.push(vs as number[]); }
  console.log(`  ★総合 α=${cronbach(totRows).alpha.toFixed(2)}(n=${cronbach(totRows).n}) スコア${mean([...totalScore.values()]).toFixed(1)}±${sd([...totalScore.values()]).toFixed(1)}`);

  // ---- 日誌ロード（連結・ID Migraine判定に使用） ----
  const logs = await fetchByIds<{ patient_id: string; recorded_date: string; had_headache: boolean; day_fulfillment: number | null; interictal_work: number | null; interictal_social: number | null; interictal_planning: number | null; interictal_emotion: number | null; symptoms: string[] | null; work_absence: boolean | null; activity_impact: number | null }>(
    "headache_logs", "patient_id, recorded_date, had_headache, day_fulfillment, interictal_work, interictal_social, interictal_planning, interictal_emotion, symptoms, work_absence, activity_impact", "patient_id", comp.map(c => c.pid));
  const logsByPid = new Map<string, typeof logs>();
  for (const l of logs) { if (!logsByPid.has(l.patient_id)) logsByPid.set(l.patient_id, []); logsByPid.get(l.patient_id)!.push(l); }
  const pidOf = new Map(comp.map(c => [c.sid, c.pid]));
  // ID Migraine（JHS54と同一：頭痛日の 悪心/光過敏/生活障害 ≥2、症状記録≥1日で判定。記録無し=判定不能null）
  function idMigPositive(pid: string): boolean | null {
    const ll = logsByPid.get(pid) ?? [];
    const haSym = ll.filter(l => l.had_headache && l.symptoms != null);
    const ha = ll.filter(l => l.had_headache);
    if (haSym.length < 1) return null;
    const nausea = haSym.some(l => l.symptoms!.includes("nausea"));
    const photo = haSym.some(l => l.symptoms!.includes("photophobia"));
    const disab = ha.some(l => l.work_absence === true || (l.activity_impact != null && l.activity_impact >= 5));
    return [nausea, photo, disab].filter(Boolean).length >= 2;
  }

  // ---- 背景設問 ----
  const qAge = qByText("年齢"), qSex = qByText("性別"), qHx = qByText("片頭痛歴");
  const qMigDays = qByText("過去4週間の片頭痛日数"), qHaDays = qByText("過去4週間の頭痛日数"), qAcuteDays = qByText("急性期治療薬使用日数");
  const qPrevWhich = qByText("予防薬を選んで"), qAcuteWhich = qByText("発作時のお薬を選んで"), qCgrp = qByText("CGRP関連");
  const qAura = qByText("前兆"), qPrevYN = qByText("予防するためのお薬"), qEmploy = qByText("就労・就学");

  const ages = comp.map(c => qAge ? numVal(c.sid, qAge.id) : null).filter((v): v is number => v != null && v > 0 && v < 120);
  const sexIdx = new Map(comp.map(c => [c.sid, qSex ? scaleVal(c.sid, qSex.id) : null]));
  const migDays = new Map(comp.map(c => [c.sid, qMigDays ? numVal(c.sid, qMigDays.id) : null]));
  const acuteDays = new Map(comp.map(c => [c.sid, qAcuteDays ? numVal(c.sid, qAcuteDays.id) : null]));

  const nFemale = [...sexIdx.values()].filter(v => v === 1).length;
  const migDaysArr = [...migDays.values()].filter((v): v is number => v != null);
  console.log(`\n【背景】年齢 ${mean(ages).toFixed(1)}±${sd(ages).toFixed(1)}歳(n=${ages.length}) / 女性 ${nFemale}/${comp.length} (${(100*nFemale/comp.length).toFixed(0)}%)`);
  console.log(`  自己申告 片頭痛日数/4週 中央値 ${median(migDaysArr).toFixed(1)}(IQR ${quantile(migDaysArr,.25).toFixed(1)}-${quantile(migDaysArr,.75).toFixed(1)}) / 急性薬日数中央値 ${median([...acuteDays.values()].filter((v):v is number=>v!=null)).toFixed(1)}`);

  // 症例定義（複合：ID Migraine陽性 / 自己申告 / 片頭痛特異薬 のいずれか）
  const drugSpecific = (sid: string) => (qAcuteWhich ? optsVal(sid, qAcuteWhich.id).some(i => i <= 6) : false) // 発作時 0..6=トリプタン/ジタン/リメゲパント
    || (qCgrp ? optsVal(sid, qCgrp.id).some(i => i <= 5) : false) // CGRP 0..5
    || (qPrevWhich ? optsVal(sid, qPrevWhich.id).some(i => i <= 3) : false); // 予防特異 アジョビ..アクイプタ
  const selfReport = (sid: string) => { const d = migDays.get(sid); const hx = qHx ? scaleVal(sid, qHx.id) : null; return (d != null && d > 0) || (hx != null && hx <= 3); }; // 片頭痛日数>0 or 片頭痛歴あり
  const idMig = (sid: string) => idMigPositive(pidOf.get(sid)!) === true;
  const isCaseDef = (sid: string) => drugSpecific(sid) || selfReport(sid) || idMig(sid);
  const nDrug = comp.filter(c => drugSpecific(c.sid)).length;
  const nId = comp.filter(c => idMig(c.sid)).length;
  const nIdIndet = comp.filter(c => idMigPositive(pidOf.get(c.sid)!) === null).length;
  const nSelf = comp.filter(c => selfReport(c.sid)).length;
  const nCase = comp.filter(c => isCaseDef(c.sid)).length;
  console.log(`  症例定義の内訳: 片頭痛特異薬 ${nDrug} / ID Migraine陽性 ${nId}（判定不能${nIdIndet}） / 自己申告片頭痛 ${nSelf}`);
  console.log(`  複合症例定義（いずれか該当）: ${nCase}/${comp.length} → 除外（いずれも非該当）${comp.length - nCase}名`);

  // ---- 併存的妥当性（アンケート内） ----
  const midasQ = Q.filter(q => q.section_title?.startsWith("MIDAS") && q.question_type === "number").sort((a, b) => a.order_index - b.order_index).slice(0, 5);
  const mibsQ = Q.filter(q => q.section_title?.startsWith("MIBS-4"));
  const midasTot = (sid: string) => { const vs = midasQ.map(q => numVal(sid, q.id)); return vs.every(v => v != null) ? (vs as number[]).reduce((a, b) => a + b, 0) : null; };
  const mibsTot = (sid: string) => { const vs = mibsQ.map(q => scaleVal(sid, q.id)); return vs.every(v => v != null) ? (vs as number[]).reduce((a, b) => a + b, 0) : null; };

  function corrWith(getY: (sid: string) => number | null, label: string) {
    const xs: number[] = [], ys: number[] = [];
    for (const { sid } of comp) { const x = totalScore.get(sid), y = getY(sid); if (x != null && y != null) { xs.push(x); ys.push(y); } }
    const sp = spearman(xs, ys), pe = pearson(xs, ys);
    console.log(`  MFI総合 ↔ ${label}: Spearman ρ=${rf(sp.r)} / Pearson r=${rf(pe.r)} (n=${xs.length})`);
    return { xs, ys };
  }
  console.log(`\n【併存的妥当性（アンケート内・負相関を期待）】`);
  corrWith(midasTot, "MIDAS");
  corrWith(mibsTot, "MIBS-4(4週)");
  corrWith(s => migDays.get(s) ?? null, "自己申告 片頭痛日数");
  corrWith(s => acuteDays.get(s) ?? null, "自己申告 急性薬日数");

  // ---- DB連結（日誌。ログは上でロード済み） ----
  const mibsDaily = (l: typeof logs[number]) => { const vs = [l.interictal_work, l.interictal_social, l.interictal_planning, l.interictal_emotion].filter((v): v is number => v != null); return vs.length ? vs.reduce((a, b) => a + b, 0) : null; };

  // 各回答者の日誌指標
  type Diary = { haRate: number; mibsMean: number | null; fulfillMean: number | null; loggedDays: number; haDays: number };
  const diary = new Map<string, Diary>();
  for (const { sid, pid } of comp) {
    const ll = logsByPid.get(pid) ?? [];
    if (ll.length < 7) continue; // 最低7日記録
    const haDays = ll.filter(l => l.had_headache).length;
    const mibsVals = ll.filter(l => !l.had_headache).map(mibsDaily).filter((v): v is number => v != null);
    const fulfillVals = ll.map(l => l.day_fulfillment).filter((v): v is number => v != null);
    diary.set(sid, { haRate: haDays / ll.length, mibsMean: mibsVals.length ? mean(mibsVals) : null, fulfillMean: fulfillVals.length ? mean(fulfillVals) : null, loggedDays: ll.length, haDays });
  }
  console.log(`\n【DB連結（日誌）】連結できた回答者: ${diary.size}/${comp.length}（7日以上記録）`);

  // ---- 回答前28日の固定ウィンドウ（査読対応：日誌評価期間の統一＋感度解析用） ----
  // 各回答者について responded_at 直前28日間の日誌のみ集計（記録日数・頭痛日割合・充実度）。
  // 記録日数の閾値（7/14/21日）や記録率≥80%での感度解析は Python 側で w28_days から絞る。
  type Win = { days: number; ha: number; haProp: number; free: number; fulfillMean: number | null; fulfillN: number; mibsMean: number | null; mibsN: number };
  const win28 = new Map<string, Win>();
  for (const { sid, pid, respondedAt } of comp) {
    if (!respondedAt) continue;
    const end = new Date(respondedAt).getTime();
    const start = end - 28 * 86400000;
    const ll = (logsByPid.get(pid) ?? []).filter(l => { const t = new Date(l.recorded_date + "T12:00:00+09:00").getTime(); return t >= start && t < end; });
    const ha = ll.filter(l => l.had_headache).length;
    const freeLogs = ll.filter(l => !l.had_headache);
    const fulfillVals = ll.map(l => l.day_fulfillment).filter((v): v is number => v != null);
    const mibsVals = freeLogs.map(mibsDaily).filter((v): v is number => v != null);
    win28.set(sid, { days: ll.length, ha, haProp: ll.length ? ha / ll.length : NaN, free: freeLogs.length, fulfillMean: fulfillVals.length ? mean(fulfillVals) : null, fulfillN: fulfillVals.length, mibsMean: mibsVals.length ? mean(mibsVals) : null, mibsN: mibsVals.length });
  }
  const w28days = [...win28.values()].map(w => w.days);
  const nW7 = w28days.filter(d => d >= 7).length, nW14 = w28days.filter(d => d >= 14).length, nW21 = w28days.filter(d => d >= 21).length, nW80 = w28days.filter(d => d >= 22.4).length;
  console.log(`  【28日固定窓】記録日数 中央値 ${median(w28days.filter(d=>d>0)).toFixed(1)}日（IQR ${quantile(w28days.filter(d=>d>0),.25).toFixed(1)}-${quantile(w28days.filter(d=>d>0),.75).toFixed(1)}）／ ≥7日:${nW7} ≥14日:${nW14} ≥21日:${nW21} ≥80%(22.4日):${nW80}（回答者${comp.length}名中）`);
  function linkCorr(get: (d: Diary) => number | null, label: string, expect: string) {
    const xs: number[] = [], ys: number[] = [];
    for (const { sid } of comp) { const x = totalScore.get(sid), d = diary.get(sid); if (x == null || !d) continue; const y = get(d); if (y != null) { xs.push(x); ys.push(y); } }
    const sp = spearman(xs, ys);
    console.log(`  MFI総合 ↔ 日誌 ${label}: ρ=${rf(sp.r)} (n=${xs.length})  [期待:${expect}]`);
  }
  linkCorr(d => d.haRate, "頭痛日の割合", "負");
  linkCorr(d => d.mibsMean, "日次MIBS-4(頭痛なし日)", "負");
  linkCorr(d => d.fulfillMean, "充実度 day_fulfillment", "正");
  // 自己申告 片頭痛日数 vs 日誌 頭痛率（自己申告と客観記録の一致）
  {
    const xs: number[] = [], ys: number[] = [];
    for (const { sid } of comp) { const m = migDays.get(sid), d = diary.get(sid); if (m != null && d) { xs.push(m); ys.push(d.haRate); } }
    console.log(`  （参考）自己申告片頭痛日数 ↔ 日誌 頭痛日割合: ρ=${rf(spearman(xs, ys).r)} (n=${xs.length})`);
  }

  // ---- 既知グループ ----
  console.log(`\n【既知グループ（MFI総合スコア）】`);
  // 片頭痛頻度（自己申告 4週日数）：低頻度EM<8 / 高頻度EM 8-14 / CM≥15
  const grp = (d: number | null) => d == null ? null : d < 8 ? "低頻度EM(<8)" : d < 15 ? "高頻度EM(8-14)" : "CM(≥15)";
  const byFreq: Record<string, number[]> = {};
  for (const { sid } of comp) { const g = grp(migDays.get(sid) ?? null); const s = totalScore.get(sid); if (g && s != null) (byFreq[g] ??= []).push(s); }
  for (const g of ["低頻度EM(<8)", "高頻度EM(8-14)", "CM(≥15)"]) if (byFreq[g]) console.log(`  ${g}: n=${byFreq[g].length} MFI ${mean(byFreq[g]).toFixed(1)}±${sd(byFreq[g]).toFixed(1)}`);
  if (byFreq["低頻度EM(<8)"] && byFreq["CM(≥15)"]) { const m = MWU(byFreq["低頻度EM(<8)"], byFreq["CM(≥15)"]); if (m) console.log(`  低頻度EM vs CM: p=${m.p.toFixed(4)} Cliff's δ=${m.delta.toFixed(2)}`); }
  // 性別
  const male = comp.filter(c => sexIdx.get(c.sid) === 0).map(c => totalScore.get(c.sid)).filter((v): v is number => v != null);
  const female = comp.filter(c => sexIdx.get(c.sid) === 1).map(c => totalScore.get(c.sid)).filter((v): v is number => v != null);
  console.log(`  男性 n=${male.length} ${mean(male).toFixed(1)} / 女性 n=${female.length} ${mean(female).toFixed(1)}` + (male.length >= 3 && female.length >= 3 ? ` (p=${MWU(male, female)?.p.toFixed(4)})` : ""));
  // 年代
  const ageBand = (sid: string) => { const a = qAge ? numVal(sid, qAge.id) : null; return a == null ? null : a < 40 ? "<40" : a < 60 ? "40-59" : "≥60"; };
  for (const b of ["<40", "40-59", "≥60"]) { const vs = comp.filter(c => ageBand(c.sid) === b).map(c => totalScore.get(c.sid)).filter((v): v is number => v != null); if (vs.length) console.log(`  年代 ${b}: n=${vs.length} MFI ${mean(vs).toFixed(1)}±${sd(vs).toFixed(1)}`); }

  // ---- 信頼度ティア別の妥当性（症例定義の効果） ----
  console.log(`\n【症例定義ティア別の妥当性（MFI総合との相関・負相関を期待）】`);
  function tier(name: string, pred: (sid: string) => boolean) {
    const S = new Set(comp.filter(c => pred(c.sid)).map(c => c.sid));
    const sp = (getY: (sid: string) => number | null) => { const xs: number[] = [], ys: number[] = []; for (const c of comp) { if (!S.has(c.sid)) continue; const x = totalScore.get(c.sid), y = getY(c.sid); if (x != null && y != null) { xs.push(x); ys.push(y); } } return { r: spearman(xs, ys).r, n: xs.length }; };
    const md = sp(midasTot), mig = sp(s => migDays.get(s) ?? null), dh = sp(s => diary.get(s)?.haRate ?? null);
    console.log(`  ${name} (n=${S.size}): ↔MIDAS ρ=${rf(md.r)}(${md.n}) / ↔自己申告片頭痛日数 ρ=${rf(mig.r)}(${mig.n}) / ↔日誌頭痛率 ρ=${rf(dh.r)}(${dh.n})`);
  }
  tier("全体", () => true);
  tier("複合症例定義", isCaseDef);
  tier("ID Migraine陽性", idMig);
  tier("片頭痛特異薬 使用者", drugSpecific);

  // ---- CSV 出力（EFA/ネットワーク用） ----
  // Table1 追加記述子（前兆・慢性相当・予防薬・CGRP・就労・MOHリスク）
  const auraPos = (sid: string) => { const a = resp.get(sid)?.get(qAura?.id ?? ""); return a ? (a.opts?.includes(0) ? 1 : 0) : ""; };
  const prevUse = (sid: string) => { const a = resp.get(sid)?.get(qPrevYN?.id ?? ""); const v = a?.opts?.[0]; return v == null ? "" : (v === 0 ? 1 : v === 1 ? 0 : ""); };
  const cgrpUse = (sid: string) => { const a = resp.get(sid)?.get(qCgrp?.id ?? ""); return a ? (a.opts?.some(i => i <= 5) ? 1 : 0) : ""; };
  const haDaysV = (sid: string) => qHaDays ? (numVal(sid, qHaDays.id) ?? "") : "";
  const employV = (sid: string) => qEmploy ? (scaleVal(sid, qEmploy.id) ?? "") : "";

  const header = ["sid", "age", "sex", "mig_days", "ha_days", "aura", "prev_use", "cgrp_use", "employ", "acute_days", "case_def", "id_mig", "drug_specific", "mfi_total", "f1", "f2", "midas", "mibs4",
    "diary_ha_rate", "diary_mibs", "diary_fulfill", "diary_logged",
    "w28_days", "w28_ha", "w28_haprop", "w28_free", "w28_fulfill", "w28_fulfilln", "w28_mibs", "w28_mibsn",
    ...domains.map((_, i) => `dom${i + 1}`), ...scored.map(q => `q${domains.indexOf(q.section_title)}_${q.order_index}`)];
  const lines = [header.join(",")];
  for (const { sid } of comp) {
    const w = win28.get(sid);
    const row = [sid.slice(0, 8), qAge ? numVal(sid, qAge.id) ?? "" : "", sexIdx.get(sid) ?? "", migDays.get(sid) ?? "", haDaysV(sid), auraPos(sid), prevUse(sid), cgrpUse(sid), employV(sid), acuteDays.get(sid) ?? "",
      isCaseDef(sid) ? 1 : 0, idMig(sid) ? 1 : 0, drugSpecific(sid) ? 1 : 0, (totalScore.get(sid) ?? "").toString(), factorScore(sid, F1D) ?? "", factorScore(sid, F2D) ?? "", midasTot(sid) ?? "", mibsTot(sid) ?? "",
      diary.get(sid)?.haRate ?? "", diary.get(sid)?.mibsMean ?? "", diary.get(sid)?.fulfillMean ?? "", diary.get(sid)?.loggedDays ?? "",
      w?.days ?? "", w?.ha ?? "", (w && !isNaN(w.haProp)) ? w.haProp : "", w?.free ?? "", w?.fulfillMean ?? "", w?.fulfillN ?? "", w?.mibsMean ?? "", w?.mibsN ?? "",
      ...domains.map(d => domScore[d].get(sid) ?? ""), ...scored.map(q => scaleVal(sid, q.id) ?? "")];
    lines.push(row.join(","));
  }
  writeFileSync(OUT_CSV, lines.join("\n"));
  console.log(`\n【CSV出力】${OUT_CSV}（${comp.length}行 × ${header.length}列）\n → EFA・ネットワーク解析はこのCSVをR/Pythonへ`);

  // ---- ファネルCSV（実施可能性・脱落分析：査読コメント6対応） ----
  // 配信(全send) → 同意(consented_at) → 開始(in_progress/completed) → 完了(completed) → 中断(in_progress)。
  // 中断者は最後に回答した設問の order_index / セクションで脱落地点を特定。年齢・性別で完了/未完了を比較。
  const FUNNEL_CSV = OUT_CSV.replace("mfi_matrix.csv", "mfi_funnel.csv");
  const qOrder = new Map(Q.map(q => [q.id, { oi: q.order_index, sec: q.section_title ?? "" }]));
  const nonTestSends = sends.filter(s => !isTest(s.patient_id));
  // 各 send の「カット時点でのステータス」（カット後の同意/完了は pending/in_progress 扱いに巻き戻す）
  const effStatus = (s: typeof sends[number]) => (s.status === "completed" && s.responded_at && s.responded_at < CUTOFF) ? "completed"
    : (s.consented_at && s.consented_at < CUTOFF) ? "in_progress" : "pending";
  const inProgIds = nonTestSends.filter(s => effStatus(s) === "in_progress").map(s => s.id);
  const inProgResp = await fetchByIds<{ send_id: string; question_id: string }>("survey_responses", "send_id, question_id", "send_id", inProgIds);
  const lastBySend = new Map<string, { n: number; maxOi: number; sec: string }>();
  for (const r of inProgResp) { const o = qOrder.get(r.question_id); if (!o) continue; const cur = lastBySend.get(r.send_id) ?? { n: 0, maxOi: -1, sec: "" }; cur.n++; if (o.oi > cur.maxOi) { cur.maxOi = o.oi; cur.sec = o.sec; } lastBySend.set(r.send_id, cur); }
  const ageFromBirth = (bd: string | null | undefined) => bd ? Math.floor((Date.now() - new Date(bd).getTime()) / (365.25 * 86400000)) : "";
  const fLines = ["sid,status,consented,n_answered,last_order_index,last_section,age,sex"];
  for (const s of nonTestSends) {
    const p = patById.get(s.patient_id); const es = effStatus(s);
    let nAns = "", maxOi = "", sec = "";
    if (es === "completed") { nAns = String(scored.length); sec = "完了"; }
    else if (es === "in_progress") { const l = lastBySend.get(s.id); if (l) { nAns = String(l.n); maxOi = String(l.maxOi); sec = l.sec; } }
    fLines.push([s.id.slice(0, 8), es, (s.consented_at && s.consented_at < CUTOFF) ? 1 : 0, nAns, maxOi, sec, ageFromBirth(p?.birthdate), p?.gender ?? ""].join(","));
  }
  writeFileSync(FUNNEL_CSV, fLines.join("\n"));
  const nConsented = nonTestSends.filter(s => effStatus(s) !== "pending").length;
  const nComp = nonTestSends.filter(s => effStatus(s) === "completed").length, nDrop = nonTestSends.filter(s => effStatus(s) === "in_progress").length;
  console.log(`【ファネルCSV】${FUNNEL_CSV}（${nonTestSends.length}件・カット${CUTOFF}）配信${nonTestSends.length}→同意/開始${nConsented}→完了${nComp}→中断${nDrop}`);
}
main().then(() => process.exit(0)).catch(e => { console.error(e); process.exit(1); });
