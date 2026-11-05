#!/usr/bin/env python3
"""Additional file 2 for the J Headache Pain submission (2026-11-05).

Reads mfi_supp.html (built by build_supp_v2.py -> build_supp_v3.py) and writes
mfi_supp_jhp.html with the changes needed for consistency with the submitted
manuscript (Rev24):
  1. Domain labels in the tables use the names in the manuscript.
  2. The auxiliary total score is removed (the manuscript does not combine the
     two factors into a total score).
  3. The pointer to an "Open practices" section, which the JHP format does not
     have, is removed.
  4. The completer vs non-completer sex comparison states its source (account
     profiles), because Table 1 of the manuscript reports survey-reported sex.
  5. Title and print styles for a PDF additional file.
Run from the repository root:  python3 analysis/build_supp_v4_jhp.py
"""
import re

src = open("mfi_supp.html", encoding="utf-8").read()
h = src

def sub(pattern, repl, count_expected, flags=0):
    global h
    h, n = re.subn(pattern, repl, h, flags=flags)
    assert n == count_expected, (pattern, n)

# 1. domain labels (exact table-cell text)
LABELS = {
    "Freedom to do activities": "Freedom of activity",
    "Hope and staying with treatment": "Hope and treatment continuity",
    "Impact on people close to you": "Impact on close others",
    "Sensing an oncoming attack (premonitory awareness)": "Premonitory-symptom awareness",
    "Trigger awareness": "Trigger recognition",
    "Preemptive action": "Pre-emptive action",
    "Preventing medication-overuse headache": "Prevention of medication overuse",
    "Overall assessment": "Global freedom evaluation",
}
for old, new in LABELS.items():
    h, n = re.subn(r"<td>" + re.escape(old) + r"</td>", "<td>" + new + "</td>", h)
    assert n > 0, old

# 2. auxiliary total score
sub(r"; total scored set α=[0-9.]+ \(auxiliary only, because the two factors were uncorrelated\)", "", 1)
sub(r"Factor and total scores were mildly negatively skewed", "Factor scores were mildly negatively skewed", 1)
sub(r"Unadjusted correlations of the MFI factors and the auxiliary total score with each measure",
    "Unadjusted correlations of the MFI factor scores with each measure", 1)
sub(r" Because the two factors were uncorrelated \(factor-score r=[-0-9.]+\), the total is an auxiliary summary only and is shown last\.", "", 1)
# drop the last column of the S3b table
i = h.index("<h2>S3b."); j = h.index("</table>", i)
block = h[i:j]
block2, n = re.subn(r"<t[hd][^>]*>(?:(?!<t[hd]).)*?</t[hd]>(</tr>)", r"\1", block, flags=re.S)
assert n == 4, n
h = h[:i] + block2 + h[j:]
# drop the total row of the S2c table
sub(r"<tr><td>Total \(auxiliary\)</td>.*?</tr>\n?", "", 1, flags=re.S)
assert "auxiliary" not in h and "Total (auxiliary)" not in h

# 3. Open practices pointer
sub(r" \(see Open practices\)", "", 1)

# 4. source of the sex comparison
sub(r"or sex \((\d+)% vs (\d+)% women\)",
    r"or sex as recorded in the account profiles (\1% vs \2% women; Table 1 of the main text reports survey-reported sex)", 1)

# 5. title and print styles
sub(r"<h1>Supplementary material — Migraine Freedom Index \(MFI\) pilot \(n=(\d+)\)</h1>",
    r"<h1>Additional file 2. Supplementary Tables S1 to S4</h1><p>Development of the Migraine Freedom Index, a patient-reported measure of migraine freedom in daily life: an exploratory linked-diary study (n=\1)</p>", 1)
PRINT = ("<style>@page{size:A4 landscape;margin:12mm}"
         "body{color:#000;font-family:'Helvetica Neue',Arial,'Hiragino Sans','Hiragino Kaku Gothic ProN',sans-serif}"
         "h2{color:#000}p.note{color:#222}th{background:none}"
         "tr{page-break-inside:avoid}h1,h2{page-break-after:avoid}</style>")
h = h.replace("</style>", "</style>" + PRINT, 1)

open("mfi_supp_jhp.html", "w", encoding="utf-8").write(h)
print("wrote mfi_supp_jhp.html")
