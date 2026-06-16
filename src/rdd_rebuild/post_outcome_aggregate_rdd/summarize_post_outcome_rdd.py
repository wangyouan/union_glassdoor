#!/usr/bin/env python
"""Script 3: Summarize post-outcome RDD results."""

import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
IN_FY = Path("outputs/rdd_rebuild/post_outcome_aggregate_rdd/firm_year_post_rdd_results.csv")
IN_FQ = Path("outputs/rdd_rebuild/post_outcome_aggregate_rdd/firm_quarter_cumulative_rdd_results.csv")
OUT = PROJ / "outputs/rdd_rebuild/post_outcome_aggregate_rdd"

fy = pd.read_csv(IN_FY)
fq = pd.read_csv(IN_FQ)
print(f"FY: {len(fy)} specs, FQ: {len(fq)} specs")

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
OC_LABELS = {"overall_rating":"Overall","career_opp":"Career Opp","comp_benefit":"Comp & Benefits",
             "senior_mgmt":"Senior Mgmt","wlb":"WLB","culture":"Culture"}

def stars(p):
    if pd.isna(p): return ""; return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

# Consistency analysis
cons_rows = []
for oc in OUTCOMES:
    fy_cur = fy[(fy["outcome"]==oc)&(fy["employee_sample"]=="current")&(fy["multi_version"]=="A")&
                (fy["bw_label"]=="global")&(fy["poly"]=="linear")&(fy["fe_spec"]=="none")]
    fq_cur = fq[(fq["outcome"]==oc)&(fq["employee_sample"]=="current")&(fq["bw_label"]=="global")&
                (fq["window"]=="0_365")&(fq["poly"]=="linear")&(fq["fe_spec"]=="none")]
    fy_m20 = fy[(fy["outcome"]==oc)&(fy["employee_sample"]=="current")&(fy["multi_version"]=="A")&
                (fy["bw_label"]=="m20")&(fy["poly"]=="linear")&(fy["fe_spec"]=="none")]

    signs = []
    for df_sub in [fy_cur, fq_cur]: signs.append(np.sign(df_sub["estimate"].values[0]) if len(df_sub)>0 else 0)
    sign_fy = signs[0]; sign_fq = signs[1]
    sign_m20 = np.sign(fy_m20["estimate"].values[0]) if len(fy_m20)>0 else 0
    sign_consistent = sign_fy == sign_fq == sign_m20 and sign_fy != 0

    p_ok = sum(1 for df_sub in [fy_cur,fq_cur] if len(df_sub)>0 and df_sub["p_value"].values[0]<0.10)
    n_ev = int(fy_cur["n_events"].values[0]) if len(fy_cur)>0 else 0
    n_gv = int(fy_cur["n_gvkeys"].values[0]) if len(fy_cur)>0 else 0
    size_ok = (n_ev>=50 and n_gv>=30)

    score = sum([sign_consistent, p_ok>=1, p_ok>=2, size_ok, oc!="diversity"])
    label = "Promising" if score>=4 else ("Borderline" if score>=3 else "Null")

    cons_rows.append({
        "outcome":oc,"sign_consistent":sign_consistent,"sign_fy":sign_fy,"sign_fq":sign_fq,
        "p_fy":fy_cur["p_value"].values[0] if len(fy_cur)>0 else np.nan,
        "p_fq":fq_cur["p_value"].values[0] if len(fq_cur)>0 else np.nan,
        "n_events":n_ev,"n_gvkeys":n_gv,"size_ok":size_ok,
        "consistency_score":score,"label":label})

df_cons = pd.DataFrame(cons_rows)
df_cons.to_csv(OUT / "post_outcome_rdd_consistency_summary.csv", index=False)
best = df_cons[df_cons["consistency_score"]>=3].sort_values("consistency_score", ascending=False)
best.to_csv(OUT / "post_outcome_rdd_best_candidates.csv", index=False)

print("\n=== Firm-Year Post RDD Main (current, A, global, linear, no FE) ===")
for _, r in fy[(fy["employee_sample"]=="current")&(fy["multi_version"]=="A")&(fy["bw_label"]=="global")&
               (fy["poly"]=="linear")&(fy["fe_spec"]=="none")].sort_values("outcome").iterrows():
    sig = stars(r["p_value"])
    print(f"  {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")

print(f"\n=== Consistency Summary ===")
for _, r in df_cons.iterrows():
    print(f"  {r['outcome']:20s}: score={r['consistency_score']} {r['label']:12s} sign_fy={int(r['sign_fy']):+d} sign_fq={int(r['sign_fq']):+d} p_fy={r['p_fy']:.3f} p_fq={r['p_fq']:.3f}")

# Build summary markdown
rpt = f"""# Post-Outcome RDD Summary
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## Firm-Year Post RDD (current, A, global, linear, no FE)

| Outcome | tau | SE | p | N Events |
|---------|-----|----|----|----------|
"""
for _, r in fy[(fy["employee_sample"]=="current")&(fy["multi_version"]=="A")&(fy["bw_label"]=="global")&
               (fy["poly"]=="linear")&(fy["fe_spec"]=="none")].sort_values("outcome").iterrows():
    rpt += f"| {OC_LABELS.get(r['outcome'],r['outcome'])} | {r['estimate']:+.3f}{stars(r['p_value'])} | ({r['se']:.3f}) | {r['p_value']:.3f} | {int(r['n_events'])} |\n"

rpt += f"""
## Consistency

| Outcome | Score | Label | FY sign | FQ sign |
|---------|-------|-------|---------|---------|
"""
for _, r in df_cons.iterrows():
    rpt += f"| {OC_LABELS.get(r['outcome'],r['outcome'])} | {r['consistency_score']} | {r['label']} | {int(r['sign_fy']):+d} | {int(r['sign_fq']):+d} |\n"

with open(OUT / "post_outcome_rdd_report.md","w") as f: f.write(rpt)
print(f"\nSaved report. Done.")
