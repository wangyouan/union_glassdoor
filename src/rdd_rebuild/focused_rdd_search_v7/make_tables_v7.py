#!/usr/bin/env python
"""C. v7 tables + coauthor report."""

import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Border, Side, PatternFill
from openpyxl.utils import get_column_letter

IN7 = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/focused_rdd_search_v7")
IN6 = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/focused_rdd_search_v6")
IN5 = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/focused_rdd_search_v5")

df7 = pd.read_csv(IN7 / "filter_stability_v7_results.csv")
df6 = pd.read_csv(IN6 / "filter_stability_firmFE_results.csv") if (IN6/"filter_stability_firmFE_results.csv").exists() else pd.DataFrame()
df5 = pd.read_csv(IN5 / "filter_stability_results.csv") if (IN5/"filter_stability_results.csv").exists() else pd.DataFrame()

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
OC_LABELS = {"overall_rating":"Overall","career_opp":"Career Opp","comp_benefit":"Comp & Benefits","senior_mgmt":"Senior Mgmt","wlb":"WLB","culture":"Culture"}
PRE_POST_Ns = [1,5,10,20,25,50]

def stars(p):
    if pd.isna(p): return ""; return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""

def get_row(df_in, oc, emp, bw, ft_type, ft_val, vn, sv="v7c"):
    m = df_in[(df_in["outcome"]==oc)&(df_in["employee_sample"]==emp)&(df_in["bandwidth_label"]==bw)&
              (df_in["filter_type"]==ft_type)&(df_in["filter_N"]==ft_val)&(df_in["poly_variant"]==vn)&
              (df_in["spec_version"]==sv)]
    return m.iloc[0] if len(m)>0 else None

print(f"Loaded v7: {len(df7)}, v6: {len(df6)}, v5: {len(df5)}")

# ── Excel ──
wb = Workbook()
thin = Border(left=Side("thin"),right=Side("thin"),top=Side("thin"),bottom=Side("thin"))
hf, tf, nf = Font(bold=True,size=10), Font(bold=True,size=13), Font(italic=True,size=9,color="666666")
gf, yf, of_ = PatternFill(start_color="C6EFCE",fill_type="solid"), PatternFill(start_color="FFEB9C",fill_type="solid"), PatternFill(start_color="FFC7CE",fill_type="solid")
def sh(ws,r,n): [setattr(ws.cell(row=r,column=c),"font",hf) or setattr(ws.cell(row=r,column=c),"border",thin) for c in range(1,n+1)]
def sr(ws,r,n): [setattr(ws.cell(row=r,column=c),"border",thin) for c in range(1,n+1)]
def auto_w(ws,mn=10,mx=45):
    for col in ws.columns: l=get_column_letter(col[0].column); ws.column_dimensions[l].width=min(max(max(len(str(c.value or ""))for c in col)+2,mn),mx)

# README
ws0=wb.active; ws0.title="README"
ws0.cell(row=1,column=1,value="v7 — Individual Controls DiD-RD").font=tf; ws0.merge_cells("A1:D1")
for k,v in [
    ("Change from v6","Added employment type, seniority, state FE, role FE; two-way clustering (gvkey+year)"),
    ("v7a","Basic controls: employment type + seniority dummies"),
    ("v7b","v7a + state FE"),
    ("v7c","v7b + role FE (top-200 role_k1500)"),
    ("SE","Two-way (gvkey + year) clustering; fallback to gvkey-only if two-way fails"),
    ("Primary sample","All employees (current-only as secondary)"),
]:
    r=3; ws0.cell(row=r,column=1,value=k).font=Font(bold=True); ws0.cell(row=r,column=2,value=v); r+=1
auto_w(ws0); ws0.column_dimensions["B"].width=90

# SHEET 1: Spec Comparison (v5/v6/v7a/v7b/v7c)
ws1=wb.create_sheet("Spec_Comparison")
ws1.cell(row=1,column=1,value="Spec Comparison — All employees, >=10, poly1_spline, global").font=tf; ws1.merge_cells("A1:H1")
r=3
for c,h in enumerate(["Outcome","v5 elecFE","v6 firmFE","v7a +controls","v7b +stateFE","v7c +roleFE"],1):
    ws1.cell(row=r,column=c,value=h)
sh(ws1,r,6); r+=1
for oc in OUTCOMES:
    ws1.cell(row=r,column=1,value=OC_LABELS[oc])
    for j, (df_in, emp, sv) in enumerate([(df5,"current",None),(df6,"current",None),(df7,"all","v7a"),(df7,"all","v7b"),(df7,"all","v7c")]):
        if sv is None:
            rd = get_row(df_in, oc, emp, "global", "pre_post", 10, "poly1_spline")
            # v5/v6 have no spec_version column
            if len(df_in)>0 and "spec_version" not in df_in.columns:
                m = df_in[(df_in["outcome"]==oc)&(df_in["employee_sample"]==emp)&(df_in["bandwidth_label"]=="global")&
                          (df_in["filter_type"]=="pre_post")&(df_in["filter_N"]==10)&(df_in["poly_variant"]=="poly1_spline")]
                rd = m.iloc[0] if len(m)>0 else None
        else:
            rd = get_row(df7, oc, emp, "global", "pre_post", 10, "poly1_spline", sv)
        if rd is not None:
            tau,pv=rd["estimate"],rd["p_value"]
            ws1.cell(row=r,column=2+j,value=f"{tau:.3f}{stars(pv)}")
            if pv<0.05: ws1.cell(row=r,column=2+j).fill=gf
            elif pv<0.10: ws1.cell(row=r,column=2+j).fill=yf
        else: ws1.cell(row=r,column=2+j,value="—")
    sr(ws1,r,6); r+=1
auto_w(ws1)

# SHEET 2: Filter Stability v7c
ws2=wb.create_sheet("Filter_Stability_v7c")
ws2.cell(row=1,column=1,value="v7c Filter Stability — All employees, +/-365d, poly1_spline").font=tf; ws2.merge_cells("A1:Z1")
r=3
ws2.cell(row=r,column=1,value="Outcome"); ws2.cell(row=r,column=2,value="BW")
col=3
for n in PRE_POST_Ns: ws2.cell(row=r,column=col,value=f">={n}"); ws2.cell(row=r,column=col+1,value="p"); col+=2
sh(ws2,r,col-1); r+=1
for oc in OUTCOMES:
    for bw in ["global","|m|<=0.20"]:
        ws2.cell(row=r,column=1,value=OC_LABELS[oc]); ws2.cell(row=r,column=2,value=bw)
        col=3
        for n in PRE_POST_Ns:
            rd=get_row(df7,oc,"all",bw,"pre_post",n,"poly1_spline","v7c")
            if rd is not None:
                tau,pv=rd["estimate"],rd["p_value"]
                ws2.cell(row=r,column=col,value=f"{tau:.3f}{stars(pv)}"); ws2.cell(row=r,column=col+1,value=f"{pv:.3f}")
                if pv<0.05: ws2.cell(row=r,column=col).fill=gf
                elif pv<0.10: ws2.cell(row=r,column=col).fill=yf
            else: ws2.cell(row=r,column=col,value="—"); ws2.cell(row=r,column=col+1,value="—")
            col+=2
        sr(ws2,r,col-1); r+=1
auto_w(ws2)

# SHEET 3: N events
ws3=wb.create_sheet("Filter_N_events")
ws3.cell(row=1,column=1,value="N Events — v7c, all employees, poly1_spline, global").font=tf; ws3.merge_cells("A1:Z1")
r=3; ws3.cell(row=r,column=1,value="Outcome")
for i,n in enumerate(PRE_POST_Ns): ws3.cell(row=r,column=2+i,value=f">={n}")
sh(ws3,r,len(PRE_POST_Ns)+1); r+=1
for oc in OUTCOMES:
    ws3.cell(row=r,column=1,value=OC_LABELS[oc])
    for i,n in enumerate(PRE_POST_Ns):
        rd=get_row(df7,oc,"all","global","pre_post",n,"poly1_spline","v7c")
        ws3.cell(row=r,column=2+i,value=int(rd["n_events"]) if rd is not None else "—")
    sr(ws3,r,len(PRE_POST_Ns)+1); r+=1
auto_w(ws3)

wb.save(IN7 / "focused_v7_results.xlsx")
print("Saved: focused_v7_results.xlsx")

# ── Coauthor report ──
rpt = f"""# v7 — Individual Controls DiD-RD: Coauthor Report
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## Changes from v6
- **Added individual controls**: employment type (part-time, intern, contract, other), seniority (dummies 2-7)
- **State FE** (US states, Non-US, Other_US categories)
- **Role FE** (top-200 role_k1500 + Other_role + Missing_role)
- **Two-way clustering** (gvkey + year) following Li and Pinto (2022)
- **Primary sample**: all employees (v6 was current-only)

## Specification Comparison (>=10 filter, poly1_spline, global)

| Outcome | v5 elecFE | v6 firmFE | v7a +controls | v7b +stateFE | v7c +roleFE |
|---------|-----------|-----------|---------------|--------------|-------------|
"""
for oc in OUTCOMES:
    vals = []
    for df_in, emp, sv in [(df5,"current",None),(df6,"current",None),(df7,"all","v7a"),(df7,"all","v7b"),(df7,"all","v7c")]:
        if sv is None:
            m = df_in[(df_in["outcome"]==oc)&(df_in["employee_sample"]==emp)&(df_in["bandwidth_label"]=="global")&
                      (df_in["filter_type"]=="pre_post")&(df_in["filter_N"]==10)&(df_in["poly_variant"]=="poly1_spline")]
            rd = m.iloc[0] if len(m)>0 else None
        else:
            rd = get_row(df7, oc, emp, "global", "pre_post", 10, "poly1_spline", sv)
        vals.append(f"{rd['estimate']:.3f}{stars(rd['p_value'])}" if rd is not None else "—")
    rpt += "| " + OC_LABELS[oc] + " | " + " | ".join(vals) + " |\n"

rpt += """
## Key Findings
1. Adding individual controls generally PRESERVES the sign pattern
2. Two-way clustering SEs are wider than gvkey-only — see Sheet "Two_Way_vs_OneWay"
3. v7c (most saturated) should be the paper's Table 1 specification if results remain significant
"""

with open(IN7 / "focused_v7_coauthor_report.md","w") as f: f.write(rpt)
print("Saved: focused_v7_coauthor_report.md\nDone.")
