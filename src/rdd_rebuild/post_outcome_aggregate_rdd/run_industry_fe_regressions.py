#!/usr/bin/env python
"""Re-run firm-year and firm-quarter with industry FE variants."""
import pandas as pd, numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/post_outcome_aggregate_rdd")
OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
OC_LABELS = {"overall_rating":"Overall","career_opp":"Career Opp","comp_benefit":"Comp & Benefits","senior_mgmt":"Senior Mgmt","wlb":"WLB","culture":"Culture"}
FE_VARIANTS = [("none",[]),("year",["election_year"]),("sic2",["sic2"]),("ff48",["ff48"]),
               ("year_sic2",["election_year","sic2"]),("year_ff48",["election_year","ff48"])]
BANDWIDTHS = [("global",None),("m20",lambda d: d[d["abs_margin"]<=0.20])]

def run_ols(data, outcome, bw_fn, poly, fe_spec, fe_cols):
    d = bw_fn(data) if bw_fn is not None else data.copy()
    d = d[d["outcome"]==outcome].dropna(subset=["post_mean","pre_mean","win","margin"])
    if len(d) < 30: return None
    d["win_margin"] = d["win"] * d["margin"]
    formula = "post_mean ~ win + margin + win_margin + pre_mean"
    for col in fe_cols:
        if col in d.columns and d[col].notna().sum() > 5:
            formula += f" + C({col})"
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["gvkey"]})
        return {"outcome":outcome,"poly":poly,"fe_spec":fe_spec,
            "estimate":m.params.get("win",np.nan),"se":m.bse.get("win",np.nan),
            "p_value":m.pvalues.get("win",np.nan),"n_events":len(d),"n_gvkeys":d["gvkey"].nunique()}
    except: return None

fy_results, fq_results = [], []
for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet")
    d = d[d["version"]=="A"]
    for bw_label, bw_fn in BANDWIDTHS:
        for fe_spec, fe_cols in FE_VARIANTS:
            for oc in OUTCOMES:
                res = run_ols(d, oc, bw_fn, "linear", fe_spec, fe_cols)
                if res: res.update({"employee_sample":emp,"bw_label":bw_label}); fy_results.append(res)
    print(f"  FY {emp}: {len(fy_results)} specs")

for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_quarter_post_rdd_data_{emp}.parquet")
    d = d[d["window"]=="0_365"]
    d = d.rename(columns={"pre_mean_365":"pre_mean"})  # align column name
    for bw_label, bw_fn in BANDWIDTHS:
        for fe_spec, fe_cols in FE_VARIANTS:
            for oc in OUTCOMES:
                res = run_ols(d, oc, bw_fn, "linear", fe_spec, fe_cols)
                if res: res.update({"employee_sample":emp,"bw_label":bw_label}); fq_results.append(res)
    print(f"  FQ {emp}: {len(fq_results)} specs")

df_fy = pd.DataFrame(fy_results)
df_fq = pd.DataFrame(fq_results)
df_fy.to_csv(OUT / "firm_year_industry_fe_results.csv", index=False)
df_fq.to_csv(OUT / "firm_quarter_industry_fe_results.csv", index=False)

# Report
print("\n=== WLB: Firm-Year FE Comparison (current, global) ===")
for fe_spec in ["none","year","sic2","ff48","year_sic2","year_ff48"]:
    r = df_fy[(df_fy["outcome"]=="wlb")&(df_fy["employee_sample"]=="current")&(df_fy["bw_label"]=="global")&(df_fy["fe_spec"]==fe_spec)]
    if len(r)>0:
        r=r.iloc[0]; sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"  {fe_spec:12s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig}")

print("\n=== All Outcomes: FY current global, no FE vs FF48 FE ===")
for oc in OUTCOMES:
    nr = df_fy[(df_fy["outcome"]==oc)&(df_fy["employee_sample"]=="current")&(df_fy["bw_label"]=="global")&(df_fy["fe_spec"]=="none")]
    fr = df_fy[(df_fy["outcome"]==oc)&(df_fy["employee_sample"]=="current")&(df_fy["bw_label"]=="global")&(df_fy["fe_spec"]=="ff48")]
    tn = nr["estimate"].values[0] if len(nr)>0 else np.nan; pn = nr["p_value"].values[0] if len(nr)>0 else np.nan
    tf = fr["estimate"].values[0] if len(fr)>0 else np.nan; pf = fr["p_value"].values[0] if len(fr)>0 else np.nan
    print(f"  {OC_LABELS.get(oc,oc):20s}: noFE tau={tn:+.4f} p={pn:.3f} | FF48 tau={tf:+.4f} p={pf:.3f}")
print("Done.")
