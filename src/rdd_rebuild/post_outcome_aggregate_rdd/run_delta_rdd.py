#!/usr/bin/env python
"""v10c: Delta RDD — DV = post_mean - pre_mean, no pre_mean control."""

import pandas as pd, numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/post_outcome_aggregate_rdd")
OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
OC_LABELS = {"overall_rating":"Overall","career_opp":"Career Opp","comp_benefit":"Comp & Benefits","senior_mgmt":"Senior Mgmt","wlb":"WLB","culture":"Culture"}
FE_VARIANTS = [("none",[]),("year",["election_year"]),("sic2",["sic2"]),("ff48",["ff48"]),
               ("year_sic2",["election_year","sic2"]),("year_ff48",["election_year","ff48"])]
BWS = [("global",None),("m20",lambda d: d[d["abs_margin"]<=0.20]),
       ("m10",lambda d: d[d["abs_margin"]<=0.10]),("m05",lambda d: d[d["abs_margin"]<=0.05])]

def run_ols(data, outcome, bw_fn, poly, fe_spec, fe_cols):
    d = bw_fn(data) if bw_fn is not None else data.copy()
    d = d[d["outcome"]==outcome].dropna(subset=["delta","win","margin"])
    if len(d) < 30: return None
    d["win_margin"] = d["win"] * d["margin"]
    formula = "delta ~ win + margin + win_margin"
    if poly == "quadratic":
        d["margin2"]=d["margin"]**2; d["win_margin2"]=d["win"]*d["margin2"]
        formula += " + margin2 + win_margin2"
    for col in fe_cols:
        if col in d.columns and d[col].notna().sum() > 5:
            formula += f" + C({col})"
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["gvkey"]})
        return {"outcome":outcome,"poly":poly,"fe_spec":fe_spec,
            "estimate":m.params.get("win",np.nan),"se":m.bse.get("win",np.nan),
            "p_value":m.pvalues.get("win",np.nan),"n_events":len(d),"n_gvkeys":d["gvkey"].nunique()}
    except: return None

# Firm-year delta
print("=== Firm-Year Delta RDD ===")
fy_results = []
for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet")
    d["delta"] = d["post_mean"] - d["pre_mean"]
    d = d[d["version"]=="A"]
    filters = [("unrestricted",None)]
    if emp == "all":
        filters += [("n5",lambda dd: dd[(dd["n_pre"]>=5)&(dd["n_post"]>=5)]),
                    ("n10",lambda dd: dd[(dd["n_pre"]>=10)&(dd["n_post"]>=10)])]
    for f_label, f_fn in filters:
        df_use = f_fn(d) if f_fn else d
        for bw_label, bw_fn in BWS:
            for poly in ["linear","quadratic"]:
                for fe_spec, fe_cols in FE_VARIANTS:
                    for oc in OUTCOMES:
                        res = run_ols(df_use, oc, bw_fn, poly, fe_spec, fe_cols)
                        if res:
                            res.update({"employee_sample":emp,"filter":f_label,"bw_label":bw_label,
                                        "dv_type":"delta","multi_version":"A"})
                            fy_results.append(res)
    # Version C
    dC = d[d["is_first"]]
    for bw_label, bw_fn in BWS[:2]:  # global, m20 only
        for fe_spec, fe_cols in FE_VARIANTS:
            for oc in OUTCOMES:
                res = run_ols(dC, oc, bw_fn, "linear", fe_spec, fe_cols)
                if res:
                    res.update({"employee_sample":emp,"filter":"unrestricted","bw_label":bw_label,
                                "dv_type":"delta","multi_version":"C"})
                    fy_results.append(res)
    print(f"  FY {emp}: {len(fy_results)} specs")

df_fy = pd.DataFrame(fy_results)
df_fy.to_csv(OUT / "firm_year_delta_rdd_results.csv", index=False)

# Firm-quarter delta
print("\n=== Firm-Quarter Delta RDD ===")
fq_results = []
for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_quarter_post_rdd_data_{emp}.parquet")
    d["delta"] = d["post_mean"] - d["pre_mean_365"]
    for wlabel in ["0_90","0_180","0_365"]:
        dw = d[d["window"]==wlabel]
        for bw_label, bw_fn in BWS[:3]:  # global, m20, m10
            for fe_spec, fe_cols in FE_VARIANTS:
                for oc in OUTCOMES:
                    res = run_ols(dw, oc, bw_fn, "linear", fe_spec, fe_cols)
                    if res:
                        res.update({"employee_sample":emp,"bw_label":bw_label,"window":wlabel,"dv_type":"delta"})
                        fq_results.append(res)
    print(f"  FQ {emp}: {len(fq_results)} specs")

df_fq = pd.DataFrame(fq_results)
df_fq.to_csv(OUT / "firm_quarter_delta_rdd_results.csv", index=False)

# ── rdrobust ──
print("\n=== rdrobust ===")
rdr_results = []
try:
    from rdrobust import rdrobust
    for emp in ["current","all"]:
        d = pd.read_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet")
        d["delta"] = d["post_mean"] - d["pre_mean"]
        for oc in OUTCOMES:
            sub = d[(d["outcome"]==oc)].dropna(subset=["delta","margin"])
            if len(sub) < 50: continue
            rdr = rdrobust(y=sub["delta"].values, x=sub["margin"].values, c=0, p=1, kernel="triangular")
            rdr_results.append({"outcome":oc,"employee_sample":emp,
                "h_left":rdr.bws.iloc[0,0],"h_right":rdr.bws.iloc[0,1],
                "N_left":int(rdr.N[0]),"N_right":int(rdr.N[1]),
                "estimate":rdr.coef["Robust"][0],"se":rdr.se["Robust"][0],"p_value":rdr.pv["Robust"][0]})
    print(f"  {len(rdr_results)} rdrobust results")
except ImportError:
    print("  rdrobust not installed — skipping")

if rdr_results:
    pd.DataFrame(rdr_results).to_csv(OUT / "delta_rdrobust_results.csv", index=False)

# ── Report ──
print("\n=== Main Table: FY delta, current, A, unrestricted, global, linear, no FE ===")
m = df_fy[(df_fy["employee_sample"]=="current")&(df_fy["bw_label"]=="global")&
          (df_fy["poly"]=="linear")&(df_fy["fe_spec"]=="none")&(df_fy["multi_version"]=="A")&
          (df_fy["filter"]=="unrestricted")]
for _, r in m.sort_values("outcome").iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {OC_LABELS.get(r['outcome'],r['outcome']):20s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")

print("\n=== WLB Bandwidth Robustness (FY, current, linear, no FE) ===")
for bw in ["global","m20","m10","m05"]:
    r = df_fy[(df_fy["employee_sample"]=="current")&(df_fy["bw_label"]==bw)&(df_fy["poly"]=="linear")&
              (df_fy["fe_spec"]=="none")&(df_fy["outcome"]=="wlb")&(df_fy["multi_version"]=="A")]
    if len(r)>0:
        r=r.iloc[0]; sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"  {bw:8s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")

if rdr_results:
    print("\n=== rdrobust (current, FY) ===")
    for _, r in pd.DataFrame(rdr_results)[pd.DataFrame(rdr_results)["employee_sample"]=="current"].iterrows():
        sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"  {OC_LABELS.get(r['outcome'],r['outcome']):20s}: h=[{r['h_left']:.3f},{r['h_right']:.3f}] tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig}")

print("Done.")
