#!/usr/bin/env python
"""Add industry FE (SIC2, FF48) and re-run post-outcome RDD with industry controls."""

import pandas as pd, numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
CTRL = PROJ / "outputs/compustat_firm_controls.parquet"
OUT = PROJ / "outputs/rdd_rebuild/post_outcome_aggregate_rdd"

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
OC_LABELS = {"overall_rating":"Overall","career_opp":"Career Opp","comp_benefit":"Comp & Benefits",
             "senior_mgmt":"Senior Mgmt","wlb":"WLB","culture":"Culture"}

# ── STEP 1: Load industry data ──
print("=== STEP 1: Industry Data ===")
ctrl = pd.read_parquet(CTRL)
print(f"  Compustat controls: {len(ctrl)} rows, {ctrl['gvkey'].nunique()} gvkeys")
print(f"  Columns: sic={ctrl['sic'].notna().sum()}, sic2={ctrl['sic2'].notna().sum()}, ff48={ctrl['ff48'].notna().sum()}")
# Build gvkey → sic2, ff48 lookup (most common value per gvkey)
gvkey_ind = ctrl.groupby("gvkey").agg(
    sic2=("sic2", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    ff48=("ff48", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
).reset_index()
gvkey_ind["sic2"] = gvkey_ind["sic2"].astype(str)
gvkey_ind["ff48"] = gvkey_ind["ff48"].astype(str)
print(f"  gvkey lookup: {len(gvkey_ind)} gvkeys, sic2 unique={gvkey_ind['sic2'].nunique()}, ff48 unique={gvkey_ind['ff48'].nunique()}")

# ── STEP 2: Merge into event-level data ──
print("\n=== STEP 2: Merge Industry into Event Data ===")
for data_file in ["firm_year_post_rdd_data_current.parquet","firm_year_post_rdd_data_all.parquet",
                   "firm_quarter_post_rdd_data_current.parquet","firm_quarter_post_rdd_data_all.parquet"]:
    fp = OUT / data_file
    if not fp.exists():
        print(f"  SKIP {data_file} (not found)")
        continue
    d = pd.read_parquet(fp)
    d["gvkey"] = d["gvkey"].astype(str)
    gvkey_ind["gvkey"] = gvkey_ind["gvkey"].astype(str)
    n_before = len(d)
    d = d.merge(gvkey_ind, on="gvkey", how="left")
    n_matched = d["sic2"].notna().sum()
    print(f"  {data_file}: {n_matched}/{n_before} rows matched ({n_matched/n_before*100:.1f}%), "
          f"sic2 unique={d['sic2'].nunique()}, ff48 unique={d['ff48'].nunique()}")
    d.to_parquet(fp, index=False)

# ── STEP 3: Re-run with industry FE ──
print("\n=== STEP 3: Re-run Firm-Year with Industry FE ===")
FE_VARIANTS = [
    ("none",[]), ("year",["election_year"]),
    ("sic2",["sic2"]), ("ff48",["ff48"]),
    ("year_sic2",["election_year","sic2"]), ("year_ff48",["election_year","ff48"])
]
BANDWIDTHS = [("global",None),("m20",lambda d: d[d["abs_margin"]<=0.20])]

def run_ols(data, outcome, bw_fn, poly, fe_spec, fe_cols):
    d = bw_fn(data) if bw_fn is not None else data.copy()
    d = d[d["outcome"]==outcome].dropna(subset=["post_mean","pre_mean","win","margin"])
    if len(d) < 30: return None
    d["win_margin"] = d["win"]*d["margin"]
    formula = "post_mean ~ win + margin + win_margin + pre_mean"
    if poly == "quadratic":
        d["margin2"]=d["margin"]**2; d["win_margin2"]=d["win"]*d["margin2"]
        formula += " + margin2 + win_margin2"
    for col in fe_cols:
        if col in d.columns and d[col].notna().sum() > 10:
            formula += f" + C({col})"
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["gvkey"]})
        return {"outcome":outcome,"poly":poly,"fe_spec":fe_spec,
            "estimate":m.params.get("win",np.nan),"se":m.bse.get("win",np.nan),
            "p_value":m.pvalues.get("win",np.nan),"n_events":len(d),"n_gvkeys":d["gvkey"].nunique()}
    except: return None

# Firm-year
fy_results = []
for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet")
    d = d[d["version"]=="A"]
    for bw_label, bw_fn in BANDWIDTHS:
        for fe_spec, fe_cols in FE_VARIANTS:
            for oc in OUTCOMES:
                res = run_ols(d, oc, bw_fn, "linear", fe_spec, fe_cols)
                if res:
                    res.update({"employee_sample":emp,"bw_label":bw_label})
                    fy_results.append(res)
    print(f"  {emp}: {len(fy_results)} FY specs")

df_fy = pd.DataFrame(fy_results)
df_fy.to_csv(OUT / "firm_year_industry_fe_results.csv", index=False)

# Firm-quarter (0_365 only)
fq_results = []
for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_quarter_post_rdd_data_{emp}.parquet")
    d = d[d["window"]=="0_365"]
    for bw_label, bw_fn in BANDWIDTHS:
        for fe_spec, fe_cols in FE_VARIANTS:
            for oc in OUTCOMES:
                res = run_ols(d, oc, bw_fn, "linear", fe_spec, fe_cols)
                if res:
                    res.update({"employee_sample":emp,"bw_label":bw_label})
                    fq_results.append(res)
    print(f"  {emp}: {len(fq_results)} FQ specs")

df_fq = pd.DataFrame(fq_results)
df_fq.to_csv(OUT / "firm_quarter_industry_fe_results.csv", index=False)

# ── REPORT ──
print("\n=== WLB: Firm-Year FE Comparison (current, global) ===")
for fe_spec in ["none","year","sic2","ff48","year_sic2","year_ff48"]:
    r = df_fy[(df_fy["outcome"]=="wlb")&(df_fy["employee_sample"]=="current")&
              (df_fy["bw_label"]=="global")&(df_fy["fe_spec"]==fe_spec)]
    if len(r)>0:
        r=r.iloc[0]
        sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"  {fe_spec:12s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig}")

print("\n=== All Outcomes: FY current global, no FE vs FF48 FE ===")
for oc in OUTCOMES:
    none_r = df_fy[(df_fy["outcome"]==oc)&(df_fy["employee_sample"]=="current")&
                   (df_fy["bw_label"]=="global")&(df_fy["fe_spec"]=="none")]
    ff48_r = df_fy[(df_fy["outcome"]==oc)&(df_fy["employee_sample"]=="current")&
                   (df_fy["bw_label"]=="global")&(df_fy["fe_spec"]=="ff48")]
    t_none = none_r["estimate"].values[0] if len(none_r)>0 else np.nan
    t_ff48 = ff48_r["estimate"].values[0] if len(ff48_r)>0 else np.nan
    p_none = none_r["p_value"].values[0] if len(none_r)>0 else np.nan
    p_ff48 = ff48_r["p_value"].values[0] if len(ff48_r)>0 else np.nan
    print(f"  {OC_LABELS.get(oc,oc):20s}: noFE tau={t_none:+.4f} p={p_none:.3f} | FF48 tau={t_ff48:+.4f} p={p_ff48:.3f}")
print("Done.")
