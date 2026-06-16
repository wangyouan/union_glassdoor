#!/usr/bin/env python
"""v10d: Delta RDD with Version B (greedy >365d) as primary."""

import pandas as pd, numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/post_outcome_aggregate_rdd")
OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
OC_LABELS = {"overall_rating":"Overall","career_opp":"Career Opp","comp_benefit":"Comp & Benefits","senior_mgmt":"Senior Mgmt","wlb":"WLB","culture":"Culture"}
FE_VARIANTS = [("none",[]),("year",["election_year"]),("sic2",["sic2"]),("ff48",["ff48"]),
               ("year_sic2",["election_year","sic2"]),("year_ff48",["election_year","ff48"])]
BWS = [("global",None),("m20",0.20),("m10",0.10),("m05",0.05)]

def assign_version_B(elec_df):
    """Greedy: keep elections spaced >365d apart, starting from first per gvkey."""
    keep = set()
    for gv, grp in elec_df.groupby("gvkey"):
        grp = grp.sort_values("election_date")
        anchor = None
        for _, row in grp.iterrows():
            if anchor is None or (row["election_date"] - anchor).days > 365:
                keep.add(row["election_id"])
                anchor = row["election_date"]
    return keep

def run_ols(data, outcome, bw_val, poly, fe_spec, fe_cols):
    d = data.copy()
    if bw_val is not None: d = d[d["abs_margin"]<=bw_val]
    d = d[d["outcome"]==outcome].dropna(subset=["delta","win","margin"])
    if len(d) < 30: return None
    d["win_margin"] = d["win"]*d["margin"]
    formula = "delta ~ win + margin + win_margin"
    if poly == "quadratic":
        d["margin2"]=d["margin"]**2; d["win_margin2"]=d["win"]*d["margin2"]
        formula += " + margin2 + win_margin2"
    for col in fe_cols:
        if col in d.columns and d[col].notna().sum() > 5:
            formula += f" + C({col})"
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["gvkey"]})
        return {"outcome":outcome,"poly":poly,"fe_spec":fe_spec,"estimate":m.params.get("win",np.nan),
                "se":m.bse.get("win",np.nan),"p_value":m.pvalues.get("win",np.nan),
                "n_events":len(d),"n_gvkeys":d["gvkey"].nunique()}
    except: return None

all_results = []
for emp in ["current","all"]:
    d = pd.read_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet")
    d["delta"] = d["post_mean"] - d["pre_mean"]

    # Version B
    elec = d[["election_id","gvkey","election_date"]].drop_duplicates()
    b_ids = assign_version_B(elec)
    # Version C
    c_ids = set(elec.groupby("gvkey")["election_date"].idxmin().apply(
        lambda i: elec.loc[i,"election_id"]))

    versions = {"A": None, "B": b_ids, "C": c_ids}
    filters = {"unrestricted": None}
    if emp == "all":
        filters.update({"n5": lambda dd: dd[(dd["n_pre"]>=5)&(dd["n_post"]>=5)],
                        "n10": lambda dd: dd[(dd["n_pre"]>=10)&(dd["n_post"]>=10)]})

    for v_label, v_ids in versions.items():
        dv = d if v_ids is None else d[d["election_id"].isin(v_ids)]
        for f_label, f_fn in filters.items():
            df_use = f_fn(dv) if f_fn else dv
            for bw_label, bw_val in BWS:
                for poly in ["linear","quadratic"]:
                    for fe_spec, fe_cols in FE_VARIANTS:
                        for oc in OUTCOMES:
                            res = run_ols(df_use, oc, bw_val, poly, fe_spec, fe_cols)
                            if res:
                                res.update({"employee_sample":emp,"filter":f_label,
                                    "bw_label":bw_label,"multi_version":v_label,"dv_type":"delta"})
                                all_results.append(res)
        n_done = sum(1 for r in all_results if r["multi_version"]==v_label and r["employee_sample"]==emp)
        print(f"  {emp} {v_label}: {n_done} specs")

df_all = pd.DataFrame(all_results)
df_all.to_csv(OUT / "delta_vABC_comparison.csv", index=False)
df_vB = df_all[df_all["multi_version"]=="B"]
df_vB.to_csv(OUT / "delta_vB_main_results.csv", index=False)
print(f"Saved: {len(df_all)} total, {len(df_vB)} Version B")

# ── rdrobust via R ──
import subprocess
rscript = """
library(rdrobust); library(nanoparquet); library(dplyr); library(readr)
setwd('/data/disk4/workspace/projects/union_glassdoor')
OUTDIR <- 'outputs/rdd_rebuild/post_outcome_aggregate_rdd'
OUTCOMES <- c('overall_rating','career_opp','comp_benefit','senior_mgmt','wlb','culture')
results <- list()
for(emp in c('current','all')){
  d <- nanoparquet::read_parquet(file.path(OUTDIR, sprintf('firm_year_post_rdd_data_%s.parquet',emp)))
  d$delta <- d$post_mean - d$pre_mean
  # Version B
  elec <- d %>% distinct(election_id, gvkey, election_date) %>% arrange(gvkey, election_date)
  keep <- data.frame(eid=integer(),stringsAsFactors=FALSE)
  for(gv in unique(elec$gvkey)){
    grp <- elec %>% filter(gvkey==gv) %>% arrange(election_date)
    anchor <- NULL
    for(i in 1:nrow(grp)){
      if(is.null(anchor) || as.numeric(difftime(grp$election_date[i],anchor,units='days'))>365){
        keep <- rbind(keep, data.frame(eid=grp$election_id[i]))
        anchor <- grp$election_date[i]
      }}}
  d <- d %>% filter(election_id %in% keep$eid)
  for(oc in OUTCOMES){
    sub <- d %>% filter(outcome==oc, !is.na(delta), !is.na(margin))
    if(nrow(sub)<50) next
    rdr <- tryCatch(rdrobust(y=sub$delta,x=sub$margin,c=0,p=1,kernel='triangular'),
                    error=function(e)NULL)
    if(!is.null(rdr)){
      results[[length(results)+1]] <- tibble(outcome=oc,employee_sample=emp,
        h_left=rdr$bws[1,1],h_right=rdr$bws[1,2],N_left=rdr$N[1],N_right=rdr$N[2],
        estimate=rdr$coef[3],se=rdr$se[3],p_value=rdr$pv[3])
    }}}
write_csv(bind_rows(results), file.path(OUTDIR, 'delta_rdrobust_vB_results.csv'))
cat(sprintf('rdrobust: %d results\\n', length(results)))
"""
with open("/tmp/rdrobust_vB.R","w") as f: f.write(rscript)
subprocess.run(["Rscript","/tmp/rdrobust_vB.R"], check=True)

# ── REPORT ──
print("\n=== MAIN TABLE: Version B, current, unrestricted, global, linear, no FE ===")
m = df_all[(df_all["multi_version"]=="B")&(df_all["employee_sample"]=="current")&
           (df_all["bw_label"]=="global")&(df_all["poly"]=="linear")&(df_all["fe_spec"]=="none")&
           (df_all["filter"]=="unrestricted")]
for _,r in m.sort_values("outcome").iterrows():
    sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {OC_LABELS.get(r['outcome'],r['outcome']):20s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")

print("\n=== WLB Bandwidth (vB, current, linear, no FE) ===")
for bw in ["global","m20","m10","m05"]:
    r = df_all[(df_all["multi_version"]=="B")&(df_all["employee_sample"]=="current")&
               (df_all["bw_label"]==bw)&(df_all["poly"]=="linear")&(df_all["fe_spec"]=="none")&
               (df_all["outcome"]=="wlb")]
    if len(r)>0:
        r=r.iloc[0]; sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"  {bw:8s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")

print("\n=== A vs B vs C: WLB, current, linear, no FE ===")
for v in ["A","B","C"]:
    vals=[]
    for bw in ["global","m20","m10","m05"]:
        r = df_all[(df_all["multi_version"]==v)&(df_all["employee_sample"]=="current")&
                   (df_all["bw_label"]==bw)&(df_all["poly"]=="linear")&(df_all["fe_spec"]=="none")&
                   (df_all["outcome"]=="wlb")]
        tau=r["estimate"].values[0] if len(r)>0 else np.nan
        pv=r["p_value"].values[0] if len(r)>0 else np.nan
        vals.append(f"{tau:+.3f}(p={pv:.3f})")
    print(f"  {v}: {' | '.join(vals)}")

print("\n=== WLB FE Robustness (vB, current, m10) ===")
for fe_spec in ["none","year","sic2","ff48","year_sic2","year_ff48"]:
    r = df_all[(df_all["multi_version"]=="B")&(df_all["employee_sample"]=="current")&
               (df_all["bw_label"]=="m10")&(df_all["poly"]=="linear")&(df_all["fe_spec"]==fe_spec)&
               (df_all["outcome"]=="wlb")]
    if len(r)>0:
        r=r.iloc[0]; sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"  {fe_spec:12s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig}")
print("Done.")
