#!/usr/bin/env python
"""Script 2: Firm-quarter post-outcome RDD."""

import pandas as pd, numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/post_outcome_aggregate_rdd"
OUT.mkdir(parents=True, exist_ok=True)
OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]

print("Loading...")
df = pd.read_parquet(SAMPLE)
df["review_date"] = pd.to_datetime(df["review_date"])
df["election_date"] = df["review_date"] - pd.to_timedelta(df["days_to_election"], unit="D")
df["days_from_election"] = df["days_to_election"]

def build_quarter_data(emp_sample):
    d = df.copy()
    if emp_sample == "current": d = d[d["is_current_employee"]==1]
    rows = []
    for oc in OUTCOMES:
        sub = d[d[oc].notna()]
        for eid, g in sub.groupby("election_id"):
            pre_mask = (g["days_from_election"]>=-365)&(g["days_from_election"]<0)
            pre_mean_365 = g.loc[pre_mask,oc].mean() if pre_mask.sum()>=1 else np.nan
            if pd.isna(pre_mean_365): continue

            # Cumulative windows
            for wl, wh, wlabel in [(0,90,"0_90"),(0,180,"0_180"),(0,365,"0_365")]:
                pm = (g["days_from_election"]>=wl)&(g["days_from_election"]<=wh)
                if pm.sum() < 1: continue
                rows.append({
                    "election_id":eid,"gvkey":str(g["gvkey"].iloc[0]),
                    "election_year":int(g["election_date"].iloc[0].year),
                    "win":int(g["win"].iloc[0]),"margin":g["margin"].iloc[0],
                    "abs_margin":abs(g["margin"].iloc[0]),
                    "outcome":oc,"pre_mean_365":pre_mean_365,
                    "post_mean":g.loc[pm,oc].mean(),"window":wlabel,
                    "n_post":int(pm.sum()),"n_pre":int(pre_mask.sum()),
                })
    return pd.DataFrame(rows)

def run_cum_ols(data, oc, bw_fn, window_label, poly, fe_spec, fe_cols):
    d = bw_fn(data) if bw_fn is not None else data.copy()
    d = d[(d["outcome"]==oc)&(d["window"]==window_label)].dropna(subset=["post_mean","pre_mean_365","win","margin"])
    if len(d) < 30: return None
    d["win_margin"] = d["win"]*d["margin"]
    formula = "post_mean ~ win + margin + win_margin + pre_mean_365"
    if poly == "quadratic":
        d["margin2"]=d["margin"]**2; d["win_margin2"]=d["win"]*d["margin2"]
        formula += " + margin2 + win_margin2"
    for col in fe_cols:
        if col in d.columns: formula += f" + C({col})"
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["gvkey"]})
        return {"outcome":oc,"window":window_label,"poly":poly,"fe_spec":fe_spec,
            "estimate":m.params.get("win",np.nan),"se":m.bse.get("win",np.nan),
            "p_value":m.pvalues.get("win",np.nan),"n_events":len(d),"n_gvkeys":d["gvkey"].nunique()}
    except: return None

# Build
for emp in ["current","all"]:
    dfq = build_quarter_data(emp)
    dfq.to_parquet(OUT / f"firm_quarter_post_rdd_data_{emp}.parquet", index=False)
    print(f"  {emp} quarter data: {len(dfq)} rows")

# Run cumulative regressions
print("\nRunning firm-quarter cumulative OLS...")
bw_funcs = {"global":None,"m20":lambda d: d[d["abs_margin"]<=0.20],"m10":lambda d: d[d["abs_margin"]<=0.10]}
results = []
for emp in ["current","all"]:
    dfq = pd.read_parquet(OUT / f"firm_quarter_post_rdd_data_{emp}.parquet")
    for bw_label, bw_fn in bw_funcs.items():
        for wlabel in ["0_90","0_180","0_365"]:
            for poly in ["linear","quadratic"]:
                for fe_spec, fe_cols in [("none",[]),("year",["election_year"])]:
                    for oc in OUTCOMES:
                        res = run_cum_ols(dfq, oc, bw_fn, wlabel, poly, fe_spec, fe_cols)
                        if res:
                            res.update({"employee_sample":emp,"bw_label":bw_label})
                            results.append(res)
        n = len(results)
        print(f"\r  {emp} {bw_label}: {n} results", end="", flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "firm_quarter_cumulative_rdd_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Main table
print("\n=== Firm-Quarter Main (current, 0_365, global, linear, no FE) ===")
m = df_r[(df_r["employee_sample"]=="current")&(df_r["bw_label"]=="global")&
         (df_r["window"]=="0_365")&(df_r["poly"]=="linear")&(df_r["fe_spec"]=="none")]
for _,r in m.sort_values("outcome").iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")
print("Done.")
