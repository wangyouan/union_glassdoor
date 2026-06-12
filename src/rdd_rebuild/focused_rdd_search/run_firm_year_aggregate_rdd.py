#!/usr/bin/env python
"""B. Firm-year aggregated RDD with gvkey-clustered SE."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = {"overall_rating":"Overall","career_opp":"Career","comp_benefit":"Comp","senior_mgmt":"Senior","wlb":"WLB","culture":"Culture"}
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]
WINDOWS = [365,180]

print("Loading and aggregating to firm-year...")
df = pd.read_parquet(SAMPLE)

def cluster_se(X, y, beta, cluster_ids, df_resid):
    resid = y - X @ beta
    n, k = X.shape
    clusters = np.unique(cluster_ids)
    G = len(clusters)
    if G < 15: return None
    meat = np.zeros((k, k))
    for g in clusters:
        mask = cluster_ids == g
        meat += (X[mask].T @ resid[mask])[:, None] @ (X[mask].T @ resid[mask])[None, :]
    bread = np.linalg.inv(X.T @ X)
    vcov = bread @ meat @ bread
    vcov *= (G/(G-1))*((n-1)/(n-k))
    return np.sqrt(np.diag(vcov))

# Build firm-year aggregate: for each election, compute delta_y = post_mean - pre_mean
# This is equivalent to event-level RDD but with the option of firm FE
def build_event_delta(oc, emp, bw_val, win_days):
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if win_days < 365: sub = sub[sub[f"within_{win_days}"]]

    rows = []
    for eid, g in sub.groupby("election_id"):
        pre = g[g["days_to_election"]<0]; post = g[g["days_to_election"]>=0]
        if len(pre)<1 or len(post)<1: continue
        rows.append({"election_id":eid,"gvkey":g["gvkey"].iloc[0],"election_year":g["election_year_elec"].iloc[0] if "election_year_elec" in g.columns else g["review_year"].iloc[0],
                     "margin":g["margin"].iloc[0],"win":g["win"].iloc[0],
                     "pre_mean":pre[oc].mean(),"post_mean":post[oc].mean(),
                     "n_pre":len(pre),"n_post":len(post)})
    ev = pd.DataFrame(rows)
    if len(ev) < 15: return None
    ev["delta"] = ev["post_mean"] - ev["pre_mean"]
    return ev

# For each firm-year, keep only earliest election
def dedup_firm_year(ev):
    ev = ev.copy()
    ev["gvkey_year"] = ev["gvkey"].astype(str) + "_" + ev["election_year"].astype(str)
    ev["multi_flag"] = ev.groupby("gvkey_year")["election_id"].transform("nunique") > 1
    ev = ev.sort_values(["gvkey_year","election_id"]).groupby("gvkey_year").first().reset_index()
    return ev

def run_delta_rdd(ev, poly_order=1, use_fe=False):
    y = ev["delta"].values; win = ev["win"].values.astype(float); m = ev["margin"].values
    mu_d, sd_d = y.mean(), y.std()
    if sd_d == 0: return None
    y_sd = (y - mu_d) / sd_d
    gvkey_ids = ev["gvkey"].values

    if poly_order == 1:
        X = np.column_stack([np.ones_like(win), win, m, win*m])
        var_names = ["Constant","Win","Margin","Win x Margin"]
    else:
        X = np.column_stack([np.ones_like(win), win, m, m**2, win*m, win*(m**2)])
        var_names = ["Constant","Win","Margin","Margin^2","Win x Margin","Win x Margin^2"]

    if use_fe:
        gvkey_u = np.unique(gvkey_ids)
        gvkey_dummies = np.column_stack([(gvkey_ids==g).astype(float) for g in gvkey_u[1:]])
        X = np.column_stack([X, gvkey_dummies])

    n, k = X.shape
    try:
        beta = np.linalg.lstsq(X, y_sd, rcond=None)[0]
        df_resid = n - k
        se_all = cluster_se(X, y_sd, beta, gvkey_ids, df_resid)
        if se_all is None:
            resid = y_sd - X@beta
            se_all = np.sqrt(np.diag(np.linalg.inv(X.T@X) * (resid@resid)/df_resid))
        tau = beta[1]; se_tau = se_all[1]
        pval = 2*stats.t.sf(abs(tau/se_tau), df_resid) if se_tau>0 else np.nan
        return {"estimate":tau,"standard_error":se_tau,"t_stat":tau/se_tau,"p_value":pval,
                "n":n,"n_events":n,"n_gvkeys":len(np.unique(gvkey_ids)),
                "n_win":int(win.sum()),"n_loss":n-int(win.sum()),
                "mu_delta":mu_d,"sd_delta":sd_d,
                "mean_delta_loss":float(y[win==0].mean()),"mean_delta_win":float(y[win==1].mean())}
    except: return None

results, ev_data = [], []
for oc, oc_label in OUTCOMES.items():
    for emp in ["current","all"]:
        for bw_label, bw_val in BANDWIDTHS:
            for wd in WINDOWS:
                ev = build_event_delta(oc, emp, bw_val, wd)
                if ev is None: continue
                ev_dedup = dedup_firm_year(ev)
                if len(ev_dedup) < 15: continue

                for poly in [1, 2]:
                    for use_fe in [False, True]:
                        for min_comments in [None, 5, 10]:
                            ev_use = ev_dedup.copy()
                            if min_comments is not None and emp == "all":
                                ev_use = ev_use[(ev_use["n_pre"]+ev_use["n_post"]) >= min_comments]
                                if len(ev_use) < 15: continue
                            res = run_delta_rdd(ev_use, poly, use_fe)
                            if res:
                                results.append({"estimator":"firm_year_aggregate","outcome":oc,"outcome_label":oc_label,
                                    "employee_sample":emp,"window_days":wd,"bandwidth_label":bw_label,"bandwidth_value":bw_val,
                                    "min_comment_rule":str(min_comments) if min_comments else "none",
                                    "specification_name":f"delta_rdd_poly{poly}" + ("_firmFE" if use_fe else ""),
                                    "fixed_effects":"firm_FE" if use_fe else "none",
                                    "polynomial_order":poly,"cluster":"gvkey",
                                    "n_observations":res["n"],"n_events":res["n_events"],
                                    "n_gvkeys":res["n_gvkeys"],"n_win_events":res["n_win"],"n_loss_events":res["n_loss"],
                                    "coefficient_of_interest":"Win","estimate":res["estimate"],
                                    "standard_error":res["standard_error"],"t_stat":res["t_stat"],"p_value":res["p_value"],
                                    "mean_depvar":res["mu_delta"],"sd_depvar":res["sd_delta"]})
    print(f"  {oc}: {len([r for r in results if r['outcome']==oc])} specs")

df_fy = pd.DataFrame(results)
df_fy.to_csv(OUT / "firm_year_aggregate_rdd_results.csv", index=False)
print(f"Saved: firm_year_aggregate_rdd_results.csv ({len(df_fy)})")

print("\n--- Firm-year: current, +/-365d, global, linear, no FE ---")
m = df_fy[(df_fy["employee_sample"]=="current")&(df_fy["window_days"]==365)&(df_fy["bandwidth_label"]=="global")&(df_fy["polynomial_order"]==1)&(df_fy["fixed_effects"]=="none")&(df_fy["min_comment_rule"]=="none")]
for _, r in m.iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: Win={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig} N={int(r['n_observations'])}")
