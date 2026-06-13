#!/usr/bin/env python
"""B. Matched aggregate RDD (firm-quarter + firm-year) using same settings as review-level."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v2"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = {"overall_rating":"Overall","career_opp":"Career","comp_benefit":"Comp","senior_mgmt":"Senior","wlb":"WLB","culture":"Culture"}
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]
WINDOWS = [365,180]
SCREENS = [("pre>=1_post>=1",1,1,None),("pre>=3_post>=3",3,3,None)]

def cluster_se(X, y, beta, cids, df_r):
    resid = y - X@beta; n,k = X.shape
    uq = np.unique(cids); G = len(uq)
    if G < 15: return None
    meat = np.zeros((k,k))
    for g in uq:
        m = cids==g; Xg=X[m]; rg=resid[m]
        meat += (Xg.T@rg)[:,None] @ (Xg.T@rg)[None,:]
    v = np.linalg.inv(X.T@X) @ meat @ np.linalg.inv(X.T@X)
    v *= (G/(G-1))*((n-1)/(n-k))
    return np.sqrt(np.diag(v))

df = pd.read_parquet(SAMPLE)

# ── Firm-quarter: same election-FE DiD as review-level ──
def run_fq(oc, emp, bw_val, wd, min_pre, min_post, poly):
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if wd < 365: sub = sub[sub[f"within_{wd}"]]

    # Apply same screening at election level
    grp = sub.groupby("election_id")["post"].agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    valid = grp[(grp["n_post"]>=min_post)&(grp["n_pre"]>=min_pre)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]

    sub["rel_quarter"] = np.floor(sub["days_to_election"]/90).astype(int).clip(-4,4)
    sub["is_post"] = (sub["days_to_election"]>=0).astype(int)
    agg = sub.groupby(["election_id","gvkey","rel_quarter","is_post","win","margin","review_year"]).agg(
        mean_rating=(oc,"mean"), n_reviews=(oc,"count")).reset_index()
    if len(agg) < 50: return None

    mu,sd = agg["mean_rating"].mean(), agg["mean_rating"].std()
    if sd==0: return None
    y = (agg["mean_rating"].values-mu)/sd
    post=agg["is_post"].values.astype(float); win=agg["win"].values.astype(float)
    margin=agg["margin"].values; eid=agg["election_id"].values; gv=agg["gvkey"].values
    year=agg["review_year"].values
    pw=post*win; pm=post*margin; pwm=post*win*margin

    X_list = [post, pw, pm, pwm]
    if poly==2: X_list += [post*(margin**2), post*win*(margin**2)]
    yu = np.unique(year)
    if len(yu)>1: X_list.append(np.column_stack([(year==yv).astype(float) for yv in yu[1:]]))
    X_raw = np.column_stack(X_list)

    eid_u, eid_inv, eid_cnt = np.unique(eid, return_inverse=True, return_counts=True)
    ym = np.bincount(eid_inv, weights=y)/eid_cnt
    Xm = np.column_stack([np.bincount(eid_inv, weights=X_raw[:,j])/eid_cnt for j in range(X_raw.shape[1])])
    y_dm, X_dm = y-ym[eid_inv], X_raw-Xm[eid_inv]
    n,k = X_dm.shape; df_resid = n-k-len(eid_u)
    if df_resid<10: return None
    try:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        se = cluster_se(X_dm, y_dm, beta, gv, df_resid)
        if se is None: return None
        tau,se_t = beta[1],se[1]
        return {"estimate":tau,"se":se_t,"p_value":2*stats.t.sf(abs(tau/se_t),df_resid) if se_t>0 else np.nan,
                "n_obs":n,"n_events":len(eid_u),"n_gvkeys":len(np.unique(gv))}
    except: return None

# ── Firm-year delta RDD ──
def run_fy(oc, emp, bw_val, wd, min_pre, min_post, poly):
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if wd < 365: sub = sub[sub[f"within_{wd}"]]

    grp = sub.groupby("election_id")["post"].agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    valid = grp[(grp["n_post"]>=min_post)&(grp["n_pre"]>=min_pre)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]

    rows = []
    for eid_e, g in sub.groupby("election_id"):
        pre=g[g["days_to_election"]<0]; post_e=g[g["days_to_election"]>=0]
        if len(pre)<min_pre or len(post_e)<min_post: continue
        rows.append({"election_id":eid_e,"gvkey":g["gvkey"].iloc[0],"margin":g["margin"].iloc[0],
                     "win":g["win"].iloc[0],"pre_mean":pre[oc].mean(),"post_mean":post_e[oc].mean()})
    ev = pd.DataFrame(rows)
    if len(ev)<15: return None
    ev["delta"] = ev["post_mean"]-ev["pre_mean"]
    mu_d,sd_d = ev["delta"].mean(), ev["delta"].std()
    if sd_d==0: return None
    y_sd = (ev["delta"].values-mu_d)/sd_d
    win=ev["win"].values.astype(float); m=ev["margin"].values; gv=ev["gvkey"].values
    X = np.column_stack([np.ones_like(win), win, m, win*m])
    if poly==2: X = np.column_stack([X, m**2, win*(m**2)])
    n,k = X.shape
    try:
        beta = np.linalg.lstsq(X, y_sd, rcond=None)[0]
        se = cluster_se(X, y_sd, beta, gv, n-k)
        if se is None: return None
        tau,se_t = beta[1],se[1]
        return {"estimate":tau,"se":se_t,"p_value":2*stats.t.sf(abs(tau/se_t),n-k) if se_t>0 else np.nan,
                "n_obs":n,"n_events":n,"n_gvkeys":len(np.unique(gv))}
    except: return None

# Run
fq_res, fy_res = [], []
n_total = len(OUTCOMES)*2*len(BANDWIDTHS)*len(WINDOWS)*2*2
n_done = 0
for oc in OUTCOMES:
    for emp in ["current","all"]:
        for bw_label, bw_val in BANDWIDTHS:
            for wd in WINDOWS:
                for th_label, min_pre, min_post, _ in SCREENS:
                    for poly in [1,2]:
                        r = run_fq(oc, emp, bw_val, wd, min_pre, min_post, poly)
                        if r:
                            fq_res.append({"outcome":oc,"employee_sample":emp,"window_days":wd,
                                "bandwidth_label":bw_label,"screening_rule":th_label,"polynomial_order":poly,
                                **r})
                        r = run_fy(oc, emp, bw_val, wd, min_pre, min_post, poly)
                        if r:
                            fy_res.append({"outcome":oc,"employee_sample":emp,"window_days":wd,
                                "bandwidth_label":bw_label,"screening_rule":th_label,"polynomial_order":poly,
                                **r})
                        n_done += 1
        pct = n_done/max(n_total,1)*100
        print(f"\r  {oc}: {pct:.0f}%", end="", flush=True)

pd.DataFrame(fq_res).to_csv(OUT / "matched_firm_quarter_results.csv", index=False)
pd.DataFrame(fy_res).to_csv(OUT / "matched_firm_year_results.csv", index=False)
print(f"\nSaved: FQ={len(fq_res)}, FY={len(fy_res)}")
