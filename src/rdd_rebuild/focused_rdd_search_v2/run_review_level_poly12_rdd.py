#!/usr/bin/env python
"""A. Review-level poly1+poly2 RDD with four screening rules, gvkey-clustered SE."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm_api
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v2"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = {"overall_rating":"Overall","career_opp":"Career","comp_benefit":"Comp","senior_mgmt":"Senior","wlb":"WLB","culture":"Culture"}
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]
WINDOWS = [365,180,90]
SCREENS = [("pre>=1_post>=1",1,1,None),("pre>=3_post>=3",3,3,None),("pre>=5_post>=5",5,5,None),("total>=10",None,None,10)]
EMP_FILTERS = ["current","all"]

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

print("Loading...")
df = pd.read_parquet(SAMPLE)

def run_one(oc, emp, bw_val, wd, min_pre, min_post, min_total, poly_order, use_fe):
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if wd < 365: sub = sub[sub[f"within_{wd}"]]

    # Screening
    grp = sub.groupby("election_id")["post"]
    stats_df = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if min_total is not None:
        stats_df["n_total"] = stats_df["n_post"]+stats_df["n_pre"]
        valid = stats_df[stats_df["n_total"]>=min_total]["election_id"]
    else:
        valid = stats_df[(stats_df["n_post"]>=min_post)&(stats_df["n_pre"]>=min_pre)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]

    if len(sub)<100 or sub["election_id"].nunique()<15 or sub["gvkey"].nunique()<10: return None

    mu, sd = sub[oc].mean(), sub[oc].std()
    if sd==0: return None
    y = (sub[oc].values-mu)/sd
    post=sub["post"].values.astype(float); win=sub["win"].values.astype(float)
    margin=sub["margin"].values; eid=sub["election_id"].values
    year=sub["review_year"].values; gv=sub["gvkey"].values

    pw=post*win; pm=post*margin; pwm=post*win*margin

    if use_fe:
        # Build X based on poly order
        if poly_order==1:
            X_raw = np.column_stack([post, pw, pm, pwm])
        else:
            m2 = margin**2; pm2 = post*m2; pwm2 = post*win*m2
            X_raw = np.column_stack([post, pw, pm, pwm, pm2, pwm2])
        # Year dummies
        yu = np.unique(year)
        if len(yu)>1:
            yd = np.column_stack([(year==yv).astype(float) for yv in yu[1:]])
            X_raw = np.column_stack([X_raw, yd])

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
            tau, se_tau = beta[1], se[1]
            pv = 2*stats.t.sf(abs(tau/se_tau), df_resid) if se_tau>0 else np.nan
            n_win = sub[sub["win"]==1]["election_id"].nunique()
            return {"estimate":tau,"se":se_tau,"t_stat":tau/se_tau,"p_value":pv,
                    "n":n,"n_events":len(eid_u),"n_gvkeys":sub["gvkey"].nunique(),
                    "n_win":n_win,"n_loss":len(eid_u)-n_win,"mu":mu,"sd":sd}
        except: return None
    else:
        # No FE
        X_list = [post, win, pw, margin, win*margin, pm, pwm]
        if poly_order==2:
            m2=margin**2; win_m2=win*m2; post_m2=post*m2; post_win_m2=post*win*m2
            X_list += [m2, win_m2, post_m2, post_win_m2]
        X = sm_api.add_constant(np.column_stack(X_list))
        n,k = X.shape
        try:
            mod = sm_api.OLS(y, X).fit()
            se = cluster_se(X, y, mod.params, gv, n-k)
            if se is None: se = np.sqrt(np.diag(mod.cov_params()))
            # Win x Post is at index 3 (const, post, win, win*post, ...)
            win_post_idx = 3
            tau, se_tau = mod.params[win_post_idx], se[win_post_idx]
            pv = 2*stats.t.sf(abs(tau/se_tau), n-k) if se_tau>0 else np.nan
            n_win = sub[sub["win"]==1]["election_id"].nunique()
            return {"estimate":tau,"se":se_tau,"t_stat":tau/se_tau,"p_value":pv,
                    "n":n,"n_events":sub["election_id"].nunique(),"n_gvkeys":sub["gvkey"].nunique(),
                    "n_win":n_win,"n_loss":sub["election_id"].nunique()-n_win,"mu":mu,"sd":sd}
        except: return None

results = []
total = len(OUTCOMES)*len(EMP_FILTERS)*len(BANDWIDTHS)*len(WINDOWS)*len(SCREENS)*2*2
n_done = 0
for oc in OUTCOMES:
    for emp in EMP_FILTERS:
        for bw_label, bw_val in BANDWIDTHS:
            for wd in WINDOWS:
                for th_label, min_pre, min_post, min_total in SCREENS:
                    for poly in [1,2]:
                        for use_fe in [True, False]:
                            res = run_one(oc, emp, bw_val, wd, min_pre, min_post, min_total, poly, use_fe)
                            if res:
                                results.append({
                                    "outcome":oc,"employee_sample":emp,"window_days":wd,
                                    "bandwidth_label":bw_label,"bandwidth_value":bw_val,
                                    "screening_rule":th_label,"polynomial_order":poly,
                                    "fixed_effects":"election_FE+year_FE" if use_fe else "none",
                                    "cluster":"gvkey","coefficient_of_interest":"Win x Post",
                                    "n_reviews":res["n"],"n_events":res["n_events"],
                                    "n_gvkeys":res["n_gvkeys"],"n_win_events":res["n_win"],
                                    "n_loss_events":res["n_loss"],
                                    "estimate":res["estimate"],"standard_error":res["se"],
                                    "t_stat":res["t_stat"],"p_value":res["p_value"],
                                    "mean_depvar":res["mu"],"sd_depvar":res["sd"]})
                            n_done += 1
        pct = n_done/max(total,1)*100
        print(f"\r  {oc}: {pct:.0f}%", end="", flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "review_level_poly12_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Quick check
print("\n--- Current, +/-365d, global, election FE, pre>=1 ---")
for poly in [1,2]:
    m = df_r[(df_r["employee_sample"]=="current")&(df_r["window_days"]==365)&(df_r["bandwidth_label"]=="global")&(df_r["screening_rule"]=="pre>=1_post>=1")&(df_r["fixed_effects"]=="election_FE+year_FE")&(df_r["polynomial_order"]==poly)]
    print(f"  poly={poly}:")
    for _,r in m.iterrows():
        sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"    {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig}")
