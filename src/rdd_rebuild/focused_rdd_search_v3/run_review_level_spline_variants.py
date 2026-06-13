#!/usr/bin/env python
"""A. Run four margin-control variants: poly1/poly2 x non-spline/spline."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm_api
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v3"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
SCREENS = [("pre>=1_post>=1",1,1,None),("pre>=3_post>=3",3,3,None),("pre>=5_post>=5",5,5,None),("total>=10",None,None,10)]
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]
WINDOWS = [365,180,90]
VARIANTS = [
    ("poly1_non_spline", 1, False),
    ("poly1_spline", 1, True),
    ("poly2_non_spline", 2, False),
    ("poly2_spline", 2, True),
]

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
print(f"Loaded {len(df):,} reviews")

def run_one(oc, emp, bw_val, wd, min_pre, min_post, min_total, poly_order, spline):
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if wd < 365: sub = sub[sub[f"within_{wd}"]]
    # Screening
    grp = sub.groupby("election_id")["post"]
    st = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if min_total is not None:
        st["n_total"] = st["n_post"]+st["n_pre"]
        valid = st[st["n_total"]>=min_total]["election_id"]
    else:
        valid = st[(st["n_post"]>=min_post)&(st["n_pre"]>=min_pre)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]
    if len(sub)<100 or sub["election_id"].nunique()<15 or sub["gvkey"].nunique()<10: return None

    mu, sd = sub[oc].mean(), sub[oc].std()
    if sd==0: return None
    y = (sub[oc].values-mu)/sd
    post=sub["post"].values.astype(float); win=sub["win"].values.astype(float)
    margin=sub["margin"].values; eid=sub["election_id"].values; gv=sub["gvkey"].values; year=sub["review_year"].values
    pw=post*win; pm=post*margin

    # Build X based on variant
    X_list = [post, pw]
    if poly_order==1:
        if spline: X_list += [pm, post*win*margin]   # spline: separate slopes
        else: X_list += [pm]                            # non-spline: common slope
    else:  # poly2
        if spline:
            X_list += [pm, post*win*margin, post*(margin**2), post*win*(margin**2)]
        else:
            X_list += [pm, post*(margin**2)]

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
        tau, se_tau = beta[1], se[1]
        pv = 2*stats.t.sf(abs(tau/se_tau), df_resid) if se_tau>0 else np.nan
        n_win = sub[sub["win"]==1]["election_id"].nunique()
        return {"estimate":tau,"se":se_tau,"t_stat":tau/se_tau,"p_value":pv,
                "n":n,"n_events":len(eid_u),"n_gvkeys":sub["gvkey"].nunique(),
                "n_win":n_win,"n_loss":len(eid_u)-n_win,"mu":mu,"sd":sd}
    except: return None

results = []
total = len(OUTCOMES)*2*len(BANDWIDTHS)*len(WINDOWS)*len(SCREENS)*len(VARIANTS)
n_done = 0
for oc in OUTCOMES:
    for emp in ["current","all"]:
        for bw_label, bw_val in BANDWIDTHS:
            for wd in WINDOWS:
                for th_label, min_pre, min_post, min_total in SCREENS:
                    for variant_name, poly_order, spline in VARIANTS:
                        res = run_one(oc, emp, bw_val, wd, min_pre, min_post, min_total, poly_order, spline)
                        if res:
                            results.append({"outcome":oc,"employee_sample":emp,"window_days":wd,
                                "bandwidth_label":bw_label,"bandwidth_value":bw_val,
                                "screening_rule":th_label,"variant":variant_name,
                                "polynomial_order":poly_order,"spline":spline,
                                "coefficient_of_interest":"Win x Post",
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
df_r.to_csv(OUT / "review_level_spline_variants_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Quick summary
print("\n--- Current, global, +/-365d, pre>=1 ---")
for vname, poly, spl in VARIANTS:
    m = df_r[(df_r["employee_sample"]=="current")&(df_r["window_days"]==365)&(df_r["bandwidth_label"]=="global")&(df_r["screening_rule"]=="pre>=1_post>=1")&(df_r["variant"]==vname)]
    print(f"  {vname}:")
    for _,r in m.iterrows():
        sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"    {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig}")
