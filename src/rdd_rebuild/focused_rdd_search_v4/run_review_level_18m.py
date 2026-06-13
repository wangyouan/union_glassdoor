#!/usr/bin/env python
"""B. Review-level regressions for ±548d window (same 4 variants as v3)."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

IN_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/focused_rdd_search_v4")
IN_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(IN_DIR / "rdd_review_event_sample_18m.parquet")

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
SCREENS = [("pre>=1_post>=1",1,1,None),("pre>=3_post>=3",3,3,None),("pre>=5_post>=5",5,5,None),("total>=10",None,None,10)]
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]
WINDOWS = [548,365,180]
VARIANTS = [("poly1_non_spline",1,False),("poly1_spline",1,True),("poly2_non_spline",2,False),("poly2_spline",2,True)]

def cluster_se(X, y, beta, cids, df_r):
    resid = y - X@beta; n,k = X.shape; uq = np.unique(cids); G = len(uq)
    if G < 15: return None
    meat = np.zeros((k,k))
    for g in uq:
        m = cids==g; Xg=X[m]; rg=resid[m]; meat += (Xg.T@rg)[:,None] @ (Xg.T@rg)[None,:]
    v = np.linalg.inv(X.T@X) @ meat @ np.linalg.inv(X.T@X)
    v *= (G/(G-1))*((n-1)/(n-k)); return np.sqrt(np.diag(v))

print(f"Loaded {len(df):,} reviews")

def run(oc, emp, bw_val, wd, min_pre, min_post, min_total, poly, spline, overlap_filter=None):
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if wd==548: sub=sub[sub["within_548"]]; wcol="within_548"
    elif wd==365: sub=sub[sub["within_365"]]
    else: sub=sub[sub["within_180"]]
    if overlap_filter is not None:
        sub = sub[sub["overlap_election"]==overlap_filter]
    # Screening
    grp = sub.groupby("election_id")["post"]
    st = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if min_total is not None: st["n_total"]=st["n_post"]+st["n_pre"]; valid=st[st["n_total"]>=min_total]["election_id"]
    else: valid=st[(st["n_post"]>=min_post)&(st["n_pre"]>=min_pre)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]
    if len(sub)<100 or sub["election_id"].nunique()<15 or sub["gvkey"].nunique()<10: return None

    mu,sd = sub[oc].mean(), sub[oc].std()
    if sd==0: return None
    y = (sub[oc].values-mu)/sd
    post=sub["post"].values.astype(float); win=sub["win"].values.astype(float)
    margin=sub["margin"].values; eid=sub["election_id"].values; gv=sub["gvkey"].values; year=sub["review_year"].values
    pw=post*win; pm=post*margin

    X_list = [post, pw]
    if poly==1:
        if spline: X_list += [pm, post*win*margin]
        else: X_list += [pm]
    else:
        if spline: X_list += [pm, post*win*margin, post*(margin**2), post*win*(margin**2)]
        else: X_list += [pm, post*(margin**2)]
    yu = np.unique(year)
    if len(yu)>1: X_list.append(np.column_stack([(year==yv).astype(float) for yv in yu[1:]]))
    X_raw = np.column_stack(X_list)

    eid_u, eid_inv, eid_cnt = np.unique(eid, return_inverse=True, return_counts=True)
    ym=np.bincount(eid_inv,weights=y)/eid_cnt
    Xm=np.column_stack([np.bincount(eid_inv,weights=X_raw[:,j])/eid_cnt for j in range(X_raw.shape[1])])
    y_dm, X_dm = y-ym[eid_inv], X_raw-Xm[eid_inv]
    n,k=X_dm.shape; df_resid=n-k-len(eid_u)
    if df_resid<10: return None
    try:
        beta=np.linalg.lstsq(X_dm,y_dm,rcond=None)[0]
        se=cluster_se(X_dm,y_dm,beta,gv,df_resid)
        if se is None: return None
        tau,se_t=beta[1],se[1]; pv=2*stats.t.sf(abs(tau/se_t),df_resid) if se_t>0 else np.nan
        return {"estimate":tau,"se":se_t,"p_value":pv,"n":n,"n_events":len(eid_u),
                "n_gvkeys":sub["gvkey"].nunique(),"mu":mu,"sd":sd}
    except: return None

results = []
total = len(OUTCOMES)*2*len(BANDWIDTHS)*len(WINDOWS)*len(SCREENS)*len(VARIANTS)
n_done = 0
for oc in OUTCOMES:
    for emp in ["current","all"]:
        for bw_label,bw_val in BANDWIDTHS:
            for wd in WINDOWS:
                for th_label,mp,mpp,mt in SCREENS:
                    for vn,po,sp in VARIANTS:
                        res = run(oc,emp,bw_val,wd,mp,mpp,mt,po,sp)
                        if res:
                            results.append({"outcome":oc,"employee_sample":emp,"window_days":wd,
                                "bandwidth_label":bw_label,"screening_rule":th_label,
                                "variant":vn,"polynomial_order":po,"spline":sp,
                                "n_reviews":res["n"],"n_events":res["n_events"],"n_gvkeys":res["n_gvkeys"],
                                "estimate":res["estimate"],"standard_error":res["se"],
                                "p_value":res["p_value"],"mean_depvar":res["mu"],"sd_depvar":res["sd"],
                                "overlap_sample":"full"})
                        # No-overlap for 548d only, baseline screening, current
                        if wd==548 and emp=="current" and th_label=="pre>=1_post>=1" and bw_label=="global":
                            res2 = run(oc,emp,bw_val,wd,mp,mpp,mt,po,sp,overlap_filter=False)
                            if res2:
                                results.append({"outcome":oc,"employee_sample":emp,"window_days":wd,
                                    "bandwidth_label":bw_label,"screening_rule":th_label,
                                    "variant":vn,"polynomial_order":po,"spline":sp,
                                    "n_reviews":res2["n"],"n_events":res2["n_events"],"n_gvkeys":res2["n_gvkeys"],
                                    "estimate":res2["estimate"],"standard_error":res2["se"],
                                    "p_value":res2["p_value"],"mean_depvar":res2["mu"],"sd_depvar":res2["sd"],
                                    "overlap_sample":"no_overlap"})
                        n_done+=1
        pct=n_done/max(total,1)*100
        print(f"\r  {oc}: {pct:.0f}%",end="",flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(IN_DIR / "review_level_18m_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Quick summary
print("\n--- Current, +/-548d, global, pre>=1 ---")
for vn,po,sp in VARIANTS:
    m = df_r[(df_r["employee_sample"]=="current")&(df_r["window_days"]==548)&(df_r["bandwidth_label"]=="global")&(df_r["screening_rule"]=="pre>=1_post>=1")&(df_r["variant"]==vn)]
    print(f"  {vn}:")
    for _,r in m.iterrows():
        sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"    {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig}")
