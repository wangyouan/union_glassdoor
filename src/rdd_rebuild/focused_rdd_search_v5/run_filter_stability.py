#!/usr/bin/env python
"""A. Filter-stability regressions with 8 thresholds, global + |m|<=0.20, ±365d + ±548d."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE_365 = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
SAMPLE_548 = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v4/rdd_review_event_sample_18m.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v5"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
VARIANTS = [("poly1_non_spline",1,False),("poly1_spline",1,True),("poly2_non_spline",2,False),("poly2_spline",2,True)]
# 8 filter thresholds
FILTERS = [
    ("pre_post_1", 1), ("pre_post_5", 5), ("pre_post_10", 10),
    ("pre_post_20", 20), ("pre_post_25", 25), ("pre_post_50", 50),
    ("total_50", None, 50), ("total_100", None, 100),
]
BANDWIDTHS = [("global",None), ("|m|<=0.20",0.20)]

def cluster_se(X, y, beta, cids, df_r):
    resid = y - X@beta; n,k = X.shape; uq = np.unique(cids); G = len(uq)
    if G < 15: return None
    meat = np.zeros((k,k))
    for g in uq:
        m = cids==g; Xg=X[m]; rg=resid[m]; meat += (Xg.T@rg)[:,None] @ (Xg.T@rg)[None,:]
    v = np.linalg.inv(X.T@X) @ meat @ np.linalg.inv(X.T@X); v *= (G/(G-1))*((n-1)/(n-k))
    return np.sqrt(np.diag(v))

def load_sample(wd):
    if wd == 365: return pd.read_parquet(SAMPLE_365)
    else: return pd.read_parquet(SAMPLE_548)

def run_one(df_in, oc, emp, bw_val, wd, ft_label, ft_val, poly, spline):
    sub = df_in[df_in[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"] == emp]
    if bw_val is not None: sub = sub[sub["abs_margin"] <= bw_val]
    if wd == 548: sub = sub[sub["within_548"]]
    elif wd == 365: pass  # all within 365 in this sample
    else: sub = sub[sub["within_180"]]

    # Filter
    grp = sub.groupby("election_id")["post"]
    st = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if ft_label.startswith("total_"):
        st["n_total"] = st["n_post"] + st["n_pre"]
        valid = st[st["n_total"] >= ft_val]["election_id"]
    else:
        valid = st[(st["n_post"] >= ft_val) & (st["n_pre"] >= ft_val)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]
    if len(sub) < 100 or sub["election_id"].nunique() < 15 or sub["gvkey"].nunique() < 10: return None

    mu, sd = sub[oc].mean(), sub[oc].std()
    if sd == 0: return None
    y = (sub[oc].values - mu) / sd
    post = sub["post"].values.astype(float); win = sub["win"].values.astype(float)
    margin = sub["margin"].values; eid = sub["election_id"].values
    gv = sub["gvkey"].values; year = sub["review_year"].values
    pw = post * win; pm = post * margin

    X_list = [post, pw]
    if poly == 1:
        if spline: X_list += [pm, post*win*margin]
        else: X_list += [pm]
    else:
        if spline: X_list += [pm, post*win*margin, post*(margin**2), post*win*(margin**2)]
        else: X_list += [pm, post*(margin**2)]
    yu = np.unique(year)
    if len(yu) > 1: X_list.append(np.column_stack([(year==yv).astype(float) for yv in yu[1:]]))
    X_raw = np.column_stack(X_list)

    eid_u, eid_inv, eid_cnt = np.unique(eid, return_inverse=True, return_counts=True)
    ym = np.bincount(eid_inv, weights=y) / eid_cnt
    Xm = np.column_stack([np.bincount(eid_inv, weights=X_raw[:,j]) / eid_cnt for j in range(X_raw.shape[1])])
    y_dm, X_dm = y - ym[eid_inv], X_raw - Xm[eid_inv]
    n, k = X_dm.shape; df_resid = n - k - len(eid_u)
    if df_resid < 10: return None
    try:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        se = cluster_se(X_dm, y_dm, beta, gv, df_resid)
        if se is None: return None
        tau, se_t = beta[1], se[1]
        pv = 2 * stats.t.sf(abs(tau/se_t), df_resid) if se_t > 0 else np.nan
        return {"estimate": tau, "standard_error": se_t, "p_value": pv,
                "n_reviews": n, "n_events": len(eid_u), "n_gvkeys": sub["gvkey"].nunique()}
    except:
        return None

FILTERS_PARSED = [
    ("pre_post", 1), ("pre_post", 5), ("pre_post", 10),
    ("pre_post", 20), ("pre_post", 25), ("pre_post", 50),
    ("total", 50), ("total", 100),
]
total = len(OUTCOMES) * 2 * len(BANDWIDTHS) * len(FILTERS_PARSED) * len(VARIANTS) * 2
n_done = 0; results = []
for wd in [365, 548]:
    df_cur = load_sample(wd)
    for oc in OUTCOMES:
        for emp in ["current", "all"]:
            for bw_label, bw_val in BANDWIDTHS:
                for ft_type, ft_val in FILTERS_PARSED:
                    ft_label = f"{ft_type}_{ft_val}"
                    for vn, po, sp in VARIANTS:
                        res = run_one(df_cur, oc, emp, bw_val, wd, ft_label, ft_val, po, sp)
                        if res:
                            results.append({
                                "outcome": oc, "window_days": wd, "employee_sample": emp,
                                "bandwidth_label": bw_label, "poly_variant": vn,
                                "polynomial_order": po, "spline": sp,
                                "filter_type": ft_type, "filter_N": ft_val,
                                "estimate": res["estimate"], "standard_error": res["standard_error"],
                                "p_value": res["p_value"],
                                "n_reviews": res["n_reviews"], "n_events": res["n_events"],
                                "n_gvkeys": res["n_gvkeys"],
                            })
                        n_done += 1
        pct = n_done / max(total, 1) * 100
        print(f"\r  +/-{wd}d {oc}: {pct:.0f}%", end="", flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "filter_stability_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Quick summary
print("\n--- Current, +/-365d, global, pre>=1, poly1_spline ---")
m = df_r[(df_r["employee_sample"]=="current")&(df_r["window_days"]==365)&(df_r["bandwidth_label"]=="global")&(df_r["filter_type"]=="pre_post")&(df_r["filter_N"]==1)&(df_r["poly_variant"]=="poly1_spline")]
for _, r in m.iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: tau={r['estimate']:+.4f} p={r['p_value']:.3f}{sig} NE={int(r['n_events'])}")
