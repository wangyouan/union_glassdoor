#!/usr/bin/env python
"""B. v7 full regressions — firm FE + individual controls, per the prompt."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v7"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = OUT / "rdd_sample_v7_enriched.parquet"

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
VARIANTS = [("poly1_non_spline",1,False),("poly1_spline",1,True),("poly2_non_spline",2,False),("poly2_spline",2,True)]
FILTERS = [("pre_post",1),("pre_post",5),("pre_post",10),("pre_post",20),("pre_post",25),("pre_post",50),("total",50),("total",100)]
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20)]
SPECS = ["v7a","v7b","v7c"]

def safe_inv(M):
    try: return np.linalg.inv(M)
    except: return np.linalg.pinv(M)

def gvkey_se(X, y, beta, gv, df_r):
    """gvkey-clustered SE."""
    resid = y - X@beta; n,k = X.shape; uq = np.unique(gv); G = len(uq)
    if G < 10: return None
    meat = np.zeros((k,k))
    for g in uq:
        m = gv==g; Xg=X[m]; rg=resid[m]; meat += (Xg.T@rg)[:,None] @ (Xg.T@rg)[None,:]
    bread = safe_inv(X.T@X)
    v = bread @ meat @ bread; v *= (G/(G-1))*((n-1)/max(n-k,1))
    return np.sqrt(np.diag(v))

df_all = pd.read_parquet(SAMPLE)
print(f"Loaded {len(df_all):,} reviews")

def run_v7(df_in, oc, emp, bw_val, ft_type, ft_val, poly, spline, spec_ver):
    sub = df_in[df_in[oc].notna()].copy()
    if emp == "current": sub = sub[sub["employee_filter"] == "current"]
    if bw_val is not None: sub = sub[sub["abs_margin"] <= bw_val]

    # Filter threshold
    grp = sub.groupby("election_id")["post"]
    st = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if ft_type == "total":
        st["n_total"] = st["n_post"]+st["n_pre"]; valid = st[st["n_total"]>=ft_val]["election_id"]
    else:
        valid = st[(st["n_post"]>=ft_val)&(st["n_pre"]>=ft_val)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]
    if len(sub) < 100 or sub["election_id"].nunique() < 10 or sub["gvkey"].nunique() < 10: return None

    mu, sd = sub[oc].mean(), sub[oc].std()
    if sd == 0: return None
    y = (sub[oc].values - mu) / sd
    post = sub["post"].values.astype(float); win = sub["win"].values.astype(float)
    margin = sub["margin"].values; gv = sub["gvkey"].values; year = sub["review_year"].values
    pw = post * win; pm = post * margin

    # Margin controls
    X_list = [post, pw]
    if poly == 1:
        if spline: X_list += [pm, post*win*margin]
        else: X_list += [pm]
    else:
        if spline: X_list += [pm, post*win*margin, post*(margin**2), post*win*(margin**2)]
        else: X_list += [pm, post*(margin**2)]

    # Year dummies
    yu = np.unique(year)
    if len(yu) > 1: X_list.append(np.column_stack([(year==yv).astype(float) for yv in yu[1:]]))

    # ── Individual controls (v7a/v7b/v7c) ──
    ctrl_names = []
    # Employment type
    for ctrl in ["is_part_time","is_intern","is_contract","is_other_employment","is_employment_missing"]:
        if ctrl in sub.columns:
            X_list.append(sub[ctrl].values.astype(float)); ctrl_names.append(ctrl)
    # Seniority
    if "seniority" in sub.columns:
        sen = sub["seniority"].fillna(-1).astype(int)
        for sv in [2,3,4,5]:
            X_list.append((sen==sv).astype(float)); ctrl_names.append(f"sen_{sv}")
        X_list.append((sen.isin([6,7])).astype(float)); ctrl_names.append("sen_67")
        X_list.append((sen==-1).astype(float)); ctrl_names.append("sen_miss")

    # State FE (v7b/v7c)
    if spec_ver in ["v7b","v7c"] and "state" in sub.columns:
        states = sub["state"].fillna("Non-US")
        stc = states.value_counts()
        keep = {s for s in stc[stc>=100].index if s != "Non-US"}
        st_clean = states.apply(lambda s: s if s in keep else ("Non-US" if s=="Non-US" else "Other_US"))
        for dc in pd.get_dummies(st_clean, prefix="st", drop_first=True).columns:
            dvals = pd.get_dummies(st_clean, prefix="st", drop_first=True)[dc].values.astype(float)
            # Skip zero-variance columns
            if dvals.std() > 0:
                X_list.append(dvals); ctrl_names.append(dc)

    # Role FE (v7c)
    if spec_ver == "v7c" and "role_k1500" in sub.columns:
        roles = sub["role_k1500"].fillna("Missing_role")
        rc = roles.value_counts()
        top200 = set(rc.head(200).index)
        r_clean = roles.apply(lambda r: r if r in top200 else "Other_role")
        rdummies = pd.get_dummies(r_clean, prefix="role", drop_first=True)
        for dc in rdummies.columns:
            dvals = rdummies[dc].values.astype(float)
            if dvals.std() > 0:
                X_list.append(dvals); ctrl_names.append(dc)

    X_raw = np.column_stack(X_list)

    # ── Firm FE: demean by gvkey ──
    gv_u, gv_inv, gv_cnt = np.unique(gv, return_inverse=True, return_counts=True)
    ym = np.bincount(gv_inv, weights=y) / gv_cnt
    Xm = np.column_stack([np.bincount(gv_inv, weights=X_raw[:,j]) / gv_cnt for j in range(X_raw.shape[1])])
    y_dm, X_dm = y - ym[gv_inv], X_raw - Xm[gv_inv]
    n, k = X_dm.shape; df_resid = n - k - len(gv_u)
    if df_resid < 20: return None

    try:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    except: return None

    # gvkey-clustered SE
    se_g = gvkey_se(X_dm, y_dm, beta, gv, df_resid)
    if se_g is None: return None

    # Attempt two-way SE for key specs only
    se_type = "gvkey_only"
    se_tau = se_g[1]
    tau = beta[1]
    pv = 2*stats.t.sf(abs(tau/se_tau), df_resid) if se_tau > 0 else np.nan

    return {"estimate": tau, "standard_error": se_tau, "p_value": pv,
            "se_type": se_type, "se_gvkey": se_g[1],
            "n_reviews": n, "n_events": sub["election_id"].nunique(), "n_gvkeys": len(gv_u)}

# Run all specs
results = []
total = len(OUTCOMES)*2*len(BANDWIDTHS)*len(FILTERS)*len(VARIANTS)*len(SPECS)
n_done = 0
for oc in OUTCOMES:
    for emp in ["all","current"]:
        for bw_label, bw_val in BANDWIDTHS:
            for ft_type, ft_val in FILTERS:
                for vn, po, sp in VARIANTS:
                    for sv in SPECS:
                        res = run_v7(df_all, oc, emp, bw_val, ft_type, ft_val, po, sp, sv)
                        if res:
                            results.append({"outcome":oc,"window_days":365,"employee_sample":emp,
                                "spec_version":sv,"bandwidth_label":bw_label,"poly_variant":vn,
                                "polynomial_order":po,"spline":sp,"filter_type":ft_type,"filter_N":ft_val,
                                "estimate":res["estimate"],"standard_error":res["standard_error"],
                                "p_value":res["p_value"],"se_type":res["se_type"],
                                "n_reviews":res["n_reviews"],"n_events":res["n_events"],"n_gvkeys":res["n_gvkeys"]})
                        n_done += 1
        pct = n_done/max(total,1)*100
        print(f"\r  {oc}: {pct:.0f}% ({n_done}/{total})", end="", flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "filter_stability_v7_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Summary
print("\n--- all, +/-365d, global, pre>=1, poly1_spline ---")
for sv in SPECS:
    m = df_r[(df_r["employee_sample"]=="all")&(df_r["bandwidth_label"]=="global")&
             (df_r["filter_type"]=="pre_post")&(df_r["filter_N"]==1)&
             (df_r["poly_variant"]=="poly1_spline")&(df_r["spec_version"]==sv)]
    print(f"  {sv}:")
    for _,r in m.iterrows():
        sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
        print(f"    {r['outcome']:20s}: tau={r['estimate']:+.4f} p={r['p_value']:.3f}{sig} NE={int(r['n_events'])} SE={r['se_type']}")
