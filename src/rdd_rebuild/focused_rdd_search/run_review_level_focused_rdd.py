#!/usr/bin/env python
"""A. Review-level focused RDD with gvkey-clustered SE."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm_api
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = {"overall_rating":"Overall Rating","career_opp":"Career Opp","comp_benefit":"Comp & Benefits",
            "senior_mgmt":"Senior Mgmt","wlb":"Work-Life Balance","culture":"Culture & Values"}
BANDWIDTHS = [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]
WINDOWS = [365,180,90]
THRESHOLDS = [("pre>=1_post>=1",1,1),("pre>=3_post>=3",3,3),("pre>=5_post>=5",5,5),("total>=10",None,None,10)]
EMP_FILTERS = ["current","all"]

print("Loading RDD sample...")
df = pd.read_parquet(SAMPLE)
print(f"  {len(df):,} reviews, {df['gvkey'].nunique()} gvkeys, {df['election_id'].nunique()} elections")

def cluster_se(X, y, beta, cluster_ids, df_resid):
    """Cluster-robust variance-covariance by gvkey."""
    resid = y - X @ beta
    n, k = X.shape
    clusters = np.unique(cluster_ids)
    G = len(clusters)
    if G < 20: return None  # too few clusters
    meat = np.zeros((k, k))
    for g in clusters:
        mask = cluster_ids == g
        X_g = X[mask]; r_g = resid[mask]
        meat += (X_g.T @ r_g)[:, None] @ (X_g.T @ r_g)[None, :]
    bread = np.linalg.inv(X.T @ X)
    vcov = bread @ meat @ bread
    # Finite-sample correction
    vcov *= (G / (G - 1)) * ((n - 1) / (n - k))
    return np.sqrt(np.diag(vcov))

def run_review(oc, emp, bw_val, win_days, min_pre, min_post, min_total=None, use_fe=True, weighted=False):
    """Review-level DiD-RD with gvkey-clustered SE."""
    sub = df[df[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"] == emp]
    if bw_val is not None: sub = sub[sub["abs_margin"] <= bw_val]
    if win_days < 365: sub = sub[sub[f"within_{win_days}"]]

    # Threshold
    grp = sub.groupby("election_id")["post"]
    eid_stats = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if min_total is not None:
        eid_stats["n_total"] = eid_stats["n_post"] + eid_stats["n_pre"]
        valid = eid_stats[eid_stats["n_total"] >= min_total]["election_id"]
    else:
        valid = eid_stats[(eid_stats["n_post"] >= min_post) & (eid_stats["n_pre"] >= min_pre)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]

    if len(sub) < 100 or sub["election_id"].nunique() < 15 or sub["gvkey"].nunique() < 10:
        return None

    mu, sd = sub[oc].mean(), sub[oc].std()
    if sd == 0: return None
    y = (sub[oc].values - mu) / sd

    post = sub["post"].values.astype(float)
    win = sub["win"].values.astype(float)
    margin = sub["margin"].values
    eid = sub["election_id"].values
    year = sub["review_year"].values
    gvkey_ids = sub["gvkey"].values

    post_win = post * win; post_margin = post * margin; post_win_margin = post * win * margin
    year_dummies = np.column_stack([(year == yv).astype(float) for yv in np.unique(year)[1:]]) if len(np.unique(year)) > 1 else np.zeros((len(y), 0))

    if use_fe:
        X_raw = np.column_stack([post, post_win, post_margin, post_win_margin] + ([year_dummies] if year_dummies.shape[1] > 0 else []))
        var_names = ["Post","Win x Post","Post x Margin","Win x Post x Margin"]
        # Demean by election
        eid_u, eid_inv, eid_cnt = np.unique(eid, return_inverse=True, return_counts=True)
        y_mean = np.bincount(eid_inv, weights=y) / eid_cnt
        X_mean = np.column_stack([np.bincount(eid_inv, weights=X_raw[:,j]) / eid_cnt for j in range(X_raw.shape[1])])
        y_dm, X_dm = y - y_mean[eid_inv], X_raw - X_mean[eid_inv]
        n, k = X_dm.shape
        df_resid = n - k - len(eid_u)
        if df_resid < 10: return None
        try:
            beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
            se_all = cluster_se(X_dm, y_dm, beta, gvkey_ids, df_resid)
            if se_all is None: return None
            res = {"Win": (np.nan, np.nan, np.nan), "Post": (beta[0], se_all[0], 2*stats.t.sf(abs(beta[0]/se_all[0]), df_resid) if se_all[0]>0 else np.nan),
                   "Win x Post": (beta[1], se_all[1], 2*stats.t.sf(abs(beta[1]/se_all[1]), df_resid) if se_all[1]>0 else np.nan),
                   "fe": "election_FE+year_FE", "coi": "Win x Post", "coi_idx": 1,
                   "n": n, "n_events": len(eid_u), "n_gvkeys": int(sub["gvkey"].nunique()),
                   "n_win": int(sub[sub["win"]==1]["election_id"].nunique()),
                   "n_loss": int(sub[sub["win"]==0]["election_id"].nunique()), "mu": mu, "sd": sd}
            if k >= 3: res["Post x Margin"] = (beta[2], se_all[2], np.nan)
            if k >= 4: res["Win x Post x Margin"] = (beta[3], se_all[3], np.nan)
            return res
        except: return None
    else:
        # No election FE
        X = sm_api.add_constant(np.column_stack([post, win, post_win, margin, win*margin, post_margin, post_win_margin]))
        var_names = ["Constant","Post","Win","Win x Post","Margin","Win x Margin","Post x Margin","Win x Post x Margin"]
        n, k = X.shape
        try:
            mod = sm_api.OLS(y, X).fit()
            se_all = cluster_se(X, y, mod.params, gvkey_ids, n-k)
            if se_all is None:
                se_all = np.sqrt(np.diag(mod.cov_params()))
            res = {}
            for i, vn in enumerate(var_names):
                pval = 2*stats.t.sf(abs(mod.params[i]/se_all[i]), n-k) if se_all[i]>0 else np.nan
                res[vn] = (mod.params[i], se_all[i], pval)
            res["fe"] = "none"; res["coi"] = "Win x Post"; res["coi_idx"] = 3
            res["n"] = n; res["n_events"] = sub["election_id"].nunique()
            res["n_gvkeys"] = int(sub["gvkey"].nunique())
            res["n_win"] = int(sub[sub["win"]==1]["election_id"].nunique())
            res["n_loss"] = int(sub[sub["win"]==0]["election_id"].nunique()); res["mu"] = mu; res["sd"] = sd
            return res
        except: return None

# Run all specs
results, coef_rows = [], []
n_total = len(OUTCOMES)*len(EMP_FILTERS)*len(BANDWIDTHS)*len(WINDOWS)*len(THRESHOLDS)*2
n_done = 0
for oc, oc_label in OUTCOMES.items():
    for emp in EMP_FILTERS:
        for bw_label, bw_val in BANDWIDTHS:
            for wd in WINDOWS:
                for th_label, min_pre, min_post, *rest in [t if len(t)>3 else (*t,None) for t in THRESHOLDS]:
                    min_total = rest[0] if rest else None
                    for use_fe in [True, False]:
                        res = run_review(oc, emp, bw_val, wd, min_pre, min_post, min_total, use_fe)
                        if res:
                            tau, se, p = res[res["coi"]]
                            results.append({"estimator":"review_level","outcome":oc,"outcome_label":oc_label,
                                "employee_sample":emp,"window_days":wd,"bandwidth_label":bw_label,
                                "bandwidth_value":bw_val,"threshold_rule":th_label,"specification_name":res["fe"],
                                "fixed_effects":res["fe"],"cluster":"gvkey","n_reviews":res["n"],
                                "n_events":res["n_events"],"n_gvkeys":res["n_gvkeys"],
                                "n_win_events":res["n_win"],"n_loss_events":res["n_loss"],
                                "coefficient_of_interest":res["coi"],"estimate":tau,"standard_error":se,
                                "t_stat":tau/se if se>0 else np.nan,"p_value":p,
                                "mean_depvar":res["mu"],"sd_depvar":res["sd"]})
                            # Full coefs
                            cr = {"outcome":oc,"employee_sample":emp,"window_days":wd,"bandwidth_label":bw_label,
                                  "threshold_rule":th_label,"fixed_effects":res["fe"],"n_reviews":res["n"],
                                  "n_events":res["n_events"],"n_gvkeys":res["n_gvkeys"]}
                            for vn in ["Win","Post","Win x Post","Margin","Win x Margin","Post x Margin","Win x Post x Margin"]:
                                if vn in res:
                                    cr[f"{vn}_coef"], cr[f"{vn}_se"], cr[f"{vn}_p"] = res[vn]
                            coef_rows.append(cr)
                        n_done += 1
        pct = n_done/max(n_total,1)*100
        print(f"\r  {oc}: {pct:.0f}%", end="", flush=True)

print(f"\n  Done: {len(results)} result rows, {len(coef_rows)} full coefficient rows")

df_r = pd.DataFrame(results)
df_c = pd.DataFrame(coef_rows)
df_r.to_csv(OUT / "review_level_focused_rdd_results.csv", index=False)
df_c.to_csv(OUT / "review_level_focused_rdd_full_coefficients.csv", index=False)
print(f"Saved: review_level_focused_rdd_results.csv ({len(df_r)})")

# Quick summary
print("\n--- Current, +/-365d, pre>=1_post>=1, election FE, global ---")
m = df_r[(df_r["employee_sample"]=="current")&(df_r["window_days"]==365)&(df_r["threshold_rule"]=="pre>=1_post>=1")&(df_r["fixed_effects"]=="election_FE+year_FE")&(df_r["bandwidth_label"]=="global")]
for _, r in m.iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig} N={int(r['n_reviews']):,} E={int(r['n_events'])}")
