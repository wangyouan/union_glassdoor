#!/usr/bin/env python
"""C. Firm-quarter aggregated RDD with gvkey-clustered SE."""

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

print("Loading and aggregating to firm-quarter...")
df_rdd = pd.read_parquet(SAMPLE)

def cluster_se(X, y, beta, cluster_ids, df_resid):
    resid = y - X @ beta; n, k = X.shape
    clusters = np.unique(cluster_ids); G = len(clusters)
    if G < 15: return None
    meat = np.zeros((k,k))
    for g in clusters:
        mask = cluster_ids == g
        meat += (X[mask].T @ resid[mask])[:,None] @ (X[mask].T @ resid[mask])[None,:]
    vcov = np.linalg.inv(X.T@X) @ meat @ np.linalg.inv(X.T@X)
    vcov *= (G/(G-1))*((n-1)/(n-k))
    return np.sqrt(np.diag(vcov))

# Build firm-quarter panel: for each election, aggregate reviews by relative quarter
def build_quarter_panel(oc, emp, bw_val, win_days):
    sub = df_rdd[df_rdd[oc].notna()].copy()
    if emp != "all": sub = sub[sub["employee_filter"]==emp]
    if bw_val is not None: sub = sub[sub["abs_margin"]<=bw_val]
    if win_days < 365: sub = sub[sub[f"within_{win_days}"]]

    sub["rel_quarter"] = np.floor(sub["days_to_election"]/90).astype(int).clip(-4, 4)
    sub["post"] = (sub["days_to_election"] >= 0).astype(int)

    grp = sub.groupby(["election_id","gvkey","rel_quarter","post","win","margin","review_year"])
    agg = grp.agg(mean_rating=(oc,"mean"), n_reviews=(oc,"count")).reset_index()
    return agg

def run_quarter_did(panel, use_election_fe=True):
    """Quarter-level DiD with election or firm FE."""
    # Keep quarters with n>=1
    panel = panel[panel["n_reviews"]>=1].copy()
    if len(panel) < 50: return None

    mu, sd = panel["mean_rating"].mean(), panel["mean_rating"].std()
    if sd == 0: return None
    y = (panel["mean_rating"].values - mu) / sd
    post = panel["post"].values.astype(float); win = panel["win"].values.astype(float)
    margin = panel["margin"].values; eid = panel["election_id"].values
    year = panel["review_year"].values; gvkey_ids = panel["gvkey"].values

    post_win = post * win; post_margin = post * margin; post_win_margin = post * win * margin
    year_dummies = np.column_stack([(year==yv).astype(float) for yv in np.unique(year)[1:]]) if len(np.unique(year))>1 else None
    X_raw = np.column_stack([post, post_win, post_margin, post_win_margin])
    if year_dummies is not None: X_raw = np.column_stack([X_raw, year_dummies])

    if use_election_fe:
        eid_u, eid_inv, eid_cnt = np.unique(eid, return_inverse=True, return_counts=True)
        y_mean = np.bincount(eid_inv, weights=y) / eid_cnt
        X_mean = np.column_stack([np.bincount(eid_inv, weights=X_raw[:,j])/eid_cnt for j in range(X_raw.shape[1])])
        y_dm, X_dm = y - y_mean[eid_inv], X_raw - X_mean[eid_inv]
        n_elections = len(eid_u)
    else:
        gv_u, gv_inv, gv_cnt = np.unique(gvkey_ids, return_inverse=True, return_counts=True)
        y_mean = np.bincount(gv_inv, weights=y) / gv_cnt
        X_mean = np.column_stack([np.bincount(gv_inv, weights=X_raw[:,j])/gv_cnt for j in range(X_raw.shape[1])])
        y_dm, X_dm = y - y_mean[gv_inv], X_raw - X_mean[gv_inv]
        n_elections = panel["election_id"].nunique()

    n, k = X_dm.shape; df_resid = n - k - n_elections
    if df_resid < 10: return None
    try:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        se_all = cluster_se(X_dm, y_dm, beta, gvkey_ids, df_resid)
        if se_all is None: return None
        tau, se_tau = beta[1], se_all[1]
        pval = 2*stats.t.sf(abs(tau/se_tau), df_resid) if se_tau>0 else np.nan
        return {"estimate":tau,"standard_error":se_tau,"t_stat":tau/se_tau,"p_value":pval,
                "n":n,"n_events":n_elections,"n_gvkeys":len(np.unique(gvkey_ids)),
                "n_win":int(panel[panel["win"]==1]["election_id"].nunique()),
                "n_loss":int(panel[panel["win"]==0]["election_id"].nunique()),
                "mu":mu,"sd":sd}
    except: return None

results = []
for oc, oc_label in OUTCOMES.items():
    for emp in ["current","all"]:
        for bw_label, bw_val in BANDWIDTHS:
            for wd in [365, 180]:
                panel = build_quarter_panel(oc, emp, bw_val, wd)
                if panel is None: continue
                for use_fe in [True, False]:
                    for min_comments in [None, 5, 10] if emp == "all" else [None]:
                        p = panel.copy()
                        if min_comments is not None:
                            p = p[p["n_reviews"] >= min_comments]
                            if len(p) < 50: continue
                        res = run_quarter_did(p, use_fe)
                        if res:
                            results.append({"estimator":"firm_quarter_aggregate","outcome":oc,"outcome_label":oc_label,
                                "employee_sample":emp,"window_days":wd,"bandwidth_label":bw_label,"bandwidth_value":bw_val,
                                "min_comment_rule":str(min_comments) if min_comments else "none",
                                "specification_name":"election_FE" if use_fe else "firm_FE",
                                "fixed_effects":"election_FE+year_FE" if use_fe else "firm_FE+year_FE",
                                "polynomial_order":1,"cluster":"gvkey",
                                "n_observations":res["n"],"n_events":res["n_events"],"n_gvkeys":res["n_gvkeys"],
                                "n_win_events":res["n_win"],"n_loss_events":res["n_loss"],
                                "coefficient_of_interest":"Win x Post","estimate":res["estimate"],
                                "standard_error":res["standard_error"],"t_stat":res["t_stat"],"p_value":res["p_value"],
                                "mean_depvar":res["mu"],"sd_depvar":res["sd"]})
    print(f"  {oc}: {len([r for r in results if r['outcome']==oc])} specs")

df_fq = pd.DataFrame(results)
df_fq.to_csv(OUT / "firm_quarter_aggregate_rdd_results.csv", index=False)
print(f"Saved: firm_quarter_aggregate_rdd_results.csv ({len(df_fq)})")

print("\n--- Firm-quarter: current, +/-365d, global, election FE ---")
m = df_fq[(df_fq["employee_sample"]=="current")&(df_fq["window_days"]==365)&(df_fq["bandwidth_label"]=="global")&(df_fq["fixed_effects"]=="election_FE+year_FE")]
for _, r in m.iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig}")
