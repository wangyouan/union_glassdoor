#!/usr/bin/env python
"""
Step 4: Review-level DiD-RD with absorbing election FE.

Uses within-transformation (demeaning by election_id) for speed.
Specification:
  rating_sd ~ post + post*win + post*margin + post*win*margin + year_FE
  (election FE absorbed via demeaning)

Bandwidths: global, |m|<=0.20, |m|<=0.10
Employee filters: current (primary), all (robustness)
Windows: +/-365, +/-180, +/-90
Thresholds: pre>=1_post>=1, pre>=3_post>=3, pre>=5_post>=5
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "rdd_rebuild"
SAMPLE = OUT / "rdd_review_event_sample_from_raw.parquet"

print("=" * 70)
print("Loading RDD review-event sample...")
df = pd.read_parquet(SAMPLE)
print(f"  {len(df):,} reviews")

outcomes = [c for c in ["overall_rating","career_opp","comp_benefit",
                         "senior_mgmt","wlb","culture","diversity"]
            if c in df.columns]

def run_review_did_rdd(data, oc, bw_val=None, win_days=365, emp="current",
                        min_pre=1, min_post=1):
    """Efficient review-level DiD-RD with absorbed election FE."""

    # Filter
    sub = data.copy()
    if bw_val is not None:
        sub = sub[sub["abs_margin"] <= bw_val]
    if win_days < 365:
        wcol = f"within_{win_days}"
        sub = sub[sub[wcol]]
    if emp != "all":
        sub = sub[sub["employee_filter"] == emp]
    sub = sub[sub[oc].notna()]

    # Apply event-level threshold
    grp_counts = sub.groupby("election_id")["post"].agg(["sum", lambda x: (~x.astype(bool)).sum()])
    grp_counts.columns = ["n_post","n_pre"]
    valid_eids = grp_counts[(grp_counts["n_pre"]>=min_pre)&(grp_counts["n_post"]>=min_post)].index
    sub = sub[sub["election_id"].isin(valid_eids)]

    n_reviews = len(sub)
    n_events = sub["election_id"].nunique()
    if n_reviews < 100 or n_events < 20:
        return None

    # Standardize outcome
    mu = sub[oc].mean()
    sd = sub[oc].std()
    if sd == 0:
        return None
    y_raw = (sub[oc].values - mu) / sd

    # Build variables
    post = sub["post"].values.astype(float)
    win = sub["win"].values.astype(float)
    margin = sub["margin"].values
    year = sub["review_year"].values
    eid = sub["election_id"].values

    # Interactions
    post_win = post * win
    post_margin = post * margin
    post_win_margin = post * win * margin

    # Year dummies
    year_unique = np.unique(year)
    year_dummies = np.column_stack([(year == y).astype(float) for y in year_unique[1:]])

    # Build X: post, post*win, post*margin, post*win*margin, then year dummies
    X_raw = np.column_stack([post, post_win, post_margin, post_win_margin])
    X_raw = np.column_stack([X_raw, year_dummies])

    # Demean by election_id (absorbs election FE)
    # For each election, subtract election-level mean from y and X
    eid_unique, eid_inverse, eid_counts = np.unique(eid, return_inverse=True, return_counts=True)
    n_elections = len(eid_unique)

    # Compute election-level means efficiently
    y_mean_e = np.bincount(eid_inverse, weights=y_raw) / eid_counts
    X_mean_e = np.column_stack([
        np.bincount(eid_inverse, weights=X_raw[:,j]) / eid_counts
        for j in range(X_raw.shape[1])
    ])

    # Within-transformation
    y_dm = y_raw - y_mean_e[eid_inverse]
    X_dm = X_raw - X_mean_e[eid_inverse]

    # OLS on demeaned data
    try:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        y_pred = X_dm @ beta
        resid = y_dm - y_pred
        n, k = X_dm.shape
        # Correct df: lost n_elections degrees of freedom
        df_resid = n - k - n_elections
        if df_resid < 10:
            return None

        # HC1 variance
        sigma2 = np.sum(resid**2) / df_resid
        XtX_inv = np.linalg.inv(X_dm.T @ X_dm)
        # HC1: n/(n-k) * (X'X)^-1 * X' diag(e^2) X * (X'X)^-1
        hc1_meat = X_dm.T @ (X_dm * resid[:, None]**2)
        vcov = (n / df_resid) * XtX_inv @ hc1_meat @ XtX_inv
        se_all = np.sqrt(np.diag(vcov))

        tau = beta[1]  # post*win
        se_tau = se_all[1]
        t_stat = tau / se_tau if se_tau > 0 else np.nan
        p_val = 2 * stats.t.sf(abs(t_stat), df=df_resid)

        n_win_e = sub[sub["win"]==1]["election_id"].nunique()
        n_loss_e = sub[sub["win"]==0]["election_id"].nunique()

        return {
            "estimator": "review_level_did_rdd",
            "outcome": oc, "employee_filter": emp,
            "window_days": win_days,
            "bandwidth_label": "global" if bw_val is None else f"|m|<={bw_val}",
            "bandwidth_value": bw_val if bw_val is not None else np.nan,
            "threshold": f"pre>={min_pre}_post>={min_post}",
            "fixed_effects": "election_FE + year_FE",
            "n_reviews": n_reviews,
            "n_events": n_events,
            "n_gvkeys": int(sub["gvkey"].nunique()),
            "n_win_events": n_win_e,
            "n_loss_events": n_loss_e,
            "estimate_tau": tau,
            "se": se_tau,
            "t_stat": t_stat,
            "p_value": p_val,
            "mean_y": float(mu),
            "sd_y": float(sd),
            "rsquared": 1 - np.var(resid)/np.var(y_dm),
        }
    except Exception as e:
        return {"error": str(e)[:100]}


# ── Run specifications ────────────────────────────────────────────────
print("\nRunning review-level DiD-RD...")
results = []
total = len(outcomes) * 2 * 3 * 3 * 3  # outcomes * emp * bandwidths * windows * thresholds
n_done = 0

for oc in outcomes:
    for emp in ["current", "all"]:
        for bw_label, bw_val in [("global",None),("|m|<=0.20",0.20),("|m|<=0.10",0.10)]:
            for wd in [365, 180, 90]:
                for mp in [1, 3, 5]:
                    res = run_review_did_rdd(df, oc, bw_val, wd, emp, min_pre=mp, min_post=mp)
                    if res is not None and "error" not in res:
                        results.append(res)
                    n_done += 1
        pct = n_done / total * 100
        print(f"\r  Progress: {pct:.0f}% ({n_done}/{total})", end="", flush=True)

print(f"\n  Completed: {len(results)} valid results")

# Save
df_rr = pd.DataFrame(results)
df_rr.to_csv(OUT / "review_level_linear_did_rdd_results.csv", index=False)
print(f"  Saved: review_level_linear_did_rdd_results.csv ({len(df_rr)} rows)")

# ── Main results ──────────────────────────────────────────────────────
print("\n--- Review-level DiD-RD: current, +/-365d, pre>=3_post>=3 ---")
mask = ((df_rr["employee_filter"]=="current") & (df_rr["window_days"]==365) &
        (df_rr["threshold"]=="pre>=3_post>=3"))
for _, r in df_rr[mask].iterrows():
    sig = "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s} | {r['bandwidth_label']:10s} | "
          f"tau={r['estimate_tau']:+.4f} (se={r['se']:.4f}) p={r['p_value']:.3f} "
          f"N={int(r['n_reviews']):,} E={int(r['n_events'])} {sig}")

# Sign consistency summary
print("\n--- Sign consistency (current, +/-365d, pre>=1_post>=1) ---")
mask2 = ((df_rr["employee_filter"]=="current") & (df_rr["window_days"]==365) &
         (df_rr["threshold"]=="pre>=1_post>=1"))
for oc in outcomes:
    sub = df_rr[mask2 & (df_rr["outcome"]==oc)]
    if len(sub) < 3:
        continue
    g = sub[sub["bandwidth_label"]=="global"]
    b20 = sub[sub["bandwidth_label"]=="|m|<=0.20"]
    b10 = sub[sub["bandwidth_label"]=="|m|<=0.10"]
    tg = g["estimate_tau"].values[0] if len(g)>0 else np.nan
    t20 = b20["estimate_tau"].values[0] if len(b20)>0 else np.nan
    t10 = b10["estimate_tau"].values[0] if len(b10)>0 else np.nan
    signs = [np.sign(t) for t in [tg,t20,t10] if not np.isnan(t)]
    cons = "YES" if len(set(signs))==1 else "NO"
    pvals = [s["p_value"] for s in [g,b20,b10] if len(s)>0]
    mp = np.median(pvals) if pvals else np.nan
    print(f"  {oc:20s} | global={tg:+.4f} bw20={t20:+.4f} bw10={t10:+.4f} | "
          f"sign_ok={cons} | median_p={mp:.3f} | n={int(sub['n_reviews'].max()):,}")

# ── Summary: Event vs Review Direction ─────────────────────────────────
print("\n--- Event-Level vs Review-Level Direction (global, current, +/-365d) ---")
df_ev = pd.read_csv(OUT / "event_level_linear_rdd_results.csv")
mask_ev = ((df_ev["employee_filter"]=="current")&(df_ev["window_days"]==365)&
           (df_ev["threshold"]=="pre>=1_post>=1")&(df_ev["bandwidth_label"]=="global")&
           (df_ev["weighted"]==True))
for oc in outcomes:
    ev = df_ev[mask_ev & (df_ev["outcome"]==oc)]
    rv = df_rr[mask2 & (df_rr["outcome"]==oc) & (df_rr["bandwidth_label"]=="global")]
    if len(ev)>0 and len(rv)>0:
        et = ev["tau"].values[0]
        rt = rv["estimate_tau"].values[0]
        agree = "YES" if np.sign(et)==np.sign(rt) else "NO"
        print(f"  {oc:20s} | event={et:+.4f} review={rt:+.4f} | agree={agree}")

print("\nDone.")
