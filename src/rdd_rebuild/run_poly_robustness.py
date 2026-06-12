#!/usr/bin/env python
"""
RDD polynomial robustness: p=1 (linear), p=2 (quadratic), p=3 (cubic).
Also linear spline (piecewise linear, already run as main spec).

Compares across polynomial orders for event-level RDD.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "rdd_rebuild"
EVENTS = OUT / "event_level_rdd_data.parquet"

print("Loading event-level data...")
df = pd.read_parquet(EVENTS)
outcomes = sorted(df["outcome"].unique())
bandwidths = [("global", None), ("|m|<=0.20", 0.20), ("|m|<=0.10", 0.10)]

# ── RDD function: supports p=1,2,3 ──────────────────────────────────
def event_rdd_poly(data, bw_val, poly_order=1, weighted=False):
    """delta = a + tau*win + poly(margin) + win*poly(margin) + e"""
    sub = data.dropna(subset=["delta", "win", "margin"])
    if bw_val is not None:
        sub = sub[sub["abs_margin"] <= bw_val]
    if len(sub) < 30 or sub["win"].nunique() < 2:
        return None

    y = sub["delta"].values
    win = sub["win"].values.astype(float)
    m = sub["margin"].values

    # Build polynomial terms
    X_list = [np.ones_like(win), win]  # const, win
    col_names = ["const", "win"]
    for p in range(1, poly_order + 1):
        X_list.append(m**p)           # margin^p
        col_names.append(f"m{p}")
        X_list.append(win * m**p)     # win × margin^p
        col_names.append(f"win_m{p}")

    X = np.column_stack(X_list)

    w = None
    if weighted and "n_pre" in sub.columns and "n_post" in sub.columns:
        n_pre = np.maximum(sub["n_pre"].values.astype(float), 1)
        n_post = np.maximum(sub["n_post"].values.astype(float), 1)
        w = 2 / (1/n_pre + 1/n_post)
        w = w / w.mean()

    try:
        mod = sm.WLS(y, X, weights=w) if w is not None else sm.OLS(y, X)
        res = mod.fit()
        n, k = X.shape
        se = np.sqrt(np.diag(res.cov_params())) * np.sqrt(n / (n - k))  # HC1

        tau = res.params[1]  # win coefficient
        se_tau = se[1]
        t = tau / se_tau if se_tau > 0 else np.nan
        pval = 2 * stats.t.sf(abs(t), df=n - k)

        return {
            "poly_order": poly_order,
            "n_events": n, "n_gvkeys": int(sub["gvkey"].nunique()),
            "n_win": int(win.sum()), "n_loss": n - int(win.sum()),
            "tau": tau, "se": se_tau, "t_stat": t, "p_value": pval,
            "aic": res.aic, "bic": res.bic,
            "rsquared": res.rsquared,
            "mean_delta_win": float(y[win==1].mean()) if win.sum()>0 else np.nan,
            "mean_delta_loss": float(y[win==0].mean()) if (1-win).sum()>0 else np.nan,
            "sd_delta": float(y.std()),
        }
    except Exception as e:
        return None


# ── Run all polynomial orders ────────────────────────────────────────
print("Running poly=1,2,3 for all outcome/filter/bandwidth/threshold combos...")
all_results = []
for oc in outcomes:
    for emp in ["current", "all"]:
        for win_days in [365, 180, 90]:
            for th in df["threshold"].unique():
                for bw_label, bw_val in bandwidths:
                    for poly_order in [1, 2, 3]:
                        for wgt in [False, True]:
                            mask = ((df["outcome"] == oc) & (df["employee_filter"] == emp) &
                                    (df["window_days"] == win_days) & (df["threshold"] == th))
                            sub = df[mask]
                            if len(sub) < 30:
                                continue
                            res = event_rdd_poly(sub, bw_val, poly_order=poly_order, weighted=wgt)
                            if res:
                                res.update({
                                    "outcome": oc, "employee_filter": emp,
                                    "window_days": win_days, "threshold": th,
                                    "bandwidth_label": bw_label, "bandwidth_value": bw_val,
                                    "weighted": wgt,
                                })
                                all_results.append(res)

df_all = pd.DataFrame(all_results)
df_all.to_csv(OUT / "event_level_rdd_poly_comparison.csv", index=False)
print(f"  Saved {len(df_all)} results across p=1,2,3")

# ── Summary: best polynomial order by AIC ────────────────────────────
print("\n--- Best polynomial order by AIC (current, ±365d, pre>=1_post>=1, weighted) ---")
mask_base = ((df_all["employee_filter"] == "current") & (df_all["window_days"] == 365) &
             (df_all["threshold"] == "pre>=1_post>=1") & (df_all["weighted"] == True))

for oc in outcomes:
    for bw_label in ["global", "|m|<=0.20", "|m|<=0.10"]:
        sub = df_all[mask_base & (df_all["outcome"] == oc) & (df_all["bandwidth_label"] == bw_label)]
        if len(sub) < 3:
            continue
        best = sub.loc[sub["aic"].idxmin()]
        print(f"  {oc:20s} | {bw_label:10s} | best_p={int(best['poly_order'])} | "
              f"τ={best['tau']:+.4f} se={best['se']:.4f} p={best['p_value']:.3f} | "
              f"AIC={best['aic']:.1f} | p=1 AIC={sub[sub['poly_order']==1]['aic'].values[0]:.1f} "
              f"p=2 AIC={sub[sub['poly_order']==2]['aic'].values[0]:.1f} "
              f"p=3 AIC={sub[sub['poly_order']==3]['aic'].values[0]:.1f}")

# ── Compact comparison table ─────────────────────────────────────────
print("\n--- Polynomial order comparison (global, current, ±365d, pre>=1_post>=1, weighted) ---")
print(f"{'Outcome':20s} | {'p=1 τ':>8s} {'p=1 se':>7s} {'p=1 p':>6s} | {'p=2 τ':>8s} {'p=2 se':>7s} {'p=2 p':>6s} | {'p=3 τ':>8s} {'p=3 se':>7s} {'p=3 p':>6s} | {'Best':>5s}")
print("-" * 110)
for oc in outcomes:
    sub = df_all[mask_base & (df_all["outcome"] == oc) & (df_all["bandwidth_label"] == "global")]
    if len(sub) < 3:
        continue
    r1 = sub[sub["poly_order"] == 1].iloc[0]
    r2 = sub[sub["poly_order"] == 2].iloc[0]
    r3 = sub[sub["poly_order"] == 3].iloc[0]
    best_p = sub.loc[sub["aic"].idxmin(), "poly_order"]
    print(f"{oc:20s} | {r1['tau']:+8.4f} {r1['se']:7.4f} {r1['p_value']:6.3f} | "
          f"{r2['tau']:+8.4f} {r2['se']:7.4f} {r2['p_value']:6.3f} | "
          f"{r3['tau']:+8.4f} {r3['se']:7.4f} {r3['p_value']:6.3f} | p={int(best_p)}")

# ── Spline specification: piecewise quadratic (knot at 0) ────────────
print("\n--- Spline (piecewise quadratic) vs Linear ---")
def event_rdd_spline(data, bw_val, weighted=False):
    """Piecewise quadratic spline: different quadratic on each side of 0."""
    sub = data.dropna(subset=["delta", "win", "margin"])
    if bw_val is not None:
        sub = sub[sub["abs_margin"] <= bw_val]
    if len(sub) < 30 or sub["win"].nunique() < 2:
        return None

    y = sub["delta"].values
    win = sub["win"].values.astype(float)
    m = sub["margin"].values
    m_pos = m * (m > 0).astype(float)   # only positive side
    m_neg = m * (m < 0).astype(float)   # only negative side

    # Spline: y = α + τ*win + β1*m_neg + β2*m_neg² + β3*m_pos + β4*m_pos² + ε
    # This allows different quadratic curves on each side
    X = np.column_stack([
        np.ones_like(win),   # const
        win,                  # jump at 0
        m_neg,               # linear left
        m_neg**2,            # quadratic left
        m_pos,               # linear right
        m_pos**2,            # quadratic right
    ])

    w = None
    if weighted:
        n_pre = np.maximum(sub["n_pre"].values.astype(float), 1)
        n_post = np.maximum(sub["n_post"].values.astype(float), 1)
        w = 2 / (1/n_pre + 1/n_post)
        w = w / w.mean()

    try:
        mod = sm.WLS(y, X, weights=w) if w is not None else sm.OLS(y, X)
        res = mod.fit()
        n, k = X.shape
        se = np.sqrt(np.diag(res.cov_params())) * np.sqrt(n / (n - k))
        tau = res.params[1]
        se_tau = se[1]
        t = tau / se_tau if se_tau > 0 else np.nan
        pval = 2 * stats.t.sf(abs(t), df=n - k)
        return {"tau": tau, "se": se_tau, "t_stat": t, "p_value": pval, "aic": res.aic}
    except:
        return None

for oc in outcomes:
    for bw_label, bw_val in [("global", None), ("|m|<=0.20", 0.20)]:
        mask = ((df["outcome"] == oc) & (df["employee_filter"] == "current") &
                (df["window_days"] == 365) & (df["threshold"] == "pre>=1_post>=1"))
        sub = df[mask]
        if len(sub) < 30:
            continue
        spline = event_rdd_spline(sub, bw_val, weighted=True)
        linear = event_rdd_poly(sub, bw_val, poly_order=1, weighted=True)
        if spline and linear:
            better = "spline" if spline["aic"] < linear["aic"] else "linear"
            print(f"  {oc:20s} | {bw_label:10s} | linear τ={linear['tau']:+.4f} p={linear['p_value']:.3f} AIC={linear['aic']:.1f} | "
                  f"spline τ={spline['tau']:+.4f} p={spline['p_value']:.3f} AIC={spline['aic']:.1f} | prefer={better}")

print("\nDone.")
