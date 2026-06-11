#!/usr/bin/env python
"""
Steps 3+5: Event-level linear RDD + rdrobust robustness.

Fast version — event-level only. Review-level DiD-RD in separate script.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "rdd_rebuild"
EVENTS = OUT / "event_level_rdd_data.parquet"

print("=" * 70)
print("LOADING EVENT-LEVEL DATA")
print("=" * 70)
df = pd.read_parquet(EVENTS)
print(f"  {len(df):,} rows")

outcomes = sorted(df["outcome"].unique())
bandwidths = [("global", None), ("|m|<=0.20", 0.20), ("|m|<=0.10", 0.10)]
employee_filters = ["current", "all"]

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: EVENT-LEVEL LINEAR RDD")
print("=" * 70)

def event_rdd(data, bw_val, weighted=False):
    """delta_y = a + tau*win + b1*margin + b2*win*margin"""
    sub = data.dropna(subset=["delta", "win", "margin"])
    if bw_val is not None:
        sub = sub[sub["abs_margin"] <= bw_val]
    if len(sub) < 20 or sub["win"].nunique() < 2:
        return None

    y = sub["delta"].values
    win = sub["win"].values.astype(float)
    margin = sub["margin"].values
    win_x_margin = win * margin

    X = sm.add_constant(np.column_stack([win, margin, win_x_margin]))
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
        vcov = res.cov_params()
        se = np.sqrt(np.diag(vcov))
        # HC1 adjustment
        adj = n / (n - k)
        se = se * np.sqrt(adj)

        tau = res.params[1]  # win coefficient
        se_tau = se[1]
        t = tau / se_tau if se_tau > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), df=n - k)

        return {
            "n_events": n, "n_gvkeys": int(sub["gvkey"].nunique()),
            "n_win": int(win.sum()), "n_loss": n - int(win.sum()),
            "tau": tau, "se": se_tau, "t_stat": t, "p_value": p,
            "mean_delta_win": float(y[win == 1].mean()) if win.sum() > 0 else np.nan,
            "mean_delta_loss": float(y[win == 0].mean()) if (1-win).sum() > 0 else np.nan,
            "sd_delta": float(y.std()), "rsquared": res.rsquared,
        }
    except Exception as e:
        return None

results = []
for oc in outcomes:
    for emp in employee_filters:
        for win_days in [365, 180, 90]:
            for th in df["threshold"].unique():
                for bw_label, bw_val in bandwidths:
                    for wgt in [False, True]:
                        mask = ((df["outcome"] == oc) & (df["employee_filter"] == emp) &
                                (df["window_days"] == win_days) & (df["threshold"] == th))
                        sub = df[mask]
                        if len(sub) < 20:
                            continue
                        res = event_rdd(sub, bw_val, weighted=wgt)
                        if res:
                            res.update({
                                "outcome": oc, "employee_filter": emp,
                                "window_days": win_days, "threshold": th,
                                "bandwidth_label": bw_label, "bandwidth_value": bw_val,
                                "weighted": wgt,
                            })
                            results.append(res)

df_er = pd.DataFrame(results)
df_er.to_csv(OUT / "event_level_linear_rdd_results.csv", index=False)
print(f"  Saved {len(df_er)} results")

# ── Main results table ──────────────────────────────────────────────
print("\n--- Event-level RDD: current, ±365d, pre>=1_post>=1, weighted ---")
best_mask = ((df_er["employee_filter"] == "current") & (df_er["window_days"] == 365) &
             (df_er["threshold"] == "pre>=1_post>=1") & (df_er["weighted"] == True))
for _, r in df_er[best_mask].iterrows():
    sig = "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.10 else ""
    print(f"  {r['outcome']:20s} | {r['bandwidth_label']:10s} | "
          f"τ={r['tau']:+.4f} (se={r['se']:.4f}), p={r['p_value']:.3f} "
          f"N={int(r['n_events'])} E={int(r['n_gvkeys'])} {sig}")

# Summarize sign consistency
print("\n--- Sign consistency (current, ±365d, pre>=1_post>=1) ---")
for oc in outcomes:
    sub = df_er[(df_er["outcome"] == oc) & (df_er["employee_filter"] == "current") &
                (df_er["window_days"] == 365) & (df_er["threshold"] == "pre>=1_post>=1")]
    if len(sub) < 3:
        continue
    g = sub[sub["bandwidth_label"] == "global"]
    b20 = sub[sub["bandwidth_label"] == "|m|<=0.20"]
    b10 = sub[sub["bandwidth_label"] == "|m|<=0.10"]
    tg = g["tau"].mean() if len(g) > 0 else np.nan
    t20 = b20["tau"].mean() if len(b20) > 0 else np.nan
    t10 = b10["tau"].mean() if len(b10) > 0 else np.nan
    signs = [np.sign(t) for t in [tg, t20, t10] if not np.isnan(t)]
    cons = "✓" if len(set(signs)) == 1 else "✗"
    p_vals = [b["p_value"].mean() for b in [g, b20, b10] if len(b) > 0]
    p_med = np.median(p_vals) if p_vals else np.nan
    print(f"  {oc:20s} | global={tg:+.4f} bw20={t20:+.4f} bw10={t10:+.4f} | "
          f"sign_ok={cons} | median_p={p_med:.3f} | n_global={int(g['n_events'].max()) if len(g)>0 else 0}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: MANUAL LOCAL-LINEAR RDD (rdrobust fallback)")
print("=" * 70)

ll_results = []
for oc in outcomes:
    for emp in employee_filters:
        for win_days in [365]:
            for th in df["threshold"].unique():
                mask = ((df["outcome"] == oc) & (df["employee_filter"] == emp) &
                        (df["window_days"] == win_days) & (df["threshold"] == th))
                sub = df[mask].dropna(subset=["delta", "margin"])
                if len(sub) < 50 or sub["win"].nunique() < 2:
                    continue

                y = sub["delta"].values
                x = sub["margin"].values
                n = len(y)

                # Silverman-like bandwidth
                h = 1.84 * np.std(x) * n**(-1/5)
                h = min(h, 0.30)

                # Find optimal h via simple CV
                for h_try in [h, 0.20, 0.15, 0.10]:
                    mask_h = np.abs(x) <= h_try
                    if mask_h.sum() < 30:
                        continue
                    x_h, y_h = x[mask_h], y[mask_h]
                    win_h = (x_h > 0).astype(float)
                    # Triangular kernel
                    w = 1 - np.abs(x_h) / h_try
                    w = w / w.sum() * len(w)

                    X_h = sm.add_constant(np.column_stack([win_h, x_h, win_h * x_h]))
                    try:
                        mod_h = sm.WLS(y_h, X_h, weights=w).fit()
                        tau_h = mod_h.params[1]
                        se_h = np.sqrt(mod_h.cov_params()[1, 1])
                        p_h = 2 * stats.t.sf(abs(tau_h / se_h), df=len(y_h)-4) if se_h > 0 else np.nan

                        ll_results.append({
                            "outcome": oc, "employee_filter": emp,
                            "window_days": win_days, "threshold": th,
                            "bandwidth": f"{h_try:.2f}",
                            "n_effective": int(mask_h.sum()),
                            "n_gvkeys": int(sub.loc[mask_h, "gvkey"].nunique()),
                            "tau": tau_h, "se": se_h,
                            "p_value": p_h,
                            "n_left": int((x_h < 0).sum()),
                            "n_right": int((x_h >= 0).sum()),
                        })
                    except:
                        pass

df_ll = pd.DataFrame(ll_results)
df_ll.to_csv(OUT / "rdrobust_event_level_results.csv", index=False)
print(f"  Saved {len(df_ll)} local-linear results")

print("\n--- Local-linear RDD (current, ±365d, pre>=1_post>=1) ---")
for oc in outcomes:
    sub = df_ll[(df_ll["outcome"] == oc) & (df_ll["employee_filter"] == "current") &
                (df_ll["window_days"] == 365) & (df_ll["threshold"] == "pre>=1_post>=1")]
    if len(sub) > 0:
        # Pick h closest to 0.20
        best = sub.iloc[(sub["bandwidth"].astype(float) - 0.20).abs().argsort().iloc[0]]
        sig = "**" if best["p_value"] < 0.05 else "*" if best["p_value"] < 0.10 else ""
        print(f"  {oc:20s} | h={best['bandwidth']} | "
              f"τ={best['tau']:+.4f} (se={best['se']:.4f}), p={best['p_value']:.3f} "
              f"N={int(best['n_effective'])} {sig}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: STABILITY SUMMARY")
print("=" * 70)

# Build summary
summary_rows = []
for oc in outcomes:
    for emp in employee_filters:
        sub = df_er[(df_er["outcome"] == oc) & (df_er["employee_filter"] == emp) &
                    (df_er["window_days"] == 365) & (df_er["threshold"] == "pre>=1_post>=1")]
        if len(sub) < 3:
            continue

        g = sub[sub["bandwidth_label"] == "global"]
        b20 = sub[sub["bandwidth_label"] == "|m|<=0.20"]
        b10 = sub[sub["bandwidth_label"] == "|m|<=0.10"]

        tg = g["tau"].mean() if len(g) > 0 else np.nan
        t20 = b20["tau"].mean() if len(b20) > 0 else np.nan
        t10 = b10["tau"].mean() if len(b10) > 0 else np.nan

        signs = [np.sign(t) for t in [tg, t20, t10] if not np.isnan(t)]
        sign_cons = len(set(signs)) == 1

        p_vals = sub["p_value"].dropna()
        median_p = p_vals.median() if len(p_vals) > 0 else np.nan
        n_sig5 = (p_vals < 0.05).sum()
        n_sig10 = (p_vals < 0.10).sum()

        summary_rows.append({
            "outcome": oc, "employee_filter": emp,
            "global_tau": tg, "bw20_tau": t20, "bw10_tau": t10,
            "sign_consistent": sign_cons,
            "sign_direction": "negative" if tg < 0 else "positive",
            "n_sig5": n_sig5, "n_sig10": n_sig10, "median_p": median_p,
            "max_n_events": int(sub["n_events"].max()),
            "max_n_gvkeys": int(sub["n_gvkeys"].max()),
        })

df_sum = pd.DataFrame(summary_rows).sort_values("median_p")
df_sum.to_csv(OUT / "rdd_rebuild_outcome_summary.csv", index=False)

print("\nStability summary (current employees, ±365d, pre>=1_post>=1):")
for _, r in df_sum.iterrows():
    if r["employee_filter"] == "current":
        cons = "✓" if r["sign_consistent"] else "✗"
        print(f"  {r['outcome']:20s} | global={r['global_tau']:+.4f} bw20={r['bw20_tau']:+.4f} bw10={r['bw10_tau']:+.4f} | "
              f"sign={cons} | sig5={int(r['n_sig5'])} sig10={int(r['n_sig10'])} | "
              f"p_med={r['median_p']:.3f} | E={int(r['max_n_events'])}")

# Best candidate
candidates = df_sum[(df_sum["sign_consistent"]) & (df_sum["max_n_gvkeys"] >= 20)]
if len(candidates) > 0:
    best = candidates.sort_values("median_p").iloc[0]
    print(f"\n★★★ Best candidate: {best['outcome']} ({best['employee_filter']}) ★★★")
    print(f"  global τ={best['global_tau']:.4f}, bw20 τ={best['bw20_tau']:.4f}, bw10 τ={best['bw10_tau']:.4f}")
    print(f"  median p={best['median_p']:.3f}")
else:
    print("\nWARNING: No outcome passes all consistency checks!")
    # Try without sign consistency
    fallback = df_sum.sort_values("median_p").iloc[0]
    print(f"Best by p-value: {fallback['outcome']} (p={fallback['median_p']:.3f}, sign_consistent={fallback['sign_consistent']})")

# Print diversity assessment
print(f"\nDiversity & Inclusion assessment:")
div_data = df_er[(df_er["outcome"] == "diversity") & (df_er["employee_filter"] == "current") &
                 (df_er["window_days"] == 365) & (df_er["threshold"] == "pre>=1_post>=1")]
if len(div_data) > 0:
    print(f"  Global: τ={div_data[div_data['bandwidth_label']=='global']['tau'].mean():.4f}, "
          f"N={int(div_data[div_data['bandwidth_label']=='global']['n_events'].max())}, "
          f"gvkeys={int(div_data[div_data['bandwidth_label']=='global']['n_gvkeys'].max())}")
    # Compare to overall_rating
    rating_data = df_er[(df_er["outcome"] == "overall_rating") & (df_er["employee_filter"] == "current") &
                        (df_er["window_days"] == 365) & (df_er["threshold"] == "pre>=1_post>=1")]
    if len(rating_data) > 0:
        print(f"  Overall rating global: τ={rating_data[rating_data['bandwidth_label']=='global']['tau'].mean():.4f}, "
              f"N={int(rating_data[rating_data['bandwidth_label']=='global']['n_events'].max())}, "
              f"gvkeys={int(rating_data[rating_data['bandwidth_label']=='global']['n_gvkeys'].max())}")

# Save stability grid
df_er["estimator"] = "event_level_linear_rdd"
cols_to_save = ["estimator", "outcome", "employee_filter", "window_days", "threshold",
                "bandwidth_label", "bandwidth_value", "weighted",
                "n_events", "n_gvkeys", "n_win", "n_loss",
                "tau", "se", "t_stat", "p_value",
                "mean_delta_win", "mean_delta_loss", "sd_delta"]
df_er[[c for c in cols_to_save if c in df_er.columns]].to_csv(
    OUT / "rdd_rebuild_stability_grid.csv", index=False)

print(f"\n{'=' * 70}")
print("EVENT-LEVEL RDD COMPLETE")
print(f"Results: {OUT}/event_level_linear_rdd_results.csv")
print(f"Summary: {OUT}/rdd_rebuild_outcome_summary.csv")
print(f"Local-linear: {OUT}/rdrobust_event_level_results.csv")
