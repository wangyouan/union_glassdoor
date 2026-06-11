#!/usr/bin/env python
"""
02b_firm_year_rdd_regressions.py
==================================
Firm-year RDD regressions for union election effects.

Uses union_margin (vote share - 0.5) as the running variable and
win_union = I(margin > 0) as the treatment indicator.

Models (global polynomial, order p = 1,2,3 for each):
  RDD1: Outcome ~ WinUnion + poly(margin) + WinUnion × poly(margin)
  RDD2: RDD1 + lagged controls (size, leverage, ROA, MTB, sales_growth)
  RDD3: RDD1 + industry FE (sic2 or ff48)
  RDD4: DiD-RDD:  ΔOutcome ~ WinUnion + poly(margin)
  RDD5: ANCOVA-RDD: Outcome_post ~ WinUnion + poly(margin) + Outcome_pre

All specifications report linear (p=1), quadratic (p=2), and cubic (p=3)
global polynomials. The preferred specification is the one with the
lowest AIC / BIC, with cubic as the default for global RDD.

Outputs:
  outputs/analysis_stability/firm_year_rdd_results.csv
  outputs/analysis_stability/firm_year_rdd_summary.csv
  outputs/analysis_stability/figures/rdd_*_plot.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs/analysis_stability"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FIRMYEAR_FILE = PROJ / "outputs/union_glassdoor_firm_year_regression.parquet"

# ── Load data ───────────────────────────────────────────────────────────
print("=" * 70)
print("Loading firm-year data for RDD analysis...")
df = pd.read_parquet(FIRMYEAR_FILE)
print(f"  Shape: {df.shape}")

# ── Key variables ───────────────────────────────────────────────────────
# Running variable: union_margin (centered at 0)
# Treatment: win_union (= 1 if union_margin > 0)
running_var = "union_margin"
treatment_var = "win_union"

# Outcomes (election year)
outcomes = {
    "GD_rating": "Overall Rating",
    "GD_career_opp": "Career Opportunities",
    "GD_comp_benefit": "Compensation & Benefits",
    "GD_senior_mgmt": "Senior Management",
    "GD_culture": "Culture & Values",
}

# Outcomes for DiD (change score)
outcomes_did = {
    "GD_rating": ("GD_rating_for1", "GD_rating_lag1"),
    "GD_career_opp": ("GD_career_opp_for1", "GD_career_opp_lag1"),
    "GD_comp_benefit": ("GD_comp_benefit_for1", "GD_comp_benefit_lag1"),
    "GD_senior_mgmt": ("GD_senior_mgmt_for1", "GD_senior_mgmt_lag1"),
    "GD_culture": ("GD_culture_for1", "GD_culture_lag1"),
}

# Lagged controls
lag_controls = {
    "L_size": "Size (lag)",
    "L_leverage": "Leverage (lag)",
    "L_roa": "ROA (lag)",
    "L_book_to_market": "MTB (lag)",
    "L_sales_growth": "Sales Growth (lag)",
}

# ── Prepare data ────────────────────────────────────────────────────────
print("\nPreparing data...")
df[running_var] = df[running_var].astype(float)
df[treatment_var] = df[treatment_var].astype(int)

# Standardize outcomes for comparability
for oc in outcomes:
    mu = df[oc].mean()
    sd = df[oc].std()
    df[f"{oc}_sd"] = (df[oc] - mu) / sd

# DiD outcomes: post - pre
for oc, (post_col, pre_col) in outcomes_did.items():
    if post_col in df.columns and pre_col in df.columns:
        df[f"{oc}_diff"] = df[post_col] - df[pre_col]
        # Standardize diff
        mu_d = df[f"{oc}_diff"].mean()
        sd_d = df[f"{oc}_diff"].std()
        df[f"{oc}_diff_sd"] = (df[f"{oc}_diff"] - mu_d) / sd_d

# Standardize running variable for numerical stability
margin_mean = df[running_var].mean()
margin_sd = df[running_var].std()
df["margin_std"] = (df[running_var] - margin_mean) / margin_sd

# Industry FE
if "sic2" in df.columns:
    df["sic2_str"] = df["sic2"].astype(str)
    ind_col = "sic2_str"
elif "ff48" in df.columns:
    df["ff48_str"] = df["ff48"].astype(str)
    ind_col = "ff48_str"
else:
    ind_col = None

print(f"  Running var: {running_var}, mean={margin_mean:.4f}, sd={margin_sd:.4f}")
print(f"  Treatment: {treatment_var}, N_treated={df[treatment_var].sum()}, N_control={(1-df[treatment_var]).sum()}")
print(f"  Industry FE: {ind_col}")

# ═════════════════════════════════════════════════════════════════════════
# RDD REGRESSION FUNCTION
# ═════════════════════════════════════════════════════════════════════════

def run_rdd_regression(y_var, data, poly_order=3, controls=None, industry_fe=False,
                       did_mode=False, ancova_pre=None):
    """
    Run RDD with global polynomial of order `poly_order`.

    Parameters
    ----------
    y_var : str — outcome variable name
    data : DataFrame
    poly_order : int — polynomial order (1, 2, or 3)
    controls : list of str or None — control variable names
    industry_fe : bool — include industry dummies
    did_mode : bool — use change score (no treatment×poly interaction)
    ancova_pre : str or None — pre-outcome variable for ANCOVA

    Returns
    -------
    dict with results
    """
    needed = [y_var, running_var, treatment_var]
    if controls:
        needed.extend(controls)
    if industry_fe and ind_col:
        needed.append(ind_col)
    if ancova_pre:
        needed.append(ancova_pre)

    subset = data.dropna(subset=needed).copy()
    if len(subset) < 30:
        return {"N": len(subset), "coef_win_union": np.nan, "se_robust": np.nan}

    N = len(subset)

    # Build polynomial terms of margin_std
    margin = subset["margin_std"].values
    win = subset[treatment_var].values.astype(float)

    X_vars = []
    # Polynomial terms
    for p in range(1, poly_order + 1):
        col_name = f"poly_{p}"
        subset[col_name] = margin ** p
        X_vars.append(col_name)

    if not did_mode:
        # Interaction terms
        for p in range(1, poly_order + 1):
            int_name = f"win_x_poly_{p}"
            subset[int_name] = win * subset[f"poly_{p}"]
            X_vars.append(int_name)

    # Treatment
    if not did_mode:
        X_vars = X_vars + [treatment_var]  # win_union

    # Controls
    if controls:
        for cc in controls:
            if cc in subset.columns and subset[cc].notna().sum() > 0:
                # Standardize controls
                c_mu = subset[cc].mean()
                c_sd = subset[cc].std()
                if c_sd > 0:
                    subset[f"{cc}_std"] = (subset[cc] - c_mu) / c_sd
                    X_vars.append(f"{cc}_std")

    # Industry FE
    if industry_fe and ind_col and ind_col in subset.columns:
        ind_dummies = pd.get_dummies(subset[ind_col], prefix="ind", drop_first=True)
        for dc in ind_dummies.columns:
            subset[dc] = ind_dummies[dc].values
            X_vars.append(dc)

    # ANCOVA: add pre-outcome
    if ancova_pre and ancova_pre in subset.columns:
        pre_mu = subset[ancova_pre].mean()
        pre_sd = subset[ancova_pre].std()
        if pre_sd > 0:
            subset["_pre_outcome_std"] = (subset[ancova_pre] - pre_mu) / pre_sd
            X_vars.append("_pre_outcome_std")

    # Drop any columns with all NaN
    X_vars = [c for c in X_vars if c in subset.columns and subset[c].notna().all()]

    # Build design matrix (sm.add_constant adds const as column 0)
    X = sm.add_constant(subset[X_vars].values)
    y = subset[y_var].values

    try:
        mod = sm.OLS(y, X).fit()

        # Find win_union index in params
        # params order: [const, X_vars[0], X_vars[1], ..., X_vars[k]]
        # So win_union is at position 1 + index of treatment_var in X_vars
        win_idx = None
        for i, v in enumerate(X_vars):
            if v == treatment_var:
                win_idx = 1 + i  # +1 for const added by sm.add_constant
                break

        if win_idx is not None and win_idx < len(mod.params):
            beta = mod.params[win_idx]
            # HC2 robust SE
            resid = mod.resid.reshape(-1, 1)
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                h = np.sum(X @ XtX_inv * X, axis=1)  # leverage
                h = np.clip(h, 0, 0.99)
                # HC2: bread = (X'X)^-1, meat = X' diag(e^2/(1-h)) X
                bread = XtX_inv
                meat = X.T @ (X * (resid**2 / (1 - h).reshape(-1, 1)))
                hc2_vcov = bread @ meat @ bread
                se = np.sqrt(np.diag(hc2_vcov))[win_idx]
            except (np.linalg.LinAlgError, ValueError):
                se = mod.bse[win_idx]
            t_stat = beta / se if se > 0 and not np.isnan(se) else np.nan
            pval = 2 * stats.t.sf(abs(t_stat), df=N - len(mod.params)) if not np.isnan(t_stat) else np.nan
        else:
            beta, se, t_stat, pval = np.nan, np.nan, np.nan, np.nan

        return {
            "N": N,
            "N_treated": int(win.sum()),
            "N_control": N - int(win.sum()),
            "coef_win_union": beta,
            "se_robust": se,
            "t_stat": t_stat,
            "p_value": pval,
            "ci_low": beta - 1.96 * se if not np.isnan(beta) else np.nan,
            "ci_high": beta + 1.96 * se if not np.isnan(beta) else np.nan,
            "rsquared": mod.rsquared,
            "rsquared_adj": mod.rsquared_adj,
            "aic": mod.aic,
            "bic": mod.bic,
            "f_stat": mod.fvalue,
            "n_params": len(mod.params),
            "mean_y": y.mean(),
            "sd_y": y.std(),
        }

    except Exception as e:
        return {"N": N, "coef_win_union": np.nan, "se_robust": np.nan, "error": str(e)[:100]}

# ═════════════════════════════════════════════════════════════════════════
# RUN ALL RDD SPECIFICATIONS
# ═════════════════════════════════════════════════════════════════════════

all_results = []

# Define specifications to run
specs = {
    "RDD1": {"controls": None, "industry_fe": False, "did_mode": False, "ancova_pre": None},
    "RDD2": {"controls": list(lag_controls.keys()), "industry_fe": False, "did_mode": False, "ancova_pre": None},
    "RDD3": {"controls": None, "industry_fe": True, "did_mode": False, "ancova_pre": None},
    "RDD4": {"controls": None, "industry_fe": False, "did_mode": True, "ancova_pre": None},
    "RDD5": {"controls": None, "industry_fe": False, "did_mode": False, "ancova_pre": None},  # ancova_pre set per outcome
}

# Outcomes to use
# For RDD1-RDD3: use standardized main-period outcomes
# For RDD4 (DiD): use diff_sd outcomes
# For RDD5 (ANCOVA): use for1 outcome with lag1 as control

for model_name, spec_base in specs.items():
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print("=" * 70)

    for poly_order in [1, 2, 3]:
        print(f"\n  --- Polynomial order = {poly_order} ---")

        for oc_short, oc_label in outcomes.items():
            if model_name == "RDD5":
                # ANCOVA: post outcome ~ treatment + poly + pre outcome
                post_col, pre_col = outcomes_did[oc_short]
                if post_col not in df.columns or pre_col not in df.columns:
                    continue
                # Standardize post outcome
                mu_post = df[post_col].mean()
                sd_post = df[post_col].std()
                if sd_post == 0:
                    continue
                df[f"{post_col}_sd"] = (df[post_col] - mu_post) / sd_post
                y_var = f"{post_col}_sd"
                spec = {**spec_base, "ancova_pre": pre_col}
                spec["controls"] = None  # No extra controls in ANCOVA
            elif model_name == "RDD4":
                y_var = f"{oc_short}_diff_sd"
                spec = spec_base
            else:
                y_var = f"{oc_short}_sd"
                spec = spec_base

            if y_var not in df.columns or df[y_var].notna().sum() < 20:
                continue

            res = run_rdd_regression(
                y_var, df,
                poly_order=poly_order,
                controls=spec["controls"],
                industry_fe=spec["industry_fe"],
                did_mode=spec["did_mode"],
                ancova_pre=spec.get("ancova_pre"),
            )

            res.update({
                "outcome": oc_short,
                "outcome_label": oc_label,
                "y_var": y_var,
                "model": model_name,
                "poly_order": poly_order,
                "spec_description": {
                    "RDD1": "Baseline RDD (no controls)",
                    "RDD2": "RDD + lagged firm controls",
                    "RDD3": "RDD + industry FE",
                    "RDD4": f"DiD-RDD: Δ({oc_short}) = post - pre",
                    "RDD5": f"ANCOVA-RDD: post ~ treatment + poly + pre({oc_short})",
                }.get(model_name, model_name),
            })

            all_results.append(res)

            if not np.isnan(res.get("coef_win_union", np.nan)):
                print(f"    {oc_short}: β={res['coef_win_union']:.4f}, "
                      f"se={res['se_robust']:.4f}, p={res['p_value']:.3f}, "
                      f"N={res['N']}, AIC={res['aic']:.1f}")

# ═════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)

# Economic magnitude: coefficient is already in SD units for RDD1-RDD3, RDD5
# For DiD (RDD4), the diff is also standardized
results_df["economic_magnitude_sd"] = results_df["coef_win_union"]
results_df["sign"] = np.sign(results_df["coef_win_union"])
results_df["significant_10"] = results_df["p_value"] < 0.10
results_df["significant_5"] = results_df["p_value"] < 0.05
results_df["significant_1"] = results_df["p_value"] < 0.01

results_df.to_csv(OUT / "firm_year_rdd_results.csv", index=False)
print(f"\n{'=' * 70}")
print(f"Saved {len(results_df)} RDD results to firm_year_rdd_results.csv")

# ═════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RDD Summary: Best specification per outcome (lowest AIC)")

summary_rows = []
for oc_short in outcomes:
    for model in ["RDD1", "RDD2", "RDD3", "RDD5"]:
        sub = results_df[(results_df["outcome"] == oc_short) & (results_df["model"] == model)]
        if len(sub) == 0:
            continue
        # Pick best polynomial order by AIC
        best = sub.loc[sub["aic"].idxmin()] if sub["aic"].notna().any() else sub.iloc[0]
        summary_rows.append({
            "outcome": oc_short,
            "outcome_label": outcomes[oc_short],
            "model": model,
            "best_poly_order": int(best["poly_order"]),
            "N": int(best["N"]),
            "coef_win_union": best["coef_win_union"],
            "se_robust": best["se_robust"],
            "p_value": best["p_value"],
            "ci_low": best["ci_low"],
            "ci_high": best["ci_high"],
            "aic": best["aic"],
            "significant_5": best["p_value"] < 0.05,
            "rsquared": best["rsquared"],
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "firm_year_rdd_summary.csv", index=False)
print(summary_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
# RDD PLOTS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating RDD plots...")

# For each outcome, plot with cubic polynomial (p=3) on the RDD1 model
for oc_short, oc_label in outcomes.items():
    y_var = f"{oc_short}_sd"
    plot_data = df.dropna(subset=[y_var, running_var, treatment_var]).copy()

    if len(plot_data) < 20:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Scatter + global cubic fit ---
    ax = axes[0]
    margin = plot_data[running_var].values
    y = plot_data[y_var].values
    win = plot_data[treatment_var].values

    # Fit cubic separately on each side
    margin_std = (margin - margin_mean) / margin_sd

    for side_sign, color, label in [(-1, "#d73027", "Union Lost"), (1, "#4575b4", "Union Won")]:
        mask = (np.sign(margin) == side_sign) if side_sign < 0 else (margin > 0)
        if mask.sum() < 10:
            continue
        m_side = margin_std[mask]
        y_side = y[mask]

        # Fit cubic
        X_poly = np.column_stack([m_side**p for p in range(1, 4)])
        X_fit = sm.add_constant(X_poly)
        mod_side = sm.OLS(y_side, X_fit).fit()
        y_pred = mod_side.fittedvalues

        # Sort for plotting
        sort_idx = np.argsort(m_side)
        ax.scatter(margin[mask], y[mask], alpha=0.3, s=10, color=color, label=f"{label} (N={mask.sum()})")
        ax.plot(margin[mask][sort_idx], y_pred[sort_idx], color=color, linewidth=2)

    # Add vertical line at cutoff
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)

    # Add discontinuity estimate
    rdd1_cubic = results_df[(results_df["outcome"] == oc_short) &
                              (results_df["model"] == "RDD1") &
                              (results_df["poly_order"] == 3)]
    if len(rdd1_cubic) > 0 and not np.isnan(rdd1_cubic.iloc[0]["coef_win_union"]):
        r = rdd1_cubic.iloc[0]
        ax.annotate(f"RDD estimate: {r['coef_win_union']:.3f} SD\n"
                    f"se={r['se_robust']:.3f}, p={r['p_value']:.3f}\n"
                    f"N={int(r['N'])}",
                    xy=(0.02, 0.98), xycoords="axes fraction",
                    va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Union Vote Margin (0 = 50% threshold)")
    ax.set_ylabel("Rating (standardized)")
    ax.set_title(f"RDD: {oc_label}\nGlobal Cubic Polynomial", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    # --- Panel 2: Binned means ± threshold ---
    ax = axes[1]
    # Create bins of the running variable
    n_bins = 20
    plot_data["margin_bin"] = pd.qcut(plot_data[running_var], n_bins, labels=False, duplicates="drop")
    binned = plot_data.groupby("margin_bin").agg(
        margin_mean=(running_var, "mean"),
        y_mean=(y_var, "mean"),
        y_se=(y_var, "sem"),
        n=("gvkey", "count"),
        win_share=(treatment_var, "mean"),
    ).reset_index()

    colors = ["#d73027" if w < 0.5 else "#4575b4" for w in binned["win_share"]]
    ms_vals = [max(1, float(n)/20) for n in binned["n"].values]
    ax.errorbar(binned["margin_mean"], binned["y_mean"], yerr=1.96 * binned["y_se"].values,
                fmt="o", capsize=3, alpha=0.7, color="gray", markersize=4)
    ax.scatter(binned["margin_mean"], binned["y_mean"], c=colors, s=[max(5, float(n)/2) for n in binned["n"].values], alpha=0.8, edgecolors="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Union Vote Margin (0 = 50% threshold)")
    ax.set_ylabel("Rating (standardized)")
    ax.set_title(f"Binned Means (±2 SE)\nMarker size ∝ N reviews", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG / f"rdd_{oc_short}_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: rdd_{oc_short}_plot.png")

# ═════════════════════════════════════════════════════════════════════════
# SPECIFICATION CHOICE TABLE (polynomial order comparison for RDD1)
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Polynomial order comparison (RDD1):")

poly_compare = results_df[(results_df["model"] == "RDD1")].pivot_table(
    index=["outcome", "poly_order"],
    values=["coef_win_union", "se_robust", "p_value", "aic", "bic", "N"],
    aggfunc="first"
).reset_index()

print(poly_compare.to_string(index=False))
poly_compare.to_csv(OUT / "firm_year_rdd_poly_comparison.csv", index=False)

print("\n" + "=" * 70)
print("02b_firm_year_rdd_regressions COMPLETE.")
print(f"Results: {OUT}/firm_year_rdd_results.csv")
print(f"Summary:  {OUT}/firm_year_rdd_summary.csv")
print(f"Plots:    {FIG}/rdd_*_plot.png")
