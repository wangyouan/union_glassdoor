#!/usr/bin/env python
"""
02_firm_year_regressions.py
================================
Firm-year panel regressions for union election effects.

Strategy: Reshape firm-year data from wide (pre/main/post per election)
to long format (pre/post per election), then run DiD regressions.

Models:
  FY1: Outcome ~ PostElection + firm FE + year FE
  FY2: Outcome ~ PostElection * UnionWin + firm FE + year FE
  FY3: FY2 + controls (size, leverage, ROA, etc.)
  FY4: Event-time bins around election year

Thresholds: no threshold, >=1, >=3, >=5, >=10 reviews

Outputs:
  outputs/analysis_stability/firm_year_regression_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from linearmodels.iv.absorbing import AbsorbingLS

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "analysis_stability"
FIRMYEAR_FILE = PROJ / "outputs" / "union_glassdoor_firm_year_regression.parquet"

# ── Load data ───────────────────────────────────────────────────────────
print("Loading firm-year data...")
df = pd.read_parquet(FIRMYEAR_FILE)
print(f"  Shape: {df.shape}")

# ── Identify outcome variables ──────────────────────────────────────────
# Core outcomes (raw, election year)
core_outcomes = ["GD_rating", "GD_career_opp", "GD_comp_benefit", "GD_senior_mgmt",
                 "GD_culture", "GD_wlb", "GD_diversity"]
# Check which are in data
available = [c for c in core_outcomes if c in df.columns]
print(f"  Available outcomes: {available}")

# Check for work-life balance naming
for c in df.columns:
    if "wlb" in c.lower() or "work_life" in c.lower() or "worklife" in c.lower():
        if "GD_" in c and "_lag1" not in c and "_for1" not in c and "_sd" not in c and "_n_" not in c:
            if c not in available:
                available.append(c)
print(f"  All outcomes to use: {available}")

# ── Reshape to long format ──────────────────────────────────────────────
print("\nReshaping to long format (pre/post per election)...")

# We need: for each election, pre (lag1) and post (for1) observations
id_cols = ["election_id", "case_number", "gvkey", "election_date", "election_year",
           "votes_for_union", "votes_against_union", "total_valid_votes",
           "union_support_rate", "win_union", "lose_union", "union_tie", "union_margin",
           "close_election_abs_margin"]

# Verify id_cols exist
id_cols = [c for c in id_cols if c in df.columns]

# Review count variables (for threshold)
review_count_vars = [c for c in df.columns if c.startswith("n_reviews") or c.startswith("n_GD_")]
# Also check for n_<outcome> pattern
n_vars = {}
for oc in available:
    oc_short = oc.replace("GD_", "")
    n_col = f"n_GD_{oc_short}"
    if n_col in df.columns:
        n_vars[oc] = n_col
    # Also check lag1 / for1
    n_col_lag1 = f"n_GD_{oc_short}_lag1"
    n_col_for1 = f"n_GD_{oc_short}_for1"
    if n_col_lag1 in df.columns:
        n_vars[f"{oc}_lag1"] = n_col_lag1
    if n_col_for1 in df.columns:
        n_vars[f"{oc}_for1"] = n_col_for1

# Use total review count if no outcome-specific count
if "n_reviews" in df.columns:
    total_review_var = "n_reviews"
elif "n_reviews_lag1" in df.columns:
    total_review_var = "n_reviews_lag1"
else:
    total_review_var = None

print(f"  Review count variables: outcome-specific={list(n_vars.keys())[:5]}...")
print(f"  Total review var: {total_review_var}")

# Build long-format data
long_rows = []
for _, row in df.iterrows():
    eid = row["election_id"]
    gvkey = row["gvkey"]

    # Pre period (lag1)
    pre_row = {"period": "pre", "post": 0, "election_id": eid}
    for c in id_cols:
        pre_row[c] = row[c]
    # Add gd_year for pre
    pre_row["gd_year"] = row.get("gd_year_lag1", np.nan)
    # Add review counts
    if total_review_var:
        pre_row["n_reviews"] = row.get(f"{total_review_var}", np.nan)
    # Add outcomes
    for oc in available:
        pre_row[oc] = row.get(f"{oc}_lag1", np.nan)
        # Also add outcome-specific review count
        oc_short = oc.replace("GD_", "")
        pre_row[f"n_{oc}"] = row.get(f"n_GD_{oc_short}_lag1", np.nan)
    # Add controls (use lagged)
    for cc in ["size", "log_me", "leverage", "cash_ratio", "roa", "profitability",
               "tangibility", "capx_at", "rd_at", "book_to_market", "sales_growth", "log_emp"]:
        pre_row[cc] = row.get(f"L_{cc}", np.nan) if f"L_{cc}" in df.columns else row.get(cc, np.nan)
    long_rows.append(pre_row)

    # Post period (for1)
    post_row = {"period": "post", "post": 1, "election_id": eid}
    for c in id_cols:
        post_row[c] = row[c]
    post_row["gd_year"] = row.get("gd_year_for1", np.nan)
    if total_review_var:
        # Use main n_reviews for post (if no for1 count)
        n_var_for1 = f"{total_review_var}_for1"
        post_row["n_reviews"] = row.get(n_var_for1 if n_var_for1 in df.columns else total_review_var, np.nan)
    for oc in available:
        post_row[oc] = row.get(f"{oc}_for1", np.nan)
        oc_short = oc.replace("GD_", "")
        post_row[f"n_{oc}"] = row.get(f"n_GD_{oc_short}_for1", np.nan)
    for cc in ["size", "log_me", "leverage", "cash_ratio", "roa", "profitability",
               "tangibility", "capx_at", "rd_at", "book_to_market", "sales_growth", "log_emp"]:
        post_row[cc] = row.get(cc, np.nan)
    long_rows.append(post_row)

df_long = pd.DataFrame(long_rows)
print(f"  Long format: {len(df_long)} rows ({len(df_long)//2} elections × 2 periods)")

# Standardize outcomes
print("\nStandardizing outcomes...")
for oc in available:
    mu = df_long[oc].mean()
    sd = df_long[oc].std()
    df_long[f"{oc}_sd"] = (df_long[oc] - mu) / sd
    print(f"  {oc}_sd: mean={df_long[f'{oc}_sd'].mean():.4f}, n={df_long[oc].notna().sum()}")

# Also create standardized versions of subpopulation outcomes
# (Only for all_reviews for now; subpopulations explored later)

# ── Regression function ─────────────────────────────────────────────────
def run_fy_regression(data, y_var, exog_vars, absorb_vars, cluster_var="gvkey"):
    """Run absorbing regression for firm-year data."""
    needed = [y_var] + absorb_vars + exog_vars
    if cluster_var:
        needed.append(cluster_var)
    subset = data.dropna(subset=needed)

    if len(subset) < 20:
        return {"N": len(subset), "coef": np.nan, "se": np.nan, "p_value": np.nan}

    y = subset[y_var].values
    X = subset[exog_vars].values.astype(float)

    absorb_df = subset[absorb_vars].copy()
    for av in absorb_vars:
        absorb_df[av] = absorb_df[av].astype(str)

    try:
        mod = AbsorbingLS(y, X, absorb=absorb_df, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", clusters=subset[cluster_var].values)

        results = {"N": len(subset), "N_firms": subset["gvkey"].nunique(),
                   "N_events": subset["election_id"].nunique()}
        for i, var_name in enumerate(exog_vars):
            results[f"coef_{var_name}"] = res.params[i]
            results[f"se_{var_name}"] = res.std_errors[i]
            results[f"t_stat_{var_name}"] = res.tstats[i]
            results[f"p_value_{var_name}"] = res.pvalues[i]
            results[f"ci_low_{var_name}"] = res.params[i] - 1.96 * res.std_errors[i]
            results[f"ci_high_{var_name}"] = res.params[i] + 1.96 * res.std_errors[i]

        # For single-variable case, also set top-level coef/se
        if len(exog_vars) == 1:
            results["coef"] = results[f"coef_{exog_vars[0]}"]
            results["se"] = results[f"se_{exog_vars[0]}"]
            results["t_stat"] = results[f"t_stat_{exog_vars[0]}"]
            results["p_value"] = results[f"p_value_{exog_vars[0]}"]
            results["ci_low"] = results[f"ci_low_{exog_vars[0]}"]
            results["ci_high"] = results[f"ci_high_{exog_vars[0]}"]

        results["mean_y"] = y.mean()
        results["sd_y"] = y.std()
        return results
    except Exception as e:
        return {"N": len(subset), "coef": np.nan, "se": np.nan, "p_value": np.nan, "error": str(e)}

# ═════════════════════════════════════════════════════════════════════════
# RUN FY1-FY4
# ═════════════════════════════════════════════════════════════════════════

thresholds = [0, 1, 3, 5, 10]
all_results = []

for min_reviews in thresholds:
    print(f"\n{'=' * 70}")
    print(f"THRESHOLD: >= {min_reviews} reviews")
    print("=" * 70)

    for oc_sd in [f"{oc}_sd" for oc in available]:
        oc_base = oc_sd.replace("_sd", "")

        # Filter by review count
        n_col = f"n_{oc_base}"
        if n_col in df_long.columns:
            mask = df_long[n_col] >= min_reviews
        elif "n_reviews" in df_long.columns:
            mask = df_long["n_reviews"] >= min_reviews
        else:
            mask = pd.Series(True, index=df_long.index)

        df_sub = df_long[mask].copy()
        if len(df_sub) < 20:
            continue

        # Standardize within this subsample
        mu_sub = df_sub[oc_base].mean()
        sd_sub = df_sub[oc_base].std()
        if sd_sub > 0:
            df_sub[f"{oc_base}_sd_sub"] = (df_sub[oc_base] - mu_sub) / sd_sub
        else:
            continue

        y_var = f"{oc_base}_sd_sub"

        # ── FY1: PostElection only ──────────────────────────────────
        res_fy1 = run_fy_regression(
            df_sub, y_var,
            exog_vars=["post"],
            absorb_vars=["gvkey", "gd_year"],
            cluster_var="gvkey"
        )
        res_fy1.update({
            "outcome": oc_sd, "model": "FY1", "min_reviews_threshold": min_reviews,
            "sample": "all", "analysis_level": "firm-year", "window": "[-1,+1] years"
        })
        all_results.append(res_fy1)

        # ── FY2: Post × UnionWin interaction ────────────────────────
        df_sub["post_x_win"] = df_sub["post"] * df_sub["win_union"].astype(float)
        res_fy2 = run_fy_regression(
            df_sub, y_var,
            exog_vars=["post", "post_x_win"],
            absorb_vars=["gvkey", "gd_year"],
            cluster_var="gvkey"
        )
        res_fy2.update({
            "outcome": oc_sd, "model": "FY2", "min_reviews_threshold": min_reviews,
            "sample": "all", "analysis_level": "firm-year", "window": "[-1,+1] years"
        })
        all_results.append(res_fy2)

        # ── FY3: With controls ──────────────────────────────────────
        control_vars = ["post"]
        for cc in ["size", "leverage", "roa", "book_to_market", "sales_growth"]:
            if cc in df_sub.columns and df_sub[cc].notna().sum() > 100:
                # Standardize control
                c_mu = df_sub[cc].mean()
                c_sd = df_sub[cc].std()
                if c_sd > 0:
                    df_sub[f"{cc}_sd"] = (df_sub[cc] - c_mu) / c_sd
                    control_vars.append(f"{cc}_sd")

        if len(control_vars) > 1:  # Only run if we have controls
            res_fy3 = run_fy_regression(
                df_sub, y_var,
                exog_vars=control_vars,
                absorb_vars=["gvkey", "gd_year"],
                cluster_var="gvkey"
            )
            res_fy3.update({
                "outcome": oc_sd, "model": "FY3", "min_reviews_threshold": min_reviews,
                "sample": "all", "analysis_level": "firm-year", "window": "[-1,+1] years"
            })
            all_results.append(res_fy3)

        # ── FY4: Event-time bins ────────────────────────────────────
        # Create relative year bins: -3, -2, -1, 0 (reference), +1, +2, +3
        df_sub["rel_year"] = df_sub["gd_year"] - df_sub["election_year"].astype(float)
        for b in range(-3, 4):
            if b != 0:  # 0 is reference
                df_sub[f"rel_year_{b}"] = (df_sub["rel_year"] == b).astype(float)

        rel_year_bins = [f"rel_year_{b}" for b in range(-3, 4) if b != 0]
        df_event = df_sub.dropna(subset=rel_year_bins + [y_var])

        if len(df_event) > 50:
            res_fy4 = run_fy_regression(
                df_event, y_var,
                exog_vars=rel_year_bins,
                absorb_vars=["gvkey", "gd_year"],
                cluster_var="gvkey"
            )
            res_fy4.update({
                "outcome": oc_sd, "model": "FY4", "min_reviews_threshold": min_reviews,
                "sample": "all", "analysis_level": "firm-year", "window": "[-3,+3] years"
            })
            # Save per-bin coefficients
            for b in rel_year_bins:
                res_fy4[f"coef_bin_{b}"] = res_fy4.get(f"coef_{b}", np.nan)
                res_fy4[f"se_bin_{b}"] = res_fy4.get(f"se_{b}", np.nan)
            all_results.append(res_fy4)

# ═════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)
results_df["economic_magnitude_sd"] = results_df["coef"]
results_df["sign"] = np.sign(results_df["coef"])
results_df["significant_10"] = results_df["p_value"] < 0.10
results_df["significant_5"] = results_df["p_value"] < 0.05
results_df["significant_1"] = results_df["p_value"] < 0.01
results_df["outcome_family"] = results_df["outcome"].str.replace("_sd", "").str.replace("GD_", "")

results_df.to_csv(OUT / "firm_year_regression_results.csv", index=False)
print(f"\n{'=' * 70}")
print(f"Saved {len(results_df)} regression results to firm_year_regression_results.csv")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n--- FY1 Summary (min_reviews >= 3) ---")
fy1_sub = results_df[(results_df["model"] == "FY1") & (results_df["min_reviews_threshold"] == 3)]
cols = ["outcome", "N", "coef", "se", "t_stat", "p_value", "significant_5"]
print(fy1_sub[cols].to_string(index=False))

print("\n--- FY1: All thresholds for GD_rating_sd ---")
fy1_rating = results_df[(results_df["model"] == "FY1") & (results_df["outcome"].str.contains("rating"))]
print(fy1_rating[["min_reviews_threshold", "N", "coef", "se", "p_value"]].to_string(index=False))

print("\n02_firm_year_regressions complete.")
