#!/usr/bin/env python
"""
01_review_level_regressions.py
================================
Review-level regressions following Li & Pinto (2025) design.

Models:
  R1: Sd_Outcome ~ PostElection + firm FE + year FE
  R2: Sd_Outcome ~ PostElection + firm FE + year FE + month FE
  R3: Sd_Outcome ~ PostElection + firm FE + year FE + job_title FE
  R4: R1 split by current/former employee status
  R5: R1 split by job category (if classification available)

Outputs:
  outputs/analysis_stability/review_regression_results.csv
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
from pathlib import Path
from linearmodels.iv.absorbing import AbsorbingLS

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "analysis_stability"
REVIEW_FILE = PROJ / "outputs" / "union_glassdoor_comment_level_window365.parquet"
TITLE_CLASSIFIED = PROJ / "outputs" / "union_classified_title_universe.csv"

# ── Load data ───────────────────────────────────────────────────────────
print("Loading review-level data...")
df = pd.read_parquet(REVIEW_FILE)
print(f"  Shape: {df.shape}")

# ── Prepare variables ───────────────────────────────────────────────────
print("\nPreparing variables...")

# Outcome variables (rating + subratings)
rating_outcomes = [
    "GD_rating", "GD_CareerOpp", "GD_CompBenefits", "GD_Management",
    "GD_WorkLife", "GD_CultureValues", "GD_diversity"
]

# Standardize outcomes (z-score)
for oc in rating_outcomes:
    mu = df[oc].mean()
    sd = df[oc].std()
    df[f"{oc}_sd"] = (df[oc] - mu) / sd
    print(f"  {oc}_sd: mean={df[f'{oc}_sd'].mean():.4f}, sd={df[f'{oc}_sd'].std():.4f}")

# Categorical outcomes → numeric
# GD_Recommend: v=1 (positive), o=2 (neutral/ok), x=0 (negative/disapprove)
# GD_CEOSupport: v=1 (approve), o=2 (no opinion), r=0 (disapprove), x=0
# GD_Outlook: v=1 (positive), o=2 (neutral), r=0 (negative), x=0
rec_map = {"v": 1, "o": 0.5, "x": 0}
ceo_map = {"v": 1, "o": 0.5, "r": 0, "x": 0}
outlook_map = {"v": 1, "o": 0.5, "r": 0, "x": 0}

df["GD_Recommend_num"] = df["GD_Recommend"].map(rec_map)
df["GD_CEOSupport_num"] = df["GD_CEOSupport"].map(ceo_map)
df["GD_Outlook_num"] = df["GD_Outlook"].map(outlook_map)

for oc in ["GD_Recommend_num", "GD_CEOSupport_num", "GD_Outlook_num"]:
    mu = df[oc].mean()
    sd = df[oc].std()
    df[f"{oc}_sd"] = (df[oc] - mu) / sd
    print(f"  {oc}_sd: mean={df[f'{oc}_sd'].mean():.4f}, n={df[oc].notna().sum()}")

# Employee status
df["is_current"] = df["GD_ReviewerStatus"].str.contains("Current", na=False)
df["is_former"] = df["GD_ReviewerStatus"].str.contains("Former", na=False)
print(f"  Current: {df['is_current'].sum()}, Former: {df['is_former'].sum()}")

# Post-election indicator already exists: post_election_comment
# Firm FE = gvkey, Year FE = year
print(f"  N gvkey: {df['gvkey'].nunique()}")
print(f"  N years: {df['year'].nunique()}")

# ── Load job title classification ───────────────────────────────────────
print("\nLoading job title classification...")
if TITLE_CLASSIFIED.exists():
    title_df = pd.read_csv(TITLE_CLASSIFIED)
    print(f"  Title classification shape: {title_df.shape}")
    print(f"  Columns: {list(title_df.columns)}")

    # Check for canonical title column
    title_col = None
    for c in ["title_canonical_en", "title_canonical", "title_normalized", "GD_JobTitle_canonical"]:
        if c in title_df.columns:
            title_col = c
            break

    if title_col:
        # Merge title classification onto reviews
        # The classification file might use GD_JobTitle as key
        if "GD_JobTitle" in title_df.columns:
            df = df.merge(title_df[["GD_JobTitle", title_col]], on="GD_JobTitle", how="left")
            print(f"  Merged {title_col} onto reviews")
            print(f"  Unique titles: {df[title_col].nunique()}")
            print(f"  Missing titles: {df[title_col].isna().sum()}")

            # For R3: use only reviews with at least K occurrences per title
            title_counts = df[title_col].value_counts()
            valid_titles = title_counts[title_counts >= 5].index
            df["title_for_fe"] = df[title_col].where(df[title_col].isin(valid_titles), "OTHER_RARE")
            print(f"  Titles with >=5 reviews: {len(valid_titles)}")
        else:
            print("  WARNING: GD_JobTitle not in classification file, skipping title FE")
    else:
        print("  WARNING: No canonical title column found")
else:
    print("  WARNING: Title classification file not found, skipping R3 and R5")

# ── Define all outcomes (standardized) ──────────────────────────────────
Sd_outcomes = [f"{oc}_sd" for oc in rating_outcomes]
Sd_outcomes += ["GD_Recommend_num_sd", "GD_CEOSupport_num_sd", "GD_Outlook_num_sd"]

# ═════════════════════════════════════════════════════════════════════════
# REGRESSION FUNCTION
# ═════════════════════════════════════════════════════════════════════════

def run_absorbing_regression(data, y_var, absorb_vars, exog_vars=None, cluster_var=None):
    """
    Run absorbing least squares with high-dimensional fixed effects.

    Parameters
    ----------
    data : DataFrame
    y_var : str — outcome variable name
    absorb_vars : list of str — categorical variables to absorb
    exog_vars : list of str — exogenous variables (besides absorb)
    cluster_var : str — cluster variable for standard errors

    Returns
    -------
    dict with coef, se, t_stat, p_value, N, etc.
    """
    subset = data.dropna(subset=[y_var] + absorb_vars + (exog_vars or []))

    if len(subset) < 50:
        return {"N": len(subset), "coef": np.nan, "se": np.nan, "t_stat": np.nan, "p_value": np.nan}

    y = subset[y_var].values
    X_cols = exog_vars or []

    # Build exogenous DataFrame
    if X_cols:
        X = subset[X_cols].values
        X = X.reshape(-1, len(X_cols)) if X.ndim == 1 else X
    else:
        # Need at least one exogenous variable; use constant
        X = np.ones((len(subset), 1))
        X_cols = ["const"]

    # Absorb
    absorb_df = subset[absorb_vars].copy()
    for av in absorb_vars:
        absorb_df[av] = absorb_df[av].astype(str)

    try:
        mod = AbsorbingLS(y, X, absorb=absorb_df, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", clusters=subset[cluster_var].values if cluster_var else None)

        # Extract coefficient for PostElection (first exog variable)
        coef_idx = 0 if "post_election_comment" in X_cols else 0
        coef = res.params[coef_idx]
        se = res.std_errors[coef_idx]
        t_stat = res.tstats[coef_idx]
        p_value = res.pvalues[coef_idx]
        # CI
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se

        return {
            "N": len(subset),
            "N_firms": subset["gvkey"].nunique(),
            "N_events": subset["election_id"].nunique(),
            "coef": coef,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_y": y.mean(),
            "sd_y": y.std(),
        }
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"N": len(subset), "coef": np.nan, "se": np.nan, "t_stat": np.nan, "p_value": np.nan,
                "error": str(e)}

# ═════════════════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ═════════════════════════════════════════════════════════════════════════

all_results = []

# ── R1: Baseline ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("R1: Sd_Outcome ~ PostElection + firm FE + year FE")
print("=" * 70)

for oc in Sd_outcomes:
    print(f"\n  Outcome: {oc}")
    # All employees
    res = run_absorbing_regression(
        df, oc,
        absorb_vars=["gvkey", "year"],
        exog_vars=["post_election_comment"],
        cluster_var="gvkey"
    )
    res.update({"outcome": oc, "model": "R1", "sample": "all", "window": "[-365,+365]"})
    all_results.append(res)
    if not np.isnan(res.get("coef", np.nan)):
        print(f"    N={res['N']}, coef={res['coef']:.4f}, se={res['se']:.4f}, t={res['t_stat']:.2f}, p={res['p_value']:.3f}")

# ── R2: Calendar time controls ──────────────────────────────────────────
print("\n" + "=" * 70)
print("R2: Sd_Outcome ~ PostElection + firm FE + year FE + month FE")
print("=" * 70)

for oc in Sd_outcomes:
    print(f"\n  Outcome: {oc}")
    res = run_absorbing_regression(
        df, oc,
        absorb_vars=["gvkey", "year", "month"],
        exog_vars=["post_election_comment"],
        cluster_var="gvkey"
    )
    res.update({"outcome": oc, "model": "R2", "sample": "all", "window": "[-365,+365]"})
    all_results.append(res)
    if not np.isnan(res.get("coef", np.nan)):
        print(f"    N={res['N']}, coef={res['coef']:.4f}, se={res['se']:.4f}, t={res['t_stat']:.2f}, p={res['p_value']:.3f}")

# ── R3: Job title FE ────────────────────────────────────────────────────
if "title_for_fe" in df.columns:
    print("\n" + "=" * 70)
    print("R3: Sd_Outcome ~ PostElection + firm FE + year FE + job_title FE")
    print("=" * 70)

    for oc in Sd_outcomes:
        print(f"\n  Outcome: {oc}")
        res = run_absorbing_regression(
            df, oc,
            absorb_vars=["gvkey", "year", "title_for_fe"],
            exog_vars=["post_election_comment"],
            cluster_var="gvkey"
        )
        res.update({"outcome": oc, "model": "R3", "sample": "all", "window": "[-365,+365]"})
        all_results.append(res)
        if not np.isnan(res.get("coef", np.nan)):
            print(f"    N={res['N']}, coef={res['coef']:.4f}, se={res['se']:.4f}, t={res['t_stat']:.2f}, p={res['p_value']:.3f}")
else:
    print("\nR3: SKIPPED (no title classification available)")

# ── R4: Current vs Former ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("R4: Split by employee status")
print("=" * 70)

for sample_name, sample_mask in [("current", df["is_current"]), ("former", df["is_former"])]:
    sample_df = df[sample_mask].copy()
    print(f"\n  --- {sample_name} employees (N={len(sample_df)}) ---")

    for oc in Sd_outcomes:
        print(f"    Outcome: {oc}")
        res = run_absorbing_regression(
            sample_df, oc,
            absorb_vars=["gvkey", "year"],
            exog_vars=["post_election_comment"],
            cluster_var="gvkey"
        )
        res.update({"outcome": oc, "model": "R4", "sample": sample_name, "window": "[-365,+365]"})
        all_results.append(res)
        if not np.isnan(res.get("coef", np.nan)):
            print(f"      N={res['N']}, coef={res['coef']:.4f}, se={res['se']:.4f}, t={res['t_stat']:.2f}, p={res['p_value']:.3f}")

# ── R5: Job category subsamples ─────────────────────────────────────────
if "title_for_fe" in df.columns:
    print("\n" + "=" * 70)
    print("R5: Split by job category (from title classification)")
    print("=" * 70)

    # Check for category columns in title classification
    title_df_cols = pd.read_csv(TITLE_CLASSIFIED).columns
    category_cols = [c for c in title_df_cols if "class" in c.lower() or "category" in c.lower()
                     or c in ["likely_unionizable", "likely_excluded", "ambiguous", "oc_likely"]]

    if category_cols:
        print(f"  Category columns found: {category_cols}")
        for cat_col in category_cols[:3]:  # Limit to first 3 to keep output manageable
            # Merge category info
            if cat_col not in df.columns:
                cat_data = pd.read_csv(TITLE_CLASSIFIED)[["GD_JobTitle", cat_col]].drop_duplicates(subset=["GD_JobTitle"])
                df_tmp = df.merge(cat_data, on="GD_JobTitle", how="left")
            else:
                df_tmp = df

            for cat_val in df_tmp[cat_col].dropna().unique()[:6]:  # Top 6 categories
                cat_df = df_tmp[df_tmp[cat_col] == cat_val]
                if len(cat_df) < 100:
                    continue
                print(f"\n  --- {cat_col}={cat_val} (N={len(cat_df)}) ---")
                for oc in Sd_outcomes[:5]:  # Top 5 outcomes
                    res = run_absorbing_regression(
                        cat_df, oc,
                        absorb_vars=["gvkey", "year"],
                        exog_vars=["post_election_comment"],
                        cluster_var="gvkey"
                    )
                    res.update({"outcome": oc, "model": "R5", "sample": f"{cat_col}_{cat_val}",
                                "window": "[-365,+365]"})
                    all_results.append(res)
    else:
        print("  No category columns found, skipping R5")
else:
    print("\nR5: SKIPPED (no title classification)")

# ═════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)

# Add derived fields
results_df["economic_magnitude_sd"] = results_df["coef"]  # outcome is already standardized
results_df["sign"] = np.sign(results_df["coef"])
results_df["significant_10"] = results_df["p_value"] < 0.10
results_df["significant_5"] = results_df["p_value"] < 0.05
results_df["significant_1"] = results_df["p_value"] < 0.01
results_df["analysis_level"] = "review-level"
results_df["outcome_family"] = results_df["outcome"].str.replace("_sd", "").str.replace("_num", "")

results_df.to_csv(OUT / "review_regression_results.csv", index=False)
print(f"\n{'=' * 70}")
print(f"Saved {len(results_df)} regression results to review_regression_results.csv")

# ── Summary table ───────────────────────────────────────────────────────
print("\n--- Summary: R1 coefficients (all outcomes, all employees) ---")
r1_summary = results_df[(results_df["model"] == "R1") & (results_df["sample"] == "all")]
print(r1_summary[["outcome", "N", "coef", "se", "t_stat", "p_value", "significant_5"]].to_string(index=False))

print("\n--- Summary: R4 current vs former (GD_rating_sd) ---")
r4_rating = results_df[(results_df["model"] == "R4") & (results_df["outcome"] == "GD_rating_sd")]
print(r4_rating[["sample", "N", "coef", "se", "t_stat", "p_value"]].to_string(index=False))

print("\n01_review_level_regressions complete.")
