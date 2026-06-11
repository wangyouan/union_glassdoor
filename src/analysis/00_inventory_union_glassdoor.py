#!/usr/bin/env python
"""
00_inventory_union_glassdoor.py
================================
Variable inventory for union_glassdoor stability analysis.

Outputs:
  outputs/analysis_stability/variable_inventory_ratings.csv        — review-level rating/subrating/sentiment variables
  outputs/analysis_stability/review_level_variable_inventory.csv   — all review-level variables with availability
  outputs/analysis_stability/subsample_outcome_inventory.csv       — firm-year subpopulation × outcome mapping
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "analysis_stability"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

REVIEW_FILE = PROJ / "outputs" / "union_glassdoor_comment_level_window365.parquet"
FIRMYEAR_FILE = PROJ / "outputs" / "union_glassdoor_firm_year_regression.parquet"
CONTROLS_FILE = PROJ / "outputs" / "compustat_firm_controls.parquet"

# ═════════════════════════════════════════════════════════════════════════
# 1. REVIEW-LEVEL VARIABLE INVENTORY
# ═════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. Loading review-level data...")
df_r = pd.read_parquet(REVIEW_FILE)
print(f"   Shape: {df_r.shape}")
print(f"   N firms (gvkey): {df_r['gvkey'].nunique()}")
print(f"   N elections: {df_r['election_id'].nunique()}")
print(f"   Date range: {df_r['review_date'].min()} to {df_r['review_date'].max()}")

# ── 1a. Full variable inventory ─────────────────────────────────────────
print("\n--- 1a. Full variable inventory ---")
rows = []
for c in df_r.columns:
    nonnull = df_r[c].notna().sum()
    dtype = df_r[c].dtype
    n_unique = df_r[c].nunique()

    # Determine variable type
    vtype = "other"
    if c.startswith("GD_"):
        if c in ["GD_rating", "GD_CareerOpp", "GD_CompBenefits", "GD_Management",
                  "GD_WorkLife", "GD_CultureValues", "GD_diversity"]:
            vtype = "rating" if c == "GD_rating" else "subrating"
        elif c in ["GD_Recommend", "GD_CEOSupport", "GD_Outlook"]:
            vtype = "categorical"
        elif c in ["GD_Pros", "GD_Cons", "GD_Advice", "GD_ReviewTitle"]:
            vtype = "text"
        elif c == "GD_ReviewerStatus":
            vtype = "employee_status"
        elif c == "GD_JobTitle":
            vtype = "job_title"
        elif c == "GD_CompanyName" or c == "GD_CompanyLink":
            vtype = "company_identifier"
        elif c == "GD_CompanyID":
            vtype = "company_id"
        else:
            vtype = "other_gd"
    elif c in ["gvkey", "conm", "tic", "cik"]:
        vtype = "firm_identifier"
    elif c in ["election_id", "case_number", "election_date"]:
        vtype = "election_identifier"
    elif c in ["votes_for_union", "votes_against_union", "total_valid_votes",
                "union_support_rate", "win_union", "lose_union", "union_tie",
                "union_margin", "close_election_abs_margin"]:
        vtype = "election_variable"
    elif c in ["days_from_election", "post_election_comment", "abs_days_from_election"]:
        vtype = "event_time"
    elif c in ["review_date", "review_id", "year", "month", "review_year"]:
        vtype = "review_identifier"
    elif c.startswith("L_"):
        vtype = "lagged_control"
    elif c in ["size", "log_me", "leverage", "cash_ratio", "roa", "profitability",
                "tangibility", "capx_at", "rd_at", "book_to_market", "sales_growth", "log_emp"]:
        vtype = "control"
    elif c in ["merge_reviews_elections", "merge_controls", "n_event_matches"]:
        vtype = "merge_metadata"

    # Compute stats for numeric variables
    stats = {}
    if pd.api.types.is_numeric_dtype(df_r[c]):
        col = df_r[c].dropna()
        stats = {
            "mean": col.mean(),
            "sd": col.std(),
            "min": col.min(),
            "p1": col.quantile(0.01),
            "p25": col.quantile(0.25),
            "median": col.median(),
            "p75": col.quantile(0.75),
            "p99": col.quantile(0.99),
            "max": col.max(),
        }

    rows.append({
        "variable_name": c,
        "variable_type": vtype,
        "nonmissing_n": nonnull,
        "nonmissing_share": nonnull / len(df_r),
        "dtype": str(dtype),
        "n_unique": n_unique,
        **stats,
    })

inv = pd.DataFrame(rows)
inv.to_csv(OUT / "review_level_variable_inventory.csv", index=False)
print(f"   Saved: review_level_variable_inventory.csv ({len(inv)} variables)")

# ── 1b. Rating/subrating inventory ──────────────────────────────────────
print("\n--- 1b. Rating & subrating inventory ---")
rating_vars = inv[inv["variable_type"].isin(["rating", "subrating"])].copy()
# Add reviewer status breakdown
for _, rv in rating_vars.iterrows():
    vname = rv["variable_name"]
    if vname in df_r.columns:
        col = df_r[vname].dropna()
        rating_vars.loc[_, "n_current_emp"] = df_r.loc[df_r["GD_ReviewerStatus"].str.contains("Current", na=False), vname].notna().sum()
        rating_vars.loc[_, "n_former_emp"] = df_r.loc[df_r["GD_ReviewerStatus"].str.contains("Former", na=False), vname].notna().sum()

rating_vars.to_csv(OUT / "variable_inventory_ratings.csv", index=False)
print(f"   Saved: variable_inventory_ratings.csv ({len(rating_vars)} variables)")
print(rating_vars[["variable_name", "nonmissing_n", "mean", "sd", "median"]].to_string(index=False))

# ── 1c. Employee status breakdown ───────────────────────────────────────
print("\n--- 1c. Employee status ---")
status_counts = df_r["GD_ReviewerStatus"].value_counts()
print(status_counts.to_string())

# Current vs former mapping
df_r["is_current"] = df_r["GD_ReviewerStatus"].str.contains("Current", na=False)
df_r["is_former"] = df_r["GD_ReviewerStatus"].str.contains("Former", na=False)
print(f"   Current employees: {df_r['is_current'].sum()} reviews")
print(f"   Former employees: {df_r['is_former'].sum()} reviews")

# ═════════════════════════════════════════════════════════════════════════
# 2. FIRM-YEAR OUTCOME INVENTORY
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. Loading firm-year data...")
df_fy = pd.read_parquet(FIRMYEAR_FILE)
print(f"   Shape: {df_fy.shape}")
print(f"   N firms (gvkey): {df_fy['gvkey'].nunique()}")
print(f"   N elections: {df_fy['election_id'].nunique()}")

# ── 2a. Identify all GD outcome columns ─────────────────────────────────
print("\n--- 2a. Identifying GD outcome columns ---")
# Pattern: [prefix_]GD_outcome[_time][_standardization]
gd_pattern = re.compile(r'^(.+_)?GD_(.+?)(_lag1|_for1|_main)?(_sdff48|_sdsic2)?$')

outcome_rows = []
for c in sorted(df_fy.columns):
    m = gd_pattern.match(c)
    if not m:
        continue

    prefix = (m.group(1) or "").rstrip("_")
    outcome_base = m.group(2)
    time_period = (m.group(3) or "").lstrip("_") if m.group(3) else "main"
    standardization = (m.group(4) or "").lstrip("_") if m.group(4) else "raw"

    # Get stats
    nonnull = df_fy[c].notna().sum()
    if nonnull > 0:
        col_data = df_fy[c].dropna()
        outcome_rows.append({
            "column_name": c,
            "prefix_group": prefix if prefix else "all_reviews",
            "outcome": outcome_base,
            "time_period": time_period,
            "standardization": standardization,
            "nonmissing_n": nonnull,
            "nonmissing_share": nonnull / len(df_fy),
            "mean": col_data.mean(),
            "sd": col_data.std(),
            "median": col_data.median(),
            "min": col_data.min(),
            "max": col_data.max(),
        })

outcome_df = pd.DataFrame(outcome_rows)
outcome_df.to_csv(OUT / "subsample_outcome_inventory.csv", index=False)
print(f"   Saved: subsample_outcome_inventory.csv ({len(outcome_df)} columns)")

# ── 2b. Summarize subpopulations ────────────────────────────────────────
print("\n--- 2b. Subpopulation summary ---")
for grp in sorted(outcome_df["prefix_group"].unique()):
    n_cols = len(outcome_df[outcome_df["prefix_group"] == grp])
    outcomes = outcome_df[outcome_df["prefix_group"] == grp]["outcome"].unique()
    nonmiss = outcome_df[outcome_df["prefix_group"] == grp]["nonmissing_n"].max()
    print(f"  {grp}: {n_cols} columns, outcomes={list(outcomes)}, max_n={nonmiss}")

# ── 2c. Identify subpopulation categories ───────────────────────────────
print("\n--- 2c. Subpopulation classification ---")
subpop_map = {
    "all_reviews": {"dimension": "all", "description": "All reviews"},
    "management": {"dimension": "job_category", "description": "Management"},
    "operations": {"dimension": "job_category", "description": "Operations"},
    "admin": {"dimension": "job_category", "description": "Administrative"},
    "sales": {"dimension": "job_category", "description": "Sales"},
    "rd": {"dimension": "job_category", "description": "R&D"},
    "client_facing": {"dimension": "job_category", "description": "Client-facing"},
    "compliance": {"dimension": "job_category", "description": "Compliance"},
    "core_business": {"dimension": "job_category", "description": "Core business"},
    "entry_level": {"dimension": "job_level", "description": "Entry level"},
    "senior": {"dimension": "job_level", "description": "Senior"},
    "high_hc": {"dimension": "job_level", "description": "High human capital"},
    "high_level_core": {"dimension": "job_level", "description": "High level core"},
    "low_level_core": {"dimension": "job_level", "description": "Low level core"},
    "non_core_staff": {"dimension": "job_category", "description": "Non-core staff"},
    "specialized": {"dimension": "job_category", "description": "Specialized"},
    "performance_linked": {"dimension": "job_category", "description": "Performance-linked"},
    "ambu": {"dimension": "union_class", "description": "Ambiguous union classification"},
    "mayu": {"dimension": "union_class", "description": "Maybe union"},
    "notu": {"dimension": "union_class", "description": "Not union"},
    "n_ambu": {"dimension": "union_class_count", "description": "Count - ambiguous"},
    "n_mayu": {"dimension": "union_class_count", "description": "Count - maybe union"},
    "n_notu": {"dimension": "union_class_count", "description": "Count - not union"},
    "n": {"dimension": "review_count", "description": "Review count"},
}

subpop_inv = []
for grp in sorted(outcome_df["prefix_group"].unique()):
    info = subpop_map.get(grp, {"dimension": "unknown", "description": grp})
    subpop_inv.append({
        "prefix_group": grp,
        "dimension": info["dimension"],
        "description": info["description"],
        "n_columns": len(outcome_df[outcome_df["prefix_group"] == grp]),
        "outcomes": ", ".join(sorted(outcome_df[outcome_df["prefix_group"] == grp]["outcome"].unique())),
        "time_periods": ", ".join(sorted(outcome_df[outcome_df["prefix_group"] == grp]["time_period"].unique())),
        "max_nonmissing": outcome_df[outcome_df["prefix_group"] == grp]["nonmissing_n"].max(),
    })

subpop_inv_df = pd.DataFrame(subpop_inv)
subpop_inv_df.to_csv(OUT / "subsample_outcome_inventory.csv", index=False)
print(f"   Saved updated subsample_outcome_inventory.csv")

# ── 2d. Main outcomes for analysis ──────────────────────────────────────
print("\n--- 2d. Core outcomes for analysis ---")
# Focus on: GD_rating, GD_career_opp, GD_comp_benefit, GD_senior_mgmt, GD_culture, GD_work_life, GD_diversity
# Time period: main (election year) for DiD
core_outcomes = ["rating", "career_opp", "comp_benefit", "senior_mgmt", "culture", "work_life", "diversity"]
for oc in core_outcomes:
    cols = outcome_df[(outcome_df["prefix_group"] == "all_reviews") &
                      (outcome_df["outcome"] == oc) &
                      (outcome_df["standardization"] == "raw")]
    for _, r in cols.iterrows():
        print(f"  {r['column_name']}: n={r['nonmissing_n']:.0f}, mean={r['mean']:.3f}, sd={r['sd']:.3f}")

# ═════════════════════════════════════════════════════════════════════════
# 3. SAMPLE SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. Sample summary")

summary_stats = {
    "dataset": ["review_level", "firm_year"],
    "n_observations": [len(df_r), len(df_fy)],
    "n_unique_gvkey": [df_r["gvkey"].nunique(), df_fy["gvkey"].nunique()],
    "n_unique_elections": [df_r["election_id"].nunique(), df_fy["election_id"].nunique()],
    "n_union_wins": [df_r[df_r["win_union"] == 1]["election_id"].nunique(),
                      df_fy[df_fy["win_union"] == 1]["election_id"].nunique()],
    "n_union_losses": [df_r[df_r["lose_union"] == 1]["election_id"].nunique(),
                        df_fy[df_fy["lose_union"] == 1]["election_id"].nunique()],
    "year_min": [int(df_r["year"].min()), int(df_fy["election_year"].min())],
    "year_max": [int(df_r["year"].max()), int(df_fy["election_year"].max())],
    "mean_rating": [df_r["GD_rating"].mean(), df_fy["GD_rating"].mean() if "GD_rating" in df_fy.columns else np.nan],
    "mean_days_from_election": [df_r["days_from_election"].mean(), np.nan],
    "pct_post_election": [df_r["post_election_comment"].mean(), np.nan],
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(OUT / "sample_summary_statistics.csv", index=False)
print(summary_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
# 4. DATA QUALITY CHECKS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. Data quality checks")

# Check if text sentiment variables exist
text_sentiment_cols = [c for c in df_r.columns if "sentiment" in c.lower()]
print(f"   Text sentiment columns: {text_sentiment_cols if text_sentiment_cols else 'NONE'}")

# Check CEO/Recommend/Outlook data
for c in ["GD_CEOSupport", "GD_Recommend", "GD_Outlook"]:
    if c in df_r.columns:
        vals = df_r[c].value_counts()
        print(f"   {c}: {dict(vals)}")

# Check job title classification files
title_files = [
    PROJ / "outputs" / "union_title_universe_normalized.csv",
    PROJ / "outputs" / "union_classified_title_universe.csv",
    PROJ / "outputs" / "union_classified_title_universe_final.csv",
]
print("\n   Job title classification files:")
for tf in title_files:
    exists = tf.exists()
    print(f"   {'✓' if exists else '✗'} {tf.name}")

# Check current/non-current split in firm-year
current_cols = [c for c in df_fy.columns if "current" in c.lower() and "GD_" in c]
former_cols = [c for c in df_fy.columns if "former" in c.lower() and "GD_" in c]
print(f"\n   Current employee GD columns in firm-year: {len(current_cols)}")
print(f"   Former employee GD columns in firm-year: {len(former_cols)}")
if current_cols:
    print(f"   Examples: {current_cols[:5]}")
if former_cols:
    print(f"   Examples: {former_cols[:5]}")

print("\n" + "=" * 70)
print("00_inventory complete.")
print(f"Outputs saved to: {OUT}")
