#!/usr/bin/env python
"""
03_event_study.py
================================
Event-study analysis around union election dates.

Review-level: relative-month bins [-12, +12], omit bin = -1
Firm-year: relative-year bins [-3, +3], omit bin = 0

Outputs:
  outputs/analysis_stability/review_eventstudy_coefficients.csv
  outputs/analysis_stability/firm_year_eventstudy_coefficients.csv
  outputs/analysis_stability/figures/review_eventstudy_<outcome>.png
  outputs/analysis_stability/figures/firm_year_eventstudy_<outcome>.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from linearmodels.iv.absorbing import AbsorbingLS

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "analysis_stability"
FIG = OUT / "figures"
REVIEW_FILE = PROJ / "outputs" / "union_glassdoor_comment_level_window365.parquet"
FIRMYEAR_FILE = PROJ / "outputs" / "union_glassdoor_firm_year_regression.parquet"

# ═════════════════════════════════════════════════════════════════════════
# 1. REVIEW-LEVEL EVENT STUDY
# ═════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. REVIEW-LEVEL EVENT STUDY")
print("=" * 70)

df_r = pd.read_parquet(REVIEW_FILE)
print(f"  Loaded {len(df_r)} reviews")

# Create relative month bins
df_r["rel_month"] = np.floor(df_r["days_from_election"] / 30).astype(int)
# Clip to [-12, +12]
df_r["rel_month_clip"] = df_r["rel_month"].clip(-12, 12)

# Create bin dummies (omit -1)
for m in range(-12, 13):
    if m != -1:
        df_r[f"rm_{m}"] = (df_r["rel_month_clip"] == m).astype(int)

rm_bins = [f"rm_{m}" for m in range(-12, 13) if m != -1]

# Core outcomes
rating_outcomes = [
    "GD_rating", "GD_CareerOpp", "GD_CompBenefits", "GD_Management",
    "GD_WorkLife", "GD_CultureValues", "GD_diversity"
]

# Standardize
for oc in rating_outcomes:
    mu = df_r[oc].mean()
    sd = df_r[oc].std()
    df_r[f"{oc}_sd"] = (df_r[oc] - mu) / sd

# Also convert categorical
rec_map = {"v": 1, "o": 0.5, "x": 0}
ceo_map = {"v": 1, "o": 0.5, "r": 0, "x": 0}
outlook_map = {"v": 1, "o": 0.5, "r": 0, "x": 0}
df_r["GD_Recommend_num"] = df_r["GD_Recommend"].map(rec_map)
df_r["GD_CEOSupport_num"] = df_r["GD_CEOSupport"].map(ceo_map)
df_r["GD_Outlook_num"] = df_r["GD_Outlook"].map(outlook_map)
for oc in ["GD_Recommend_num", "GD_CEOSupport_num", "GD_Outlook_num"]:
    mu = df_r[oc].mean()
    sd = df_r[oc].std()
    df_r[f"{oc}_sd"] = (df_r[oc] - mu) / sd

Sd_outcomes = [f"{oc}_sd" for oc in rating_outcomes] + \
              ["GD_Recommend_num_sd", "GD_CEOSupport_num_sd", "GD_Outlook_num_sd"]

# ── Estimate event-study ────────────────────────────────────────────────
print("\nEstimating review-level event-study regressions...")

all_es_coefs = []

for oc_sd in Sd_outcomes:
    print(f"\n  Outcome: {oc_sd}")
    subset = df_r.dropna(subset=[oc_sd] + rm_bins + ["gvkey", "year"])

    if len(subset) < 100:
        print(f"    Insufficient data (N={len(subset)})")
        continue

    y = subset[oc_sd].values
    X = subset[rm_bins].values.astype(float)
    absorb_df = subset[["gvkey", "year"]].copy()
    absorb_df["gvkey"] = absorb_df["gvkey"].astype(str)
    absorb_df["year"] = absorb_df["year"].astype(str)

    try:
        mod = AbsorbingLS(y, X, absorb=absorb_df, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", clusters=subset["gvkey"].values)

        for i, bin_name in enumerate(rm_bins):
            all_es_coefs.append({
                "outcome": oc_sd,
                "bin": bin_name,
                "rel_month": int(bin_name.replace("rm_", "")),
                "coef": res.params[i],
                "se": res.std_errors[i],
                "t_stat": res.tstats[i],
                "p_value": res.pvalues[i],
                "ci_low": res.params[i] - 1.96 * res.std_errors[i],
                "ci_high": res.params[i] + 1.96 * res.std_errors[i],
                "N": len(subset),
                "N_firms": subset["gvkey"].nunique(),
            })
        print(f"    N={len(subset)}, firms={subset['gvkey'].nunique()}")
    except Exception as e:
        print(f"    ERROR: {e}")

es_df = pd.DataFrame(all_es_coefs)
es_df.to_csv(OUT / "review_eventstudy_coefficients.csv", index=False)
print(f"\n  Saved {len(es_df)} coefficients to review_eventstudy_coefficients.csv")

# ── Event-study plots ───────────────────────────────────────────────────
print("\nGenerating event-study plots...")

# Select top outcomes to plot (all rating/subrating + significant ones)
plot_outcomes = [f"{oc}_sd" for oc in rating_outcomes]
plot_titles = {
    "GD_rating_sd": "Overall Rating",
    "GD_CareerOpp_sd": "Career Opportunities",
    "GD_CompBenefits_sd": "Compensation & Benefits",
    "GD_Management_sd": "Senior Management",
    "GD_WorkLife_sd": "Work-Life Balance",
    "GD_CultureValues_sd": "Culture & Values",
    "GD_diversity_sd": "Diversity & Inclusion",
}

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, oc_sd in enumerate(plot_outcomes):
    ax = axes[idx]
    oc_data = es_df[es_df["outcome"] == oc_sd].sort_values("rel_month")

    if len(oc_data) == 0:
        ax.set_title(f"{plot_titles.get(oc_sd, oc_sd)} — No data")
        continue

    rel_months = oc_data["rel_month"].values
    coefs = oc_data["coef"].values
    ci_low = oc_data["ci_low"].values
    ci_high = oc_data["ci_high"].values

    ax.plot(rel_months, coefs, "b-", linewidth=1.5, label="Coefficient")
    ax.fill_between(rel_months, ci_low, ci_high, alpha=0.2, color="blue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(x=-0.5, color="gray", linestyle=":", linewidth=0.8)  # election boundary

    ax.set_title(plot_titles.get(oc_sd, oc_sd), fontweight="bold")
    ax.set_xlabel("Months relative to election")
    ax.set_ylabel("Standardized coefficient")
    ax.set_xlim(-12.5, 12.5)

    # Mark omitted bin
    ax.axvspan(-1.5, -0.5, alpha=0.1, color="gray")
    ax.text(-1, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "ref", ha="center", fontsize=7, color="gray")

    # Find max abs ylim
    max_abs = max(abs(np.nanmin(ci_low)), abs(np.nanmax(ci_high))) * 1.2
    ax.set_ylim(-max_abs, max_abs)

    # Annotate
    n_firms = oc_data["N_firms"].iloc[0]
    ax.annotate(f"N firms={int(n_firms)}", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=7, color="gray")

axes[-1].remove()  # Remove extra subplot
plt.tight_layout()
plt.savefig(FIG / "review_eventstudy_all_outcomes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: review_eventstudy_all_outcomes.png")

# ── Individual outcome plots ────────────────────────────────────────────
for oc_sd in plot_outcomes:
    oc_data = es_df[es_df["outcome"] == oc_sd].sort_values("rel_month")
    if len(oc_data) == 0:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    rel_months = oc_data["rel_month"].values
    coefs = oc_data["coef"].values
    ci_low = oc_data["ci_low"].values
    ci_high = oc_data["ci_high"].values

    ax.plot(rel_months, coefs, "b-", linewidth=1.5)
    ax.fill_between(rel_months, ci_low, ci_high, alpha=0.2, color="blue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(x=-0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axvspan(-1.5, -0.5, alpha=0.1, color="gray")

    max_abs = max(abs(np.nanmin(ci_low)), abs(np.nanmax(ci_high))) * 1.2
    ax.set_ylim(-max_abs, max_abs)
    ax.set_xlim(-12.5, 12.5)

    title = plot_titles.get(oc_sd, oc_sd)
    ax.set_title(f"Event Study: {title}\nAround Union Election (±12 months)", fontweight="bold")
    ax.set_xlabel("Months relative to election (bin -1 omitted)")
    ax.set_ylabel("Standardized Rating Score")

    oc_short = oc_sd.replace("_sd", "").replace("GD_", "")
    plt.tight_layout()
    plt.savefig(FIG / f"review_eventstudy_{oc_short}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: review_eventstudy_{oc_short}.png")

# ═════════════════════════════════════════════════════════════════════════
# 2. FIRM-YEAR EVENT STUDY
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. FIRM-YEAR EVENT STUDY")
print("=" * 70)

df_fy = pd.read_parquet(FIRMYEAR_FILE)
print(f"  Loaded {len(df_fy)} elections")

# Create long-format with available periods (only lag1 / main / for1)
long_rows = []
for _, row in df_fy.iterrows():
    eid = row["election_id"]

    for period_label, period_col, oc_suffix in [
        ("pre", "gd_year_lag1", "_lag1"),
        ("event", "gd_year", ""),
        ("post", "gd_year_for1", "_for1"),
    ]:
        yr = row.get(period_col, np.nan)
        if pd.isna(yr):
            continue

        r = {
            "election_id": eid,
            "gvkey": row["gvkey"],
            "election_year": row["election_year"],
            "gd_year": yr,
            "rel_year": int(yr - row["election_year"]) if not pd.isna(yr) else np.nan,
            "win_union": row["win_union"],
            "union_support_rate": row["union_support_rate"],
            "period": period_label,
        }

        # Core outcomes
        for oc in ["rating", "career_opp", "comp_benefit", "senior_mgmt", "culture", "wlb"]:
            r[f"GD_{oc}"] = row.get(f"GD_{oc}{oc_suffix}", np.nan)
            r[f"n_GD_{oc}"] = row.get(f"n_GD_{oc}{oc_suffix}", np.nan)

        # Review count
        r["n_reviews"] = row.get(f"n_reviews{oc_suffix}", np.nan)

        long_rows.append(r)

df_fy_long = pd.DataFrame(long_rows)
print(f"  Long format: {len(df_fy_long)} rows (pre/event/post per election)")
print(f"  Unique rel_years: {sorted(df_fy_long['rel_year'].dropna().unique())}")

# Standardize
for oc in ["rating", "career_opp", "comp_benefit", "senior_mgmt", "culture", "wlb"]:
    if f"GD_{oc}" in df_fy_long.columns:
        col = df_fy_long[f"GD_{oc}"].dropna()
        if len(col) > 0:
            mu = col.mean()
            sd = col.std()
            df_fy_long[f"GD_{oc}_sd"] = (df_fy_long[f"GD_{oc}"] - mu) / sd

# ── Estimate event-study ────────────────────────────────────────────────
print("\nEstimating firm-year event-study regressions...")

# Only use rel_year -1 and +1 (omit 0 = event year)
available_rel_years = sorted([y for y in df_fy_long["rel_year"].dropna().unique() if y != 0])
print(f"  Available rel_years (excluding 0): {available_rel_years}")

for y in available_rel_years:
    df_fy_long[f"ry_{y}"] = (df_fy_long["rel_year"] == y).astype(int)

ry_bins = [f"ry_{y}" for y in available_rel_years]

fy_es_coefs = []
for oc in ["rating", "career_opp", "comp_benefit", "senior_mgmt", "culture", "wlb"]:
    oc_sd = f"GD_{oc}_sd"
    print(f"\n  Outcome: {oc_sd}")

    # Also filter by min reviews (>=3)
    subset = df_fy_long.dropna(subset=[oc_sd] + ry_bins + ["gvkey", "gd_year"])
    n_oc_col = f"n_GD_{oc}"
    if n_oc_col in df_fy_long.columns:
        subset = subset[subset[n_oc_col] >= 3]
    elif "n_reviews" in df_fy_long.columns:
        subset = subset[subset["n_reviews"] >= 3]

    if len(subset) < 50:
        print(f"    Insufficient data (N={len(subset)})")
        continue

    y_vals = subset[oc_sd].values
    X = subset[ry_bins].values.astype(float)
    absorb_df = subset[["gvkey", "gd_year"]].copy().astype(str)

    try:
        mod = AbsorbingLS(y_vals, X, absorb=absorb_df, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", clusters=subset["gvkey"].values)

        for i, bin_name in enumerate(ry_bins):
            fy_es_coefs.append({
                "outcome": oc_sd,
                "bin": bin_name,
                "rel_year": int(bin_name.replace("ry_", "")),
                "coef": res.params[i],
                "se": res.std_errors[i],
                "t_stat": res.tstats[i],
                "p_value": res.pvalues[i],
                "ci_low": res.params[i] - 1.96 * res.std_errors[i],
                "ci_high": res.params[i] + 1.96 * res.std_errors[i],
                "N": len(subset),
                "N_firms": subset["gvkey"].nunique(),
            })
        print(f"    N={len(subset)}, firms={subset['gvkey'].nunique()}")
    except Exception as e:
        print(f"    ERROR: {e}")

fy_es_df = pd.DataFrame(fy_es_coefs)
fy_es_df.to_csv(OUT / "firm_year_eventstudy_coefficients.csv", index=False)
print(f"\n  Saved {len(fy_es_df)} coefficients to firm_year_eventstudy_coefficients.csv")

# ── Firm-year event study plots ─────────────────────────────────────────
print("\nGenerating firm-year event-study plots...")

fy_titles = {
    "GD_rating_sd": "Overall Rating",
    "GD_career_opp_sd": "Career Opportunities",
    "GD_comp_benefit_sd": "Compensation & Benefits",
    "GD_senior_mgmt_sd": "Senior Management",
    "GD_culture_sd": "Culture & Values",
    "GD_wlb_sd": "Work-Life Balance",
}

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

for idx, oc_sd in enumerate(fy_titles.keys()):
    ax = axes[idx]
    oc_data = fy_es_df[fy_es_df["outcome"] == oc_sd].sort_values("rel_year")

    if len(oc_data) == 0:
        ax.set_title(f"{fy_titles[oc_sd]} — No data")
        continue

    rel_years = oc_data["rel_year"].values
    coefs = oc_data["coef"].values
    ci_low = oc_data["ci_low"].values
    ci_high = oc_data["ci_high"].values

    ax.plot(rel_years, coefs, "b-o", markersize=6, linewidth=1.5)
    ax.fill_between(rel_years, ci_low, ci_high, alpha=0.2, color="blue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(x=-0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axvspan(-0.5, 0.5, alpha=0.1, color="gray")
    ax.text(-0.05, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "ref", ha="center", fontsize=8, color="gray")

    max_abs = max(abs(np.nanmin(ci_low)), abs(np.nanmax(ci_high))) * 1.3
    ax.set_ylim(-max_abs, max_abs)
    ax.set_xlim(-3.5, 3.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_title(fy_titles[oc_sd], fontweight="bold")
    ax.set_xlabel("Years relative to election year")
    ax.set_ylabel("Standardized Rating (firm-year avg)")

    n_firms = oc_data["N_firms"].iloc[0] if len(oc_data) > 0 else 0
    ax.annotate(f"N firms={int(n_firms)}", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=7, color="gray")

plt.tight_layout()
plt.savefig(FIG / "firm_year_eventstudy_all_outcomes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: firm_year_eventstudy_all_outcomes.png")

# ── Combined review + firm-year comparison ──────────────────────────────
print("\nGenerating comparison plot: Review-level vs Firm-year event study")
# Map review outcomes to firm-year outcomes
outcome_map = {
    "GD_rating_sd": "GD_rating_sd",
    "GD_CareerOpp_sd": "GD_career_opp_sd",
    "GD_CompBenefits_sd": "GD_comp_benefit_sd",
    "GD_Management_sd": "GD_senior_mgmt_sd",
    "GD_WorkLife_sd": "GD_wlb_sd",
    "GD_CultureValues_sd": "GD_culture_sd",
}

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

for idx, (rev_oc, fy_oc) in enumerate(outcome_map.items()):
    ax = axes[idx]

    # Review-level (monthly)
    rev_data = es_df[es_df["outcome"] == rev_oc].sort_values("rel_month")
    if len(rev_data) > 0:
        # Scale months to years for comparison
        rev_years = rev_data["rel_month"].values / 12
        ax.plot(rev_years, rev_data["coef"].values, "b-", linewidth=1, alpha=0.5,
                label="Review-level (monthly)")
        ax.fill_between(rev_years, rev_data["ci_low"].values, rev_data["ci_high"].values,
                        alpha=0.1, color="blue")

    # Firm-year (annual)
    fy_data = fy_es_df[fy_es_df["outcome"] == fy_oc].sort_values("rel_year")
    if len(fy_data) > 0:
        ax.errorbar(fy_data["rel_year"].values, fy_data["coef"].values,
                    yerr=1.96 * fy_data["se"].values, fmt="ro-", capsize=4,
                    linewidth=2, markersize=8, label="Firm-year")

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(x=-0.04, color="gray", linestyle=":", linewidth=0.8)  # election

    title = plot_titles.get(rev_oc, rev_oc)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Years relative to election")
    ax.set_ylabel("Standardized coefficient")
    ax.legend(fontsize=7, loc="best")
    ax.axvspan(-0.08, 0, alpha=0.05, color="gray")

plt.tight_layout()
plt.savefig(FIG / "comparison_review_vs_firmyear_eventstudy.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: comparison_review_vs_firmyear_eventstudy.png")

print("\n" + "=" * 70)
print("03_event_study complete.")
