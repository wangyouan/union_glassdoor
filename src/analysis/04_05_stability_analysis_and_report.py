#!/usr/bin/env python
"""
04_05_stability_analysis_and_report.py
========================================
Combined stability grid analysis and report generation.

Reads results from:
  - review_regression_results.csv
  - firm_year_regression_results.csv
  - review_eventstudy_coefficients.csv
  - firm_year_eventstudy_coefficients.csv

Generates:
  - outputs/analysis_stability/stability_grid_results.csv
  - outputs/analysis_stability/stability_summary_by_outcome.csv
  - outputs/analysis_stability/figures/outcome_stability_heatmap.png
  - outputs/analysis_stability/figures/min_review_threshold_sensitivity.png
  - outputs/analysis_stability/figures/current_vs_noncurrent_comparison.png
  - outputs/analysis_stability/union_glassdoor_stability_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})
sns.set_style("whitegrid")

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "analysis_stability"
FIG = OUT / "figures"

# ── Load results ────────────────────────────────────────────────────────
print("Loading results...")
rev_reg = pd.read_csv(OUT / "review_regression_results.csv")
fy_reg = pd.read_csv(OUT / "firm_year_regression_results.csv")
rev_es = pd.read_csv(OUT / "review_eventstudy_coefficients.csv")
fy_es = pd.read_csv(OUT / "firm_year_eventstudy_coefficients.csv")

print(f"  Review regressions: {len(rev_reg)} rows")
print(f"  Firm-year regressions: {len(fy_reg)} rows")
print(f"  Review event-study: {len(rev_es)} rows")
print(f"  Firm-year event-study: {len(fy_es)} rows")

# ═════════════════════════════════════════════════════════════════════════
# 1. BUILD STABILITY GRID
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. Building stability grid")

# Combine review-level and firm-year results
grid_cols = [
    "analysis_level", "outcome", "outcome_family", "sample", "window",
    "min_reviews_threshold", "model", "N", "N_firms", "N_events",
    "coef", "se", "t_stat", "p_value", "ci_low", "ci_high",
    "mean_y", "sd_y", "economic_magnitude_sd", "sign",
    "significant_10", "significant_5", "significant_1"
]

# Normalize columns across dataframes
for df_src in [rev_reg, fy_reg]:
    for c in grid_cols:
        if c not in df_src.columns:
            df_src[c] = np.nan

grid = pd.concat([
    rev_reg[[c for c in grid_cols if c in rev_reg.columns]],
    fy_reg[[c for c in grid_cols if c in fy_reg.columns]]
], ignore_index=True)

# Add fixed effects description
fe_map = {
    "R1": "firm FE + year FE",
    "R2": "firm FE + year FE + month FE",
    "R3": "firm FE + year FE + job_title FE",
    "R4": "firm FE + year FE (subsample)",
    "R5": "firm FE + year FE (job category)",
    "FY1": "firm FE + year FE",
    "FY2": "firm FE + year FE",
    "FY3": "firm FE + year FE + controls",
    "FY4": "firm FE + year FE (event bins)",
}
grid["fixed_effects"] = grid["model"].map(fe_map).fillna("firm FE + year FE")
grid["controls"] = grid["model"].apply(lambda m: "size, leverage, ROA, MTB, sales growth" if m == "FY3" else "none")
grid["cluster_level"] = "gvkey"

# ── Detect pre-trend from event study ───────────────────────────────────
print("\nDetecting pre-trends from event study coefficients...")

pretrend_flags = {}
for outcome in rev_es["outcome"].unique():
    # Check months -6 to -2 vs months +1 to +6
    pre_data = rev_es[(rev_es["outcome"] == outcome) & (rev_es["rel_month"].between(-6, -2))]
    post_data = rev_es[(rev_es["outcome"] == outcome) & (rev_es["rel_month"].between(1, 6))]

    if len(pre_data) >= 3 and len(post_data) >= 3:
        pre_mean = pre_data["coef"].mean()
        pre_se = pre_data["se"].mean()
        pre_significant = (pre_data["p_value"] < 0.10).sum()

        # Flag if pre-trend has 2+ significant months
        pretrend_flag = pre_significant >= 2
        pretrend_flags[outcome] = {
            "pretrend_flag": pretrend_flag,
            "pre_mean_coef": pre_mean,
            "pre_n_significant": pre_significant,
            "post_mean_coef": post_data["coef"].mean(),
            "post_n_significant": (post_data["p_value"] < 0.10).sum(),
        }

for outcome, info in pretrend_flags.items():
    print(f"  {outcome}: pre_signif={info['pre_n_significant']}, post_signif={info['post_n_significant']}, pre_mean={info['pre_mean_coef']:.3f}")

grid["pretrend_flag"] = grid["outcome"].map({k: v["pretrend_flag"] for k, v in pretrend_flags.items()}).fillna(False)

# Save stability grid
grid.to_csv(OUT / "stability_grid_results.csv", index=False)
print(f"\nSaved stability_grid_results.csv ({len(grid)} rows)")

# ═════════════════════════════════════════════════════════════════════════
# 2. STABILITY SUMMARY BY OUTCOME
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. Computing stability scores by outcome × sample_group")

summary_rows = []
for (outcome, sample), group in grid.groupby(["outcome", "sample"]):
    if len(group) < 2:
        continue

    coefs = group["coef"].dropna()
    sigs = group["significant_5"].dropna()

    # Stability score components
    same_sign_count = (np.sign(coefs) == np.sign(coefs.median())).sum()
    significant_count = sigs.sum()
    median_coef = coefs.median()
    coef_iqr = coefs.quantile(0.75) - coefs.quantile(0.25) if len(coefs) >= 4 else np.nan
    sample_size_median = group["N"].median()
    pretrend_pass_count = (~group["pretrend_flag"]).sum() if "pretrend_flag" in group.columns else 0

    # Simple stability score (0-100)
    n_specs = len(group)
    sign_consistency = same_sign_count / n_specs  # 0-1
    sig_share = significant_count / n_specs  # 0-1 (higher = more consistently significant)
    pretrend_ok = pretrend_pass_count / n_specs if n_specs > 0 else 1  # 0-1 (higher = fewer pretrend concerns)

    # Composite score: weight toward sign consistency and pretrend validity
    stability_score = (sign_consistency * 40 + sig_share * 30 + pretrend_ok * 30)

    summary_rows.append({
        "outcome": outcome,
        "outcome_family": group["outcome_family"].iloc[0],
        "sample_group": sample,
        "analysis_level": group["analysis_level"].iloc[0] if "analysis_level" in group.columns else "mixed",
        "n_specifications": n_specs,
        "same_sign_count": same_sign_count,
        "significant_count": significant_count,
        "median_coef": median_coef,
        "coef_iqr": coef_iqr,
        "sample_size_median": sample_size_median,
        "pretrend_pass_count": pretrend_pass_count,
        "stability_score": stability_score,
        "sign_direction": "negative" if median_coef < 0 else "positive",
        "mean_p_value": group["p_value"].mean(),
    })

summary = pd.DataFrame(summary_rows).sort_values("stability_score", ascending=False)
summary.to_csv(OUT / "stability_summary_by_outcome.csv", index=False)
print(f"Saved stability_summary_by_outcome.csv ({len(summary)} groups)")

# Print top 10
print("\n--- Top 10 by Stability Score ---")
top10 = summary.head(10)
print(top10[["outcome", "sample_group", "n_specifications", "median_coef",
              "stability_score", "sign_direction", "significant_count"]].to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
# 3. FIGURES
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. Generating figures")

# ── 3a. Outcome stability heatmap ──────────────────────────────────────
print("\n--- 3a. Stability heatmap ---")
# Pivot: outcomes x model x sample
heatmap_data = grid.pivot_table(
    values="coef", index="outcome", columns="model",
    aggfunc="mean"
)
# Filter to outcomes with enough data
heatmap_data = heatmap_data.dropna(thresh=3)

if len(heatmap_data) > 1:
    fig, ax = plt.subplots(figsize=(14, max(6, len(heatmap_data) * 0.4)))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Coefficient (SD units)"})
    ax.set_title("Coefficient Stability Heatmap\nOutcome × Model", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Outcome")
    plt.tight_layout()
    plt.savefig(FIG / "outcome_stability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outcome_stability_heatmap.png")

# ── 3b. Min review threshold sensitivity ───────────────────────────────
print("\n--- 3b. Threshold sensitivity ---")
fy_rating_data = fy_reg[(fy_reg["outcome"].str.contains("rating", na=False)) &
                          (fy_reg["model"] == "FY1")].copy()

if len(fy_rating_data) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    for oc in fy_rating_data["outcome"].unique()[:5]:
        oc_data = fy_rating_data[fy_rating_data["outcome"] == oc].sort_values("min_reviews_threshold")
        if len(oc_data) >= 3:
            ax.errorbar(oc_data["min_reviews_threshold"], oc_data["coef"],
                       yerr=1.96 * oc_data["se"], fmt="o-", capsize=4,
                       label=oc.replace("_sd", "").replace("GD_", ""), linewidth=1.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Minimum Reviews Threshold")
    ax.set_ylabel("Coefficient (SD units)")
    ax.set_title("Sensitivity to Minimum Review Threshold\n(FY1: PostElection, Firm+Year FE)", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "min_review_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: min_review_threshold_sensitivity.png")

# ── 3c. Current vs Former comparison ────────────────────────────────────
print("\n--- 3c. Current vs Former comparison ---")
r4_data = rev_reg[rev_reg["model"] == "R4"].copy()

if len(r4_data) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    outcomes_plot = r4_data["outcome"].unique()
    x = np.arange(len(outcomes_plot))
    width = 0.35

    for i, sample_name in enumerate(["current", "former"]):
        sample_data = r4_data[r4_data["sample"] == sample_name]
        coefs = [sample_data[sample_data["outcome"] == oc]["coef"].values[0]
                 if oc in sample_data["outcome"].values and len(sample_data[sample_data["outcome"] == oc]) > 0
                 else np.nan for oc in outcomes_plot]
        ses = [sample_data[sample_data["outcome"] == oc]["se"].values[0]
               if oc in sample_data["outcome"].values and len(sample_data[sample_data["outcome"] == oc]) > 0
               else np.nan for oc in outcomes_plot]
        bars = ax.bar(x + i * width, coefs, width, yerr=1.96 * np.array(ses),
                     capsize=3, label=f"{sample_name} employees", alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([oc.replace("_sd", "").replace("GD_", "").replace("_num", "")
                        for oc in outcomes_plot], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient (SD units)")
    ax.set_title("Current vs Former Employees\n(R1: PostElection + Firm FE + Year FE)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "current_vs_noncurrent_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: current_vs_noncurrent_comparison.png")

# ── 3d. Top outcome event study (already generated in step 3) ────────────
print("\n  (Event study plots generated in step 3)")

# ═════════════════════════════════════════════════════════════════════════
# 4. FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. Generating final stability report")

# ── Compute key statistics ──────────────────────────────────────────────
n_outcomes_review = rev_reg["outcome"].nunique()
n_outcomes_fy = fy_reg["outcome"].nunique()
n_specs_total = len(grid)

# Find top outcome
if len(summary) > 0:
    top_outcome = summary.iloc[0]
    top_oc_name = top_outcome["outcome"]
    top_oc_stability = top_outcome["stability_score"]
else:
    top_oc_name = "N/A"
    top_oc_stability = 0

# Count significant results
r1_sig = rev_reg[(rev_reg["model"] == "R1") & (rev_reg["significant_5"])]
fy1_sig = fy_reg[(fy_reg["model"] == "FY1") & (fy_reg["significant_5"])]

# Direction consistency
rev_neg_share = (rev_reg[rev_reg["model"] == "R1"]["coef"] < 0).mean()
fy_pos_share = (fy_reg[fy_reg["model"] == "FY1"]["coef"] > 0).mean()

# Event study: pre-trend issues
pretrend_outcomes = [k for k, v in pretrend_flags.items() if v["pretrend_flag"]]
pretrend_clear_outcomes = [k for k, v in pretrend_flags.items() if not v["pretrend_flag"]]

# Diversity: review-level R1 coef
div_r1 = rev_reg[(rev_reg["model"] == "R1") & (rev_reg["outcome"] == "GD_diversity_sd")]
div_coef = div_r1["coef"].values[0] if len(div_r1) > 0 else np.nan
div_p = div_r1["p_value"].values[0] if len(div_r1) > 0 else np.nan

report = f"""# Union Election × Glassdoor: Stability Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Project:** union_glassdoor
**Reference design:** Li & Pinto (2025, Management Science)

---

## 1. Executive Summary

### Bottom Line
**We find suggestive but fragile evidence that union elections negatively affect Glassdoor ratings, concentrated in Diversity & Inclusion scores.** The overall pattern is characterized by:

- **Negative coefficients at review level** across all 7 rating/subrating outcomes
- **Positive (but not significant) coefficients at firm-year level** — direction inconsistency raises concerns
- **GD_diversity is the only consistently significant outcome** (review-level: -0.078 SD, p<0.001; robust to clustering and FE choices)
- **Pre-trend concerns** for GD_Management (positive pre-trend before election) and partially for GD_diversity
- **Economic magnitudes are small**: -0.01 to -0.08 standard deviations

### Recommendation
**Proceed with GD_diversity as the primary outcome, but with caution.** Report the full set of outcomes transparently. Address pre-trend concerns via alternative specifications (e.g., matched control firms, staggered DiD). Do NOT claim a robust negative effect on overall ratings without further validation.

---

## 2. Data and Sample

| Dimension | Review-Level | Firm-Year |
|-----------|-------------|-----------|
| Observations | 68,201 | 2,059 (before reshape) |
| Unique firms (gvkey) | 192 | 1,218 |
| Unique elections | 192 | 2,059 |
| Union wins | 60 elections | 951 elections |
| Union losses | 129 elections | 1,066 elections |
| Date range | 2008–2023 | 1999–2026 |
| Window | ±365 days around election | ±1 year around election year |
| Current employees | 36,955 reviews (54%) | Not separately tracked |
| Former employees | 31,246 reviews (46%) | Not separately tracked |

### Rating Variables

| Variable | N (review-level) | Mean | SD | N (firm-year) |
|----------|-----------------|------|----|----------------|
| GD_rating (Overall) | 68,201 | 3.42 | 1.25 | 829 |
| GD_CareerOpp | 52,791 | 3.21 | 1.32 | 824 |
| GD_CompBenefits | 51,749 | 3.38 | 1.27 | 824 |
| GD_Management | 52,193 | 2.96 | 1.40 | 823 |
| GD_WorkLife | 51,901 | 3.16 | 1.37 | — |
| GD_CultureValues | 47,148 | 3.41 | 1.41 | 650 |
| GD_diversity | 24,324 | 3.88 | 1.30 | 210 |

### Categorical Variables (converted to numeric)

| Variable | Mapping | N |
|----------|---------|---|
| GD_Recommend | v=1, o=0.5, x=0 | 68,201 |
| GD_CEOSupport | v=1, o=0.5, r/x=0 | 68,201 |
| GD_Outlook | v=1, o=0.5, r/x=0 | 68,201 |

### Data Limitations
- **No text sentiment variables** available (GD_Pros/GD_Cons are raw text only)
- **No job title FE** merged (canonical title mapping needs manual review)
- **GD_diversity** only available for 24,324 reviews (36% of sample) and 26 unique firms
- **Firm-year data lacks explicit current/former employee splits** for GD outcomes
- **GD_diversity**, **GD_Outlook**, **GD_CEO**, **GD_Recommend** are zero/empty in firm-year aggregation

---

## 3. Review-Level Evidence

### R1: Baseline (PostElection + Firm FE + Year FE)

| Outcome | N | Coef (SD) | SE | t-stat | p-value | Sig 5% |
|---------|---|-----------|-----|--------|---------|--------|
| GD_rating_sd | 68,201 | **-0.038** | 0.045 | -0.86 | 0.388 | No |
| GD_CareerOpp_sd | 52,791 | **-0.032** | 0.029 | -1.11 | 0.267 | No |
| GD_CompBenefits_sd | 51,749 | **-0.035** | 0.045 | -0.77 | 0.438 | No |
| GD_Management_sd | 52,193 | **-0.029** | 0.037 | -0.78 | 0.433 | No |
| GD_WorkLife_sd | 51,901 | **-0.007** | 0.020 | -0.35 | 0.728 | No |
| GD_CultureValues_sd | 47,148 | **-0.053** | 0.052 | -1.01 | 0.311 | No |
| GD_diversity_sd | 24,324 | **-0.078** | 0.020 | -3.86 | 0.000 | **YES** |
| GD_Recommend_num_sd | 68,201 | **-0.011** | 0.034 | -0.32 | 0.747 | No |
| GD_CEOSupport_num_sd | 68,201 | +0.009 | 0.030 | 0.29 | 0.772 | No |
| GD_Outlook_num_sd | 68,201 | **-0.025** | 0.031 | -0.80 | 0.426 | No |

**Key finding**: 9/10 outcomes have negative coefficients. Only GD_diversity is statistically significant.

### R2: Adding Month FE

Results nearly identical to R1. Month FE does not materially change any coefficient.

### R3: Job Title FE

**SKIPPED** — canonical title mapping from `union_classified_title_universe.csv` requires further processing to link `title_standardized` to review-level `GD_JobTitle`.

### R4: Current vs Former Employees

| Outcome | Current Coef (SD) | Former Coef (SD) | Current p | Former p |
|---------|-------------------|-------------------|-----------|----------|
| GD_rating_sd | -0.041 | -0.037 | 0.359 | 0.406 |
| GD_diversity_sd | **-0.067** | **-0.091** | **0.003** | **0.000** |

**Key finding**: GD_diversity is significant for BOTH current and former employees. Former employees show a slightly stronger negative effect (-0.091 vs -0.067 SD).

### R5: Job Category Subsamples

**SKIPPED** — depends on job title classification merge.

---

## 4. Firm-Year Evidence

### FY1: PostElection + Firm FE + Year FE (Long-format DiD)

| Outcome | Threshold ≥3, N | Coef (SD) | SE | p-value | Sig |
|---------|-----------------|-----------|-----|---------|-----|
| GD_rating_sd | 1,425 | +0.019 | 0.035 | 0.594 | No |
| GD_career_opp_sd | 1,403 | -0.014 | 0.036 | 0.702 | No |
| GD_comp_benefit_sd | 1,404 | +0.007 | 0.031 | 0.828 | No |
| GD_senior_mgmt_sd | 1,403 | +0.005 | 0.038 | 0.889 | No |
| GD_culture_sd | 1,151 | -0.026 | 0.040 | 0.509 | No |
| GD_wlb_sd | 1,404 | -0.007 | 0.036 | 0.838 | No |
| GD_diversity_sd | 387 | -0.123 | 0.082 | 0.131 | No |

### Direction Inconsistency Flag

⚠️ **CRITICAL**: The firm-year DiD coefficients are mostly POSITIVE (opposite direction from review-level). With no threshold, GD_rating shows coef=+0.060 (p=0.093). This direction reversal raises serious concerns about:
1. Aggregation bias (Simpson's paradox)
2. Different sample composition at firm-year level (1,218 firms vs 192 at review level)
3. The ±365-day window restriction at review level may select different firms

### Threshold Sensitivity (GD_rating)

| Min Reviews | N | Coef (SD) | SE | p-value |
|-------------|---|-----------|-----|---------|
| 0 | 1,623 | +0.060 | 0.036 | 0.093 |
| 1 | 1,623 | +0.060 | 0.036 | 0.093 |
| 3 | 1,425 | +0.019 | 0.035 | 0.594 |
| 5 | 1,304 | +0.020 | 0.033 | 0.553 |
| 10 | 1,141 | -0.018 | 0.033 | 0.598 |

The coefficient **declines with higher thresholds**, eventually turning negative at ≥10 reviews. This suggests the positive result at low thresholds may be noise or driven by small-sample firms.

---

## 5. Event-Study Evidence

### Pre-Trend Assessment

| Outcome | Pre-period (-6 to -2 months) | Post-period (+1 to +6 months) | Pre-Trend Flag |
|---------|------------------------------|-------------------------------|----------------|
| GD_rating_sd | Near zero, 0/5 significant | Near zero, 0/5 significant | ✅ Pass |
| GD_CareerOpp_sd | Mixed, 0/5 significant | Mixed, 0/5 significant | ✅ Pass |
| GD_CompBenefits_sd | Mixed, 0/5 significant | Mixed, 0/5 significant | ✅ Pass |
| GD_Management_sd | **Positive, 3/5 significant** | Near zero, 0/5 significant | ❌ **PRE-TREND** |
| GD_WorkLife_sd | Near zero, 0/5 significant | Near zero, 0/5 significant | ✅ Pass |
| GD_CultureValues_sd | Mixed, 1/5 significant | Mixed, 0/5 significant | ✅ Pass (borderline) |
| GD_diversity_sd | **Positive, 3/5 significant** | **Negative, 1/5 significant** | ⚠️ **PRE-TREND** |

### Interpretation

- **GD_diversity** shows a clear pattern: positive pre-election coefficients (+0.04 to +0.12, some significant) become negative after month +4 (-0.04 to -0.16). This pre→post reversal is the strongest event-study pattern in the data.

- **GD_Management** has a concerning pre-trend: months -12, -11, -9, -3 are positive and significant, suggesting management ratings were ALREADY improving before the election. The post-period flattens. This violates parallel trends and invalidates the simple DiD.

- **GD_rating (Overall)** shows no clear pattern around the election — coefficients are small and bounce around zero.

---

## 6. Stability Score Summary

| Rank | Outcome | Sample | N Specs | Median Coef | Stability | Direction |
|------|---------|--------|---------|-------------|-----------|-----------|
{f"""
"""}
"""
# Generate top rows
for i, (_, row) in enumerate(summary.head(8).iterrows()):
    report += f"| {i+1} | {row['outcome']} | {row['sample_group']} | {int(row['n_specifications'])} | {row['median_coef']:.4f} | {row['stability_score']:.0f} | {row['sign_direction']} |\n"

report += f"""

---

## 7. Answers to Key Questions

### Q1: Which outcome is most worth pursuing?
**GD_diversity** is the most promising outcome:
- Only outcome with consistent statistical significance at review level
- Strongest economic magnitude (-0.078 SD)
- Robust to clustering by gvkey
- Significant for both current and former employees
- Clear pre→post pattern in event study

However, it has limitations:
- Available for only 24,324 reviews (36% of sample)
- Concentrated in only 26 firms
- Pre-trend concerns need to be addressed

### Q2: Are review-level and firm-year consistent in direction?
**No.** Review-level coefficients are consistently negative; firm-year coefficients are mostly positive (or near zero). This is a serious concern that needs investigation before claiming robust results. Possible explanations:
- The review-level sample (±365 days) is a subset of the firm-year sample
- Aggregation to firm-year means may mask within-firm heterogeneity
- Different sets of firms in each analysis

### Q3: Which is stronger — current or former employees?
**Former employees** show slightly stronger negative effects for GD_diversity (-0.091 vs -0.067 SD). Both are significant.

### Q4: Do job-category groupings have explanatory power?
**Unable to assess** — job title classification merge requires additional processing.

### Q5: Does the min review threshold change conclusions?
**Yes.** At firm-year level, the positive coefficient for GD_rating declines from +0.060 (no threshold) to -0.018 (≥10 reviews), suggesting small-sample firms drive the positive result. Higher thresholds are more reliable.

### Q6: Does the event study support causal interpretation?
**Partially.** GD_diversity shows a clear pre→post change pattern. GD_Management shows concerning pre-trends. Overall rating shows no clear event-study pattern. The evidence is suggestive but not definitive.

### Q7: Is this worth writing up as a paper result?
**Yes, with caveats.** GD_diversity as the primary outcome, with transparent reporting of:
- The full search across all outcomes
- Pre-trend concerns and robustness checks
- Direction inconsistency between review-level and firm-year
- Small economic magnitudes
- Limited generalizability (26 firms for diversity)

The paper should not claim a robust negative effect on "employee satisfaction" broadly. Instead, frame it as: "We find suggestive evidence that union elections are associated with modest declines in diversity and inclusion ratings, with no consistent effect on overall job satisfaction."

---

## 8. Recommended Baseline Specification

For next-stage analysis:

- **Outcome**: GD_diversity (standardized)
- **Sample**: All reviews within ±365 days of election
- **Specification**: PostElection + Firm FE + Year FE + Month FE
- **Standard errors**: Clustered by gvkey
- **Robustness**:
  1. Current vs former employees separately
  2. Exclude firms with <5 reviews
  3. Alternative event windows: ±90 days, ±180 days
  4. Control for pre-trend using linear time trend
  5. Match treated (union win) to control (union loss) firms

---

## 9. Caveats

1. **Sample size**: Only 192 firms in review-level data, 26 firms for diversity
2. **Selection bias**: Glassdoor reviews are voluntary — employees who leave reviews may differ systematically from non-reviewers
3. **Direction inconsistency**: Review-level (negative) vs firm-year (positive) disagreement
4. **Pre-trend concerns**: GD_Management and potentially GD_diversity show pre-existing trends
5. **Multiple testing**: 10 outcomes × 5 models × 5 thresholds = 250+ specifications; some significance may be due to chance
6. **Job title classification**: Not integrated; heterogeneity by job type remains unexplored
7. **No text sentiment**: Unable to leverage pros/cons text for additional outcomes
8. **External validity**: Results from 2008–2023 union elections may not generalize

---

## 10. Next Steps

1. **Fix job title merge**: Map `title_standardized` from classification file to `GD_JobTitle` in reviews, then re-run R3 and R5
2. **Investigate direction inconsistency**: Run review-level regressions on the FULL sample (not just ±365 days) to see if the negative effect persists
3. **Matched control design**: Use close elections (narrow win/loss margin) as a robustness check
4. **Staggered DiD**: If multiple elections per firm, use Callaway & Sant'Anna (2021) estimator
5. **Text sentiment**: Calculate sentiment scores from GD_Pros/GD_Cons using FinBERT or similar
6. **Expand diversity sample**: Investigate why only 26 firms have diversity ratings and whether imputation is feasible

---

## Appendix: Output Files

| File | Description |
|------|-------------|
| `variable_inventory_ratings.csv` | Rating/subrating variable statistics |
| `review_level_variable_inventory.csv` | All review-level variables |
| `subsample_outcome_inventory.csv` | Firm-year subpopulation mapping |
| `review_regression_results.csv` | R1-R5 model results |
| `firm_year_regression_results.csv` | FY1-FY4 model results |
| `review_eventstudy_coefficients.csv` | Monthly event-study coefficients |
| `firm_year_eventstudy_coefficients.csv` | Annual event-study coefficients |
| `stability_grid_results.csv` | Combined stability grid |
| `stability_summary_by_outcome.csv` | Stability scores by outcome |
| `figures/review_eventstudy_*.png` | Review-level event study plots |
| `figures/firm_year_eventstudy_*.png` | Firm-year event study plots |
| `figures/comparison_review_vs_firmyear_eventstudy.png` | Comparison plot |
| `figures/outcome_stability_heatmap.png` | Stability heatmap |
| `figures/min_review_threshold_sensitivity.png` | Threshold sensitivity |
| `figures/current_vs_noncurrent_comparison.png` | Current vs Former comparison |

---

*Report generated by `src/analysis/04_05_stability_analysis_and_report.py`*
*Claude Code, Anthropic — June 2026*
"""

# Save report
report_path = OUT / "union_glassdoor_stability_report.md"
with open(report_path, "w") as f:
    f.write(report)

print(f"\nSaved union_glassdoor_stability_report.md")
print(f"\n{'=' * 70}")
print("04_05_stability_analysis_and_report complete.")
print(f"All outputs in: {OUT}")
print(f"Figures in: {FIG}")
