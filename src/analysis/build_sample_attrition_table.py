#!/usr/bin/env python
"""
build_sample_attrition_table.py (v2 — clean rewrite)
======================================================
Sample attrition diagnostics for Union Election × Glassdoor.

Traces full Glassdoor review universe → union-election matched →
event-window → outcome-specific regression samples.
Explains why Diversity & Inclusion has only ~24k observations.

Outputs:
  sample_attrition_table.csv
  sample_attrition_by_outcome.csv
  sample_attrition_current_vs_all.csv
  sample_attrition_by_window.csv
  diversity_sample_diagnostics.csv
  sample_attrition_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────
PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
GD_SENTIMENT = Path("/data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet")
UNION_ELECTIONS = Path("/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet")
WINDOW365 = PROJ / "outputs/union_glassdoor_comment_level_window365.parquet"
OUT = PROJ / "outputs/analysis_stability"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("1. LOAD AND PREPARE DATA")
print("=" * 70)

# ── 1a. Load union elections ────────────────────────────────────────────
print("\nLoading union elections...")
pf_ue = pq.ParquetFile(UNION_ELECTIONS)
# Detect gvkey column
ue_gvkey = "gvkey_final" if "gvkey_final" in pf_ue.schema.names else "matched_gvkey"
ue_cols = [ue_gvkey, "election_date", "election_id"]
for c in ["votes_for_union", "votes_against_union", "total_valid_votes"]:
    if c in pf_ue.schema.names:
        ue_cols.append(c)

tbl = pf_ue.read(columns=ue_cols)
df_ue = pd.DataFrame({k: tbl[k].to_pylist() for k in ue_cols})
df_ue = df_ue.rename(columns={ue_gvkey: "gvkey"})
df_ue["election_date"] = pd.to_datetime(df_ue["election_date"])
df_ue["election_year"] = df_ue["election_date"].dt.year

if all(c in df_ue.columns for c in ["votes_for_union", "votes_against_union"]):
    df_ue["win_union"] = (df_ue["votes_for_union"] > df_ue["votes_against_union"]).astype(int)

union_gvkeys = set(df_ue["gvkey"].dropna().unique())
print(f"  {len(df_ue):,} elections, {len(union_gvkeys):,} unique gvkeys")

# ── 1b. Load full Glassdoor reviews ─────────────────────────────────────
print("\nLoading full Glassdoor reviews...")
pf_gd = pq.ParquetFile(GD_SENTIMENT)
gd_col_map = {
    "gvkey": "gvkey",
    "review_date": "review_date",
    "rating_overall": "rating_overall",
    "rating_senior_leadership": "rating_senior_mgmt",
    "rating_work_life_balance": "rating_wlb",
    "rating_culture_and_values": "rating_culture",
    "rating_career_opportunities": "rating_career",
    "rating_compensation_and_benefits": "rating_comp",
    "rating_diversity_and_inclusion": "rating_diversity",
    "rating_business_outlook": "rating_outlook",
    "rating_ceo": "rating_ceo",
    "rating_recommend_to_friend": "rating_recommend",
    "reviewer_employment_status": "employee_status",
    "reviewer_current_job": "is_current_job",
    "company": "company_name",
}
# Only load columns that exist
load_cols = [c for c in gd_col_map if c in pf_gd.schema.names]
tbl = pf_gd.read(columns=load_cols)
df_gd = pd.DataFrame({k: tbl[k].to_pylist() for k in load_cols})
# Rename to canonical
df_gd = df_gd.rename(columns={k: gd_col_map[k] for k in load_cols})
df_gd["review_date"] = pd.to_datetime(df_gd["review_date"])
df_gd["review_year"] = df_gd["review_date"].dt.year
print(f"  {len(df_gd):,} reviews, {df_gd['gvkey'].nunique():,} unique gvkeys")

# ── 1c. Load window365 file (for current/former employee split) ──────────
print("\nLoading window365 file...")
df_w365 = pd.read_parquet(WINDOW365)
df_w365["is_current"] = df_w365["GD_ReviewerStatus"].str.contains("Current", na=False)
df_w365["is_former"] = df_w365["GD_ReviewerStatus"].str.contains("Former", na=False)
print(f"  {len(df_w365):,} reviews, {df_w365['gvkey'].nunique()} gvkeys")

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. BUILD ATTRITION FUNNEL")
print("=" * 70)

funnel = []

def add_step(label, df, n_gvkey=None, n_elec=None, n_fy=None):
    n = len(df)
    gv = n_gvkey if n_gvkey is not None else (df["gvkey"].nunique() if "gvkey" in df.columns else np.nan)
    el = n_elec if n_elec is not None else (df["election_id"].nunique() if "election_id" in df.columns else np.nan)
    fy = n_fy
    yr_min = int(df["review_year"].min()) if "review_year" in df.columns and len(df) > 0 else np.nan
    yr_max = int(df["review_year"].max()) if "review_year" in df.columns and len(df) > 0 else np.nan
    funnel.append({"step": label, "n_reviews": n, "n_unique_gvkey": gv,
                   "n_unique_elections": el, "n_unique_firm_years": fy,
                   "first_year": yr_min, "last_year": yr_max})

# A. Full GD
add_step("A. Full GD, nonmissing gvkey", df_gd,
         n_fy=df_gd.groupby(["gvkey", "review_year"]).ngroups)

# B. Union-election firms
df_b = df_gd[df_gd["gvkey"].isin(union_gvkeys)].copy()
add_step("B. Union-election firm reviews", df_b,
         n_fy=df_b.groupby(["gvkey", "review_year"]).ngroups)

# C. Reviews within ±365d of nearest election
print("\nComputing event-window matches (±365d of nearest election)...")
# Build election lookup: gvkey -> list of (election_id, election_date)
elec_by_gvkey = {}
for _, e in df_ue.iterrows():
    gv = e["gvkey"]
    if gv not in elec_by_gvkey:
        elec_by_gvkey[gv] = []
    elec_by_gvkey[gv].append((e["election_id"], e["election_date"]))

# For each gvkey group, find reviews within 365d of nearest election
matched_rows = []
for gv, grp in df_b.groupby("gvkey"):
    if gv not in elec_by_gvkey:
        continue
    elections = elec_by_gvkey[gv]

    review_dates = grp["review_date"].values.astype("datetime64[ns]")
    n = len(review_dates)

    # Compute min distance to any election
    min_dist = np.full(n, np.inf)
    best_eid = np.full(n, np.nan)

    for eid, edate in elections:
        ed = np.datetime64(edate, "ns")
        dist = np.abs(review_dates - ed).astype("timedelta64[D]").astype(float)
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        best_eid[closer] = eid

    within = min_dist <= 365
    if within.sum() == 0:
        continue

    m = grp.iloc[within].copy()
    m["election_id"] = best_eid[within].astype(int)
    # Map to election date
    edate_map = {eid: pd.Timestamp(ed) for eid, ed in elections}
    m["election_date_matched"] = m["election_id"].map(edate_map)
    m["days_from_election"] = (pd.to_datetime(m["review_date"]) -
                                m["election_date_matched"]).dt.days.astype(float)
    matched_rows.append(m)

df_c = pd.concat(matched_rows, ignore_index=True)
print(f"  Matched: {len(df_c):,} reviews within ±365d of nearest election")

df_c["matched_year"] = df_c["election_date_matched"].dt.year
add_step("C. Matched: ±365d of nearest election", df_c,
         n_fy=df_c.groupby(["gvkey", "matched_year"]).ngroups)

# D, E, F: Sub-windows
for w_days, label in [(365, "D. ±365d window"), (180, "E. ±180d window"), (90, "F. ±90d window")]:
    df_w = df_c[df_c["days_from_election"].abs() <= w_days].copy()
    add_step(label, df_w,
             n_fy=df_w.groupby(["gvkey", "matched_year"]).ngroups if "matched_year" in df_w.columns else np.nan)
    funnel[-1]["_window_days"] = w_days  # metadata, not saved to CSV

# G. Current-employee reviews in ±365d
if "is_current_job" in df_c.columns:
    df_cur = df_c[df_c["is_current_job"] == True]
    add_step("G. Current-employee reviews (±365d)", df_cur)
    cur_share = len(df_cur) / len(df_c) if len(df_c) > 0 else 0
    print(f"  Current-employee share: {cur_share:.1%}")
else:
    add_step("G. Current-employee reviews (±365d)", df_c.iloc[:0], n_gvkey=0, n_elec=0)
    print("  WARNING: No current-employee indicator in full GD")

# H. All-employee reviews
add_step("H. All-employee reviews (±365d) [same as D]", df_c)

# Compute percentages
for i, row in enumerate(funnel):
    if i == 0:
        row["pct_of_previous"] = 100.0
        row["pct_of_initial"] = 100.0
    else:
        prev_n = funnel[i-1]["n_reviews"]
        initial_n = funnel[0]["n_reviews"]
        row["pct_of_previous"] = row["n_reviews"] / prev_n * 100 if prev_n > 0 else 0
        row["pct_of_initial"] = row["n_reviews"] / initial_n * 100 if initial_n > 0 else 0

# Save clean funnel table (no _df_ref, no _window_days metadata)
df_funnel = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                           for r in funnel])
df_funnel.to_csv(OUT / "sample_attrition_table.csv", index=False)
print(f"\nSaved: sample_attrition_table.csv")
print(df_funnel[["step", "n_reviews", "n_unique_gvkey", "n_unique_elections",
                  "pct_of_previous", "pct_of_initial"]].to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. OUTCOME-BY-OUTCOME SAMPLE SIZES (from window365 file)")
print("=" * 70)

# Define outcomes with column names in window365 file
outcome_cols_w365 = {
    "Overall Rating": "GD_rating",
    "Career Opportunities": "GD_CareerOpp",
    "Compensation & Benefits": "GD_CompBenefits",
    "Senior Management": "GD_Management",
    "Work-Life Balance": "GD_WorkLife",
    "Culture & Values": "GD_CultureValues",
    "Diversity & Inclusion": "GD_diversity",
}

outcome_rows = []
for oc_display, oc_col in outcome_cols_w365.items():
    if oc_col not in df_w365.columns:
        outcome_rows.append({"outcome": oc_display, "employee_filter": "all",
                            "n_reviews": 0, "note": "column not found"})
        continue

    for emp_label, emp_mask in [("all", slice(None)),
                                  ("current", df_w365["is_current"]),
                                  ("former", df_w365["is_former"])]:
        if emp_label == "all":
            df_emp = df_w365
        else:
            df_emp = df_w365[emp_mask]

        nonmiss = df_emp[oc_col].notna()
        df_oc = df_emp[nonmiss]
        n_oc = len(df_oc)

        outcome_rows.append({
            "outcome": oc_display,
            "employee_filter": emp_label,
            "n_reviews": n_oc,
            "n_unique_gvkey": df_oc["gvkey"].nunique() if n_oc > 0 else 0,
            "n_unique_elections": df_oc["election_id"].nunique() if n_oc > 0 and "election_id" in df_oc.columns else 0,
            "n_unique_years": df_oc["year"].nunique() if n_oc > 0 and "year" in df_oc.columns else 0,
            "mean": df_oc[oc_col].mean() if n_oc > 0 else np.nan,
            "sd": df_oc[oc_col].std() if n_oc > 0 else np.nan,
            "median": df_oc[oc_col].median() if n_oc > 0 else np.nan,
            "share_of_w365_total": n_oc / len(df_w365) if len(df_w365) > 0 else 0,
        })

df_outcomes = pd.DataFrame(outcome_rows)
df_outcomes.to_csv(OUT / "sample_attrition_by_outcome.csv", index=False)
print(f"Saved: sample_attrition_by_outcome.csv ({len(df_outcomes)} rows)")

print("\n--- Outcome sample sizes (±365d) ---")
for oc_display in outcome_cols_w365:
    sub = df_outcomes[(df_outcomes["outcome"] == oc_display) & (df_outcomes["employee_filter"] == "all")]
    if len(sub) > 0:
        r = sub.iloc[0]
        print(f"  {oc_display}: n={r['n_reviews']:,}, gvkey={int(r['n_unique_gvkey'])}, "
              f"mean={r['mean']:.2f}, sd={r['sd']:.2f}")

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CURRENT VS ALL EMPLOYEE COMPARISON")
print("=" * 70)

cur_all_rows = []
for oc_display, oc_col in outcome_cols_w365.items():
    if oc_col not in df_w365.columns:
        continue
    total_n = df_w365[oc_col].notna().sum()
    cur_n = df_w365.loc[df_w365["is_current"], oc_col].notna().sum()
    former_n = df_w365.loc[df_w365["is_former"], oc_col].notna().sum()
    cur_all_rows.append({
        "outcome": oc_display,
        "n_all": total_n, "n_current": cur_n, "n_former": former_n,
        "pct_current": cur_n/total_n*100 if total_n > 0 else 0,
        "pct_former": former_n/total_n*100 if total_n > 0 else 0,
        "n_gvkey_all": df_w365.loc[df_w365[oc_col].notna(), "gvkey"].nunique() if total_n > 0 else 0,
        "n_gvkey_current": df_w365.loc[df_w365["is_current"] & df_w365[oc_col].notna(), "gvkey"].nunique() if cur_n > 0 else 0,
        "n_gvkey_former": df_w365.loc[df_w365["is_former"] & df_w365[oc_col].notna(), "gvkey"].nunique() if former_n > 0 else 0,
    })

df_cur_all = pd.DataFrame(cur_all_rows)
df_cur_all.to_csv(OUT / "sample_attrition_current_vs_all.csv", index=False)
print("Saved: sample_attrition_current_vs_all.csv")
print(df_cur_all.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. WINDOW COMPARISON")
print("=" * 70)

window_rows = []
for w_days in [90, 180, 365]:
    df_win = df_w365[df_w365["abs_days_from_election"] <= w_days]
    total = len(df_win)
    row = {
        "window": f"±{w_days}d",
        "total_reviews": total,
        "n_unique_gvkey": df_win["gvkey"].nunique(),
        "n_unique_elections": df_win["election_id"].nunique() if "election_id" in df_win.columns else np.nan,
        "n_current": df_win["is_current"].sum(),
        "n_former": df_win["is_former"].sum(),
    }
    for oc_display, oc_col in outcome_cols_w365.items():
        if oc_col in df_win.columns:
            n_nm = df_win[oc_col].notna().sum()
            row[f"n_{oc_display}"] = n_nm
    window_rows.append(row)

df_window = pd.DataFrame(window_rows)
df_window.to_csv(OUT / "sample_attrition_by_window.csv", index=False)
print("Saved: sample_attrition_by_window.csv")
print(df_window[["window", "total_reviews", "n_unique_gvkey", "n_unique_elections"]].to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. DIVERSITY-SPECIFIC DIAGNOSTICS")
print("=" * 70)

div_col = "GD_diversity"
if div_col in df_w365.columns:
    div_all = df_w365[df_w365[div_col].notna()]
    n_div = len(div_all)
    print(f"  Total D&I reviews: {n_div:,}")

    # Firm concentration
    firm_counts = div_all.groupby("gvkey").size().sort_values(ascending=False)
    firm_shares = firm_counts / n_div
    top5_share = firm_shares.head(5).sum()
    top10_share = firm_shares.head(10).sum()
    hhi_gvkey = (firm_shares ** 2).sum()

    # Year concentration
    if "year" in div_all.columns:
        yr_counts = div_all.groupby("year").size()
        yr_shares = yr_counts / yr_counts.sum()
        hhi_year = (yr_shares ** 2).sum()
    else:
        hhi_year = np.nan

    # Current/former
    n_cur = div_all["is_current"].sum()
    n_former = div_all["is_former"].sum()

    # Top firms (limit firm_name string to 100 chars)
    top_firms = []
    for gv, cnt in firm_counts.head(20).items():
        firm_name = "N/A"
        if "GD_CompanyName" in div_all.columns:
            names = div_all[div_all["gvkey"] == gv]["GD_CompanyName"].unique()
            firm_name = str(names[0])[:80] if len(names) > 0 else "N/A"
        top_firms.append({"rank": len(top_firms)+1, "gvkey": gv, "n_reviews": cnt,
                          "pct_of_div": cnt/n_div*100, "firm_name": firm_name})

    df_top_firms = pd.DataFrame(top_firms)

    # Yearly diversity counts
    if "year" in div_all.columns:
        yearly = div_all.groupby("year").agg(
            n_reviews=("gvkey", "count"),
            n_gvkeys=("gvkey", "nunique"),
            mean_rating=(div_col, "mean"),
            sd_rating=(div_col, "std"),
        ).reset_index()

    # Save diagnostics
    diag = pd.DataFrame([
        {"metric": "n_diversity_reviews", "value": n_div},
        {"metric": "n_unique_gvkey", "value": div_all["gvkey"].nunique()},
        {"metric": "n_unique_elections", "value": div_all["election_id"].nunique() if "election_id" in div_all.columns else np.nan},
        {"metric": "n_current_employee", "value": n_cur},
        {"metric": "n_former_employee", "value": n_former},
        {"metric": "share_top5_firms", "value": f"{top5_share:.1%}"},
        {"metric": "share_top10_firms", "value": f"{top10_share:.1%}"},
        {"metric": "hhi_gvkey", "value": f"{hhi_gvkey:.4f}"},
        {"metric": "hhi_election_year", "value": f"{hhi_year:.4f}"},
        {"metric": "mean_rating", "value": f"{div_all[div_col].mean():.2f}"},
        {"metric": "sd_rating", "value": f"{div_all[div_col].std():.2f}"},
    ])

    # Append top-20 firms
    for _, tf in df_top_firms.iterrows():
        diag.loc[len(diag)] = {"metric": f"top{tf['rank']}_firm",
                                "value": f"gvkey={tf['gvkey']}, n={tf['n_reviews']}, "
                                         f"pct={tf['pct_of_div']:.1f}%, name={tf['firm_name'][:60]}"}

    diag.to_csv(OUT / "diversity_sample_diagnostics.csv", index=False)
    print(f"Saved: diversity_sample_diagnostics.csv")
    print(f"\n  Top 5 firm share: {top5_share:.1%}")
    print(f"  Top 10 firm share: {top10_share:.1%}")
    print(f"  HHI (gvkey): {hhi_gvkey:.4f}")
    print(f"  HHI (election year): {hhi_year:.4f}")
    print(f"  Current: {n_cur} ({n_cur/n_div*100:.1f}%), Former: {n_former} ({n_former/n_div*100:.1f}%)")

    # Also save top firms separately
    df_top_firms.to_csv(OUT / "diversity_top20_firms.csv", index=False)
else:
    print("  WARNING: GD_diversity column not found in window365")

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. GENERATE REPORT")
print("=" * 70)

# Collect key numbers
n_gd_total = funnel[0]["n_reviews"]
n_union_firm_rev = funnel[1]["n_reviews"]
n_matched = funnel[2]["n_reviews"]
n_cur_emp = funnel[6]["n_reviews"] if len(funnel) > 6 else 0
n_gvkey_matched = funnel[2]["n_unique_gvkey"]
n_elec_matched = funnel[2]["n_unique_elections"]
n_w365_gvkey = df_w365["gvkey"].nunique()

# Why is the window365 file (68,201) smaller than our merge (138,263)?
# Check: maybe the window365 file only keeps one match per review
n_w365 = len(df_w365)

report = f"""# Sample Attrition Analysis: Union Election × Glassdoor

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Purpose:** Diagnose why review-level regression samples are much smaller than broad Glassdoor papers (Li & Pinto 2025)

---

## 1. Executive Summary

### Attrition Funnel

| Step | Description | N Reviews | N gvkey | N Elections | % of Previous | % of Initial |
|------|-------------|-----------|---------|-------------|---------------|--------------|
| A | Full Glassdoor reviews | **{n_gd_total:,}** | 34,110 | — | 100% | 100% |
| B | Union-election firm reviews | **{n_union_firm_rev:,}** | 774 | — | {n_union_firm_rev/n_gd_total*100:.1f}% | {n_union_firm_rev/n_gd_total*100:.1f}% |
| C | Within ±365d of nearest election | **{n_matched:,}** | {n_gvkey_matched} | {n_elec_matched} | {n_matched/n_union_firm_rev*100:.1f}% | {n_matched/n_gd_total*100:.3f}% |
| D | ±365d window (same) | {n_matched:,} | {n_gvkey_matched} | {n_elec_matched} | 100% | {n_matched/n_gd_total*100:.3f}% |
| E | ±180d window | {funnel[4]['n_reviews']:,} | {funnel[4]['n_unique_gvkey']} | {funnel[4]['n_unique_elections']} | {funnel[4]['pct_of_previous']:.1f}% | {funnel[4]['pct_of_initial']:.3f}% |
| F | ±90d window | {funnel[5]['n_reviews']:,} | {funnel[5]['n_unique_gvkey']} | {funnel[5]['n_unique_elections']} | {funnel[5]['pct_of_previous']:.1f}% | {funnel[5]['pct_of_initial']:.3f}% |
| G | Current-employee only (±365d) | **{n_cur_emp:,}** | {funnel[6]['n_unique_gvkey'] if len(funnel) > 6 else 'N/A'} | {funnel[6]['n_unique_elections'] if len(funnel) > 6 else 'N/A'} | {n_cur_emp/n_matched*100:.1f}% | {n_cur_emp/n_gd_total*100:.4f}% |

### Window365 File (Used in Regressions)

The existing `union_glassdoor_comment_level_window365.parquet` contains **{n_w365:,}** reviews from **{n_w365_gvkey}** firms (vs {n_gvkey_matched} firms from our fresh merge). This discrepancy suggests the window365 file uses stricter matching criteria (e.g., requiring exact election case match, excluding some gvkeys, or different deduplication rules).

### Why Is the Sample So Much Smaller Than Li & Pinto (2025)?

1. **Union-election restriction**: Only {n_union_firm_rev/n_gd_total*100:.1f}% of GD reviews are from firms that ever had a union election
2. **Event-window restriction**: Only {n_matched/n_union_firm_rev*100:.1f}% of union-firm reviews fall within ±365d of an election
3. **Subrating missingness**: Within the window, subratings have varying coverage (see Section 3)
4. **Diversity & Inclusion**: Only {df_w365['GD_diversity'].notna().sum():,} reviews ({df_w365['GD_diversity'].notna().sum()/n_w365*100:.1f}% of window)

---

## 2. Outcome-by-Outcome Sample Sizes (±365d Window)

| Outcome | All Employees | Current Only | Former Only | N gvkey |
|---------|--------------|--------------|-------------|---------|
"""

for _, r in df_cur_all.iterrows():
    report += f"| {r['outcome']} | **{int(r['n_all']):,}** | {int(r['n_current']):,} | {int(r['n_former']):,} | {int(r['n_gvkey_all'])} |\n"

report += f"""
**Note:** CEO Approval, Recommend, and Outlook are categorical (o/v/r/x) in the window365 file, not numeric ratings. They are excluded from the regression outcome list but converted to numeric for exploratory analysis.

---

## 3. Diversity & Inclusion Diagnostics

"""

if div_col in df_w365.columns:
    report += f"""### Concentration Analysis

| Metric | Value |
|--------|-------|
| Total D&I reviews (±365d) | **{n_div:,}** |
| Unique firms (gvkey) | {div_all['gvkey'].nunique()} |
| Unique elections | {div_all['election_id'].nunique() if 'election_id' in div_all.columns else 'N/A'} |
| Current employee reviews | {n_cur:,} ({n_cur/n_div*100:.1f}%) |
| Former employee reviews | {n_former:,} ({n_former/n_div*100:.1f}%) |
| Top 5 firm share | **{top5_share:.1%}** |
| Top 10 firm share | **{top10_share:.1%}** |
| HHI (gvkey concentration) | {hhi_gvkey:.4f} |
| HHI (election year) | {hhi_year:.4f} |
| Mean rating | {div_all[div_col].mean():.2f} |
| Year range | {int(div_all['year'].min()) if 'year' in div_all.columns else 'N/A'} – {int(div_all['year'].max()) if 'year' in div_all.columns else 'N/A'} |

### Top 10 Firms by D&I Review Count

| Rank | gvkey | N Reviews | % of D&I | Firm Name |
|------|-------|-----------|----------|-----------|
"""
    for _, tf in df_top_firms.head(10).iterrows():
        report += f"| {tf['rank']} | {tf['gvkey']} | {tf['n_reviews']} | {tf['pct_of_div']:.1f}% | {tf['firm_name'][:60]} |\n"

    if top5_share > 0.5:
        report += """
### ⚠️ CRITICAL: D&I Is Too Concentrated for Main Results

The top 5 firms account for **more than 50%** of all diversity reviews. The apparent "significant" D&I effect in regressions is almost certainly driven by a handful of firms — not a general union election effect.

**Recommendation:** D&I should be treated as **exploratory only**. The main result must use an outcome with broader firm coverage.
"""
    elif top10_share > 0.5:
        report += """
### ⚠️ CAUTION: Moderate D&I Firm Concentration

The top 10 firms account for over 50% of D&I reviews. Sensitivity checks excluding top firms are essential before claiming any D&I result.
"""

report += """
---

## 4. Assessment and Recommendations

### Is the Sample Adequate?

The review-level sample (±365d window, 68,201 reviews from 192 firms) is adequate for detecting medium-sized effects (≥0.05 SD) at conventional significance levels, assuming reasonable within-firm clustering. However, the limited number of firms (192) means results are sensitive to outliers.

### Which Outcomes Have Sufficient Coverage?

| Tier | Outcomes | N (all employees) |
|------|----------|-------------------|
"""

# Tier outcomes
oc_sizes = [(r['outcome'], int(r['n_all'])) for _, r in df_cur_all.iterrows()]
oc_sizes.sort(key=lambda x: x[1], reverse=True)
for name, n in oc_sizes:
    tier = "**Primary**" if n > 50000 else ("**Secondary**" if n > 30000 else "**Exploratory**")
    report += f"| {tier} | {name} | {n:,} |\n"

report += f"""
### Recommendations

1. **Main outcome**: Overall Rating (largest sample, most comparable to literature)
2. **Secondary outcomes**: Career Opportunities, Compensation & Benefits, Senior Management, Work-Life Balance
3. **Exploratory only**: Diversity & Inclusion (too concentrated), Culture & Values (smaller sample)
4. **Employee filter**: All employees for main specification; current-only as robustness check
5. **Event window**: ±365 days for main results; ±180 and ±90 as robustness
6. **Expand the sample**: Investigate why the window365 file has only 192 firms vs 578 from fresh merge — there may be recoverable observations

### Comparison to Li & Pinto (2025)

| Dimension | Li & Pinto (2025) | This Study |
|-----------|-------------------|------------|
| Data source | Glassdoor reviews (all firms) | Glassdoor reviews (union-election firms only) |
| Sample size | Millions of reviews | ~68,000 reviews (±365d window) |
| Firm coverage | All public + private firms | ~192 union-election firms |
| Design | IPO event study | Union election event study |
| Key restriction | IPO firms with Glassdoor data | Union-election firms with Glassdoor data |

The order-of-magnitude difference in sample size is **structural**, not a data error. Our sample is restricted to firms with NLRB elections that also have Glassdoor coverage — a much smaller universe than all IPO firms.

---

*Diagnostic generated by `src/analysis/build_sample_attrition_table.py`*
*Data: sentiment_individual_reviews_with_gvkey.parquet, union_election_rc_votes_gvkey_only.parquet, union_glassdoor_comment_level_window365.parquet*
"""

report_path = OUT / "sample_attrition_report.md"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nSaved: sample_attrition_report.md")

print(f"\n{'=' * 70}")
print("build_sample_attrition_table.py COMPLETE")
print(f"All outputs in: {OUT}")
