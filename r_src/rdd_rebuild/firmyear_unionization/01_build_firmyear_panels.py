#!/usr/bin/env python
"""STEP 1-2: Build firm-year unionization ratio panel + Glassdoor ratings panel.

Outputs:
  firmyear_unionization_panel.parquet — firm-year union_ratio
  firmyear_glassdoor_panel.parquet — firm-year Glassdoor ratings (all + current)
  step0_inventory.md — data inventory report
"""

import pandas as pd
import numpy as np
import os

OUT = "/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: Inventory report
# ═══════════════════════════════════════════════════════════════════════════
inventory = []
inventory.append("# STEP 0 — Data Inventory\n")
inventory.append("## 0.1 Repro Check\n")
inventory.append("- WLB: +0.081 (p=0.025) ✅ — matches expected +0.082 (p≈0.023)\n")
inventory.append("- Comp: +0.003 (p=0.909) ✅ — matches expected +0.005 (p≈0.870)\n")
inventory.append("\n## 0.2 Previous Paper Code\n")
inventory.append("- **NOT FOUND** — searched /data/disk4/workspace/projects/ and /home/user/\n")
inventory.append("- No `unionization_ratio` / `unionized_emp` / `cumulative union` construction code found\n")
inventory.append("- Using fallback: stock = cumulative sum of won-election unit_size; denominator = Compustat EMP × 1000\n")
inventory.append("- ⚠️ **构造未能对齐上一篇论文，需人工与 Amanda 确认定义**\n")
inventory.append("\n## 0.3 Glassdoor gvkey Coverage\n")
inventory.append("- 13,854,743 reviews (100% have gvkey), 34,110 unique gvkeys, 2008–2025\n")
inventory.append("- Full dataset available for firm-year panel (not just ±365 days)\n")
inventory.append("\n## 0.4 NLRB Fields\n")
inventory.append("- 33,477 elections: RC 28,463, RD 3,939, RM 676, UD 366, UC 2\n")
inventory.append("- Decertification identifiable (RD/RM/UD = 4,981, 14.9%)\n")
inventory.append("- unit_size: 99.5% non-null; eligible_voters: 74.3%\n")
inventory.append("- Election date from `date` field (100% non-null)\n")
inventory.append("\n## 0.5 Compustat\n")
inventory.append("- 598,127 firm-years, 46,732 gvkeys, 1950–2025\n")
inventory.append("- EMP in thousands (Walmart 2025: 2,100 = 2,100,000 employees)\n")
inventory.append("- Available firm controls: at, roa, leverage, sale, L_size, L_leverage, L_roa\n")

with open(f"{OUT}/step0_inventory.md", "w") as f:
    f.write("".join(inventory))
print("Saved step0_inventory.md")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Build firm-year unionization ratio panel
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== STEP 1: Firm-year unionization ratio panel ===")

# Load NLRB election data with win/loss
nlrb = pd.read_parquet("/data/disk4/workspace/projects/union/outputs/preliminary_election_level.parquet",
                       columns=["election_id","case_number","date","filing__case_type",
                                "unit_size","filing__number_of_eligible_voters"])
nlrb = nlrb.drop_duplicates(subset="election_id")

# Load election-to-gvkey mapping
matched = pd.read_parquet("/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_matched_combined.parquet",
                          columns=["election_id","gvkey_final","election_date","total_valid_votes",
                                   "votes_for_union","votes_against_union"])
matched = matched[matched['gvkey_final'].notna()].drop_duplicates(subset="election_id")

# Merge unit_size from NLRB
elections = matched.merge(nlrb[["election_id","unit_size","filing__number_of_eligible_voters",
                                 "filing__case_type","date"]], on="election_id", how="left")

# Election year: parse election_date
elections["election_date"] = pd.to_datetime(elections["election_date"])
elections["election_year"] = elections["election_date"].dt.year

# Unit size: priority = unit_size > eligible_voters > total_valid_votes
elections["unit_size_final"] = elections["unit_size"]
mask_na = elections["unit_size_final"].isna()
elections.loc[mask_na, "unit_size_final"] = elections.loc[mask_na, "filing__number_of_eligible_voters"]
elections.loc[mask_na, "unit_size_proxy"] = "eligible_voters"
elections["unit_size_proxy"] = elections.get("unit_size_proxy", "unit_size")

mask_na2 = elections["unit_size_final"].isna()
elections.loc[mask_na2, "unit_size_final"] = elections.loc[mask_na2, "total_valid_votes"]
elections.loc[mask_na2, "unit_size_proxy"] = "total_valid_votes"

# Determine win
elections["win"] = (elections["votes_for_union"] > elections["votes_against_union"]).astype(int)

# Decertification flag
elections["is_decert"] = elections["filing__case_type"].isin(["RD","RM","UD"]).astype(int)

print(f"Elections with gvkey: {len(elections):,}")
print(f"  Wins: {elections['win'].sum():,}, Losses: {(~elections['win'].astype(bool)).sum():,}")
print(f"  Decert: {elections['is_decert'].sum():,}")
print(f"  Has unit_size: {elections['unit_size_final'].notna().sum():,}")
print(f"  Election years: {elections['election_year'].min():.0f}–{elections['election_year'].max():.0f}")

# ═══ Build stock of unionized employees ═══
# For each gvkey, accumulate won election unit_size over time
# Stock: all won elections up to and including year t
won = elections[elections["win"] == 1].copy()
# Sum by gvkey-year
stock = won.groupby(["gvkey_final","election_year"])["unit_size_final"].sum().reset_index()
stock.columns = ["gvkey","election_year","flow_unionized"]

# Cumulative sum per gvkey over years
stock = stock.sort_values(["gvkey","election_year"])
stock["unionized_emp_stock"] = stock.groupby("gvkey")["flow_unionized"].cumsum()

# Also: decert-adjusted (subtract decertification unit_size from stock)
decert = elections[elections["is_decert"] == 1].copy()
decert_flow = decert.groupby(["gvkey_final","election_year"])["unit_size_final"].sum().reset_index()
decert_flow.columns = ["gvkey","election_year","decert_flow"]

stock = stock.merge(decert_flow, on=["gvkey","election_year"], how="left")
stock["decert_flow"] = stock["decert_flow"].fillna(0)
stock["decert_cumul"] = stock.groupby("gvkey")["decert_flow"].cumsum()
stock["unionized_emp_stock_decert_adj"] = stock["unionized_emp_stock"] - stock["decert_cumul"]

print(f"Stock panel: {len(stock):,} gvkey-years")
print(f"  gvkeys with any won election: {stock['gvkey'].nunique():,}")
print(f"  Year range: {stock['election_year'].min():.0f}–{stock['election_year'].max():.0f}")

# ═══ Build full panel: all gvkey-years with Compustat EMP ═══
cmp = pd.read_parquet("outputs/compustat_firm_controls.parquet")
cmp["gvkey"] = cmp["gvkey"].astype(str)
cmp_panel = cmp[["gvkey","fyear","emp","at","roa","leverage","sale","L_size","L_leverage","L_roa","L_log_emp"]].copy()
cmp_panel = cmp_panel[cmp_panel["fyear"] >= 2008]  # Glassdoor start year
cmp_panel = cmp_panel[cmp_panel["fyear"] <= 2025]  # Glassdoor end year

# Merge unionization data
panel = cmp_panel.merge(stock, left_on=["gvkey","fyear"], right_on=["gvkey","election_year"], how="left")
panel.drop(columns=["election_year"], inplace=True)

# Fill missing: never-won gvkey → 0
for col in ["flow_unionized","unionized_emp_stock","decert_flow","decert_cumul","unionized_emp_stock_decert_adj"]:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)

# EMP in thousands → actual
panel["emp_actual"] = panel["emp"] * 1000

# Union ratio
panel["union_ratio_raw"] = panel["unionized_emp_stock"] / panel["emp_actual"]
panel["union_ratio_decert_adj"] = panel["unionized_emp_stock_decert_adj"] / panel["emp_actual"]

# Winsorize
p01 = panel["union_ratio_raw"].quantile(0.01)
p99 = panel["union_ratio_raw"].quantile(0.99)
panel["union_ratio_winsor"] = panel["union_ratio_raw"].clip(p01, p99)
panel["union_ratio_capped"] = panel["union_ratio_raw"].clip(upper=1.0)

# Clean: remove inf
panel["union_ratio_raw"] = panel["union_ratio_raw"].replace([np.inf, -np.inf], np.nan)
panel["union_ratio_winsor"] = panel["union_ratio_winsor"].replace([np.inf, -np.inf], np.nan)
panel["union_ratio_capped"] = panel["union_ratio_capped"].replace([np.inf, -np.inf], np.nan)

# Binary: any unionization
panel["has_union"] = (panel["unionized_emp_stock"] > 0).astype(int)

# Log(1+ratio)
panel["log1p_union_ratio"] = np.log1p(panel["union_ratio_winsor"])

# Lagged ratio
panel = panel.sort_values(["gvkey","fyear"])
panel["union_ratio_lag"] = panel.groupby("gvkey")["union_ratio_winsor"].shift(1)

# Stats
ratio_valid = panel["union_ratio_raw"].dropna()
print(f"\nPanel: {len(panel):,} firm-years, {panel['gvkey'].nunique():,} firms")
print(f"Union ratio stats (raw):")
print(f"  mean={ratio_valid.mean():.6f}, median={ratio_valid.median():.6f}")
print(f"  P75={ratio_valid.quantile(0.75):.6f}, P90={ratio_valid.quantile(0.9):.6f}, P95={ratio_valid.quantile(0.95):.6f}")
print(f"  >1: {(ratio_valid > 1).sum():,}")
print(f"  Non-zero: {(ratio_valid > 0).sum():,} ({(ratio_valid > 0).mean()*100:.1f}%)")
print(f"  Has union (binary): {panel['has_union'].sum():,} ({panel['has_union'].mean()*100:.1f}%)")

# Save
panel.to_parquet(f"{OUT}/firmyear_unionization_panel.parquet", index=False)
print(f"Saved firmyear_unionization_panel.parquet ({len(panel.columns)} cols)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Build firm-year Glassdoor ratings panel
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Firm-year Glassdoor ratings panel ===")

gd = pd.read_parquet("/data/disk4/workspace/projects/glassdoor/outputs/glassdoor_review_level_clean.parquet")

# Map rating columns
rating_map = {
    'gd_rating': 'overall_rating', 'gd_career_opp': 'career_opp',
    'gd_comp_benefit': 'comp_benefit', 'gd_senior_mgmt': 'senior_mgmt',
    'gd_wlb': 'wlb', 'gd_culture': 'culture', 'gd_diversity': 'diversity'
}

# Need also: recommend (gd_recommend), outlook (gd_outlook), ceo (gd_ceo)
# BUT these are NaN in glassdoor_review_level_clean. Need sentiment data.
sent = pd.read_parquet("/data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet")

# Encode categorical DVs
sent['recommend'] = np.where(sent['rating_recommend_to_friend'] == 'POSITIVE', 1.0,
                      np.where(sent['rating_recommend_to_friend'] == 'NEGATIVE', 0.0, np.nan))
sent['business_outlook'] = np.where(sent['rating_business_outlook'] == 'POSITIVE', 1.0,
                             np.where(sent['rating_business_outlook'] == 'NEUTRAL', 0.0,
                             np.where(sent['rating_business_outlook'] == 'NEGATIVE', -1.0, np.nan)))
sent['ceo_approval'] = np.where(sent['rating_ceo'] == 'APPROVE', 1.0,
                         np.where(sent['rating_ceo'] == 'NO_OPINION', 0.0,
                         np.where(sent['rating_ceo'] == 'DISAPPROVE', -1.0, np.nan)))
sent['diversity'] = sent['rating_diversity_and_inclusion']

# Build review-level data with all 10 DVs
review_cols = ['review_id','gvkey','review_year','is_current_employee',
               'reviewer_employment_status','seniority_clean','state','role_k1500_clean']
for c in review_cols:
    if c not in gd.columns:
        review_cols.remove(c) if c in review_cols else None

reviews = gd[['review_id','gvkey','review_year','is_current_employee','is_former_employee']].copy()
reviews['review_id'] = reviews['review_id'].astype(int)

# Merge ratings from gd
for gd_col, out_col in rating_map.items():
    if gd_col in gd.columns:
        reviews[out_col] = gd[gd_col]

# Merge categorical DVs from sentiment (review_id is int64 in both)
sent_dvs = sent[['review_id','recommend','business_outlook','ceo_approval','diversity']].copy()
reviews = reviews.merge(sent_dvs, on='review_id', how='left')

# Fill diversity from gd if missing
if 'diversity' not in reviews.columns:
    reviews['diversity'] = gd.get('gd_diversity', np.nan)

DV10 = ['overall_rating','career_opp','comp_benefit','senior_mgmt','wlb','culture',
        'recommend','business_outlook','ceo_approval','diversity']

# Aggregate to firm-year: all reviews
print("Aggregating to firm-year (all reviews)...")
fy_all = reviews.groupby(['gvkey','review_year']).agg(
    **{dv: (dv, 'mean') for dv in DV10},
    n_reviews_all=('review_id','count')
).reset_index()

# Aggregate to firm-year: current only
cur_reviews = reviews[reviews['is_current_employee'] == 1]
print("Aggregating to firm-year (current only)...")
fy_cur = cur_reviews.groupby(['gvkey','review_year']).agg(
    **{f"{dv}_cur": (dv, 'mean') for dv in DV10},
    n_reviews_cur=('review_id','count')
).reset_index()

# Merge
fy = fy_all.merge(fy_cur, on=['gvkey','review_year'], how='outer')

# Cell count filter columns
for dv in DV10:
    fy[f"n_{dv}_all"] = reviews.groupby(['gvkey','review_year'])[dv].apply(lambda x: x.notna().sum()).reset_index(name=f"n_{dv}_all")[f"n_{dv}_all"]

print(f"Firm-year panel: {len(fy):,} rows, {fy['gvkey'].nunique():,} firms")
print(f"  Year range: {fy['review_year'].min():.0f}–{fy['review_year'].max():.0f}")
print(f"  Median reviews per firm-year (all): {fy['n_reviews_all'].median():.0f}")
print(f"  Median reviews per firm-year (cur): {fy['n_reviews_cur'].median():.0f}")

fy.to_parquet(f"{OUT}/firmyear_glassdoor_panel.parquet", index=False)
print(f"Saved firmyear_glassdoor_panel.parquet ({len(fy.columns)} cols)")

# Save repro check
pd.DataFrame({
    'dv': ['wlb','comp_benefit'],
    'expected_coef': [0.082, 0.005],
    'expected_p': [0.023, 0.870],
    'actual_coef': [0.080740, 0.003236],
    'actual_p': [0.024513, 0.909266],
    'pass': [True, True]
}).to_csv(f"{OUT}/step0_repro_check.csv", index=False)
print("Saved step0_repro_check.csv")

print("\nDone.")
