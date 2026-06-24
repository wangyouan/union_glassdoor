#!/usr/bin/env python
"""STEP 0: Field inventory, new DV encoding, missing rates, reproducibility check.

Outputs:
  - dv_field_inventory.csv: all DVs with encoding, missing rates
  - enriched_sample.parquet: main sample + new DVs merged
"""

import pandas as pd
import numpy as np
import os, sys

OUT = "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624"
os.makedirs(OUT, exist_ok=True)

# ─── Load data ──────────────────────────────────────────────────────────────
print("Loading main sample...")
main = pd.read_parquet("outputs/rdd_rebuild/focused_rdd_search_v7/rdd_sample_v7_enriched.parquet")
print(f"  Main sample: {main.shape}")

print("Loading sentiment data...")
sent = pd.read_parquet("/data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet")
print(f"  Sentiment: {sent.shape}")

# ─── 1. Map review_id between main and sentiment ────────────────────────────
# main.review_id is int64, sent.review_id is int64 — should match directly
print("\n=== Merge check ===")
print(f"Main review_ids: {main['review_id'].nunique():,}")
print(f"Sent review_ids: {sent['review_id'].nunique():,}")
overlap = set(main['review_id']) & set(sent['review_id'])
print(f"Overlap: {len(overlap):,} ({len(overlap)/main['review_id'].nunique()*100:.1f}% of main)")

# ─── 2. Encode new DVs from sentiment data ──────────────────────────────────
print("\n=== Encoding new DVs ===")

# recommend: POSITIVE→1, NEGATIVE→0, no neutral observed
sent['recommend'] = np.where(sent['rating_recommend_to_friend'] == 'POSITIVE', 1.0,
                      np.where(sent['rating_recommend_to_friend'] == 'NEGATIVE', 0.0, np.nan))
print(f"  recommend: POSITIVE={sent['recommend'].eq(1).sum():,}, NEGATIVE={sent['recommend'].eq(0).sum():,}, NA={sent['recommend'].isna().sum():,}")

# business_outlook: POSITIVE→+1, NEUTRAL→0, NEGATIVE→-1
sent['business_outlook'] = np.where(sent['rating_business_outlook'] == 'POSITIVE', 1.0,
                             np.where(sent['rating_business_outlook'] == 'NEUTRAL', 0.0,
                             np.where(sent['rating_business_outlook'] == 'NEGATIVE', -1.0, np.nan)))
print(f"  business_outlook: +1={sent['business_outlook'].eq(1).sum():,}, 0={sent['business_outlook'].eq(0).sum():,}, -1={sent['business_outlook'].eq(-1).sum():,}, NA={sent['business_outlook'].isna().sum():,}")

# ceo_approval: APPROVE→+1, NO_OPINION→0, DISAPPROVE→-1
sent['ceo_approval'] = np.where(sent['rating_ceo'] == 'APPROVE', 1.0,
                         np.where(sent['rating_ceo'] == 'NO_OPINION', 0.0,
                         np.where(sent['rating_ceo'] == 'DISAPPROVE', -1.0, np.nan)))
print(f"  ceo_approval: +1={sent['ceo_approval'].eq(1).sum():,}, 0={sent['ceo_approval'].eq(0).sum():,}, -1={sent['ceo_approval'].eq(-1).sum():,}, NA={sent['ceo_approval'].isna().sum():,}")

# diversity: already in main as 'diversity', but verify it matches rating_diversity_and_inclusion
# Also carry forward from sentiment if missing in main
sent['diversity_raw'] = sent['rating_diversity_and_inclusion']  # 1-5 float

# ─── 3. Merge new DVs into main sample ──────────────────────────────────────
# Only merge NEW columns not already in main sample
merge_cols = ['review_id', 'recommend', 'business_outlook', 'ceo_approval', 'rating_diversity_and_inclusion']
main2 = main.merge(sent[merge_cols], on='review_id', how='left', validate='1:1')

# Use diversity from main if present, else from sentiment
main2['diversity'] = main2['diversity'].fillna(main2['rating_diversity_and_inclusion'])
main2.drop(columns=['rating_diversity_and_inclusion'], inplace=True)

print(f"\n  After merge: {main2.shape}")
print(f"  recommend non-null in merged: {main2['recommend'].notna().sum():,} ({main2['recommend'].notna().mean()*100:.1f}%)")
print(f"  business_outlook non-null in merged: {main2['business_outlook'].notna().sum():,} ({main2['business_outlook'].notna().mean()*100:.1f}%)")
print(f"  ceo_approval non-null in merged: {main2['ceo_approval'].notna().sum():,} ({main2['ceo_approval'].notna().mean()*100:.1f}%)")
print(f"  diversity non-null in merged: {main2['diversity'].notna().sum():,} ({main2['diversity'].notna().mean()*100:.1f}%)")

# ─── 4. Field inventory table ───────────────────────────────────────────────
dv_list = ['overall_rating', 'career_opp', 'comp_benefit', 'senior_mgmt', 'wlb', 'culture',
           'recommend', 'business_outlook', 'ceo_approval', 'diversity']

inventory_rows = []
for dv in dv_list:
    n_nonnull = main2[dv].notna().sum()
    n_total = len(main2)
    miss_rate = (1 - n_nonnull/n_total) * 100
    dtype = main2[dv].dtype
    unique_vals = main2[dv].dropna().nunique()
    if unique_vals <= 20:
        vc = main2[dv].value_counts().sort_index()
        val_dist = ", ".join(f"{v}:{c}" for v, c in vc.items())
    else:
        val_dist = f"min={main2[dv].min():.3f}, max={main2[dv].max():.3f}, mean={main2[dv].mean():.3f}"
    inventory_rows.append({
        'dv': dv,
        'type': 'rating' if dv not in ['recommend', 'business_outlook', 'ceo_approval'] else 'categorical',
        'encoding': '1-5 stars' if dv not in ['recommend', 'business_outlook', 'ceo_approval'] else
                     '1/0 rec' if dv == 'recommend' else '+1/0/-1',
        'dtype': str(dtype),
        'non_null': n_nonnull,
        'missing_rate_pct': round(miss_rate, 2),
        'unique_values': unique_vals,
        'distribution': val_dist
    })

inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv(f"{OUT}/dv_field_inventory.csv", index=False)
print("\n=== DV Field Inventory ===")
print(inv_df.to_string(index=False))

# ─── 5. Text/behavioral variables coverage ──────────────────────────────────
print("\n=== Additional variables coverage (in merged sample) ===")
extra_vars = ['review_pros', 'review_cons', 'review_summary']
for col in ['reviewer_current_job', 'reviewer_employment_status', 'reviewer_length_of_employment',
            'seniority', 'role_k1500', 'state_x', 'state_y', 'is_us_review']:
    if col in main2.columns:
        n = main2[col].notna().sum()
        print(f"  {col}: {n:,} / {len(main2):,} ({n/len(main2)*100:.1f}%)")
    else:
        print(f"  {col}: NOT IN MAIN SAMPLE")

# Also check text columns in main
for col in ['pros_text', 'cons_text', 'advice_text', 'pros_len', 'cons_len']:
    if col in main2.columns:
        n = main2[col].notna().sum()
        print(f"  {col}: {n:,} / {len(main2):,} ({n/len(main2)*100:.1f}%)")

# ─── 6. Save enriched sample ────────────────────────────────────────────────
main2.to_parquet(f"{OUT}/enriched_sample.parquet", index=False)
print(f"\nSaved enriched sample: {OUT}/enriched_sample.parquet")

# ─── 7. Reproducibility check ───────────────────────────────────────────────
# Current-only + total>=10: WLB ≈ +0.082 (p≈0.023), Comp ≈ +0.005 (p≈0.870)
print("\n=== Reproducibility check ===")
print("Running R v7c for WLB & Comp on current-only total>=10...")
print("(This will be done by the R companion script 01_repro_check.R)")

# Save instruction for R
repro_check_r = f"""#!/usr/bin/env Rscript
# Reproducibility check: v7c on current + total>=10
# WLB ≈ +0.082 (p≈0.023), Comp ≈ +0.005 (p≈0.870)

library(fixest)
library(dplyr)
library(nanoparquet)

df <- nanoparquet::read_parquet("{OUT}/enriched_sample.parquet")
cat("Total rows:", nrow(df), "\\n")

# current only
cur <- df[df$is_current_employee == 1, ]
cat("Current rows:", nrow(cur), "\\n")

# Make state_clean and role_clean from existing columns
# state: use state_x or state_y
cur$state_clean <- ifelse(!is.na(cur$state_x), cur$state_x, cur$state_y)
cur$state_clean[is.na(cur$state_clean)] <- "UNKNOWN"

# role: use role_k1500
cur$role_clean <- ifelse(!is.na(cur$role_k1500), as.character(cur$role_k1500), "UNKNOWN")

# emp_status: from is_current_employee=1, but also check reviewer_employment_status
# For current-only sample, emp_status could differentiate Regular/Part-time etc.
cur$emp_status <- ifelse(!is.na(cur$reviewer_employment_status), as.character(cur$reviewer_employment_status), "UNKNOWN")

# seniority_f: factor
cur$seniority_f <- as.character(cur$seniority)
cur$seniority_f[is.na(cur$seniority_f)] <- "UNKNOWN"

# total>=10 filter: count non-NA wlb per election in current sample
election_counts <- cur %>%
  group_by(election_id) %>%
  summarise(total_reviews = n(), .groups = 'drop')

elections_keep <- election_counts$election_id[election_counts$total_reviews >= 10]
cat("Elections with total>=10 in current:", length(elections_keep), "\\n")

cur10 <- cur[cur$election_id %in% elections_keep, ]
cat("Current total>=10 rows:", nrow(cur10), "\\n")

# v7c spec
fe_vars <- c("gvkey", "review_year", "state_clean", "role_clean")

for (dv in c("wlb", "comp_benefit")) {{
  cat("\\n=== ", dv, " ===\\n")
  fml <- as.formula(paste(dv, "~ win + post + win_post + post:margin + emp_status + seniority_f |",
                          paste(fe_vars, collapse=" + ")))
  m <- feols(fml, data = cur10, cluster = ~gvkey + review_year)
  print(summary(m))
}}

cat("\\n=== Expected: WLB ≈ +0.082 (p≈0.023), Comp ≈ +0.005 (p≈0.870) ===\\n")
"""

with open(f"{OUT}/../01_repro_check.R", "w") as f:
    f.write(repro_check_r)

print("R repro script written. Run with:")
print(f"  Rscript {OUT}/../01_repro_check.R")
print("\nDone.")
