#!/usr/bin/env python
"""A. Enrich RDD sample with individual covariates from full Glassdoor data."""

import pandas as pd, numpy as np
from pathlib import Path

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
GD_FULL = Path("/data/disk4/workspace/projects/glassdoor/outputs/sentiment_individual_reviews_with_gvkey.parquet")
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v7"
OUT.mkdir(parents=True, exist_ok=True)

US_STATES = {"Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky",
    "Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi",
    "Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico",
    "New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
    "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
    "District of Columbia","Puerto Rico","Guam","Virgin Islands","American Samoa","Northern Mariana Islands"}

print("1. Loading RDD sample...")
df_rdd = pd.read_parquet(SAMPLE)
n_before = len(df_rdd)
print(f"  {n_before:,} reviews, {df_rdd['gvkey'].nunique()} gvkeys")

# Check if review_id exists
has_review_id = "review_id" in df_rdd.columns
print(f"  review_id in RDD sample: {has_review_id}")

print("2. Loading individual covariates from full GD...")
import pyarrow.parquet as pq
pf = pq.ParquetFile(GD_FULL)
gd_cols = ["review_id"]
for c in ["reviewer_employment_status","seniority","role_k1500","reviewer_length_of_employment","state"]:
    if c in pf.schema.names: gd_cols.append(c)
tbl = pf.read(columns=gd_cols)
df_cov = pd.DataFrame({k: tbl[k].to_pylist() for k in gd_cols})
print(f"  {len(df_cov):,} rows, columns: {list(df_cov.columns)}")

# Cast review_id to same type
df_rdd["review_id"] = df_rdd["review_id"].astype("int64")
df_cov["review_id"] = df_cov["review_id"].astype("int64")

print("3. Merging...")
df_enriched = df_rdd.merge(df_cov, on="review_id", how="left")
merge_rate = df_enriched["reviewer_employment_status"].notna().mean()
print(f"  Merge rate: {merge_rate:.1%}")

# Derived columns
if "reviewer_employment_status" in df_enriched.columns:
    emp = df_enriched["reviewer_employment_status"].fillna("UNKNOWN")
    df_enriched["is_regular"] = (emp == "REGULAR").astype(int)
    df_enriched["is_part_time"] = (emp == "PART_TIME").astype(int)
    df_enriched["is_intern"] = (emp == "INTERN").astype(int)
    df_enriched["is_contract"] = (emp == "CONTRACT").astype(int)
    other_emp = ~emp.isin(["REGULAR","PART_TIME","INTERN","CONTRACT","UNKNOWN"])
    df_enriched["is_other_employment"] = other_emp.astype(int)
    df_enriched["is_employment_missing"] = (emp == "UNKNOWN").astype(int)

# Seniority
if "seniority" in df_enriched.columns:
    df_enriched["is_seniority_missing"] = df_enriched["seniority"].isna().astype(int)
    df_enriched["seniority"] = df_enriched["seniority"].fillna(-1).astype(int)

# US state filter — check if `state` column exists in RDD or GD
if "state" in df_enriched.columns:
    df_enriched["is_us_review"] = df_enriched["state"].fillna("").apply(lambda s: s in US_STATES).astype(int)
elif "state" in df_cov.columns:
    # state was merged as a suffix
    st_col = [c for c in df_enriched.columns if c.startswith("state")][0] if any(c.startswith("state") for c in df_enriched.columns) else None
    if st_col:
        df_enriched["is_us_review"] = df_enriched[st_col].fillna("").apply(lambda s: s in US_STATES).astype(int)
else:
    df_enriched["is_us_review"] = 0

print(f"  is_us_review: {df_enriched['is_us_review'].mean():.1%}")

# Save
df_enriched.to_parquet(OUT / "rdd_sample_v7_enriched.parquet", index=False)
print(f"  Saved: rdd_sample_v7_enriched.parquet ({len(df_enriched):,} rows)")

# Coverage report
print("\n4. Covariate coverage:")
for col in ["reviewer_employment_status","seniority","role_k1500","reviewer_length_of_employment","state"]:
    if col in df_enriched.columns:
        pct = df_enriched[col].notna().mean()
        print(f"  {col}: {pct:.1%} non-missing")
for col in ["is_regular","is_part_time","is_intern","is_contract","is_other_employment","is_employment_missing"]:
    if col in df_enriched.columns:
        print(f"  {col}: mean={df_enriched[col].mean():.3f}")
print(f"  is_us_review: {df_enriched['is_us_review'].mean():.1%}")
print(f"  N reviews: {len(df_enriched):,}")
print(f"  N elections: {df_enriched['election_id'].nunique()}")
print("Done.")
