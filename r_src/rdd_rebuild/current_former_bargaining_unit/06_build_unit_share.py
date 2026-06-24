#!/usr/bin/env python
"""STEP 10: Build unit_share from unit_size and firm EMP."""

import pandas as pd
import numpy as np

OUT = "/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/current_former_bargaining_unit/20260624"

# Load NLRB unit data
nlrb = pd.read_parquet("/data/disk4/workspace/projects/union/outputs/preliminary_election_level.parquet",
                       columns=["election_id","unit_size","filing__number_of_eligible_voters"])
nlrb = nlrb.drop_duplicates(subset="election_id")

# Load enriched sample for gvkey and election year
enriched = pd.read_parquet(f"{OUT}/enriched_sample.parquet",
                           columns=["election_id","gvkey","election_year_elec"])
elections = enriched[["election_id","gvkey","election_year_elec"]].drop_duplicates(subset="election_id")

# Merge unit_size
elections = elections.merge(nlrb, on="election_id", how="left")

# Load Compustat
cmp = pd.read_parquet("outputs/compustat_firm_controls.parquet")
cmp_emp = cmp[["gvkey","fyear","emp"]].drop_duplicates()
cmp_emp["gvkey"] = cmp_emp["gvkey"].astype(str)

# For each election, find EMP at election_year - 1 (preferred) or election_year - 2
elections["fyear_target"] = elections["election_year_elec"] - 1
elections = elections.merge(cmp_emp, left_on=["gvkey","fyear_target"],
                            right_on=["gvkey","fyear"], how="left")
elections.drop(columns=["fyear"], inplace=True)

# If missing, try year - 2
mask_missing = elections["emp"].isna()
elections.loc[mask_missing, "fyear_target"] = elections.loc[mask_missing, "election_year_elec"] - 2
# Merge for missing rows
miss_eids = elections[mask_missing][["election_id","gvkey","fyear_target"]].copy()
miss_emp = miss_eids.merge(cmp_emp, left_on=["gvkey","fyear_target"],
                           right_on=["gvkey","fyear"], how="left")
miss_map = dict(zip(miss_emp["election_id"], miss_emp["emp"]))
for eid, emp_val in miss_map.items():
    elections.loc[elections["election_id"]==eid, "emp"] = emp_val

# Use eligible voters if unit_size missing, flag it
elections["unit_size_source"] = "unit_size"
mask_no_unit = elections["unit_size"].isna()
elections.loc[mask_no_unit, "unit_size"] = elections.loc[mask_no_unit, "filing__number_of_eligible_voters"]
elections.loc[mask_no_unit, "unit_size_source"] = "eligible_voters"

# Unit share — EMP is in thousands (Compustat standard)
# Convert EMP to actual count by multiplying by 1000
elections["emp_actual"] = elections["emp"] * 1000
elections["unit_share_raw"] = elections["unit_size"] / elections["emp_actual"]

# Winsorize at 1/99
p01 = elections["unit_share_raw"].quantile(0.01)
p99 = elections["unit_share_raw"].quantile(0.99)
elections["unit_share_winsor"] = elections["unit_share_raw"].clip(lower=p01, upper=p99)

# Cap at 1.0
elections["unit_share_capped"] = elections["unit_share_raw"].clip(upper=1.0)

# Stats
print(f"Elections: {len(elections):,}")
print(f"Has unit_size: {elections['unit_size'].notna().sum():,}")
print(f"Has EMP: {elections['emp'].notna().sum():,}")
print(f"EMP (note: units! raw values): mean={elections['emp'].mean():.1f}, med={elections['emp'].median():.1f}")
print(f"unit_size: mean={elections['unit_size'].mean():.1f}, med={elections['unit_size'].median():.1f}")
print(f"Has unit_share_raw: {elections['unit_share_raw'].notna().sum():,}")
print(f"unit_share_raw: mean={elections['unit_share_raw'].mean():.6f}, med={elections['unit_share_raw'].median():.6f}")
print(f"  P25={elections['unit_share_raw'].quantile(0.25):.6f}, P75={elections['unit_share_raw'].quantile(0.75):.6f}")
print(f"  P90={elections['unit_share_raw'].quantile(0.9):.6f}")
print(f"  >1: {(elections['unit_share_raw']>1).sum():,}")
print(f"unit_share_winsor: mean={elections['unit_share_winsor'].mean():.6f}, med={elections['unit_share_winsor'].median():.6f}")
print(f"unit_share_capped: mean={elections['unit_share_capped'].mean():.6f}, med={elections['unit_share_capped'].median():.6f}")

# Save
out_cols = ["election_id","gvkey","election_year_elec",
            "unit_size","unit_size_source","emp","emp_actual",
            "unit_share_raw","unit_share_winsor","unit_share_capped"]
elections[out_cols].to_csv(f"{OUT}/unit_share_election_data.csv", index=False)
print(f"Saved {OUT}/unit_share_election_data.csv")
