#!/usr/bin/env python
"""A. Rebuild RDD event sample with ±548-day window."""

import pandas as pd, numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import json

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
GD_CLEAN = Path("/data/disk4/workspace/projects/glassdoor/outputs/glassdoor_review_level_clean.parquet")
UNION_FILE = Path("/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet")
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v4"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_DAYS = 548

print("1. Loading union elections...")
pf_ue = pq.ParquetFile(UNION_FILE)
gvkey_col = "gvkey_final" if "gvkey_final" in pf_ue.schema.names else "matched_gvkey"
ue_cols = [gvkey_col,"election_id","election_date","votes_for_union","votes_against_union","total_valid_votes"]
tbl = pf_ue.read(columns=ue_cols)
df_ue = pd.DataFrame({k: tbl[k].to_pylist() for k in ue_cols})
df_ue = df_ue.rename(columns={gvkey_col:"gvkey"})
df_ue["election_date"] = pd.to_datetime(df_ue["election_date"])
df_ue["election_year"] = df_ue["election_date"].dt.year
df_ue["vote_share"] = df_ue["votes_for_union"]/(df_ue["votes_for_union"]+df_ue["votes_against_union"])
df_ue["margin"] = df_ue["vote_share"]-0.5
df_ue["win"] = (df_ue["margin"]>0).astype(int); df_ue["abs_margin"] = df_ue["margin"].abs()
union_gvkeys = set(df_ue["gvkey"].dropna().unique())
print(f"  {len(df_ue)} elections, {len(union_gvkeys)} gvkeys")

print("2. Loading GD reviews...")
pf_gd = pq.ParquetFile(GD_CLEAN)
gd_cols = ["gvkey","review_id","review_date_clean","review_year","review_month",
           "is_current_employee","is_former_employee",
           "gd_rating","gd_career_opp","gd_comp_benefit","gd_senior_mgmt","gd_wlb","gd_culture"]
gd_cols = [c for c in gd_cols if c in pf_gd.schema.names]
rename = {"review_date_clean":"review_date","gd_rating":"overall_rating","gd_career_opp":"career_opp",
          "gd_comp_benefit":"comp_benefit","gd_senior_mgmt":"senior_mgmt","gd_wlb":"wlb","gd_culture":"culture"}
tbl = pf_gd.read(columns=gd_cols)
df_gd = pd.DataFrame({k: tbl[k].to_pylist() for k in gd_cols}).rename(columns=rename)
df_gd["review_date"] = pd.to_datetime(df_gd["review_date"])
df_gd["review_year"] = df_gd["review_date"].dt.year
df_gd["employee_filter"] = "all"
if "is_current_employee" in df_gd.columns:
    df_gd.loc[df_gd["is_current_employee"].astype(bool),"employee_filter"] = "current"
if "is_former_employee" in df_gd.columns:
    df_gd.loc[df_gd["is_former_employee"].astype(bool),"employee_filter"] = "former"

# Filter union-firm reviews
df_match = df_gd[df_gd["gvkey"].isin(union_gvkeys)].copy()
n_ue_firm_gvkey = df_match["gvkey"].nunique()
del df_gd

print("3. Building election lookup...")
elec_idx = {}
for _, e in df_ue.iterrows():
    gv = e["gvkey"]
    if gv not in elec_idx: elec_idx[gv] = []
    elec_idx[gv].append(e.to_dict())

print(f"4. Matching reviews to nearest election within ±{WINDOW_DAYS}d...")
matched = []
for gv, grp in df_match.groupby("gvkey"):
    if gv not in elec_idx: continue
    elections = elec_idx[gv]
    review_dates = grp["review_date"].values.astype("datetime64[ns]")
    elec_dates = [np.datetime64(e["election_date"],"ns") for e in elections]
    n_rev = len(review_dates); best_dist = np.full(n_rev, np.inf); best_ei = np.full(n_rev, -1, dtype=int)
    for ei, ed in enumerate(elec_dates):
        d = np.abs(review_dates-ed).astype("timedelta64[D]").astype(float)
        better = d < best_dist; best_dist[better] = d[better]; best_ei[better] = ei
    within = best_dist <= WINDOW_DAYS
    if within.sum()==0: continue
    keep = np.where(within)[0]
    m = grp.iloc[keep].copy()
    m["days_to_election"] = [(review_dates[i]-elec_dates[best_ei[i]]).astype("timedelta64[D]").astype(float) for i in keep]
    for field in ["election_id","margin","win","abs_margin","vote_share","votes_for_union","votes_against_union","total_valid_votes","election_year"]:
        m[field] = [elections[best_ei[i]][field] for i in keep]
    matched.append(m)

df_rdd = pd.concat(matched, ignore_index=True)
df_rdd["post"] = (df_rdd["days_to_election"]>=0).astype(int)
df_rdd["event_time_month"] = np.floor(df_rdd["days_to_election"]/30).astype(int)
df_rdd["within_548"] = True
df_rdd["within_365"] = df_rdd["days_to_election"].abs()<=365
df_rdd["within_180"] = df_rdd["days_to_election"].abs()<=180
df_rdd["review_year"] = pd.to_datetime(df_rdd["review_date"]).dt.year
print(f"  Matched: {len(df_rdd):,} reviews, {df_rdd['gvkey'].nunique()} gvkeys, {df_rdd['election_id'].nunique()} elections")

# Overlap diagnostics
print("5. Overlap diagnostics...")
elections_per_gvkey = df_ue.groupby("gvkey").size()
multi_election_gvkeys = set(elections_per_gvkey[elections_per_gvkey>1].index)

# For each election, check if there's another election at same gvkey within 548 days
overlap_events = set()
for gv in multi_election_gvkeys:
    gv_elecs = df_ue[df_ue["gvkey"]==gv].sort_values("election_date")
    dates = gv_elecs["election_date"].values
    for i in range(len(dates)):
        for j in range(len(dates)):
            if i!=j and abs((dates[i]-dates[j]).astype("timedelta64[D]").astype(float)) <= WINDOW_DAYS:
                overlap_events.add(gv_elecs["election_id"].iloc[i])

share_overlap_events = len(overlap_events)/len(df_ue)*100
rdd_elec = set(df_rdd["election_id"].unique())
overlap_in_rdd = overlap_events & rdd_elec
share_rdd_overlap = len(overlap_in_rdd)/len(rdd_elec)*100 if len(rdd_elec)>0 else 0
print(f"  Elections with same-firm election within {WINDOW_DAYS}d: {len(overlap_events)} ({share_overlap_events:.1f}%)")
print(f"  RDD-sample elections with overlap: {len(overlap_in_rdd)} ({share_rdd_overlap:.1f}%)")

# No-overlap flag
df_rdd["overlap_election"] = df_rdd["election_id"].isin(overlap_in_rdd)

# Save
out_cols = ["gvkey","review_id","review_date","review_year","employee_filter",
            "is_current_employee","is_former_employee",
            "overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture",
            "election_id","election_year","margin","win","abs_margin","vote_share",
            "days_to_election","post","event_time_month",
            "within_548","within_365","within_180","overlap_election"]
out_cols = [c for c in out_cols if c in df_rdd.columns]
df_rdd[out_cols].to_parquet(OUT / "rdd_review_event_sample_18m.parquet", index=False)
print(f"  Saved: rdd_review_event_sample_18m.parquet ({len(df_rdd):,} x {len(out_cols)})")

# Diagnostics CSV
diag = pd.DataFrame([
    {"metric":"Total GD reviews","value":pf_gd.metadata.num_rows},
    {"metric":"Union-firm reviews","value":len(df_match)},
    {"metric":f"Matched within +/-{WINDOW_DAYS}d","value":len(df_rdd)},
    {"metric":"Unique gvkeys in RDD sample","value":df_rdd["gvkey"].nunique()},
    {"metric":"Unique elections","value":df_rdd["election_id"].nunique()},
    {"metric":"Reviews within +/-365d","value":int(df_rdd["within_365"].sum())},
    {"metric":"Reviews within +/-180d","value":int(df_rdd["within_180"].sum())},
    {"metric":"Current employee reviews","value":int((df_rdd["employee_filter"]=="current").sum())},
    {"metric":f"Elections with another election within {WINDOW_DAYS}d","value":len(overlap_events)},
    {"metric":"Share of elections with overlap","value":f"{share_overlap_events:.1f}%"},
    {"metric":"RDD-sample elections with overlap","value":f"{share_rdd_overlap:.1f}%"},
])
diag.to_csv(OUT / "sample_18m_diagnostics.csv", index=False)
print("  Saved: sample_18m_diagnostics.csv")
print("Done.")
