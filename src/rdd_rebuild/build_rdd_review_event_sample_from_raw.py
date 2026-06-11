#!/usr/bin/env python
"""
Step 1: Build RDD review-event sample from raw Glassdoor + union election data.

Matches GD reviews to union elections by gvkey, assigns each review to the
nearest election within ±365 days. Builds all derived RDD variables.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "rdd_rebuild"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

GD_CLEAN = Path("/data/disk4/workspace/projects/glassdoor/outputs/glassdoor_review_level_clean.parquet")
UNION_FILE = Path("/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet")
OLD_W365 = PROJ / "outputs/union_glassdoor_comment_level_window365.parquet"

# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LOAD UNION ELECTION DATA")
print("=" * 70)

pf_ue = pq.ParquetFile(UNION_FILE)
ue_schema = pf_ue.schema.names

# Detect gvkey column
gvkey_col_ue = "gvkey_final" if "gvkey_final" in ue_schema else "matched_gvkey"
print(f"  Using gvkey column: {gvkey_col_ue}")

# Load election data
ue_load_cols = [gvkey_col_ue, "election_id", "election_date",
                "votes_for_union", "votes_against_union", "total_valid_votes"]
ue_load_cols = [c for c in ue_load_cols if c in ue_schema]
# Add optional columns
for opt in ["case_number", "employer_name"]:
    if opt in ue_schema:
        ue_load_cols.append(opt)

tbl = pf_ue.read(columns=ue_load_cols)
df_ue = pd.DataFrame({k: tbl[k].to_pylist() for k in ue_load_cols})
df_ue = df_ue.rename(columns={gvkey_col_ue: "gvkey"})
df_ue["election_date"] = pd.to_datetime(df_ue["election_date"])
df_ue["election_year"] = df_ue["election_date"].dt.year

# Compute RDD variables
df_ue["vote_share"] = df_ue["votes_for_union"] / (df_ue["votes_for_union"] + df_ue["votes_against_union"])
df_ue["margin"] = df_ue["vote_share"] - 0.5
df_ue["win"] = (df_ue["margin"] > 0).astype(int)
df_ue["abs_margin"] = df_ue["margin"].abs()

print(f"  Elections: {len(df_ue):,}")
print(f"  Unique gvkeys: {df_ue['gvkey'].nunique():,}")
print(f"  Wins: {df_ue['win'].sum():.0f}, Losses: {(1-df_ue['win']).sum():.0f}")
union_gvkeys = set(df_ue["gvkey"].dropna().unique())

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. LOAD GLASSDOOR REVIEWS (CLEAN FILE)")
print("=" * 70)

pf_gd = pq.ParquetFile(GD_CLEAN)
gd_schema = pf_gd.schema.names

# Columns to load and their canonical names
gd_col_map = {
    "gvkey": "gvkey",
    "review_id": "review_id",
    "review_date_clean": "review_date",
    "review_year": "review_year",
    "review_month": "review_month",
    "company": "company_name",
    "state": "state",
    "job_title_raw": "job_title_raw",
    "job_title_clean": "job_title_clean",
    "is_current_employee": "is_current_employee",
    "is_former_employee": "is_former_employee",
    "gd_rating": "overall_rating",
    "gd_career_opp": "career_opp",
    "gd_comp_benefit": "comp_benefit",
    "gd_senior_mgmt": "senior_mgmt",
    "gd_wlb": "wlb",
    "gd_culture": "culture",
    "gd_diversity": "diversity",
}
gd_load_cols = [k for k in gd_col_map if k in gd_schema]
gd_rename = {k: v for k, v in gd_col_map.items() if k in gd_schema}

print(f"  Loading {len(gd_load_cols)} columns from {pf_gd.metadata.num_rows:,} rows...")
tbl_gd = pf_gd.read(columns=gd_load_cols)
df_gd = pd.DataFrame({k: tbl_gd[k].to_pylist() for k in gd_load_cols})
df_gd = df_gd.rename(columns=gd_rename)
df_gd["review_date"] = pd.to_datetime(df_gd["review_date"])
df_gd["review_year"] = df_gd["review_date"].dt.year
df_gd["review_month"] = df_gd["review_date"].dt.month

n_gd_total = len(df_gd)
n_gd_gvkeys = df_gd["gvkey"].nunique()

# Employee status
if "is_current_employee" in df_gd.columns:
    df_gd["employee_filter"] = "all"
    mask_cur = df_gd["is_current_employee"].astype(bool)
    mask_for = df_gd["is_former_employee"].astype(bool) if "is_former_employee" in df_gd.columns else ~mask_cur
    df_gd.loc[mask_cur, "employee_filter"] = "current"
    df_gd.loc[mask_for, "employee_filter"] = "former"
    print(f"  Employee split: {df_gd['employee_filter'].value_counts().to_dict()}")
else:
    df_gd["employee_filter"] = "all"
    print("  WARNING: No employee status columns found")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. FILTER TO UNION-ELECTION FIRMS")
print("=" * 70)

df_gd["in_union"] = df_gd["gvkey"].isin(union_gvkeys)
n_ue_firm_rev = df_gd["in_union"].sum()
n_ue_firm_gvkey = df_gd.loc[df_gd["in_union"], "gvkey"].nunique()
print(f"  Union-firm reviews: {n_ue_firm_rev:,} ({n_ue_firm_rev/n_gd_total*100:.1f}%)")
print(f"  Unique gvkeys: {n_ue_firm_gvkey}")

df_match = df_gd[df_gd["in_union"]].copy()
del df_gd

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. MATCH REVIEWS TO NEAREST ELECTION (±365d)")
print("=" * 70)

# Build per-gvkey election index
elec_idx = {}
for _, e in df_ue.iterrows():
    gv = e["gvkey"]
    if gv not in elec_idx:
        elec_idx[gv] = []
    elec_idx[gv].append(e.to_dict())

# For each gvkey, find nearest election per review
matched_parts = []
n_overlap = 0

for gv, grp in df_match.groupby("gvkey"):
    if gv not in elec_idx:
        continue

    elections = elec_idx[gv]
    n_elec = len(elections)
    n_rev = len(grp)
    review_dates_ns = grp["review_date"].values.astype("datetime64[ns]")

    # Compute distance to each election
    best_dist = np.full(n_rev, np.inf)
    best_ei = np.full(n_rev, -1, dtype=int)

    # For overlap counting
    elec_dates_ns = [np.datetime64(e["election_date"], "ns") for e in elections]

    for ei, e_dt in enumerate(elec_dates_ns):
        dist = np.abs(review_dates_ns - e_dt).astype("timedelta64[D]").astype(float)
        better = dist < best_dist
        best_dist[better] = dist[better]
        best_ei[better] = ei

    # Keep within 365d
    within = best_dist <= 365
    if within.sum() == 0:
        continue

    # Count reviews within 365d of multiple elections
    for ri in range(n_rev):
        n_within = sum(1 for e_dt in elec_dates_ns
                       if abs(review_dates_ns[ri] - e_dt).astype("timedelta64[D]").astype(float) <= 365)
        if n_within > 1:
            n_overlap += 1

    keep_mask = np.where(within)[0]
    m = grp.iloc[keep_mask].copy()

    # Days to assigned election
    m["days_to_election"] = [
        (review_dates_ns[i] - elec_dates_ns[best_ei[i]]).astype("timedelta64[D]").astype(float)
        for i in keep_mask
    ]

    # Election attributes
    for field in ["election_id", "margin", "win", "abs_margin", "vote_share",
                   "votes_for_union", "votes_against_union", "total_valid_votes"]:
        m[field] = [elections[best_ei[i]][field] for i in keep_mask]

    for field in ["case_number", "employer_name"]:
        m[field] = [elections[best_ei[i]].get(field, "N/A") for i in keep_mask]

    m["election_year_elec"] = [elections[best_ei[i]]["election_year"] for i in keep_mask]

    matched_parts.append(m)

df_rdd = pd.concat(matched_parts, ignore_index=True)
print(f"  Matched: {len(df_rdd):,} reviews")
print(f"  Unique gvkeys: {df_rdd['gvkey'].nunique()}")
print(f"  Unique elections: {df_rdd['election_id'].nunique()}")
print(f"  Overlapping windows: {n_overlap:,} reviews")

del df_match

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. BUILD DERIVED VARIABLES")
print("=" * 70)

df_rdd["post"] = (df_rdd["days_to_election"] >= 0).astype(int)
df_rdd["event_time_month"] = np.floor(df_rdd["days_to_election"] / 30).astype(int)
df_rdd["within_365"] = True
df_rdd["within_180"] = df_rdd["days_to_election"].abs() <= 180
df_rdd["within_90"] = df_rdd["days_to_election"].abs() <= 90

outcome_cols = [c for c in ["overall_rating", "career_opp", "comp_benefit",
                              "senior_mgmt", "wlb", "culture", "diversity"]
                if c in df_rdd.columns]

print(f"  Post share: {df_rdd['post'].mean():.3f}")
print(f"  Event months: [{df_rdd['event_time_month'].min():.0f}, {df_rdd['event_time_month'].max():.0f}]")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. SAVE SAMPLE")
print("=" * 70)

out_cols = [
    "gvkey", "review_id", "review_date", "review_year", "review_month",
    "company_name", "state", "job_title_raw", "job_title_clean",
    "employee_filter", "is_current_employee", "is_former_employee",
    "election_id", "case_number", "election_year_elec",
    "margin", "win", "abs_margin", "vote_share",
    "votes_for_union", "votes_against_union", "total_valid_votes",
    "days_to_election", "post", "event_time_month",
    "within_365", "within_180", "within_90",
] + outcome_cols
out_cols = [c for c in out_cols if c in df_rdd.columns]

df_rdd[out_cols].to_parquet(OUT / "rdd_review_event_sample_from_raw.parquet", index=False)
print(f"  Saved: {len(df_rdd):,} rows × {len(out_cols)} cols")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. DIAGNOSTICS")
print("=" * 70)

# Attrition
attrition_rows = [
    {"step": "A. Full GD clean reviews", "n": n_gd_total, "gvkey": n_gd_gvkeys},
    {"step": "B. Union-election firm reviews", "n": int(n_ue_firm_rev), "gvkey": int(n_ue_firm_gvkey)},
    {"step": "C. ±365d of nearest election", "n": len(df_rdd), "gvkey": int(df_rdd["gvkey"].nunique()),
     "elections": int(df_rdd["election_id"].nunique())},
]

for w, label in [(365, "D. ±365d"), (180, "E. ±180d"), (90, "F. ±90d")]:
    col = "within_365" if w == 365 else f"within_{w}"
    sub = df_rdd if w == 365 else df_rdd[df_rdd[col]]
    cur = int((sub["employee_filter"] == "current").sum()) if "employee_filter" in sub.columns else 0
    attrition_rows.append({
        "step": label, "n": len(sub), "gvkey": int(sub["gvkey"].nunique()),
        "elections": int(sub["election_id"].nunique()), "current": cur,
    })

for i, r in enumerate(attrition_rows):
    r["pct_initial"] = r["n"] / attrition_rows[0]["n"] * 100
    if i > 0:
        r["pct_prev"] = r["n"] / attrition_rows[i-1]["n"] * 100

df_att = pd.DataFrame(attrition_rows)
df_att.to_csv(OUT / "rdd_review_event_sample_from_raw_attrition.csv", index=False)
print("\nAttrition:")
print(df_att[["step", "n", "gvkey", "pct_initial"]].to_string(index=False))

# Bandwidth diagnostics
print("\nBandwidth diagnostics (±365d):")
for label, bw in [("global", None), ("|m|<=0.30", 0.30), ("|m|<=0.20", 0.20),
                   ("|m|<=0.10", 0.10), ("|m|<=0.05", 0.05)]:
    sub = df_rdd if bw is None else df_rdd[df_rdd["abs_margin"] <= bw]
    cur = int((sub["employee_filter"] == "current").sum())
    n_elec = sub["election_id"].nunique()
    n_win = sub[sub["win"] == 1]["election_id"].nunique()
    n_loss = sub[sub["win"] == 0]["election_id"].nunique()
    print(f"  {label}: reviews={len(sub):,}, gvkey={sub['gvkey'].nunique()}, "
          f"elections={n_elec} (w={n_win}, l={n_loss}), current={cur:,}")

# Outcome coverage
print("\nOutcome coverage (±365d, all employees):")
oc_diag = {}
for oc in outcome_cols:
    sub = df_rdd[df_rdd[oc].notna()]
    cur = int((sub["employee_filter"] == "current").sum())
    oc_diag[oc] = {"n": int(len(sub)), "gvkey": int(sub["gvkey"].nunique()),
                    "elections": int(sub["election_id"].nunique()),
                    "current": cur, "mean": float(sub[oc].mean()), "sd": float(sub[oc].std())}
    print(f"  {oc}: n={len(sub):,}, gvkey={sub['gvkey'].nunique()}, "
          f"cur={cur:,}, mean={sub[oc].mean():.2f}")

# Compare with old
if OLD_W365.exists():
    old = pd.read_parquet(OLD_W365)
    print(f"\nOld w365: {len(old):,} reviews, {old['gvkey'].nunique()} gvkeys")
    print(f"New: {len(df_rdd):,} reviews, {df_rdd['gvkey'].nunique()} gvkeys")
    print(f"Ratio: {len(df_rdd)/len(old):.1f}x reviews, {df_rdd['gvkey'].nunique()/old['gvkey'].nunique():.1f}x gvkeys")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. SAVE DIAGNOSTICS AND REPORT")
print("=" * 70)

# JSON
diag_json = {
    "build_time": datetime.now().isoformat(),
    "n_gd_total": int(n_gd_total),
    "n_ue_total": len(df_ue),
    "n_rdd_sample": len(df_rdd),
    "n_rdd_gvkeys": int(df_rdd["gvkey"].nunique()),
    "n_rdd_elections": int(df_rdd["election_id"].nunique()),
    "n_overlap_reviews": int(n_overlap),
    "outcomes": oc_diag,
}
with open(OUT / "rdd_review_event_sample_from_raw_diagnostics.json", "w") as f:
    json.dump(diag_json, f, indent=2, default=str)

# Markdown
rpt = f"""# RDD Review-Event Sample Build Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Attrition

| Step | N Reviews | N gvkeys | N Elections | % Initial |
|------|-----------|----------|-------------|-----------|
"""
for _, r in df_att.iterrows():
    elec = f"{int(r['elections']):,}" if "elections" in r and pd.notna(r["elections"]) else "—"
    rpt += f"| {r['step']} | **{int(r['n']):,}** | {int(r['gvkey'])} | {elec} | {r['pct_initial']:.1f}% |\n"

rpt += f"""
## Bandwidth Diagnostics (±365d)

| Bandwidth | N Reviews | N gvkeys | N Elections | Win | Loss | Current |
|-----------|-----------|----------|-------------|-----|------|---------|
"""
for label, bw in [("global", None), ("|m|<=0.30", 0.30), ("|m|<=0.20", 0.20),
                   ("|m|<=0.10", 0.10), ("|m|<=0.05", 0.05)]:
    sub = df_rdd if bw is None else df_rdd[df_rdd["abs_margin"] <= bw]
    cur = int((sub["employee_filter"] == "current").sum())
    n_elec = sub["election_id"].nunique()
    n_win = sub[sub["win"] == 1]["election_id"].nunique()
    n_loss = sub[sub["win"] == 0]["election_id"].nunique()
    rpt += f"| {label} | {len(sub):,} | {sub['gvkey'].nunique()} | {n_elec} | {n_win} | {n_loss} | {cur:,} |\n"

rpt += f"""
## Outcome Coverage (±365d, all employees)

| Outcome | N | gvkey | Elections | Current | Mean | SD |
|---------|----|-------|-----------|---------|------|-----|
"""
for oc, d in oc_diag.items():
    rpt += f"| {oc} | {d['n']:,} | {d['gvkey']} | {d['elections']} | {d['current']:,} | {d['mean']:.2f} | {d['sd']:.2f} |\n"

if OLD_W365.exists():
    old = pd.read_parquet(OLD_W365)
    rpt += f"""
## vs Old window365

| | Old | New | Ratio |
|---|-----|-----|-------|
| Reviews | {len(old):,} | {len(df_rdd):,} | {len(df_rdd)/len(old):.1f}x |
| gvkeys | {old['gvkey'].nunique()} | {df_rdd['gvkey'].nunique()} | {df_rdd['gvkey'].nunique()/old['gvkey'].nunique():.1f}x |
| Elections | {old['election_id'].nunique()} | {df_rdd['election_id'].nunique()} | {df_rdd['election_id'].nunique()/old['election_id'].nunique():.1f}x |
"""

with open(OUT / "rdd_review_event_sample_build_report.md", "w") as f:
    f.write(rpt)

print("  Saved: diagnostics JSON and report")
print(f"\n{'=' * 70}")
print("STEP 1 COMPLETE")
print(f"Sample: {OUT}/rdd_review_event_sample_from_raw.parquet")
