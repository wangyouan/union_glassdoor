#!/usr/bin/env python
"""
Step 2: Build event-level pre/post outcome data from RDD review-event sample.

For each election×outcome×filter×window, compute pre_mean_y, post_mean_y, delta_y.
Apply min-review thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
OUT = PROJ / "outputs" / "rdd_rebuild"
SAMPLE = OUT / "rdd_review_event_sample_from_raw.parquet"

print("Loading RDD review-event sample...")
df = pd.read_parquet(SAMPLE)
print(f"  {len(df):,} reviews, {df['gvkey'].nunique()} gvkeys, {df['election_id'].nunique()} elections")

outcome_cols = [c for c in ["overall_rating", "career_opp", "comp_benefit",
                              "senior_mgmt", "wlb", "culture", "diversity"]
                if c in df.columns]
employee_filters = ["current", "all"]  # former is diagnostic only
windows = [365, 180, 90]
thresholds = [
    ("pre>=1_post>=1", 1, 1, None),
    ("pre>=3_post>=3", 3, 3, None),
    ("pre>=5_post>=5", 5, 5, None),
    ("total>=5", None, None, 5),
    ("total>=10", None, None, 10),
]

# Build event-level data
all_rows = []

for oc in outcome_cols:
    print(f"\nOutcome: {oc}")
    for emp in employee_filters:
        mask_emp = slice(None) if emp == "all" else (df["employee_filter"] == emp)
        df_emp = df.loc[mask_emp].copy()

        for win_days in windows:
            col_w = f"within_{win_days}" if win_days < 365 else "within_365"
            mask_w = df_emp[col_w] if win_days < 365 else pd.Series(True, index=df_emp.index)
            df_ew = df_emp.loc[mask_w]

            # Group by election
            grp = df_ew[df_ew[oc].notna()].groupby("election_id")

            for eid, g in grp:
                # Take election-level attributes from first row
                e = g.iloc[0]
                pre_mask = g["days_to_election"] < 0
                post_mask = g["days_to_election"] >= 0

                n_pre = pre_mask.sum()
                n_post = post_mask.sum()
                n_total = n_pre + n_post

                if n_pre == 0 or n_post == 0:
                    continue

                pre_mean = g.loc[pre_mask, oc].mean()
                post_mean = g.loc[post_mask, oc].mean()
                delta = post_mean - pre_mean

                row = {
                    "outcome": oc,
                    "employee_filter": emp,
                    "window_days": win_days,
                    "election_id": eid,
                    "gvkey": e["gvkey"],
                    "margin": e["margin"],
                    "abs_margin": e["abs_margin"],
                    "win": e["win"],
                    "vote_share": e["vote_share"],
                    "election_year": e["election_year_elec"] if "election_year_elec" in g.columns else e["review_year"],
                    "case_number": e.get("case_number", "N/A"),
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "n_total": n_total,
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "delta": delta,
                }
                all_rows.append(row)

df_events = pd.DataFrame(all_rows)
print(f"\nTotal event-level rows (before thresholds): {len(df_events):,}")

# Apply thresholds
final_rows = []
for _, row in df_events.iterrows():
    for th_label, min_pre, min_post, min_total in thresholds:
        ok = True
        if min_pre is not None and row["n_pre"] < min_pre:
            ok = False
        if min_post is not None and row["n_post"] < min_post:
            ok = False
        if min_total is not None and row["n_total"] < min_total:
            ok = False
        if ok:
            r = row.to_dict()
            r["threshold"] = th_label
            final_rows.append(r)

df_final = pd.DataFrame(final_rows)
print(f"After thresholds: {len(df_final):,} rows")

# Save
df_final.to_parquet(OUT / "event_level_rdd_data.parquet", index=False)
print(f"Saved: event_level_rdd_data.parquet ({len(df_final):,} rows × {len(df_final.columns)} cols)")

# Diagnostics
print("\n--- Event counts by outcome/filter/window/threshold ---")
diag_rows = []
for (oc, emp, win_days, th), grp in df_final.groupby(["outcome", "employee_filter", "window_days", "threshold"]):
    n_elec = len(grp)
    n_gvkey = grp["gvkey"].nunique()
    n_win = grp["win"].sum()
    n_loss = n_elec - n_win
    mean_delta = grp["delta"].mean()
    sd_delta = grp["delta"].std()
    diag_rows.append({
        "outcome": oc, "employee_filter": emp, "window_days": win_days, "threshold": th,
        "n_events": n_elec, "n_gvkeys": n_gvkey,
        "n_win": int(n_win), "n_loss": int(n_loss),
        "mean_delta": mean_delta, "sd_delta": sd_delta,
        "mean_n_pre": grp["n_pre"].mean(), "mean_n_post": grp["n_post"].mean(),
    })

    if n_elec >= 30 and n_gvkey >= 20:
        print(f"  {oc} | {emp} | ±{win_days}d | {th}: "
              f"events={n_elec}, gvkeys={n_gvkey}, w={int(n_win)}, l={int(n_loss)}, "
              f"delta={mean_delta:.3f} (sd={sd_delta:.3f})")

df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv(OUT / "event_level_rdd_data_diagnostics.csv", index=False)

print(f"\nStep 2 complete. {len(df_final):,} event-level observations saved.")
