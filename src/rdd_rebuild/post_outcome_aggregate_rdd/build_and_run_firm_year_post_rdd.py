#!/usr/bin/env python
"""Script 1: Firm-year post-outcome RDD. DV=post_mean, PreRating control, gvkey-clustered SE."""

import pandas as pd, numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/post_outcome_aggregate_rdd"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
BANDWIDTHS = [("global",None),("m20",0.20),("m10",0.10),("m05",0.05)]
POLYS = ["linear","quadratic"]
FE_VARIANTS = [("none",[]),("year",["election_year"])]  # no industry column available

print("Loading RDD sample...")
df = pd.read_parquet(SAMPLE)
df["review_date"] = pd.to_datetime(df["review_date"])
# Reconstruct election_date
df["election_date"] = df["review_date"] - pd.to_timedelta(df["days_to_election"], unit="D")
df["election_year"] = df["election_date"].dt.year
df["days_from_election"] = df["days_to_election"]  # already computed
print(f"  {len(df):,} reviews, {df['gvkey'].nunique()} gvkeys, {df['election_id'].nunique()} elections")

def build_event_dataset(emp_sample):
    """Aggregate reviews to event level."""
    d = df.copy()
    if emp_sample == "current": d = d[d["is_current_employee"] == 1]

    rows = []
    for oc in OUTCOMES:
        sub = d[d[oc].notna()]
        for eid, g in sub.groupby("election_id"):
            pre = g[(g["days_from_election"] >= -365) & (g["days_from_election"] < 0)]
            post = g[(g["days_from_election"] >= 0) & (g["days_from_election"] <= 365)]
            n_pre, n_post = len(pre), len(post)
            if n_pre < 1 or n_post < 1: continue
            rows.append({
                "election_id": eid, "gvkey": str(g["gvkey"].iloc[0]),
                "election_date": g["election_date"].iloc[0],
                "election_year": int(g["election_date"].iloc[0].year),
                "win": int(g["win"].iloc[0]), "margin": g["margin"].iloc[0],
                "abs_margin": abs(g["margin"].iloc[0]),
                "outcome": oc,
                "pre_mean": pre[oc].mean(), "post_mean": post[oc].mean(),
                "n_pre": n_pre, "n_post": n_post, "n_total": n_pre + n_post,
            })
    return pd.DataFrame(rows)

def assign_versions(ev):
    """Assign multi-election versions A, B, C."""
    ev = ev.copy()
    # Step 1: deduplicate to election-level for version assignment
    elec_info = ev[["election_id","gvkey","election_date"]].drop_duplicates()
    elec_info["version"] = "A"
    # Version B: elections isolated from any other same-gvkey election within +/-365d
    for gv, grp in elec_info.groupby("gvkey"):
        dates = grp["election_date"].values
        eids = grp["election_id"].values
        for i, d in enumerate(dates):
            others = np.delete(dates, i)
            if len(others) == 0:
                elec_info.loc[grp.index[i], "version"] = "B"
            elif np.min(np.abs((others - d).astype("timedelta64[D]").astype(float))) > 365:
                elec_info.loc[grp.index[i], "version"] = "B"
    # Version C: first election per gvkey
    first_eids = elec_info.groupby("gvkey")["election_date"].idxmin()
    elec_info["is_first"] = elec_info.index.isin(first_eids)
    # Merge back
    ev = ev.drop(columns=["version","is_first"], errors="ignore").merge(
        elec_info[["election_id","version","is_first"]], on="election_id", how="left")
    return ev

def run_ols(data, outcome, bw_label, bw_fn, poly, fe_spec, fe_cols):
    d = bw_fn(data) if bw_fn is not None else data.copy()
    d = d[d["outcome"] == outcome].dropna(subset=["post_mean","pre_mean","win","margin"])
    if len(d) < 30: return None
    d["win_margin"] = d["win"] * d["margin"]
    formula = f"post_mean ~ win + margin + win_margin + pre_mean"
    if poly == "quadratic":
        d["margin2"] = d["margin"]**2; d["win_margin2"] = d["win"] * d["margin2"]
        formula += " + margin2 + win_margin2"
    for col in fe_cols:
        if col in d.columns:
            formula += f" + C({col})"
    try:
        m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["gvkey"]})
        return {
            "outcome": outcome, "bw_label": bw_label, "poly": poly, "fe_spec": fe_spec,
            "estimate": m.params.get("win", np.nan), "se": m.bse.get("win", np.nan),
            "p_value": m.pvalues.get("win", np.nan), "n_events": len(d),
            "n_gvkeys": d["gvkey"].nunique(),
            "pre_rating_coef": m.params.get("pre_mean", np.nan), "r_squared": m.rsquared}
    except: return None

# Build event datasets
print("\nBuilding event-level datasets...")
for emp in ["current", "all"]:
    ev = build_event_dataset(emp)
    ev = assign_versions(ev)
    ev.to_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet", index=False)
    print(f"  {emp}: {len(ev)} rows, {ev['gvkey'].nunique()} gvkeys, {ev['election_id'].nunique()} elections")

# Run regressions
print("\nRunning firm-year regressions...")
bw_funcs = {
    "global": None, "m20": lambda d: d[d["abs_margin"] <= 0.20],
    "m10": lambda d: d[d["abs_margin"] <= 0.10], "m05": lambda d: d[d["abs_margin"] <= 0.05]
}
results = []
for emp in ["current", "all"]:
    ev = pd.read_parquet(OUT / f"firm_year_post_rdd_data_{emp}.parquet")
    # Filters
    filters = [("unrestricted", lambda d: d)]  # already filtered at build
    if emp == "all":
        filters += [("n5", lambda d: d[(d["n_pre"]>=5)&(d["n_post"]>=5)]),
                    ("n10", lambda d: d[(d["n_pre"]>=10)&(d["n_post"]>=10)])]
    for f_label, f_fn in filters:
        d_f = f_fn(ev)
        versions = {"A": d_f[d_f["version"]=="A"], "B": d_f[d_f["version"]=="B"],
                    "C": d_f[d_f["is_first"]]}
        for v_label, d_v in versions.items():
            if v_label in ["B","C"]:  # only for global and m20
                bws_to_use = [("global",None),("m20",0.20)]
            else:
                bws_to_use = BANDWIDTHS
            for bw_label, bw_thresh in bws_to_use:
                bw_fn = bw_funcs[bw_label]
                for poly in POLYS:
                    for fe_spec, fe_cols in FE_VARIANTS:
                        for oc in OUTCOMES:
                            res = run_ols(d_v, oc, bw_label, bw_fn, poly, fe_spec, fe_cols)
                            if res:
                                res.update({"employee_sample": emp, "filter": f_label,
                                    "multi_version": v_label})
                                results.append(res)
        n_done = len(results)
        pct = min(100, n_done / 2000 * 100)
        print(f"\r  {emp} {f_label}: {n_done} results ({pct:.0f}%)", end="", flush=True)

df_r = pd.DataFrame(results)
df_r.to_csv(OUT / "firm_year_post_rdd_results.csv", index=False)
print(f"\nSaved {len(df_r)} results")

# Main table
print("\n=== Firm-Year Main (current, A, global, linear, no FE) ===")
m = df_r[(df_r["employee_sample"]=="current")&(df_r["multi_version"]=="A")&
         (df_r["bw_label"]=="global")&(df_r["poly"]=="linear")&(df_r["fe_spec"]=="none")&
         (df_r["filter"]=="unrestricted")]
for _, r in m.sort_values("outcome").iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: tau={r['estimate']:+.4f} se={r['se']:.4f} p={r['p_value']:.3f}{sig} E={int(r['n_events'])}")

print("\nDone.")
