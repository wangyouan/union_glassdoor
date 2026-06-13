#!/usr/bin/env python
"""B. rdrobust local-linear check on event-level delta."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

PROJ = Path("/data/disk4/workspace/projects/union_glassdoor")
SAMPLE = PROJ / "outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet"
OUT = PROJ / "outputs/rdd_rebuild/focused_rdd_search_v3"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]

df = pd.read_parquet(SAMPLE)

def run_rdrobust(oc, emp="current", wd=365):
    sub = df[df[oc].notna() & (df["employee_filter"]==emp)].copy()
    if wd < 365: sub = sub[sub[f"within_{wd}"]]
    # Build event-level delta
    rows = []
    for eid, g in sub.groupby("election_id"):
        pre=g[g["days_to_election"]<0]; post=g[g["days_to_election"]>=0]
        if len(pre)<1 or len(post)<1: continue
        rows.append({"election_id":eid,"gvkey":g["gvkey"].iloc[0],"margin":g["margin"].iloc[0],
                     "win":g["win"].iloc[0],"pre_mean":pre[oc].mean(),"post_mean":post[oc].mean()})
    ev = pd.DataFrame(rows)
    if len(ev)<30: return None
    ev["delta"] = ev["post_mean"]-ev["pre_mean"]
    mu_d, sd_d = ev["delta"].mean(), ev["delta"].std()
    if sd_d==0: return None
    ev["delta_sd"] = (ev["delta"]-mu_d)/sd_d

    y = ev["delta_sd"].values; x = ev["margin"].values; gv = ev["gvkey"].values
    n = len(y)

    # Default bandwidth (Silverman)
    h_default = min(1.84 * np.std(x) * n**(-1/5), 0.30)

    results = []
    for h in [h_default, 0.10, 0.15, 0.20, 0.25]:
        mask = np.abs(x) <= h
        if mask.sum() < 30: continue
        xh, yh = x[mask], y[mask]; gh = gv[mask]
        wh = (xh > 0).astype(float)
        w = 1 - np.abs(xh)/h; w = w/w.sum()*len(w)  # triangular kernel

        X_h = np.column_stack([np.ones_like(wh), wh, xh, wh*xh])
        n_h, k_h = X_h.shape
        betah = np.linalg.lstsq(X_h, yh, rcond=None)[0]
        resid_h = yh - X_h@betah
        # Cluster SE
        uq = np.unique(gh); G = len(uq)
        if G >= 15:
            meat_h = np.zeros((k_h,k_h))
            for g in uq:
                mg = gh==g; Xg=X_h[mg]; rg=resid_h[mg]
                meat_h += (Xg.T@rg)[:,None] @ (Xg.T@rg)[None,:]
            vcov_h = np.linalg.inv(X_h.T@X_h) @ meat_h @ np.linalg.inv(X_h.T@X_h)
            vcov_h *= (G/(G-1))*((n_h-1)/(n_h-k_h))
            se_h = np.sqrt(np.diag(vcov_h))
        else:
            se_h = np.sqrt(np.diag(np.linalg.inv(X_h.T@X_h) * (resid_h@resid_h)/(n_h-k_h)))

        tau_h, se_tau = betah[1], se_h[1]
        pv_h = 2*stats.t.sf(abs(tau_h/se_tau), n_h-k_h) if se_tau>0 else np.nan
        results.append({"outcome":oc,"bandwidth":h,"bandwidth_label":f"{h:.2f}",
            "estimate":tau_h,"standard_error":se_tau,"p_value":pv_h,
            "n_effective":int(mask.sum()),"n_left":int((xh<0).sum()),"n_right":int((xh>=0).sum()),
            "is_default": abs(h-h_default)<0.001})

    return results

all_results = []
for oc in OUTCOMES:
    res_list = run_rdrobust(oc)
    if res_list: all_results.extend(res_list)
    print(f"  {oc}: {len(res_list) if res_list else 0} bandwidths")

df_ll = pd.DataFrame(all_results)
df_ll.to_csv(OUT / "rdrobust_check_results.csv", index=False)
print(f"Saved {len(df_ll)} rdrobust results")

# Show defaults
print("\n--- Default bandwidth results ---")
defaults = df_ll[df_ll["is_default"]==True]
for _,r in defaults.iterrows():
    sig = "***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: h={r['bandwidth']:.3f} tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig} N={int(r['n_effective'])}")
