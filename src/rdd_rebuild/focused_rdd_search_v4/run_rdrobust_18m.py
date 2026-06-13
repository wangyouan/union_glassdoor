#!/usr/bin/env python
"""C. rdrobust check on ±548d event-level delta."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

IN_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/focused_rdd_search_v4")
df = pd.read_parquet(IN_DIR / "rdd_review_event_sample_18m.parquet")
OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]

def run_ll(oc, wd, emp="current"):
    sub = df[df[oc].notna()&(df["employee_filter"]==emp)].copy()
    if wd==548: sub=sub[sub["within_548"]]
    elif wd==365: sub=sub[sub["within_365"]]
    else: sub=sub[sub["within_180"]]
    rows = []
    for eid,g in sub.groupby("election_id"):
        pre=g[g["days_to_election"]<0]; post=g[g["days_to_election"]>=0]
        if len(pre)<1 or len(post)<1: continue
        rows.append({"election_id":eid,"gvkey":g["gvkey"].iloc[0],"margin":g["margin"].iloc[0],
                     "win":g["win"].iloc[0],"pre_mean":pre[oc].mean(),"post_mean":post[oc].mean()})
    ev=pd.DataFrame(rows)
    if len(ev)<30: return []
    ev["delta"]=ev["post_mean"]-ev["pre_mean"]; mu_d,sd_d=ev["delta"].mean(),ev["delta"].std()
    if sd_d==0: return []
    ev["delta_sd"]=(ev["delta"]-mu_d)/sd_d
    y=ev["delta_sd"].values; x=ev["margin"].values; gv=ev["gvkey"].values; n=len(y)
    h_def=min(1.84*np.std(x)*n**(-1/5),0.30)
    res=[]
    for h in [h_def,0.10,0.15,0.20,0.25]:
        mask=np.abs(x)<=h
        if mask.sum()<30: continue
        xh,yh,gh=x[mask],y[mask],gv[mask]; wh=(xh>0).astype(float)
        w=1-np.abs(xh)/h; w=w/w.sum()*len(w)
        Xh=np.column_stack([np.ones_like(wh),wh,xh,wh*xh]); nh,kh=Xh.shape
        betah=np.linalg.lstsq(Xh,yh,rcond=None)[0]; resid_h=yh-Xh@betah
        uq=np.unique(gh); G=len(uq)
        if G>=15:
            meat_h=np.zeros((kh,kh))
            for g in uq: mg=gh==g; Xg=Xh[mg]; rg=resid_h[mg]; meat_h+=(Xg.T@rg)[:,None]@(Xg.T@rg)[None,:]
            vcov_h=np.linalg.inv(Xh.T@Xh)@meat_h@np.linalg.inv(Xh.T@Xh); vcov_h*=(G/(G-1))*((nh-1)/(nh-kh))
            se_h=np.sqrt(np.diag(vcov_h))
        else: se_h=np.sqrt(np.diag(np.linalg.inv(Xh.T@Xh)*(resid_h@resid_h)/(nh-kh)))
        tau_h,se_t=betah[1],se_h[1]
        res.append({"outcome":oc,"window_days":wd,"bandwidth":h,
            "estimate":tau_h,"standard_error":se_t,"p_value":2*stats.t.sf(abs(tau_h/se_t),nh-kh) if se_t>0 else np.nan,
            "n_effective":int(mask.sum()),"is_default":abs(h-h_def)<0.001})
    return res

all_ll=[]
for oc in OUTCOMES:
    for wd in [548,365]:
        rlist=run_ll(oc,wd)
        if rlist: all_ll.extend(rlist)
    print(f"  {oc}: {len([r for r in all_ll if r['outcome']==oc])} bw-points")

df_ll=pd.DataFrame(all_ll)
df_ll.to_csv(IN_DIR / "rdrobust_18m_results.csv", index=False)
print(f"Saved {len(df_ll)} results")
print("\n--- Default bw, ±548d ---")
for _,r in df_ll[(df_ll["is_default"])&(df_ll["window_days"]==548)].iterrows():
    sig="***" if r["p_value"]<0.01 else "**" if r["p_value"]<0.05 else "*" if r["p_value"]<0.10 else ""
    print(f"  {r['outcome']:20s}: h={r['bandwidth']:.3f} tau={r['estimate']:+.4f} se={r['standard_error']:.4f} p={r['p_value']:.3f}{sig}")
