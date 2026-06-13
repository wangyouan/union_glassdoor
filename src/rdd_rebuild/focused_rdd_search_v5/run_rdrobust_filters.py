#!/usr/bin/env python
"""B. rdrobust check per filter threshold, ±365d, current employees."""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

IN_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/focused_rdd_search_v5")
df = pd.read_parquet("/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/rdd_review_event_sample_from_raw.parquet")
OUTCOMES = ["overall_rating","career_opp","comp_benefit","senior_mgmt","wlb","culture"]
FILTERS = [("pre_post",1),("pre_post",5),("pre_post",10),("pre_post",20),("pre_post",25),("pre_post",50),("total",50),("total",100)]

def run_ll(oc, ft_type, ft_val):
    sub = df[df[oc].notna() & (df["employee_filter"]=="current")].copy()
    grp = sub.groupby("election_id")["post"]
    st = grp.agg(n_post="sum", n_pre=lambda x: (~x.astype(bool)).sum()).reset_index()
    if ft_type == "total":
        st["n_total"] = st["n_post"]+st["n_pre"]; valid = st[st["n_total"]>=ft_val]["election_id"]
    else:
        valid = st[(st["n_post"]>=ft_val)&(st["n_pre"]>=ft_val)]["election_id"]
    sub = sub[sub["election_id"].isin(valid)]
    rows=[]
    for eid,g in sub.groupby("election_id"):
        pre=g[g["days_to_election"]<0]; post=g[g["days_to_election"]>=0]
        if len(pre)<1 or len(post)<1: continue
        rows.append({"election_id":eid,"gvkey":g["gvkey"].iloc[0],"margin":g["margin"].iloc[0],
                     "win":g["win"].iloc[0],"pre_mean":pre[oc].mean(),"post_mean":post[oc].mean()})
    ev=pd.DataFrame(rows)
    if len(ev)<20: return None
    ev["delta"]=ev["post_mean"]-ev["pre_mean"]; mu_d,sd_d=ev["delta"].mean(),ev["delta"].std()
    if sd_d==0: return None
    ev["delta_sd"]=(ev["delta"]-mu_d)/sd_d
    y=ev["delta_sd"].values; x=ev["margin"].values; gv=ev["gvkey"].values; n=len(y)
    h=min(1.84*np.std(x)*n**(-1/5),0.30)
    mask=np.abs(x)<=h
    if mask.sum()<20: return None
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
    return {"estimate":tau_h,"standard_error":se_t,"p_value":2*stats.t.sf(abs(tau_h/se_t),nh-kh) if se_t>0 else np.nan,
            "n_effective":int(mask.sum()),"bandwidth":h}

results=[]
for oc in OUTCOMES:
    for ft_type,ft_val in FILTERS:
        r=run_ll(oc,ft_type,ft_val)
        if r: results.append({"outcome":oc,"window_days":365,"filter_type":ft_type,"filter_N":ft_val,**r})
    print(f"  {oc}: {len([x for x in results if x['outcome']==oc])} filters")

df_ll=pd.DataFrame(results)
df_ll.to_csv(IN_DIR / "rdrobust_filter_results.csv", index=False)
print(f"Saved {len(df_ll)} results")
