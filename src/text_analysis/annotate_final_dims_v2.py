#!/usr/bin/env python
"""Re-annotate missing dimensions in small groups (2 per prompt) for speed."""

import pandas as pd, json, subprocess, sys, time
from pathlib import Path

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
OUT2 = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260619/text_analysis")
OUT2.mkdir(parents=True, exist_ok=True)
CKPT = OUT2 / "final_dims_ckpt.csv"

df = pd.read_csv(OUT / "annotation_results_all_dims.csv")
txt = pd.read_parquet(OUT / "text_analysis_sample.parquet")[["review_id","review_pros","review_cons"]]
df = df.merge(txt, on="review_id", how="left")

# Group dimensions in groups of 2 for faster inference
DIM_GROUPS = [
    {
        "working_conditions": "workload, hours, scheduling, overtime, pace, work-life balance, shifts, flexibility, burnout",
        "voice_fairness": "fair treatment, favoritism, respect, dignity, equal treatment, retaliation, discrimination, bias"
    },
    {
        "career_development": "promotions, career paths, training, skill development, mentorship, growth opportunities, internal mobility, raises",
        "mgmt_communication": "management communication, transparency, information sharing, whether management listens, feedback, clarity, direction"
    },
    {
        "union_sentiment": "unions, labor unions, strikes, collective action, organizing, union campaigns, union membership",
        "bargaining_contract": "contract negotiations, collective bargaining, labor agreements, union contracts, wage negotiations, grievance procedures, seniority rules"
    }
]

def query(prompt):
    try:
        r = subprocess.run(["ollama","run","qwen2.5:3b",prompt], capture_output=True, text=True, timeout=180)
        s=r.stdout.find("{"); e=r.stdout.rfind("}")+1
        return json.loads(r.stdout[s:e]) if s>=0 and e>s else {}
    except:
        return {}

# Resume
done = set()
if CKPT.exists():
    cp = pd.read_csv(CKPT)
    done = set(cp["review_id"].values)
    for c in cp.columns:
        if c != "review_id" and c not in df.columns:
            df[c] = cp[c].values[:len(df)]
    print(f"Resuming: {len(done)} done")

t0, n_done = time.time(), 0
for i in range(len(df)):
    rid = df.at[i, "review_id"]
    if rid in done: n_done += 1; continue
    text = (str(df.at[i, "review_pros"]) + " " + str(df.at[i, "review_cons"]))[:500]

    for group in DIM_GROUPS:
        dims = list(group.keys())
        fields = []
        for d in dims:
            fields += [f'"{d}_mention": 0/1. Mentions {group[d]}?', f'"{d}_positive": 0/1. Positive?', f'"{d}_negative": 0/1. Negative?']
        prompt = f"Review text: {text}\n\nFor each dimension, answer 0 or 1.\n" + "\n".join(fields) + "\n\nReply JSON only."
        resp = query(prompt)
        for d in dims:
            df.at[i, f"{d}_mention"] = int(resp.get(f"{d}_mention", 0) or 0)
            df.at[i, f"{d}_positive"] = int(resp.get(f"{d}_positive", 0) or 0)
            df.at[i, f"{d}_negative"] = int(resp.get(f"{d}_negative", 0) or 0)

    n_done += 1
    if n_done % 50 == 0:
        all_dims = []
        for g in DIM_GROUPS:
            for d in g: all_dims += [f"{d}_{x}" for x in ["mention","positive","negative"]]
        df[["review_id"]+all_dims].iloc[:i+1].to_csv(CKPT, index=False)
        elapsed = time.time()-t0
        rate = (n_done-len(done))/elapsed*60
        eta = (len(df)-n_done)/max(rate,0.01)
        print(f"  [{n_done}/{len(df)}] {rate:.1f}/min, ETA {eta/60:.1f}h", flush=True)

all_dims = []
for g in DIM_GROUPS:
    for d in g: all_dims += [f"{d}_{x}" for x in ["mention","positive","negative"]]
df[["review_id"]+all_dims].to_csv(OUT2 / "annotation_final_dims.csv", index=False)
print(f"Saved annotation_final_dims.csv ({len(df)} rows)", flush=True)
