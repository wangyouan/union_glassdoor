#!/usr/bin/env python
"""Annotate remaining dimensions: working_conditions, voice, mgmt, career, union, bargaining.
Fixed: review-level, balanced sentiment prompt, single ollama call per review, checkpointing."""

import pandas as pd, json, subprocess, sys, time
from pathlib import Path

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
OUT2 = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260619/text_analysis")
OUT2.mkdir(parents=True, exist_ok=True)
CKPT = OUT2 / "final_dims_checkpoint.csv"
FINAL = OUT2 / "annotation_final_dims.csv"

# Load existing annotations + text
df = pd.read_csv(OUT / "annotation_results_all_dims.csv")
txt = pd.read_parquet(OUT / "text_analysis_sample.parquet")[["review_id","review_pros","review_cons"]]
df = df.merge(txt, on="review_id", how="left")

# New dimensions
NEW_DIMS = {
    "working_conditions": "workload, hours, scheduling, overtime, pace, pressure, flexibility, shift patterns, weekend work - anything about your daily work conditions and time",
    "voice_fairness": "fairness of processes, whether workers can voice concerns, retaliation, favoritism, equal treatment, respect, dignity, discrimination",
    "mgmt_communication": "quality of management communication, transparency, information sharing, whether management listens, feedback from supervisors, clarity of expectations",
    "career_development": "promotions, career paths, training, skill development, mentorship, growth opportunities, internal mobility, job rotation",
    "union_sentiment": "any mention of unions, labor unions, strikes, collective action, labor organizing, union campaigns, workplace organizing, union membership",
    "bargaining_contract": "contract negotiations, wages being negotiated, benefits negotiations, collective bargaining, labor agreements, union contracts, seniority provisions, grievance procedures",
}

DIMS_ORDER = list(NEW_DIMS.keys())

def make_prompt(text):
    """Review-level prompt asking for mention + positive/negative sentiment."""
    parts = []
    for d in DIMS_ORDER:
        desc = NEW_DIMS[d]
        parts.append(f'"{d}_mention": 0 or 1. Does this text mention {desc}?')
        parts.append(f'"{d}_positive": 0 or 1. Does the text express POSITIVE feeling about {desc[:40]}?')
        parts.append(f'"{d}_negative": 0 or 1. Does the text express NEGATIVE feeling about {desc[:40]}?')

    return f"""Analyze this Glassdoor review as a whole (pros + cons combined).

{chr(10).join(parts)}

Reply ONLY in JSON. TEXT: {text[:800]}"""

def query(prompt):
    try:
        r = subprocess.run(["ollama","run","qwen2.5:3b",prompt], capture_output=True, text=True, timeout=120)
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
        if c != "review_id" and c in df.columns:
            df[c] = cp[c].values[:len(df)]
    print(f"Resuming: {len(done)} already done")

t0 = time.time()
for i in range(len(df)):
    rid = df.at[i, "review_id"]
    if rid in done:
        continue

    text = (str(df.at[i, "review_pros"]) + " " + str(df.at[i, "review_cons"]))[:800]
    resp = query(make_prompt(text)) if text.strip() else {}

    for d in DIMS_ORDER:
        mention = int(resp.get(f"{d}_mention", 0) or 0)
        pos = int(resp.get(f"{d}_positive", 0) or 0)
        neg = int(resp.get(f"{d}_negative", 0) or 0)
        df.at[i, f"{d}_mention"] = mention
        df.at[i, f"{d}_positive"] = pos
        df.at[i, f"{d}_negative"] = neg

    if (i+1) % 50 == 0:
        cols = ["review_id"] + [f"{d}_{x}" for d in DIMS_ORDER for x in ["mention","positive","negative"]]
        df[cols].iloc[:i+1].to_csv(CKPT, index=False)
        elapsed = time.time()-t0
        rate = (i+1-len(done))/elapsed*60
        eta = (len(df)-i-1)/max(rate,0.01)
        print(f"  [{i+1}/{len(df)}] {rate:.1f}/min ETA {eta/60:.1f}h", flush=True)

# Final save
cols = ["review_id"] + [f"{d}_{x}" for d in DIMS_ORDER for x in ["mention","positive","negative"]]
df[cols].to_csv(FINAL, index=False)
print(f"Saved annotation_final_dims.csv ({len(df)} rows)", flush=True)
