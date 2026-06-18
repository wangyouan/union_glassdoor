#!/usr/bin/env python
"""LLM Annotation via Ollama — WLB mention/sentiment extraction from pros/cons text.
Saves checkpoint every 50 reviews for fault tolerance."""

import pandas as pd, json, time, subprocess, numpy as np, sys
from pathlib import Path

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
SAMPLE = OUT / "annotation_sample.csv"
CHECKPOINT = OUT / "annotation_checkpoint.csv"

df = pd.read_csv(SAMPLE)
print(f"Loaded {len(df)} reviews")

# Load checkpoint
done_ids = set()
if CHECKPOINT.exists():
    cp = pd.read_csv(CHECKPOINT)
    done_ids = set(cp["review_id"].values)
    print(f"Checkpoint: {len(done_ids)} already done")

# Build prompt
def make_prompt(text):
    return f"""You are analyzing a Glassdoor employee review. For the TEXT below, answer:

1. WLB_MENTION: Does this text mention anything about work-life balance, working hours, schedule flexibility, overtime, burnout, time off, weekends, or shift length?
   Answer YES or NO.

2. WLB_SENTIMENT: If the text mentions WLB, is the sentiment POSITIVE, NEGATIVE, or NEUTRAL? If no WLB mention, answer NONE.

Reply ONLY in this exact JSON format: {{"WLB_MENTION": "YES/NO", "WLB_SENTIMENT": "POSITIVE/NEGATIVE/NEUTRAL/NONE"}}

TEXT: {text}"""

def query_ollama(prompt, model="qwen2.5:3b"):
    """Call ollama and return parsed JSON."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=120
        )
        raw = result.stdout.strip()
        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        return {"WLB_MENTION": "PARSE_ERROR", "WLB_SENTIMENT": "PARSE_ERROR"}
    except Exception as e:
        return {"WLB_MENTION": "ERROR", "WLB_SENTIMENT": str(e)[:50]}

# Annotate
results = []
if CHECKPOINT.exists():
    results = pd.read_csv(CHECKPOINT).to_dict("records")

start_time = time.time()
for i, row in df.iterrows():
    rid = row["review_id"]
    if rid in done_ids:
        continue

    pros = str(row.get("review_pros", ""))[:500]
    cons = str(row.get("review_cons", ""))[:500]

    # Annotate pros
    resp_pros = query_ollama(make_prompt(pros)) if pros.strip() else {"WLB_MENTION": "NO", "WLB_SENTIMENT": "NONE"}
    resp_cons = query_ollama(make_prompt(cons)) if cons.strip() else {"WLB_MENTION": "NO", "WLB_SENTIMENT": "NONE"}

    results.append({
        "review_id": rid,
        "wlb_pros_mention": resp_pros.get("WLB_MENTION", "ERROR"),
        "wlb_pros_sentiment": resp_pros.get("WLB_SENTIMENT", "ERROR"),
        "wlb_cons_mention": resp_cons.get("WLB_MENTION", "ERROR"),
        "wlb_cons_sentiment": resp_cons.get("WLB_SENTIMENT", "ERROR"),
    })

    # Checkpoint every 50 reviews
    if len(results) % 50 == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
        elapsed = time.time() - start_time
        rate = len(results) / elapsed * 60
        eta = (len(df) - len(results)) / max(rate, 0.01)
        print(f"  [{len(results)}/{len(df)}] {rate:.1f} reviews/min, ETA: {eta/60:.1f}h", flush=True)

# Final save
pd.DataFrame(results).to_csv(OUT / "annotation_results.csv", index=False)
print(f"\nDone! {len(results)} reviews annotated.")
print(f"Saved: annotation_results.csv")
