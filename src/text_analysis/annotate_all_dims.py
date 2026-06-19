#!/usr/bin/env python
"""Annotate remaining 5 dimensions (schedule, work_intensity, pay, benefits, job_security)
for the same 2,986 reviews. WLB already done — this adds the missing columns."""

import pandas as pd, json, time, subprocess, sys
from pathlib import Path

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
CHECKPOINT = OUT / "annotation_all_dims_checkpoint.csv"

df = pd.read_csv(OUT / "annotation_results.csv")
text = pd.read_parquet(OUT / "text_analysis_sample.parquet")[["review_id","review_pros","review_cons"]]
df = df.merge(text, on="review_id", how="left")

# Resume from checkpoint
done_ids = set()
if CHECKPOINT.exists():
    cp = pd.read_csv(CHECKPOINT)
    done_ids = set(cp["review_id"].values)
    # Merge already-completed results
    for c in ["schedule_mention","schedule_sent","work_intensity_mention","work_intensity_sent",
              "pay_mention","pay_sent","benefits_mention","benefits_sent",
              "job_security_mention","job_security_sent"]:
        if c in cp.columns:
            df[c] = cp[c]
    print(f"Resuming from checkpoint: {len(done_ids)} already done")
else:
    for c in ["schedule_mention","schedule_sent","work_intensity_mention","work_intensity_sent",
              "pay_mention","pay_sent","benefits_mention","benefits_sent",
              "job_security_mention","job_security_sent"]:
        df[c] = "PENDING"

def make_prompt(text, dims):
    """Build prompt for multiple dimensions at once."""
    dim_desc = {
        "schedule": "working hours, shift schedules, overtime, weekends, holidays, time off, PTO, flexibility in when you work",
        "work_intensity": "workload, work pressure, pace of work, understaffing, tight deadlines, work-life balance challenges",
        "pay": "salary, wages, hourly pay, raises, bonuses, pay increases, compensation level, pay fairness",
        "benefits": "health insurance, 401k, retirement, vacation days, sick leave, parental leave, tuition reimbursement, perks",
        "job_security": "layoffs, job stability, fear of losing job, restructuring, downsizing, long-term employment prospects"
    }
    parts = []
    for d in dims:
        parts.append(f'"{d}_mention": Does this text mention {dim_desc[d]}? Answer 0 (no) or 1 (yes).')
        parts.append(f'"{d}_sentiment": If mentioned, is the sentiment -1 (negative/complaint), 0 (neutral/mixed), or 0 if not mentioned.')

    return f"""Analyze this Glassdoor review text. For each dimension, determine mention (0/1) and sentiment (-1/0).

{chr(10).join(parts)}

Reply ONLY in JSON format with these EXACT keys: {json.dumps([f"{d}_mention" for d in dims] + [f"{d}_sentiment" for d in dims])}

TEXT: {text}"""

def query_ollama(prompt, model="qwen2.5:3b"):
    try:
        result = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=120)
        raw = result.stdout.strip()
        s = raw.find("{"); e = raw.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(raw[s:e])
        return {}
    except:
        return {}

# Dimensions to annotate
DIMS = ["schedule", "work_intensity", "pay", "benefits", "job_security"]

start_time = time.time()
n_done = 0
for i, row in df.iterrows():
    rid = row["review_id"]
    if rid in done_ids:
        n_done += 1
        continue

    pros = str(row.get("review_pros", ""))[:500]
    cons = str(row.get("review_cons", ""))[:500]

    # Query all 5 dims in one prompt per text
    resp_pros = query_ollama(make_prompt(pros, DIMS)) if pros.strip() else {}
    resp_cons = query_ollama(make_prompt(cons, DIMS)) if cons.strip() else {}

    for d in DIMS:
        df.at[i, f"{d}_mention"] = f"{d}_pros"  # temporary, will be combined below
        # Combine: mention if either pros or cons mentions
        mention_pros = int(resp_pros.get(f"{d}_mention", 0) or 0)
        mention_cons = int(resp_cons.get(f"{d}_mention", 0) or 0)
        df.at[i, f"{d}_mention"] = "YES" if (mention_pros or mention_cons) else "NO"

        # Sentiment: aggregate from pros and cons
        sent_pros = int(resp_pros.get(f"{d}_sentiment", 0) or 0)
        sent_cons = int(resp_cons.get(f"{d}_sentiment", 0) or 0)
        if sent_pros > 0 and sent_cons >= 0:
            df.at[i, f"{d}_sent"] = "POSITIVE"
        elif sent_cons < 0 and sent_pros <= 0:
            df.at[i, f"{d}_sent"] = "NEGATIVE"
        elif sent_pros == 0 and sent_cons == 0:
            df.at[i, f"{d}_sent"] = "NONE"
        else:
            df.at[i, f"{d}_sent"] = "NEUTRAL"

    n_done += 1

    # Checkpoint every 50
    if n_done % 50 == 0:
        # Save all annotation columns
        save_cols = ["review_id", "wlb_pros_mention","wlb_pros_sentiment","wlb_cons_mention","wlb_cons_sentiment"]
        for d in DIMS:
            save_cols += [f"{d}_mention", f"{d}_sent"]
        df[save_cols].to_csv(CHECKPOINT, index=False)

        elapsed = time.time() - start_time
        rate = (n_done - len(done_ids)) / elapsed * 60
        remaining = len(df) - n_done - len(done_ids)
        eta = remaining / max(rate, 0.01)
        print(f"  [{n_done}/{len(df)}] {rate:.1f} reviews/min, ETA: {eta/60:.1f}h", flush=True)

# Final save: merge with original WLB results
orig = pd.read_csv(OUT / "annotation_results.csv")
for d in DIMS:
    orig[f"{d}_mention"] = df[f"{d}_mention"]
    orig[f"{d}_sent"] = df[f"{d}_sent"]
orig.to_csv(OUT / "annotation_results_all_dims.csv", index=False)
print(f"\nDone! Saved annotation_results_all_dims.csv ({len(orig)} rows)")
