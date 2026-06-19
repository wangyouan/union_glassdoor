#!/usr/bin/env python
"""Full-sample BERT inference for pay/benefits — sharded with checkpointing."""

import pandas as pd, numpy as np, torch, os, sys, time
torch.set_num_threads(4); os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

TA = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
MODEL_DIR = TA / "bert_models"
BATCH = 256; SHARD_SIZE = 50000

print("Loading data...")
df = pd.read_parquet(TA / "text_analysis_sample.parquet")
texts = (df["review_pros"].fillna("") + " [SEP] " + df["review_cons"].fillna("")).tolist()
n = len(texts)
print(f"{n:,} texts")

def predict_sharded(task_name, model_path, n_shards=None):
    """Sharded inference with checkpointing."""
    # Load columns if already partially done
    col_name = f"{task_name}_prob"
    ckpt_file = TA / f"shard_ckpt_{task_name}.txt"
    done_shards = set()
    if ckpt_file.exists():
        with open(ckpt_file) as f: done_shards = set(int(x) for x in f.read().strip().split(",") if x)

    if col_name not in df.columns:
        df[col_name] = np.nan

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR / model_path)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / model_path)
    model.eval()

    total_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    t0 = time.time()

    for shard in range(total_shards):
        if shard in done_shards:
            continue

        start = shard * SHARD_SIZE; end = min(start + SHARD_SIZE, n)
        batch_texts = texts[start:end]

        probs = []
        for i in range(0, len(batch_texts), BATCH):
            batch_t = batch_texts[i:i+BATCH]
            enc = tokenizer(batch_t, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs.extend(torch.softmax(out.logits, -1)[:, 1].tolist())

        df.iloc[start:end, df.columns.get_loc(col_name)] = probs

        # Save checkpoint
        done_shards.add(shard)
        with open(ckpt_file, "w") as f: f.write(",".join(str(s) for s in sorted(done_shards)))

        elapsed = time.time() - t0
        rate = (len(done_shards) * SHARD_SIZE) / elapsed
        eta = (total_shards - len(done_shards)) * SHARD_SIZE / rate / 60 if rate > 0 else 0
        print(f"  [{task_name}] shard {shard+1}/{total_shards} ({end}/{n}) {rate:.0f}/s ETA {eta:.0f}m", flush=True)
        df.to_parquet(TA / "full_sample_with_text_predictions.parquet", index=False)  # incremental save

    elapsed = time.time() - t0
    print(f"  [{task_name}] DONE in {elapsed/60:.1f}m", flush=True)

# Run all 4
for task, model_name in [("pay_mention","pay_mention"),("pay_neg","pay_neg"),
                           ("benefits_mention","benefits_mention"),("benefits_neg","benefits_neg")]:
    predict_sharded(task, model_name)

# ── Build text measures ──
print("\nBuilding text measures...")
df["pay_complaint"] = df["pay_mention_prob"] * df["pay_neg_prob"]
df["benefits_complaint"] = df["benefits_mention_prob"] * df["benefits_neg_prob"]
for c in ["pay_complaint","benefits_complaint"]:
    df[f"{c}_std"] = (df[c] - df[c].mean()) / df[c].std()

df.to_parquet(TA / "full_sample_with_text_predictions.parquet", index=False)
print(f"Saved with new columns. Pay complaint mean={df['pay_complaint'].mean():.4f}, "
      f"benefits complaint mean={df['benefits_complaint'].mean():.4f}")
