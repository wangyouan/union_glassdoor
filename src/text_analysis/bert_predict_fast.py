#!/usr/bin/env python
"""Fast BERT inference — batch tokenize + predict, minimal overhead."""

import pandas as pd, numpy as np, torch, os, sys, time
torch.set_num_threads(4); os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
MODEL_DIR = OUT / "bert_models"
BATCH = 256  # larger batch, no DataLoader overhead

# Load data
print("Loading...", flush=True)
df = pd.read_parquet(OUT / "text_analysis_sample.parquet")
texts = (df["review_pros"].fillna("") + " " + df["review_cons"].fillna("")).tolist()
n = len(texts)
print(f"{n:,} texts", flush=True)

# ── WLB Mention ──
print("\n=== WLB Mention ===", flush=True)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR / "wlb_mention")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / "wlb_mention")
model.eval()

mention_preds = []
t0 = time.time()
for i in range(0, n, BATCH):
    batch_texts = texts[i:i+BATCH]
    enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    mention_preds.extend(out.logits.argmax(-1).tolist())

    if (i // BATCH) % 20 == 0:
        elapsed = time.time() - t0
        rate = (i + BATCH) / elapsed if elapsed > 0 else 0
        eta = (n - i - BATCH) / max(rate, 1)
        print(f"  [{min(i+BATCH,n)}/{n}] {rate:.0f}/s ETA {eta/60:.0f}m", flush=True)

df["wlb_mention_bert"] = mention_preds
print(f"Done in {(time.time()-t0)/60:.1f}m", flush=True)

# ── WLB Sentiment ──
print("\n=== WLB Sentiment ===", flush=True)
tokenizer2 = DistilBertTokenizer.from_pretrained(MODEL_DIR / "wlb_sentiment")
model2 = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / "wlb_sentiment")
model2.eval()

sent_preds = []
t0 = time.time()
for i in range(0, n, BATCH):
    batch_texts = texts[i:i+BATCH]
    enc = tokenizer2(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model2(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    sent_preds.extend(out.logits.argmax(-1).tolist())

    if (i // BATCH) % 20 == 0:
        elapsed = time.time() - t0
        rate = (i + BATCH) / elapsed if elapsed > 0 else 0
        print(f"  [{min(i+BATCH,n)}/{n}] {rate:.0f}/s ETA {eta/60:.0f}m", flush=True)

df["wlb_sentiment_bert"] = sent_preds

# Map: 0=neg, 1=pos, 2=neutral → -1/0/1
df["wlb_net_text_bert"] = 0
mask = df["wlb_mention_bert"] == 1
df.loc[mask, "wlb_net_text_bert"] = df.loc[mask, "wlb_sentiment_bert"].map({0: -1, 1: 1, 2: 0})

# Stats
print(f"\nMention rate: {df['wlb_mention_bert'].mean()*100:.1f}%", flush=True)
print("Net WLB:", dict(df["wlb_net_text_bert"].value_counts().sort_index()), flush=True)

# Quick DiD-RD
print("\n=== BERT WLB by Post x Win ===", flush=True)
for p in [0, 1]:
    for w in [0, 1]:
        s = df[(df["post"] == p) & (df["win"] == w)]
        print(f"  Post={p} Win={w}: mention={s['wlb_mention_bert'].mean():.3f} net={s['wlb_net_text_bert'].mean():+.4f}", flush=True)

df.to_parquet(OUT / "full_sample_bert_predictions.parquet", index=False)
print(f"\nSaved ({len(df):,} rows)", flush=True)
