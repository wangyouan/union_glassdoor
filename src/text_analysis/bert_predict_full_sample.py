#!/usr/bin/env python
"""BERT inference on full 490k sample — mention + sentiment prediction."""
import pandas as pd, numpy as np, torch, os
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
MODEL_DIR = OUT / "bert_models"
BATCH_SIZE = 32  # larger batch for inference

# Load data
print("Loading data...")
df = pd.read_parquet(OUT / "text_analysis_sample.parquet")
texts = (df["review_pros"].fillna("") + " " + df["review_cons"].fillna("")).tolist()
print(f"{len(texts):,} reviews loaded")

# Load models
mention_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR / "wlb_mention")
mention_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / "wlb_mention")
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR / "wlb_sentiment")
sentiment_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / "wlb_sentiment")

# ── Inference ──
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts, self.tokenizer, self.max_len = texts, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

def predict(model, tokenizer, texts, batch_size=32):
    ds = TextDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for input_ids, attn in tqdm(dl, desc="Predicting"):
            outputs = model(input_ids=input_ids, attention_mask=attn)
            all_preds.extend(outputs.logits.argmax(-1).tolist())
    return all_preds

print("\nPredicting WLB mention...")
df["wlb_mention_bert"] = predict(mention_model, mention_tokenizer, texts)

print("Predicting WLB sentiment...")
df["wlb_sentiment_bert"] = predict(sentiment_model, sentiment_tokenizer, texts)

# Map sentiment: 0=neg, 1=pos, 2=neutral
df["wlb_net_text_bert"] = 0
df.loc[df["wlb_mention_bert"]==1, "wlb_net_text_bert"] = \
    df.loc[df["wlb_mention_bert"]==1, "wlb_sentiment_bert"].map({0:-1, 1:1, 2:0})

print(f"\nMention rate: {df['wlb_mention_bert'].mean()*100:.1f}%")
print(f"Net WLB text distribution:")
print(df["wlb_net_text_bert"].value_counts().sort_index())

# Quick DiD-RD stats
print("\n=== BERT WLB by Post x Win ===")
for post_v in [0,1]:
    for win_v in [0,1]:
        sub = df[(df['post']==post_v)&(df['win']==win_v)]
        m = sub["wlb_mention_bert"].mean()
        net = sub["wlb_net_text_bert"].mean()
        print(f"  Post={post_v} Win={win_v}: mention={m:.3f} net={net:+.4f} N={len(sub):,}")

df.to_parquet(OUT / "full_sample_bert_predictions.parquet", index=False)
print(f"\nSaved full_sample_bert_predictions.parquet ({len(df):,} rows)")
