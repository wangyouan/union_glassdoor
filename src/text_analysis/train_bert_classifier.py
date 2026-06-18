#!/usr/bin/env python
"""CPU-friendly BERT classifier: DistilBERT + 3 epochs + small batches.
Predicts WLB mention (binary) and WLB sentiment (3-class) from pros+cons text."""

import pandas as pd, numpy as np, torch, json, os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
MODEL_DIR = OUT / "bert_models"; MODEL_DIR.mkdir(exist_ok=True)
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 8  # small for CPU
EPOCHS = 3
LEARNING_RATE = 2e-5

# ── Load and prepare data ──
print("Loading annotation data...")
df = pd.read_csv(OUT / "annotation_results.csv")
text = pd.read_parquet(OUT / "text_analysis_sample.parquet")[["review_id","review_pros","review_cons"]]
df = df.merge(text, on="review_id", how="left")

# Combine pros + cons
df["text"] = df["review_pros"].fillna("") + " [SEP] " + df["review_cons"].fillna("")

# Labels: mention
df["wlb_mention_label"] = ((df["wlb_pros_mention"] == "YES") | (df["wlb_cons_mention"] == "YES")).astype(int)

# Labels: sentiment (-1=neg, 0=none/neutral, 1=pos)
def get_sentiment(row):
    pros_neg = (row["wlb_pros_mention"] == "YES" and row["wlb_pros_sentiment"] == "NEGATIVE")
    cons_pos = (row["wlb_cons_mention"] == "YES" and row["wlb_cons_sentiment"] == "POSITIVE")
    pros_pos = (row["wlb_pros_mention"] == "YES" and row["wlb_pros_sentiment"] == "POSITIVE")
    cons_neg = (row["wlb_cons_mention"] == "YES" and row["wlb_cons_sentiment"] == "NEGATIVE")
    if (pros_pos or cons_pos) and not (pros_neg or cons_neg):
        return 1  # positive
    elif (pros_neg or cons_neg) and not (pros_pos or cons_pos):
        return -1  # negative
    elif (pros_pos or cons_pos) and (pros_neg or cons_neg):
        return 0  # mixed, treat as neutral
    return 0  # no mention or neutral

df["wlb_sentiment_label"] = df.apply(get_sentiment, axis=1)
df["sentiment_class"] = df["wlb_sentiment_label"].map({-1: 0, 0: 1, 1: 2})

# Train/val/test split (from annotation_sample.csv)
split_df = pd.read_csv(OUT / "annotation_sample.csv")[["review_id","split"]]
df = df.merge(split_df, on="review_id", how="left")
df["split"] = df["split"].fillna("train")

print(f"Train: {(df['split']=='train').sum()}, Val: {(df['split']=='val').sum()}, Test: {(df['split']=='test').sum()}")
print(f"Mention distribution: {df['wlb_mention_label'].value_counts().to_dict()}")
print(f"Sentiment distribution: {dict(zip([-1,0,1], np.bincount(df['sentiment_class'], minlength=3)))}")

# ── Dataset ──
class WLBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Train function ──
def train_model(task_name, label_col, num_labels, output_name):
    print(f"\n{'='*50}\nTraining {task_name} ({num_labels} classes)\n{'='*50}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    train_df = df[df["split"]=="train"]; val_df = df[df["split"]=="val"]
    train_ds = WLBDataset(train_df["text"].values, train_df[label_col].values, tokenizer)
    val_ds = WLBDataset(val_df["text"].values, val_df[label_col].values, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    device = torch.device("cpu")
    model.to(device)
    best_val_f1 = 0

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device),
                           labels=batch["labels"].to(device))
            loss = outputs.loss; loss.backward()
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for batch in val_dl:
                outputs = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device))
                preds.extend(outputs.logits.argmax(-1).tolist())
                trues.extend(batch["labels"].tolist())

        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="macro")
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f}, val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(MODEL_DIR / output_name)
            tokenizer.save_pretrained(MODEL_DIR / output_name)

    print(f"Best val F1: {best_val_f1:.3f}, model saved to {MODEL_DIR/output_name}")
    return model, tokenizer

# ── Train two models ──
# 1. Mention classifier (binary)
mention_model, mention_tokenizer = train_model("WLB Mention", "wlb_mention_label", 2, "wlb_mention")
# 2. Sentiment classifier (3-class)
sentiment_model, sentiment_tokenizer = train_model("WLB Sentiment", "sentiment_class", 3, "wlb_sentiment")
print(f"\nDone at {datetime.now()}")
