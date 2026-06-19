#!/usr/bin/env python
"""Train 4 BERT classifiers for pay/benefits mention + negative sentiment."""

import pandas as pd, numpy as np, torch, os, json, time, sys
torch.set_num_threads(4); os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

TA = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
MODEL_DIR = TA / "bert_models"; MODEL_DIR.mkdir(exist_ok=True)
BATCH, EPOCHS, LR = 16, 3, 2e-5
MODEL_NAME = "distilbert-base-uncased"

# ── Load data ──
print("Loading...")
ann = pd.read_csv(TA / "annotation_results_all_dims.csv")
smp = pd.read_csv(TA / "annotation_sample.csv")[["review_id","review_pros","review_cons","split"]]
df = ann.merge(smp, on="review_id", how="inner")
df["text"] = (df["review_pros"].fillna("") + " [SEP] " + df["review_cons"].fillna("")).str.strip()

# ── Labels ──
df["pay_mention_label"] = (df["pay_mention"] == "YES").astype(int)
df["benefits_mention_label"] = (df["benefits_mention"] == "YES").astype(int)
df["pay_neg_label"] = ((df["pay_mention"] == "YES") & (df["pay_sent"] == "NEGATIVE")).astype(int)
df["benefits_neg_label"] = ((df["benefits_mention"] == "YES") & (df["benefits_sent"] == "NEGATIVE")).astype(int)

for task, col in [("pay_mention","pay_mention_label"),("pay_neg","pay_neg_label"),
                   ("benefits_mention","benefits_mention_label"),("benefits_neg","benefits_neg_label")]:
    print(f"  {task}: {dict(df[col].value_counts())}")

class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(str(self.texts[i]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}

def train(task_name, label_col, output_dir):
    """Train a binary DistilBERT classifier."""
    out_path = MODEL_DIR / output_dir
    if out_path.exists():
        print(f"  {task_name}: model exists, skip")
        return
    print(f"\n=== Training {task_name} ===")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_df = df[df["split"] == "train"]; val_df = df[df["split"] == "val"]
    train_ds = TextDS(train_df["text"].values, train_df[label_col].values, tokenizer)
    val_ds = TextDS(val_df["text"].values, val_df[label_col].values, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            out.loss.backward(); optimizer.step(); scheduler.step(); total_loss += out.loss.item()
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for batch in val_dl:
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                preds.extend(out.logits.argmax(-1).tolist()); trues.extend(batch["labels"].tolist())
        acc, f1 = accuracy_score(trues, preds), f1_score(trues, preds, average="binary")
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f} val_acc={acc:.3f} val_f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1; model.save_pretrained(out_path); tokenizer.save_pretrained(out_path)
    print(f"  Best F1={best_f1:.3f}, saved to {out_path}")

# Train 4 models
for task, col, out in [("pay_mention","pay_mention_label","pay_mention"),
                        ("pay_neg","pay_neg_label","pay_neg"),
                        ("benefits_mention","benefits_mention_label","benefits_mention"),
                        ("benefits_neg","benefits_neg_label","benefits_neg")]:
    train(task, col, out)
print("\nAll models trained.")
