#!/usr/bin/env python
"""Train BERT classifiers for wc/vf/mc (mention + complaint), then predict on full sample."""

import pandas as pd, numpy as np, torch, os, time
torch.set_num_threads(4); os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

TA = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
TA2 = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260619/text_analysis")
MODEL_DIR = TA / "bert_models"; MODEL_DIR.mkdir(exist_ok=True)
BATCH, EPOCHS, LR = 16, 3, 2e-5
MODEL_NAME = "distilbert-base-uncased"
LOAD_KWARGS = {"local_files_only": True}
SHARD_SIZE = 50000

# ── Load data ──
print("Loading...")
ann = pd.read_csv(TA2 / "annotation_final_dims.csv")
smp = pd.read_csv(TA / "annotation_sample.csv")[["review_id","review_pros","review_cons","split"]]
df = ann.merge(smp, on="review_id", how="inner")
df["text"] = (df["review_pros"].fillna("") + " [SEP] " + df["review_cons"].fillna("")).str.strip()

# Full sample for inference
full = pd.read_parquet(TA / "text_analysis_sample.parquet")
full_texts = (full["review_pros"].fillna("") + " [SEP] " + full["review_cons"].fillna("")).tolist()

# Dimensions to process
DIMS = {
    "wc": "working conditions",
    "vf": "voice and fairness",
    "mc": "management communication"
}

# ── Build labels ──
for d in DIMS:
    df[f"{d}_mention_label"] = df[f"{d}_mention"].astype(int).clip(0, 1)
    df[f"{d}_complaint_label"] = ((df[f"{d}_mention"].astype(int) == 1) & (df[f"{d}_negative"].astype(int) == 1)).astype(int).clip(0, 1)
    print(f"  {d}: mention={df[f'{d}_mention_label'].sum()}, complaint={df[f'{d}_complaint_label'].sum()}, unique={df[f'{d}_complaint_label'].unique()}")

class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(str(self.texts[i]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}

def train(task_name, label_col, output_dir):
    out_path = MODEL_DIR / output_dir
    if out_path.exists():
        print(f"  {task_name}: exists, skip"); return
    print(f"\n=== {task_name} ===")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, local_files_only=True)
    train_df = df[df["split"]=="train"]; val_df = df[df["split"]=="val"]
    train_ds = TextDS(train_df["text"].values, train_df[label_col].values, tokenizer)
    val_ds = TextDS(val_df["text"].values, val_df[label_col].values, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            out.loss.backward(); optimizer.step(); scheduler.step()
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for batch in val_dl:
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                preds.extend(out.logits.argmax(-1).tolist()); trues.extend(batch["labels"].tolist())
        f1 = f1_score(trues, preds, average="binary"); print(f"  Epoch {epoch+1}: acc={accuracy_score(trues,preds):.3f} f1={f1:.3f}")
        if f1 > best_f1: best_f1 = f1; model.save_pretrained(out_path); tokenizer.save_pretrained(out_path)
    print(f"  Best F1={best_f1:.3f}")

# ── Train all models ──
for d in DIMS:
    train(f"{d}_mention", f"{d}_mention_label", f"{d}_mention")
    train(f"{d}_complaint", f"{d}_complaint_label", f"{d}_complaint")

# ── Predict on full sample ──
print("\n=== Full-sample inference ===")
for d in DIMS:
    for task in ["mention","complaint"]:
        col = f"{d}_{task}_prob"
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR / f"{d}_{task}", local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / f"{d}_{task}", local_files_only=True); model.eval()
        probs = []
        for i in range(0, len(full_texts), BATCH*2):
            batch_t = full_texts[i:i+BATCH*2]
            enc = tokenizer(batch_t, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs.extend(torch.softmax(out.logits, -1)[:, 1].tolist())
            if (i // (BATCH*2)) % 50 == 0:
                print(f"  {d}_{task}: {i+BATCH*2}/{len(full_texts)}", flush=True)
        full[col] = probs
    full[f"{d}_complaint_score"] = full[f"{d}_mention_prob"] * full[f"{d}_complaint_prob"]
    full[f"{d}_complaint_std"] = (full[f"{d}_complaint_score"] - full[f"{d}_complaint_score"].mean()) / full[f"{d}_complaint_score"].std()
    print(f"  {d}: complaint mean={full[f'{d}_complaint_score'].mean():.4f}")

full.to_parquet(TA / "full_sample_with_text_predictions.parquet", index=False)
print(f"Saved {len(full):,} rows with wc/vf/mc predictions")
