#!/usr/bin/env python
"""CPU-friendly classifier: TF-IDF + LogisticRegression for WLB mention/sentiment.
Predicts on full 490k sample. No GPU needed."""

import pandas as pd, numpy as np, pickle, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

OUT = Path("/data/disk4/workspace/projects/union_glassdoor/outputs/20260618/text_analysis")
MODEL_DIR = OUT / "sklearn_models"; MODEL_DIR.mkdir(exist_ok=True)

# Load annotation + text
print("Loading data...")
df = pd.read_csv(OUT / "annotation_results.csv")
text = pd.read_parquet(OUT / "text_analysis_sample.parquet")[["review_id","review_pros","review_cons"]]
df = df.merge(text, on="review_id", how="left")
df["text"] = df["review_pros"].fillna("") + " " + df["review_cons"].fillna("")

# Labels
df["wlb_mention"] = ((df["wlb_pros_mention"]=="YES") | (df["wlb_cons_mention"]=="YES")).astype(int)
df["wlb_net_sentiment"] = np.where(
    (df["wlb_pros_mention"]=="YES") & (df["wlb_pros_sentiment"]=="POSITIVE"), 1,
    np.where((df["wlb_cons_mention"]=="YES") & (df["wlb_cons_sentiment"]=="NEGATIVE"), -1, 0))
df["sentiment_class"] = df["wlb_net_sentiment"].map({-1:0, 0:1, 1:2})

# Load split
split_df = pd.read_csv(OUT / "annotation_sample.csv")[["review_id","split"]]
df = df.merge(split_df, on="review_id", how="left"); df["split"] = df["split"].fillna("train")
print(f"Train: {(df['split']=='train').sum()}, Val: {(df['split']=='val').sum()}, Test: {(df['split']=='test').sum()}")

# ── Train mention classifier ──
print("\n=== WLB Mention Classifier ===")
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
X_train = vec.fit_transform(df[df["split"]=="train"]["text"])
X_val = vec.transform(df[df["split"]=="val"]["text"])
X_test = vec.transform(df[df["split"]=="test"]["text"])
y_train = df[df["split"]=="train"]["wlb_mention"]
y_val = df[df["split"]=="val"]["wlb_mention"]
y_test = df[df["split"]=="test"]["wlb_mention"]

clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
clf.fit(X_train, y_train)
val_pred = clf.predict(X_val); test_pred = clf.predict(X_test)
print(f"Val  acc={accuracy_score(y_val,val_pred):.3f} f1={f1_score(y_val,val_pred):.3f}")
print(f"Test acc={accuracy_score(y_test,test_pred):.3f} f1={f1_score(y_test,test_pred):.3f}")
print(classification_report(y_test, test_pred, target_names=["No mention","Mention"]))

# Save model
with open(MODEL_DIR / "wlb_mention_vectorizer.pkl","wb") as f: pickle.dump(vec, f)
with open(MODEL_DIR / "wlb_mention_classifier.pkl","wb") as f: pickle.dump(clf, f)

# ── Predict on full sample ──
print("\n=== Predicting on full 490k sample ===")
full = pd.read_parquet(OUT / "text_analysis_sample.parquet")
full_text = full["review_pros"].fillna("") + " " + full["review_cons"].fillna("")
X_full = vec.transform(full_text)

full["wlb_mention_pred"] = clf.predict(X_full)
full["wlb_mention_prob"] = clf.predict_proba(X_full)[:,1]

mention_rate = full["wlb_mention_pred"].mean()
print(f"Predicted WLB mention rate: {mention_rate*100:.1f}% ({full['wlb_mention_pred'].sum():,} reviews)")

# ── Predict net sentiment on WLB-mentioned reviews only ──
# For simplicity, use a simple keyword-based approach since sentiment labels are sparse
# The 2986 annotations have only ~140 negative + ~120 positive WLB mentions
# TF-IDF + LogisticRegression won't learn well from this
# Instead, use a keyword sentiment approach

print("\n=== Sentiment prediction (keyword-based) ===")
# Build keyword sentiment from labeled data
wlb_pros_words = df[df["wlb_pros_sentiment"]=="POSITIVE"]["review_pros"].dropna()
wlb_cons_words = df[df["wlb_cons_sentiment"]=="NEGATIVE"]["review_cons"].dropna()

# Simple approach: use TF-IDF + LogisticRegression for sentiment too (on mentioned texts only)
sent_vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words="english")
mentioned = df[df["wlb_mention"]==1]
X_sent_train = sent_vec.fit_transform(mentioned[mentioned["split"]=="train"]["text"])
X_sent_test = sent_vec.transform(mentioned[mentioned["split"]=="test"]["text"])
y_sent_train = mentioned[mentioned["split"]=="train"]["sentiment_class"]
y_sent_test = mentioned[mentioned["split"]=="test"]["sentiment_class"]

sent_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
sent_clf.fit(X_sent_train, y_sent_train)
sent_pred = sent_clf.predict(X_sent_test)
print(f"Sentiment test acc={accuracy_score(y_sent_test,sent_pred):.3f} f1={f1_score(y_sent_test,sent_pred,average='macro'):.3f}")

# Predict sentiment on full sample for WLB-mentioned reviews
wlb_mask = full["wlb_mention_pred"] == 1
if wlb_mask.sum() > 0:
    X_sent_full = sent_vec.transform(full_text[wlb_mask])
    sent_preds = sent_clf.predict(X_sent_full)  # 0=neg, 1=neutral, 2=pos
    full["wlb_sentiment_pred"] = 1  # neutral default
    full.loc[wlb_mask, "wlb_sentiment_pred"] = np.where(sent_preds==2, 1, np.where(sent_preds==0, -1, 0))
else:
    full["wlb_sentiment_pred"] = 0

# Save
full.to_parquet(OUT / "full_sample_with_text_predictions.parquet", index=False)
print(f"Saved full_sample_with_text_predictions.parquet ({len(full):,} rows)")

# Distribution
print("\n=== Prediction Distribution ===")
print(f"WLB mention predicted: {full['wlb_mention_pred'].mean()*100:.1f}%")
sent_dist = full.groupby("wlb_sentiment_pred").size()
print(f"Sentiment: {sent_dist.to_dict()}")
print("Done.")
