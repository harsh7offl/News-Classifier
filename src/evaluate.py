"""
Evaluate fine-tuned DistilBERT on AG News test set.

Usage:
    python src/evaluate.py --model_dir ./outputs/model

Outputs:
    - Classification report
    - Confusion matrix (saved to outputs/confusion_matrix.png)
    - Baseline comparison (TF-IDF + LogReg)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_agnews, AGNewsDataset, preprocess_text, get_label_map


# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_transformer(model, tokenizer, texts, batch_size=32, device="cpu"):
    """Run inference with the fine-tuned transformer model."""
    model.eval()
    model.to(device)
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)

    return np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """Plot and save a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Normalised) — DistilBERT")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"📊 Confusion matrix saved to: {save_path}")


# ── Baseline ──────────────────────────────────────────────────────────────────

def run_baseline(X_train, y_train, X_test, y_test):
    """TF-IDF + Logistic Regression baseline for comparison."""
    print("\n⚖️  Running TF-IDF + LogReg baseline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}")
    return acc, f1


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    label_map = get_label_map()
    label_names = [label_map[i] for i in range(len(label_map))]

    print("📦 Loading test data...")
    test_texts, test_labels = load_agnews(split="test")
    test_texts = [preprocess_text(t) for t in test_texts]

    # ── Transformer evaluation ─────────────────────────────────────────────
    print(f"\n🤖 Loading model from: {args.model_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    print("🔍 Running inference on test set...")
    preds = predict_transformer(model, tokenizer, test_texts,
                                batch_size=64, device=device)

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")

    print("\n── DistilBERT Results ──────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=label_names))

    plot_confusion_matrix(
        test_labels, preds, label_names,
        save_path="./outputs/confusion_matrix.png"
    )

    # ── Baseline for comparison ────────────────────────────────────────────
    train_texts, train_labels = load_agnews(split="train")
    train_texts = [preprocess_text(t) for t in train_texts]
    base_acc, base_f1 = run_baseline(train_texts, train_labels,
                                     test_texts, test_labels)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n── Model Comparison ─────────────────────────────")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1 Macro':>10}")
    print("-" * 52)
    print(f"{'TF-IDF + LogReg (baseline)':<30} {base_acc:>10.4f} {base_f1:>10.4f}")
    print(f"{'DistilBERT (fine-tuned)':<30} {acc:>10.4f} {f1:>10.4f}")
    print(f"{'Improvement':<30} {acc - base_acc:>+10.4f} {f1 - base_f1:>+10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./outputs/model")
    args = parser.parse_args()
    evaluate(args)
