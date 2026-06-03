"""
Fine-tuning DistilBERT on AG News classification.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("✅ Python started")
print(f"   Python version: {sys.version}")
print(f"   Working dir: {os.getcwd()}")

try:
    print("📦 Importing libraries...")
    import argparse
    import wandb
    import torch
    import numpy as np
    print(f"   torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    print("   transformers ok")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    print("   sklearn ok")

    from data.dataset import load_agnews, AGNewsDataset, preprocess_text, get_label_map
    print("   data.dataset ok")

except Exception as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./outputs/model"
LOGS_DIR = "./outputs/logs"
NUM_LABELS = 4


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    try:
        print("\n🔧 Initialising WandB...")
        wandb.init(
            project="news-classifier",
            name=f"distilbert-agnews-ep{args.epochs}",
            config=vars(args),
        )
        print("   wandb ok")
    except Exception as e:
        print(f"⚠️  WandB init failed ({e}) — continuing without it")
        os.environ["WANDB_MODE"] = "disabled"

    try:
        print("\n📦 Loading dataset...")
        texts, labels = load_agnews(split="train", max_samples=args.max_samples)
        texts = [preprocess_text(t) for t in texts]

        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )
        print(f"   Train: {len(X_train)} | Val: {len(X_val)}")

        print("\n🔤 Tokenizing...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=128)
        val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=128)
        train_dataset = AGNewsDataset(train_enc, y_train)
        val_dataset   = AGNewsDataset(val_enc,   y_val)
        print("   Tokenizing done")

        print("\n🤖 Loading model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=get_label_map(),
            label2id={v: k for k, v in get_label_map().items()},
        )
        print("   Model loaded")

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_dir=LOGS_DIR,
            logging_steps=10,
            report_to="wandb",
            fp16=torch.cuda.is_available(),
            seed=42,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("\n🚀 Training started...")
        trainer.train()

        print("\n💾 Saving model and tokenizer...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Training complete! Model saved to: {OUTPUT_DIR}")
        wandb.finish()

    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on AG News")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    print(f"\n⚙️  Args: epochs={args.epochs}, batch_size={args.batch_size}, max_samples={args.max_samples}")
    train(args)