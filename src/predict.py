"""
Single-text and batch inference with the fine-tuned model.

Usage:
    python src/predict.py --text "Apple launches new MacBook with M4 chip"
    python src/predict.py --file inputs.txt
"""

import os
import argparse
import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

MODEL_DIR = os.environ.get("MODEL_DIR", "./outputs/model")


def load_model(model_dir: str = MODEL_DIR):
    """Load tokenizer and model from disk."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(texts: list[str], tokenizer, model, device: str = "cpu"):
    """
    Predict class labels and confidence scores.

    Args:
        texts: list of input strings
        tokenizer, model: loaded HuggingFace objects
        device: 'cpu' or 'cuda'

    Returns:
        list of dicts with keys: text, label, confidence, all_scores
    """
    model.to(device)
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs = F.softmax(logits, dim=-1).cpu().numpy()
    pred_ids = probs.argmax(axis=-1)

    id2label = model.config.id2label
    results = []
    for i, text in enumerate(texts):
        pred_label = id2label[pred_ids[i]]
        results.append({
            "text": text,
            "label": pred_label,
            "confidence": float(probs[i][pred_ids[i]]),
            "all_scores": {id2label[j]: float(probs[i][j])
                           for j in range(len(id2label))},
        })
    return results


def pretty_print(result: dict):
    print(f"\n📰 Text      : {result['text'][:80]}...")
    print(f"🏷️  Predicted : {result['label']}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print("📊 All scores:")
    for label, score in sorted(result["all_scores"].items(),
                                key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"   {label:<12} {score:.2%}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on news text")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to classify")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to .txt file with one text per line")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    print(f"⚙️  Loading model from {args.model_dir}...")
    tokenizer, model = load_model(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Demo examples
        texts = [
            "Apple launches new MacBook Pro with M4 chip at WWDC",
            "Manchester City defeats Real Madrid 3-1 in Champions League final",
            "Federal Reserve raises interest rates by 25 basis points",
            "Scientists discover new exoplanet that may support liquid water",
        ]

    results = predict(texts, tokenizer, model, device=device)
    for r in results:
        pretty_print(r)
