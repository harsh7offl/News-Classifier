"""
FastAPI REST endpoint for the news classifier.

Run locally:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    POST /predict        — classify a single text
    POST /predict/batch  — classify a list of texts
    GET  /health         — health check
    GET  /labels         — list all class labels
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import load_model, predict

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="News Classifier API",
    description="Classify news headlines into World / Sports / Business / Sci-Tech",
    version="1.0.0",
)

MODEL_DIR = os.environ.get("MODEL_DIR", "./outputs/model")
tokenizer, model = None, None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
def load():
    global tokenizer, model
    tokenizer, model = load_model(MODEL_DIR)
    print(f"✅ Model loaded from {MODEL_DIR} on {device}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., example="Apple announces new M4 MacBook Pro")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50)


class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    all_scores: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/labels")
def labels():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"labels": list(model.config.id2label.values())}


@app.post("/predict", response_model=PredictionResult)
def predict_single(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    results = predict([req.text], tokenizer, model, device)
    return results[0]


@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(req: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = predict(req.texts, tokenizer, model, device)
    return results
