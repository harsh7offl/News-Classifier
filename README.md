# 📰 News Classifier — Fine-tuned DistilBERT

> End-to-end NLP pipeline: Fine-tuning → Evaluation → REST API → Docker

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-94.41%25-brightgreen)

---

## 🎯 Problem Statement

News platforms generate thousands of articles daily. This project automatically
classifies news into **World / Sports / Business / Sci-Tech** using a fine-tuned
DistilBERT model, deployed as a production REST API.

---

## 🏗️ Architecture

```
Raw Text → Preprocessing → DistilBERT Tokenizer → Fine-tuned Classifier → FastAPI
```

---

## 📊 Results

| Model | Accuracy | F1 Macro |
|---|---|---|
| TF-IDF + LogReg (baseline) | 0.9224 | 0.9222 |
| **DistilBERT (fine-tuned)** | **0.9441** | **0.9441** |
| Improvement | +0.0217 | +0.0219 |

### Per-Class Performance
| Class | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.96 | 0.95 | 0.96 |
| Sports | 0.99 | 0.99 | 0.99 |
| Business | 0.91 | 0.91 | 0.91 |
| Sci/Tech | 0.92 | 0.93 | 0.92 |

> Trained on only 2,000 samples (1.7% of dataset) — still beats the TF-IDF baseline trained on 120k samples

---

## 📁 Project Structure

```
news-classifier/
├── data/
│   └── dataset.py          # AG News loading & preprocessing
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── src/
│   ├── train.py            # Fine-tuning with HuggingFace Trainer + WandB
│   ├── evaluate.py         # Metrics, confusion matrix, baseline comparison
│   ├── predict.py          # Inference (single & batch)
│   └── baseline.py         # Standalone TF-IDF + LogReg baseline
├── api/
│   └── app.py              # FastAPI REST endpoints
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/harsh7offl/News-Classifier.git
cd News-Classifier
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train
```bash
python src/train.py --epochs 3 --batch_size 16 --max_samples 5000
```

### Evaluate
```bash
python src/evaluate.py --model_dir ./outputs/model
```

### Predict
```bash
python src/predict.py --text "Tesla stock drops after earnings miss"
```

### Run API
```bash
uvicorn api.app:app --reload --port 8000
# Swagger UI → http://127.0.0.1:8000/docs
```

### Docker
```bash
docker build -t news-classifier .
docker run -p 8000:8000 news-classifier
```

---

## 🔬 Key Design Decisions

| Decision | Rationale |
|---|---|
| DistilBERT over BERT | 40% smaller, 60% faster, 97% of BERT performance |
| max_length=128 | AG News headlines are short — covers 95%+ of samples |
| Macro F1 as metric | Balanced classes — penalises per-class failures equally |
| Early stopping patience=2 | Prevents overfitting automatically |
| TF-IDF baseline | Validates that transformer adds real value |

---

## 📈 Experiment Tracking

Training curves tracked with Weights & Biases:
👉 [View WandB Dashboard](https://wandb.ai/harshavarthanpk-psg-college-of-technology/news-classifier)

---

## 👤 Author

**Harshavarthan** — Data Scientist @ Strand Life Sciences  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/harsh7offl)
