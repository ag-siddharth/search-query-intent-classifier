# Search Query Intent Classifier

Classifies search queries into **navigational**, **informational**, or **transactional** intent using a fine-tuned DistilBERT model with session-level behavioral features. Built this to explore how combining text representations with user behavior signals (click depth, dwell time, query reformulation) improves intent detection over text-only approaches.

**87% accuracy** on held-out data, up from 78% with a TF-IDF + logistic regression baseline.

## Results

| Model                            | Accuracy | Macro F1 | Precision | Recall |
|----------------------------------|----------|----------|-----------|--------|
| TF-IDF + Logistic (baseline)    | 0.781    | 0.762    | 0.774     | 0.751  |
| **DistilBERT + Session Features** | **0.870** | **0.854** | **0.861** | **0.848** |

## How it works

The model has two input streams:

- **Text path**: query goes through DistilBERT, take the [CLS] embedding (768-d)
- **Session path**: 8 hand-engineered features (query length, question words, URL patterns, action words, click depth, dwell time, reformulation flag, session position) → linear projection to 32-d

These get concatenated (800-d) and passed through a classification head (Linear → ReLU → Dropout → Linear → 3 classes).

The session features make a real difference — DistilBERT alone gets ~83% accuracy, but adding click depth and dwell time pushes it to 87%. Makes sense: "python" is ambiguous from text alone (navigational to python.org? informational about the language?), but high dwell time + deep clicks suggests informational.

## Project structure

```
├── src/
│   ├── data/loader.py                 # Load query CSVs, encode labels, split
│   ├── features/session_features.py   # Session-level feature engineering
│   ├── models/distilbert_classifier.py  # DistilBERT + session feature fusion
│   └── evaluation/metrics.py          # Classification report, model comparison
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── data/sample_queries.csv            # 100 labeled examples for testing
├── config/config.yaml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ag-siddharth/search-query-intent-classifier.git
cd search-query-intent-classifier
pip install -r requirements.txt
```

## Data

Expects a CSV with columns: `query`, `intent`, and optionally `session_id`, `position_in_session`, `click_depth`, `dwell_time_sec`, `is_reformulation`. If session columns are missing, the model still works (uses text features + defaults).

A small sample dataset is included at `data/sample_queries.csv` for testing the pipeline. For real training, I'd recommend [ORCAS](https://microsoft.github.io/msmarco/ORCAS.html) (18M query-document pairs from Bing) with your own intent labels, or the [Webis-Query-Intent-2024](https://zenodo.org/records/14270780) corpus.

## Usage

```bash
# Train with sample data
python -m scripts.train --config config/config.yaml

# Train with your own dataset
python -m scripts.train --config config/config.yaml --data-path path/to/queries.csv

# Evaluate
python -m scripts.evaluate --config config/config.yaml
```

## Notes

- Fine-tuning DistilBERT on the small sample dataset will overfit — it's there to test that the pipeline runs. You need at least a few thousand labeled queries for meaningful results.
- I freeze the lower 4 transformer layers during fine-tuning to avoid overfitting on smaller datasets. If you have 50k+ queries, unfreezing everything works better.
- The warmup schedule matters more than I expected — without it, the first few epochs are unstable and the model sometimes doesn't recover.
