"""Evaluate the trained intent classifier on the test set.

Usage:
    python -m scripts.evaluate --config config/config.yaml
"""

import argparse
import logging
import pickle
from pathlib import Path

import yaml

from src.data.loader import encode_labels, load_queries, split_data, INTENT_LABELS
from src.evaluation.metrics import evaluate_classifier, comparison_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate intent classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = args.data_path or cfg["data"]["path"]
    model_dir = Path(cfg["output"]["model_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data and reproduce same split ----
    df = load_queries(data_path)
    df, _ = encode_labels(df)
    _, _, test_df = split_data(
        df,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_seed=cfg["data"]["random_seed"],
    )

    # ---- Load saved artifacts ----
    with open(model_dir / "trainer.pkl", "rb") as f:
        trainer = pickle.load(f)
    with open(model_dir / "feature_engineer.pkl", "rb") as f:
        fe = pickle.load(f)
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # ---- Evaluate ----
    test_feats = fe.transform(test_df)
    y_true = test_df["label"].values
    y_pred = trainer.predict(test_df["query"].tolist(), test_feats)

    result = evaluate_classifier(
        "DistilBERT + Session Features", y_true, y_pred, label_names=INTENT_LABELS
    )

    # ---- Save report ----
    report = comparison_report([result])
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        f.write("\n\nDetailed Report:\n")
        f.write(result.report)

    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
