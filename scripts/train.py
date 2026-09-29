"""Train the search query intent classifier.

Usage:
    python -m scripts.train --config config/config.yaml
    python -m scripts.train --config config/config.yaml --data-path data/my_queries.csv
"""

import argparse
import logging
import pickle
from pathlib import Path

import yaml

from src.data.loader import encode_labels, load_queries, split_data
from src.features.session_features import SessionFeatureEngineer
from src.models.distilbert_classifier import IntentClassifierTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data-path", type=str, default=None, help="Override data path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_path = args.data_path or cfg["data"]["path"]

    # ---- Load and split ----
    df = load_queries(data_path)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(
        df,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_seed=cfg["data"]["random_seed"],
    )

    # ---- Feature engineering ----
    fe = SessionFeatureEngineer()
    train_feats = fe.fit_transform(train_df)
    val_feats = fe.transform(val_df)

    # ---- Train DistilBERT + session features ----
    trainer = IntentClassifierTrainer(cfg["model"])
    trainer.fit(
        train_queries=train_df["query"].tolist(),
        train_feats=train_feats,
        y_train=train_df["label"].values,
        val_queries=val_df["query"].tolist(),
        val_feats=val_feats,
        y_val=val_df["label"].values,
    )

    # ---- Save artifacts ----
    model_dir = Path(cfg["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "trainer.pkl", "wb") as f:
        pickle.dump(trainer, f)
    with open(model_dir / "feature_engineer.pkl", "wb") as f:
        pickle.dump(fe, f)
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    logger.info("All artifacts saved to %s", model_dir)


if __name__ == "__main__":
    main()
