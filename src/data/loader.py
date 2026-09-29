"""Data loader for search query intent classification.

Expects a CSV with at minimum: query, intent.
Optional session columns: session_id, position_in_session, click_depth,
dwell_time_sec, is_reformulation.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

INTENT_LABELS = ["navigational", "informational", "transactional"]

SESSION_COLS = [
    "position_in_session",
    "click_depth",
    "dwell_time_sec",
    "is_reformulation",
]


def load_queries(path: str) -> pd.DataFrame:
    """Load a query intent dataset from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    assert "query" in df.columns, "CSV must have a 'query' column"
    assert "intent" in df.columns, "CSV must have an 'intent' column"

    df["intent"] = df["intent"].str.lower().str.strip()
    valid_mask = df["intent"].isin(INTENT_LABELS)
    if not valid_mask.all():
        dropped = (~valid_mask).sum()
        logger.warning("Dropping %d rows with unknown intent labels", dropped)
        df = df[valid_mask].reset_index(drop=True)

    logger.info("Loaded %d queries (%s)", len(df), df["intent"].value_counts().to_dict())
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode intent labels as integers."""
    le = LabelEncoder()
    le.fit(INTENT_LABELS)
    df = df.copy()
    df["label"] = le.transform(df["intent"])
    return df, le


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df["label"]
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=random_seed, stratify=train_val["label"]
    )
    logger.info("Split — train: %d, val: %d, test: %d", len(train), len(val), len(test))
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
