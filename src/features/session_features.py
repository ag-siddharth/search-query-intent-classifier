"""Session-level feature engineering for query intent classification.

Extracts both text-level features (query length, question words, URL/action
patterns) and behavioral features from search session logs (click depth,
dwell time, reformulation, position in session).
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "is", "are", "can", "do", "does"}
ACTION_WORDS = {"buy", "order", "purchase", "download", "install", "subscribe", "sign", "register",
                "book", "reserve", "hire", "rent", "apply", "schedule", "donate", "get"}
URL_PATTERN = re.compile(r"\.(com|org|net|edu|gov|io|tv|co)\b|^www\.", re.IGNORECASE)


def extract_query_features(query: str) -> dict:
    """Extract text-based features from a single query string."""
    tokens = query.lower().split()
    return {
        "query_length": len(tokens),
        "has_question_word": int(bool(tokens and tokens[0] in QUESTION_WORDS)),
        "has_url_pattern": int(bool(URL_PATTERN.search(query))),
        "has_action_word": int(any(t in ACTION_WORDS for t in tokens)),
    }


class SessionFeatureEngineer:
    """Engineer session-level features from query logs.

    Produces an 8-dimensional feature vector per query combining
    text-derived signals with behavioral session data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract all features into a numpy array."""
        # Query-level text features
        text_feats = df["query"].apply(extract_query_features).apply(pd.Series)

        # Session-level features (use defaults if columns are missing)
        session_feats = pd.DataFrame(index=df.index)
        session_feats["click_depth"] = df.get("click_depth", pd.Series(1.0, index=df.index))
        session_feats["dwell_time_sec"] = df.get("dwell_time_sec", pd.Series(60.0, index=df.index))
        session_feats["is_reformulation"] = df.get("is_reformulation", pd.Series(0, index=df.index))
        session_feats["session_position"] = df.get("position_in_session", pd.Series(1, index=df.index))

        # log-transform dwell time — distribution is super right-skewed
        session_feats["dwell_time_sec"] = np.log1p(session_feats["dwell_time_sec"])

        combined = pd.concat([text_feats, session_feats], axis=1)
        return combined.values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training data and return scaled features."""
        raw = self._build_feature_matrix(df)
        scaled = self.scaler.fit_transform(raw)
        self._is_fitted = True
        return scaled.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform using previously fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        raw = self._build_feature_matrix(df)
        return self.scaler.transform(raw).astype(np.float32)
