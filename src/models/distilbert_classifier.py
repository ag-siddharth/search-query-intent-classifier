"""DistilBERT classifier with session feature fusion for intent classification.

Takes query text through DistilBERT ([CLS] pooling) and concatenates it with
hand-engineered session features before the classification head.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QueryIntentDataset(Dataset):
    """PyTorch dataset combining tokenized queries with session features."""

    def __init__(self, queries: list[str], session_feats: np.ndarray,
                 labels: np.ndarray, tokenizer: DistilBertTokenizer, max_length: int = 64):
        self.encodings = tokenizer(
            queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        self.session_feats = torch.tensor(session_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "session_feats": self.session_feats[idx],
            "label": self.labels[idx],
        }


class IntentClassifier(nn.Module):
    """DistilBERT + session features → intent classification."""

    def __init__(
        self,
        pretrained: str = "distilbert-base-uncased",
        session_feat_dim: int = 8,
        session_proj_dim: int = 32,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained)
        bert_dim = self.bert.config.dim  # 768

        self.session_proj = nn.Sequential(
            nn.Linear(session_feat_dim, session_proj_dim),
            nn.ReLU(),
        )

        fused_dim = bert_dim + session_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, session_feats):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        session_proj = self.session_proj(session_feats)
        fused = torch.cat([cls_embed, session_proj], dim=1)
        return self.classifier(fused)


class IntentClassifierTrainer:
    """Training wrapper for the DistilBERT intent classifier."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.get("pretrained", "distilbert-base-uncased"))
        self.model: IntentClassifier | None = None

    def _build_loader(self, queries, session_feats, labels, shuffle=True):
        dataset = QueryIntentDataset(
            queries, session_feats, labels, self.tokenizer, self.config.get("max_length", 64)
        )
        return DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=shuffle,
            num_workers=0,
        )

    def fit(
        self,
        train_queries: list[str], train_feats: np.ndarray, y_train: np.ndarray,
        val_queries: list[str] | None = None,
        val_feats: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "IntentClassifierTrainer":
        self.model = IntentClassifier(
            pretrained=self.config.get("pretrained", "distilbert-base-uncased"),
            session_feat_dim=self.config.get("session_feature_dim", 8),
            session_proj_dim=self.config.get("session_projection_dim", 32),
            hidden_dim=self.config.get("hidden_dim", 256),
            num_classes=3,
            dropout=self.config.get("dropout", 0.3),
        ).to(self.device)

        train_loader = self._build_loader(train_queries, train_feats, y_train, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        total_steps = len(train_loader) * self.config.get("epochs", 5)
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        criterion = nn.CrossEntropyLoss()
        epochs = self.config.get("epochs", 5)
        patience = self.config.get("patience", 2)
        best_val_loss = float("inf")
        wait = 0

        logger.info("Training DistilBERT classifier (%d epochs, device: %s)", epochs, self.device)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = self.model(batch["input_ids"], batch["attention_mask"], batch["session_feats"])
                loss = criterion(logits, batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(batch["label"])

            avg_loss = total_loss / len(y_train)
            msg = f"Epoch {epoch}/{epochs} — train loss: {avg_loss:.4f}"

            if val_queries is not None:
                val_loss = self._eval_loss(val_queries, val_feats, y_val, criterion)
                msg += f" | val loss: {val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

            logger.info(msg)

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        return self

    def _eval_loss(self, queries, feats, labels, criterion):
        self.model.eval()
        loader = self._build_loader(queries, feats, labels, shuffle=False)
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(batch["input_ids"], batch["attention_mask"], batch["session_feats"])
                total += criterion(logits, batch["label"]).item() * len(batch["label"])
        return total / len(labels)

    def predict(self, queries: list[str], session_feats: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        self.model.eval()
        dummy_labels = np.zeros(len(queries), dtype=np.int64)
        loader = self._build_loader(queries, session_feats, dummy_labels, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(batch["input_ids"], batch["attention_mask"], batch["session_feats"])
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, queries: list[str], session_feats: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        self.model.eval()
        dummy_labels = np.zeros(len(queries), dtype=np.int64)
        loader = self._build_loader(queries, session_feats, dummy_labels, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(batch["input_ids"], batch["attention_mask"], batch["session_feats"])
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs)
