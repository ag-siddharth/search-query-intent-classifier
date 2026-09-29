"""Evaluation metrics for intent classification."""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    report: str
    confusion: np.ndarray


def evaluate_classifier(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> ClassificationResult:
    """Compute all classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(
        "%s — Accuracy: %.3f, Macro F1: %.3f, Precision: %.3f, Recall: %.3f",
        name, acc, f1, prec, rec,
    )
    logger.info("\n%s", report)

    return ClassificationResult(
        name=name,
        accuracy=acc,
        macro_precision=prec,
        macro_recall=rec,
        macro_f1=f1,
        report=report,
        confusion=cm,
    )


def comparison_report(results: list[ClassificationResult]) -> str:
    """Generate a side-by-side comparison table."""
    lines = [
        "=" * 72,
        "MODEL COMPARISON REPORT",
        "=" * 72,
        f"{'Model':<35} {'Accuracy':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}",
        "-" * 72,
    ]
    for r in sorted(results, key=lambda x: x.macro_f1, reverse=True):
        lines.append(
            f"{r.name:<35} {r.accuracy:>8.3f} {r.macro_f1:>8.3f} "
            f"{r.macro_precision:>8.3f} {r.macro_recall:>8.3f}"
        )
    lines.append("=" * 72)

    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report
