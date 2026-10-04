"""Evaluation metrics shared by all experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from src.datasets.schema import BIN_HALLUCINATED, LABELS


def multiclass_metrics(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = LABELS) -> dict:
    labels = [l for l in labels if l in set(y_true) | set(y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {l: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def binary_metrics(y_true: list[str], y_pred: list[str], positive: str = BIN_HALLUCINATED) -> dict:
    """Hallucination is the positive class -> recall == hallucination detection rate."""
    t = np.array([y == positive for y in y_true])
    p = np.array([y == positive for y in y_pred])
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y_true),
        "accuracy": float((t == p).mean()),
        "precision": prec,
        "recall": rec,
        "hallucination_detection_rate": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }


def best_threshold(scores: np.ndarray, is_positive: np.ndarray, higher_is_positive: bool = False) -> float:
    """Threshold maximising binary F1 (tune on validation only)."""
    best_t, best_f = 0.5, -1.0
    for t in np.unique(scores):
        pred = scores >= t if higher_is_positive else scores < t
        tp = (pred & is_positive).sum()
        fp = (pred & ~is_positive).sum()
        fn = (~pred & is_positive).sum()
        f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t
