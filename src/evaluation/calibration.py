"""Probability calibration for the v2 detector.

Temperature scaling is fitted on val only, with two-fold cross-fitting so the
reported ECE/Brier are out-of-fold. The base classifier is fitted on train with
the v1-selected regularisation strength; no test data is read here.

The functions are deliberately small and framework-free beyond numpy/scipy so the
metrics can be unit-tested without loading models.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    chosen = probabilities[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    one_hot = np.zeros_like(probabilities, dtype=np.float64)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def calibration_bins(y_true: np.ndarray, probabilities: np.ndarray,
                     n_bins: int = 10) -> list[dict]:
    """Reliability bins over the top-class confidence."""
    y_true = np.asarray(y_true, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-open bins, with the final bin closed at 1.0.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for number in range(n_bins):
        mask = index == number
        bins.append({
            'lower': float(edges[number]),
            'upper': float(edges[number + 1]),
            'n': int(mask.sum()),
            'mean_confidence': float(confidence[mask].mean()) if mask.any() else None,
            'accuracy': float(correct[mask].mean()) if mask.any() else None,
        })
    return bins


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray,
                               n_bins: int = 10) -> float:
    bins = calibration_bins(y_true, probabilities, n_bins=n_bins)
    total = len(probabilities)
    return float(sum(bin_['n'] / total * abs(bin_['accuracy'] - bin_['mean_confidence'])
                     for bin_ in bins if bin_['n']))


def class_calibration_error(y_true: np.ndarray, probabilities: np.ndarray,
                            class_index: int, n_bins: int = 10) -> float:
    """One-vs-rest ECE for a class (useful when a class is systematically overconfident)."""
    binary = (np.asarray(y_true) == class_index).astype(np.int64)
    scores = probabilities[:, class_index]
    confidence = np.where(scores >= 0.5, scores, 1.0 - scores)
    predicted = (scores >= 0.5).astype(np.int64)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    total, error = len(binary), 0.0
    for number in range(n_bins):
        mask = index == number
        if mask.any():
            error += mask.sum() / total * abs(float((predicted[mask] == binary[mask]).mean())
                                              - float(confidence[mask].mean()))
    return float(error)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Minimise NLL over T > 0; returns the fitted temperature."""
    from scipy.optimize import minimize_scalar

    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(y_true, softmax(logits, np.exp(log_temperature)))

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(10.0)), method='bounded')
    if not result.success:
        raise RuntimeError(f'temperature fit failed: {result.message}')
    return float(np.exp(result.x))


def cross_fit_temperature(logits: np.ndarray, y_true: np.ndarray, n_splits: int = 2,
                          seed: int = 42) -> tuple[np.ndarray, list[float], float]:
    """Out-of-fold calibrated probabilities, per-fold temperatures, and the full-fit T."""
    from sklearn.model_selection import StratifiedKFold

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    oof = np.zeros_like(logits)
    temperatures = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(logits, y_true):
        temperature = fit_temperature(logits[train_index], y_true[train_index])
        temperatures.append(temperature)
        oof[test_index] = softmax(logits[test_index], temperature)
    return oof, temperatures, fit_temperature(logits, y_true)


def summary(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    return {
        'n': int(len(y_true)),
        'accuracy': float((predicted == y_true).mean()),
        'nll': negative_log_likelihood(y_true, probabilities),
        'brier': brier_score(y_true, probabilities),
        'ece_10bin': expected_calibration_error(y_true, probabilities, n_bins=n_bins),
        'mean_confidence': float(probabilities.max(axis=1).mean()),
    }
