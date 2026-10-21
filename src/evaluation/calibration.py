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


def fit_temperature_brier(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature fitted directly on Brier score rather than NLL."""
    from scipy.optimize import minimize_scalar

    def objective(log_temperature: float) -> float:
        return brier_score(y_true, softmax(logits, np.exp(log_temperature)))

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(10.0)), method='bounded')
    if not result.success:
        raise RuntimeError(f'Brier temperature fit failed: {result.message}')
    return float(np.exp(result.x))


def cross_fit_temperature_brier(logits: np.ndarray, y_true: np.ndarray, n_splits: int = 2,
                                seed: int = 42) -> tuple[np.ndarray, list[float], float]:
    from sklearn.model_selection import StratifiedKFold

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    oof = np.zeros_like(logits)
    temperatures = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(logits, y_true):
        temperature = fit_temperature_brier(logits[train_index], y_true[train_index])
        temperatures.append(temperature)
        oof[test_index] = softmax(logits[test_index], temperature)
    return oof, temperatures, fit_temperature_brier(logits, y_true)


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


def fit_vector_scaling(logits: np.ndarray, y_true: np.ndarray,
                       l2: float = 1e-3) -> dict:
    """Per-class scale and bias on logits, fitted by penalised multinomial NLL."""
    from scipy.optimize import minimize

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n_classes = logits.shape[1]

    def objective(theta: np.ndarray) -> float:
        weights, bias = theta[:n_classes], theta[n_classes:]
        probabilities = softmax(logits * weights + bias)
        penalty = l2 * (np.sum((weights - 1.0) ** 2) + np.sum(bias ** 2))
        return negative_log_likelihood(y_true, probabilities) + penalty

    start = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
    result = minimize(objective, start, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(f'vector-scaling fit failed: {result.message}')
    return {'weights': result.x[:n_classes].tolist(),
            'bias': result.x[n_classes:].tolist()}


def apply_vector_scaling(logits: np.ndarray, parameters: dict) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.asarray(parameters['weights'], dtype=np.float64)
    bias = np.asarray(parameters['bias'], dtype=np.float64)
    return softmax(logits * weights + bias)


def fit_vector_scaling_brier(logits: np.ndarray, y_true: np.ndarray,
                             l2: float = 1e-3) -> dict:
    """Per-class scale/bias fitted directly on Brier score."""
    from scipy.optimize import minimize

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n_classes = logits.shape[1]

    def objective(theta: np.ndarray) -> float:
        weights, bias = theta[:n_classes], theta[n_classes:]
        probabilities = softmax(logits * weights + bias)
        penalty = l2 * (np.sum((weights - 1.0) ** 2) + np.sum(bias ** 2))
        return brier_score(y_true, probabilities) + penalty

    start = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
    result = minimize(objective, start, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(f'Brier vector-scaling fit failed: {result.message}')
    return {'weights': result.x[:n_classes].tolist(),
            'bias': result.x[n_classes:].tolist()}


def cross_fit_vector_scaling_brier(logits: np.ndarray, y_true: np.ndarray,
                                   n_splits: int = 2, seed: int = 42
                                   ) -> tuple[np.ndarray, list[dict], dict]:
    from sklearn.model_selection import StratifiedKFold

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    oof = np.zeros_like(logits)
    parameters = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(logits, y_true):
        fitted = fit_vector_scaling_brier(logits[train_index], y_true[train_index])
        parameters.append(fitted)
        oof[test_index] = apply_vector_scaling(logits[test_index], fitted)
    return oof, parameters, fit_vector_scaling_brier(logits, y_true)


def cross_fit_vector_scaling(logits: np.ndarray, y_true: np.ndarray, n_splits: int = 2,
                             seed: int = 42) -> tuple[np.ndarray, list[dict], dict]:
    from sklearn.model_selection import StratifiedKFold

    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    oof = np.zeros_like(logits)
    parameters = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(logits, y_true):
        fitted = fit_vector_scaling(logits[train_index], y_true[train_index])
        parameters.append(fitted)
        oof[test_index] = apply_vector_scaling(logits[test_index], fitted)
    return oof, parameters, fit_vector_scaling(logits, y_true)


def fit_isotonic(probabilities: np.ndarray, y_true: np.ndarray) -> list:
    """One-vs-rest isotonic maps for each class; None when a fold has one class."""
    from sklearn.isotonic import IsotonicRegression

    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    models = []
    for class_index in range(probabilities.shape[1]):
        target = (y_true == class_index).astype(np.float64)
        if np.unique(target).size < 2:
            models.append(None)
            continue
        models.append(IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                      .fit(probabilities[:, class_index], target))
    return models


def apply_isotonic(probabilities: np.ndarray, models: list) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    output = np.zeros_like(probabilities)
    for class_index, model in enumerate(models):
        output[:, class_index] = (model.predict(probabilities[:, class_index])
                                  if model is not None else probabilities[:, class_index])
    output = np.clip(output, 1e-9, None)
    return output / output.sum(axis=1, keepdims=True)


def isotonic_to_parameters(models: list) -> list[dict | None]:
    """Serialise fitted isotonic maps as interpolation thresholds."""
    parameters = []
    for model in models:
        if model is None:
            parameters.append(None)
        else:
            parameters.append({'x': np.asarray(model.X_thresholds_).tolist(),
                               'y': np.asarray(model.y_thresholds_).tolist()})
    return parameters


def apply_isotonic_parameters(probabilities: np.ndarray,
                              parameters: list[dict | None]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    output = np.zeros_like(probabilities)
    for class_index, parameter in enumerate(parameters):
        if parameter is None:
            output[:, class_index] = probabilities[:, class_index]
        else:
            output[:, class_index] = np.interp(probabilities[:, class_index],
                                               parameter['x'], parameter['y'])
    output = np.clip(output, 1e-9, None)
    return output / output.sum(axis=1, keepdims=True)


def cross_fit_isotonic(probabilities: np.ndarray, y_true: np.ndarray, n_splits: int = 2,
                       seed: int = 42) -> tuple[np.ndarray, list[list], list]:
    from sklearn.model_selection import StratifiedKFold

    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    oof = np.zeros_like(probabilities)
    fold_models = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(probabilities, y_true):
        models = fit_isotonic(probabilities[train_index], y_true[train_index])
        fold_models.append(models)
        oof[test_index] = apply_isotonic(probabilities[test_index], models)
    return oof, fold_models, fit_isotonic(probabilities, y_true)


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
