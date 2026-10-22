"""Selective prediction for the v2 detector.

Policy family: answer the argmax class only when its calibrated probability clears
a class-specific threshold; otherwise abstain. Thresholds are selected on val under
the pre-registered precision constraints and frozen before the test pass.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

LABELS = ('SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT')
VERIFIABLE = (0, 1)


def clopper_pearson_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    if n == 0:
        return 0.0
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha / 2, successes, n - successes + 1))


def policy_metrics(y_true: np.ndarray, probabilities: np.ndarray,
                   thresholds: np.ndarray) -> dict:
    """Legacy v2 metrics, preserved for frozen experiment reproduction.

    ``per_class.precision`` and ``nei_precision`` are selected GOLD-subgroup
    accuracy, not predicted-class precision. Use reliability_metrics from
    src.evaluation.reliability for new reliability reports.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(y_true)), predicted]
    answered = confidence >= np.asarray(thresholds)[predicted]

    def rate(mask: np.ndarray) -> float | None:
        return float(mask.mean()) if mask.size else None

    def precision(mask: np.ndarray) -> float | None:
        return float((predicted[mask] == y_true[mask]).mean()) if mask.any() else None

    verifiable = np.isin(y_true, VERIFIABLE)
    nei = y_true == 2
    answered_verifiable = answered & verifiable
    answered_nei = answered & nei
    correct_verifiable = int((predicted[answered_verifiable] == y_true[answered_verifiable]).sum())
    n_verifiable_answered = int(answered_verifiable.sum())

    per_class = {}
    for index, label in enumerate(LABELS):
        mask = y_true == index
        selected = answered & mask
        per_class[label] = {
            'n': int(mask.sum()),
            'coverage': (float(selected.sum() / mask.sum()) if mask.any() else None),
            'precision': precision(selected),
        }

    return {
        'coverage': rate(answered),
        'answered_precision': precision(answered),
        'n_answered': int(answered.sum()),
        'verifiable_coverage': (n_verifiable_answered / int(verifiable.sum())
                                if verifiable.any() else None),
        'verifiable_precision': (correct_verifiable / n_verifiable_answered
                                 if n_verifiable_answered else None),
        'verifiable_precision_lower_95': clopper_pearson_lower(
            correct_verifiable, n_verifiable_answered),
        'nei_coverage': (float(answered_nei.sum() / nei.sum()) if nei.any() else None),
        'nei_precision': precision(answered_nei),
        'nei_abstention': (float(((~answered) & nei).sum() / nei.sum())
                           if nei.any() else None),
        'per_class': per_class,
    }


def select_policy(y_true: np.ndarray, probabilities: np.ndarray,
                  min_overall_precision: float = 0.90,
                  min_verifiable_precision: float = 0.94,
                  min_coverage: float = 0.50,
                  grid_start: float = 0.30, grid_stop: float = 0.95,
                  grid_step: float = 0.01) -> dict:
    """Search class-conditional thresholds, with the pre-registered fallback."""
    grid = np.round(np.arange(grid_start, grid_stop + 1e-9, grid_step), 2)
    candidates = []
    for supported in grid:
        for contradicted in grid:
            for insufficient in grid:
                thresholds = np.array([supported, contradicted, insufficient])
                metrics = policy_metrics(y_true, probabilities, thresholds)
                candidate = {'thresholds': thresholds.tolist(), **metrics}
                candidates.append(candidate)

    def meets(candidate: dict, require_overall: bool) -> bool:
        if candidate['verifiable_precision_lower_95'] is None:
            return False
        ok = candidate['verifiable_precision_lower_95'] >= min_verifiable_precision
        if require_overall:
            ok = ok and (candidate['answered_precision'] or 0.0) >= min_overall_precision
        return bool(ok)

    both = [candidate for candidate in candidates
            if candidate['coverage'] >= min_coverage and meets(candidate, True)]
    if both:
        chosen = max(both, key=lambda candidate: (candidate['coverage'],
                                                  candidate['verifiable_precision_lower_95'],
                                                  candidate['answered_precision'] or 0.0))
        mode = 'both_constraints'
    else:
        fallback = [candidate for candidate in candidates if meets(candidate, False)]
        if not fallback:
            raise ValueError('no policy meets the verifiable-precision constraint')
        chosen = max(fallback, key=lambda candidate: (candidate['coverage'],
                                                      candidate['verifiable_precision_lower_95'],
                                                      candidate['answered_precision'] or 0.0,
                                                      sum(candidate['thresholds'])))
        mode = 'fallback_verifiable_only'

    chosen['selection_mode'] = mode
    chosen['feasible_both_at_min_coverage'] = bool(both)
    chosen['constraints'] = {
        'min_overall_precision': min_overall_precision,
        'min_verifiable_precision': min_verifiable_precision,
        'min_coverage': min_coverage,
        'verifiable_precision_uses': 'clopper-pearson 95% lower bound',
    }
    return chosen


def risk_coverage_curve(y_true: np.ndarray, probabilities: np.ndarray,
                        start: float = 0.30, stop: float = 0.95,
                        step: float = 0.01) -> list[dict]:
    """Global-confidence curve for context; the selected policy is class-specific."""
    y_true = np.asarray(y_true, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    points = []
    for threshold in np.round(np.arange(start, stop + 1e-9, step), 2):
        answered = confidence >= threshold
        if answered.sum() == 0:
            continue
        verifiable = answered & np.isin(y_true, VERIFIABLE)
        points.append({
            'threshold': float(threshold),
            'coverage': float(answered.mean()),
            'answered_precision': float((predicted[answered] == y_true[answered]).mean()),
            'verifiable_coverage': (float(verifiable.sum() / np.isin(
                y_true, VERIFIABLE).sum()) if np.isin(y_true, VERIFIABLE).any() else None),
            'verifiable_precision': (float((predicted[verifiable] == y_true[verifiable]).mean())
                                     if verifiable.any() else None),
            'nei_coverage': float((answered & (y_true == 2)).sum() / max((y_true == 2).sum(), 1)),
        })
    return points
