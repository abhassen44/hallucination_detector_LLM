"""Reliability of accepted verdicts; abstention is separate from the NEI label."""
from __future__ import annotations

import numpy as np

from src.datasets.schema import LABELS, INSUFFICIENT


def reliability_metrics(gold, verdicts, accepted) -> dict:
    gold, verdicts = np.asarray(gold), np.asarray(verdicts)
    accepted = np.asarray(accepted, dtype=bool)
    if gold.ndim != 1 or gold.shape != verdicts.shape or gold.shape != accepted.shape:
        raise ValueError('Gold, verdicts and acceptance must be aligned one-dimensional arrays')
    if not set(gold.tolist()) <= set(LABELS) or not set(verdicts.tolist()) <= set(LABELS):
        raise ValueError('Unknown verdict label')

    def fraction(n, d):
        return float(n / d) if d else None

    correct = gold == verdicts
    nei = gold == INSUFFICIENT
    false_assertions = accepted & nei & (verdicts != INSUFFICIENT)
    by_prediction, by_gold = {}, {}
    for label in LABELS:
        predicted = accepted & (verdicts == label)
        actual = gold == label
        selected = accepted & actual
        by_prediction[label] = {
            'accepted': int(predicted.sum()), 'correct': int((predicted & correct).sum()),
            'precision': fraction((predicted & correct).sum(), predicted.sum()),
        }
        by_gold[label] = {
            'n': int(actual.sum()), 'accepted': int(selected.sum()),
            'acceptance_rate': fraction(selected.sum(), actual.sum()),
            'accepted_accuracy': fraction((selected & correct).sum(), selected.sum()),
            'correct_rejected': int((actual & correct & ~accepted).sum()),
        }
    return {
        'schema': 'accepted-verdict-reliability-v1', 'n': len(gold),
        'n_accepted': int(accepted.sum()), 'n_abstained': int((~accepted).sum()),
        'acceptance_rate': fraction(accepted.sum(), len(gold)),
        'accepted_accuracy': fraction((correct & accepted).sum(), accepted.sum()),
        'per_predicted_class': by_prediction, 'per_gold_class': by_gold,
        'nei_false_assertions': int(false_assertions.sum()),
        'nei_false_assertion_rate': fraction(false_assertions.sum(), nei.sum()),
        'nei_share_of_accepted_errors': fraction(false_assertions.sum(),
                                                 (accepted & ~correct).sum()),
    }
