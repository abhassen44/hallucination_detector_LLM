import numpy as np

from scripts.v3.answerability_v3 import (CLASS_THRESHOLDS, answerability_policy,
                                      meets_constraints)


def _metrics(**overrides):
    metrics = {
        'verifiable_precision': 0.95,
        'verifiable_precision_lower_95': 0.91,
        'verifiable_coverage': 0.82,
        'coverage': 0.60,
        'answered_precision': 0.76,
        'nei_coverage': 0.38,
    }
    metrics.update(overrides)
    return metrics


def test_answerability_policy_requires_both_gates():
    probabilities = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05],
                              [0.2, 0.7, 0.1]])
    p_answer = np.array([0.9, 0.2, 0.9])

    answered = answerability_policy(probabilities, p_answer, tau=0.5)

    assert answered.tolist() == [True, False, True]
    assert CLASS_THRESHOLDS.tolist() == [0.5, 0.38, 0.84]


def test_answerability_policy_can_abstain_on_class_confidence():
    probabilities = np.array([[0.45, 0.3, 0.25]])
    p_answer = np.array([0.9])

    assert answerability_policy(probabilities, p_answer, tau=0.5).tolist() == [False]


def test_constraints_require_all_six_values():
    assert meets_constraints(_metrics()) is True
    assert meets_constraints(_metrics(nei_coverage=0.41)) is False
    assert meets_constraints(_metrics(answered_precision=0.74)) is False
    assert meets_constraints(_metrics(verifiable_coverage=0.79)) is False
    assert meets_constraints(_metrics(verifiable_precision_lower_95=0.89)) is False
