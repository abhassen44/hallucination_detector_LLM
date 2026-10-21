import numpy as np
import pytest

from src.evaluation.selective import (clopper_pearson_lower, policy_metrics,
                                      risk_coverage_curve, select_policy)


def test_clopper_pearson_lower_is_below_the_point_estimate():
    assert clopper_pearson_lower(95, 100) < 0.95
    assert clopper_pearson_lower(95, 100) > 0.88
    assert clopper_pearson_lower(0, 10) == 0.0
    assert clopper_pearson_lower(0, 0) == 0.0


def test_policy_metrics_separates_verifiable_and_nei():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array([
        [0.9, 0.05, 0.05], [0.4, 0.3, 0.3],
        [0.1, 0.8, 0.1], [0.2, 0.3, 0.5],
        [0.8, 0.1, 0.1], [0.1, 0.1, 0.8],
    ])
    thresholds = np.array([0.5, 0.5, 0.5])

    metrics = policy_metrics(labels, probabilities, thresholds)

    assert metrics['n_answered'] == 5  # rows 1, 3, 4, 5, 6
    assert metrics['coverage'] == 5 / 6
    assert metrics['verifiable_coverage'] == 0.75  # rows 1, 3, 4 of four verifiable
    assert metrics['verifiable_precision'] == 2 / 3  # row 4 is wrong
    assert metrics['nei_coverage'] == 1.0  # both NEI rows were answered
    assert metrics['per_class']['SUPPORTED']['coverage'] == 0.5


def test_select_policy_uses_fallback_when_overall_target_is_infeasible():
    labels = np.array([0, 1, 2] * 10)
    probabilities = np.zeros((30, 3))
    for index, label in enumerate(labels):
        if label == 2:  # confident but wrong NEI answers keep overall precision low
            probabilities[index, 0] = 0.9
            probabilities[index, 2] = 0.05
        else:
            probabilities[index, label] = 0.9
    probabilities[:, 0] += 0.01
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    chosen = select_policy(labels, probabilities, min_overall_precision=0.99,
                           min_verifiable_precision=0.8, min_coverage=0.9,
                           grid_start=0.3, grid_stop=0.9, grid_step=0.3)

    assert chosen['selection_mode'] == 'fallback_verifiable_only'
    assert chosen['feasible_both_at_min_coverage'] is False
    assert len(chosen['thresholds']) == 3


def test_risk_coverage_curve_is_monotone_in_threshold():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array([
        [0.9, 0.05, 0.05], [0.2, 0.5, 0.3],
        [0.1, 0.8, 0.1], [0.2, 0.3, 0.5],
        [0.8, 0.1, 0.1], [0.1, 0.1, 0.8],
    ])
    curve = risk_coverage_curve(labels, probabilities, start=0.3, stop=0.9, step=0.1)
    coverages = [point['coverage'] for point in curve]
    assert coverages == sorted(coverages, reverse=True)
    assert all(0 <= point['coverage'] <= 1 for point in curve)
