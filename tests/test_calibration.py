import numpy as np
import pytest

from src.evaluation.calibration import (brier_score, cross_fit_temperature,
                                        expected_calibration_error, fit_temperature,
                                        negative_log_likelihood, softmax, summary)


def _one_hot(labels, n_classes=3):
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def test_softmax_is_a_distribution_and_respects_temperature():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    for temperature in (0.5, 1.0, 2.0):
        probabilities = softmax(logits, temperature)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert (probabilities >= 0).all()
    sharp = softmax(logits, 0.5)[0, 0]
    smooth = softmax(logits, 2.0)[0, 0]
    assert sharp > smooth


def test_perfect_predictions_have_zero_brier_and_ece():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = _one_hot(labels)
    assert brier_score(labels, probabilities) == 0.0
    assert expected_calibration_error(labels, probabilities) == 0.0
    assert negative_log_likelihood(labels, probabilities) < 1e-10


def test_overconfidence_raises_ece():
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    confident_wrong = np.tile([0.98, 0.01, 0.01], (len(labels), 1))
    assert expected_calibration_error(labels, confident_wrong) > 0.5


def test_temperature_fit_recovers_a_synthetic_temperature():
    rng = np.random.default_rng(42)
    true_temperature = 1.8
    logits = rng.normal(size=(6000, 3)) * 2.0
    probabilities = softmax(logits, true_temperature)
    labels = np.array([rng.choice(3, p=row) for row in probabilities])

    fitted = fit_temperature(logits, labels)

    assert 1.4 < fitted < 2.3


def test_cross_fit_returns_out_of_fold_probabilities():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(200, 3))
    labels = rng.integers(0, 3, size=200)
    labels[:3] = [0, 1, 2]

    oof, temperatures, full = cross_fit_temperature(logits, labels, n_splits=2, seed=42)

    assert oof.shape == logits.shape
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert len(temperatures) == 2 and all(t > 0 for t in temperatures)
    assert full > 0


def test_summary_reports_the_expected_fields():
    labels = np.array([0, 1, 2])
    probabilities = _one_hot(labels)
    result = summary(labels, probabilities)
    assert result['n'] == 3
    assert result['accuracy'] == 1.0
    assert set(result) == {'n', 'accuracy', 'nll', 'brier', 'ece_10bin', 'mean_confidence'}
