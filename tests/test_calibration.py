import numpy as np
import pytest

from src.evaluation.calibration import (apply_isotonic, apply_isotonic_parameters,
                                        apply_vector_scaling, brier_score, cross_fit_isotonic,
                                        cross_fit_temperature, cross_fit_temperature_brier,
                                        cross_fit_vector_scaling, cross_fit_vector_scaling_brier,
                                        expected_calibration_error, fit_isotonic,
                                        fit_temperature, fit_temperature_brier,
                                        fit_vector_scaling, fit_vector_scaling_brier,
                                        isotonic_to_parameters, negative_log_likelihood,
                                        softmax, summary)


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


def test_brier_temperature_does_not_worsen_brier_on_overconfident_data():
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(1500, 3)) * 3.0
    labels = np.array([rng.choice(3, p=row) for row in softmax(logits, 2.5)])
    raw = softmax(logits)
    temperature = fit_temperature_brier(logits, labels)
    calibrated = softmax(logits, temperature)

    assert temperature > 0
    assert brier_score(labels, calibrated) <= brier_score(labels, raw) + 1e-9


def test_cross_fit_brier_temperature_returns_oof_rows():
    rng = np.random.default_rng(12)
    logits = rng.normal(size=(200, 3))
    labels = rng.integers(0, 3, size=200)
    labels[:3] = [0, 1, 2]

    oof, temperatures, final = cross_fit_temperature_brier(logits, labels, n_splits=2)

    assert oof.shape == logits.shape
    assert len(temperatures) == 2 and final > 0


def test_brier_vector_scaling_returns_probabilities():
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(400, 3))
    labels = np.array([rng.choice(3, p=row) for row in softmax(logits)])

    parameters = fit_vector_scaling_brier(logits, labels)
    calibrated = apply_vector_scaling(logits, parameters)
    oof, folds, final = cross_fit_vector_scaling_brier(logits, labels, n_splits=2)

    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert oof.shape == logits.shape
    assert len(folds) == 2 and len(final['weights']) == 3


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


def test_vector_scaling_returns_probabilities_and_can_improve_nll():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(2000, 3)) * 2.0
    labels = np.array([rng.choice(3, p=row) for row in softmax(logits)])

    parameters = fit_vector_scaling(logits, labels)
    calibrated = apply_vector_scaling(logits, parameters)

    assert calibrated.shape == logits.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert negative_log_likelihood(labels, calibrated) <= negative_log_likelihood(
        labels, softmax(logits)) + 1e-6
    assert len(parameters['weights']) == 3 and len(parameters['bias']) == 3


def test_cross_fit_vector_scaling_is_out_of_fold():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(300, 3))
    labels = rng.integers(0, 3, size=300)
    labels[:3] = [0, 1, 2]

    oof, folds, final = cross_fit_vector_scaling(logits, labels, n_splits=2, seed=42)

    assert oof.shape == logits.shape
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert len(folds) == 2 and len(final['weights']) == 3


def test_isotonic_maps_preserve_a_distribution():
    rng = np.random.default_rng(5)
    probabilities = softmax(rng.normal(size=(400, 3)))
    labels = np.array([rng.choice(3, p=row) for row in probabilities])
    models = fit_isotonic(probabilities, labels)
    calibrated = apply_isotonic(probabilities, models)

    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert (calibrated >= 0).all()


def test_isotonic_parameters_round_trip():
    rng = np.random.default_rng(7)
    probabilities = softmax(rng.normal(size=(300, 3)))
    labels = np.array([rng.choice(3, p=row) for row in probabilities])
    models = fit_isotonic(probabilities, labels)

    parameters = isotonic_to_parameters(models)
    direct = apply_isotonic(probabilities, models)
    restored = apply_isotonic_parameters(probabilities, parameters)

    assert np.allclose(direct, restored, atol=1e-8)


def test_cross_fit_isotonic_returns_oof_rows():
    rng = np.random.default_rng(6)
    probabilities = softmax(rng.normal(size=(200, 3)))
    labels = rng.integers(0, 3, size=200)
    labels[:3] = [0, 1, 2]

    oof, folds, final = cross_fit_isotonic(probabilities, labels, n_splits=2, seed=42)

    assert oof.shape == probabilities.shape
    assert len(folds) == 2 and len(final) == 3


def test_summary_reports_the_expected_fields():
    labels = np.array([0, 1, 2])
    probabilities = _one_hot(labels)
    result = summary(labels, probabilities)
    assert result['n'] == 3
    assert result['accuracy'] == 1.0
    assert set(result) == {'n', 'accuracy', 'nll', 'brier', 'ece_10bin', 'mean_confidence'}
