import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.datasets.schema import LABELS, to_binary
from src.evaluation.significance import (
    _binary_f1, _encode, _encode_binary, _macro_f1, mcnemar, paired_delta,
)


def test_macro_f1_matches_sklearn():
    y = _encode(['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT', 'SUPPORTED', 'CONTRADICTED'])
    p = _encode(['SUPPORTED', 'SUPPORTED', 'INSUFFICIENT', 'CONTRADICTED', 'CONTRADICTED'])
    expected = f1_score([LABELS[i] for i in y], [LABELS[i] for i in p],
                        labels=list(LABELS), average='macro', zero_division=0)
    assert _macro_f1(y, p) == pytest.approx(expected)


def test_binary_f1_matches_sklearn():
    y = _encode_binary([to_binary(label) for label in
                        ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT', 'SUPPORTED']])
    p = _encode_binary([to_binary(label) for label in
                        ['SUPPORTED', 'SUPPORTED', 'INSUFFICIENT', 'CONTRADICTED']])
    assert _binary_f1(y, p) == pytest.approx(f1_score(y, p, pos_label=1, zero_division=0))


def test_mcnemar_counts_and_direction():
    candidate = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    reference = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    result = mcnemar(candidate, reference)
    assert result['favouring_candidate'] == 2
    assert result['favouring_reference'] == 1
    assert 0 <= result['p_chi2'] <= 1
    assert 0 <= result['p_exact_binomial'] <= 1


def test_mcnemar_identical_systems_is_null():
    correct = np.array([1, 0, 1, 0], dtype=bool)
    result = mcnemar(correct, correct)
    assert result['discordant'] == 0
    assert result['p_chi2'] == 1.0
    assert result['p_exact_binomial'] == 1.0


def test_paired_bootstrap_is_deterministic_and_directional():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 3, n)
    reference = np.where(rng.random(n) < 0.5, y, (y + 1) % 3)
    candidate = np.where(rng.random(n) < 0.9, y, (y + 1) % 3)
    first = paired_delta(y, candidate, reference, _macro_f1, n_boot=300, seed=7)
    second = paired_delta(y, candidate, reference, _macro_f1, n_boot=300, seed=7)
    assert first == second
    assert first['observed'] > 0
    assert first['lower_95'] > 0
    assert first['p_two_sided'] <= 0.05


def test_paired_bootstrap_identical_systems_centres_on_zero():
    y = np.array([0, 1, 2, 0, 1, 2])
    result = paired_delta(y, y.copy(), y.copy(), _macro_f1, n_boot=200, seed=1)
    assert result['observed'] == 0
    assert result['lower_95'] == 0 and result['upper_95'] == 0
    assert result['p_two_sided'] == 1.0
