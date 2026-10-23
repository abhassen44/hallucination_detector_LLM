import json

import numpy as np
import pytest

from scripts.v5.evaluate_results_v5 import evaluate
from scripts.v5.evaluate_v5_test import _jsonable
from src.utils.config import PROJECT_ROOT


def test_jsonable_handles_numpy_scalars():
    payload = {
        'flag': np.bool_(True),
        'scalar': np.float64(0.5),
        'integer': np.int64(3),
        'vector': np.array([1, 2]),
        'nested': [{'value': np.float32(0.25)}],
    }

    result = _jsonable(payload)

    assert result == {
        'flag': True,
        'scalar': 0.5,
        'integer': 3,
        'vector': [1, 2],
        'nested': [{'value': pytest.approx(0.25)}],
    }
    json.dumps(result)


def test_v5_freeze_and_test_report_are_consistent():
    freeze = json.loads((PROJECT_ROOT / 'experiments/v5/freeze.json').read_text(
        encoding='utf-8'))
    report = json.loads((PROJECT_ROOT / 'experiments/v5/test_metrics.json').read_text(
        encoding='utf-8'))

    assert freeze['test_evaluated'] is True
    assert freeze['test_attempts'] == report['attempts']
    assert freeze['test_recovered_after_error'] == report['recovered_after_error']
    assert freeze['answerability']['tau'] == pytest.approx(0.62)
    assert report['test_pass'] == 1
    assert report['metrics']['n'] == 1998
    assert all(report['test_gates'].values())
    assert report['hashes']['test_rerank']
    assert report['hashes']['test_grounding_mc']
    assert report['hashes']['test_grounding_hhem']


def test_v5_results_report_recomputes_frozen_metrics():
    result = evaluate()
    assert result['test']['test_pass'] == 1
    assert result['test']['attempts'] == 2
    assert result['distribution']['nei']['total'] == 666
    assert all(check['match'] for check in result['checks'].values())
