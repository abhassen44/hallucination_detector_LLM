"""The improvement studies must stay val-only and never report a test score."""
import json

import pytest

from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/stage5_learned'
RERANK = OUT / 'val_study_rerank.json'
CONTEXT = OUT / 'val_study_context.json'


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.skipif(not RERANK.exists(), reason='Rerank study not generated')
def test_rerank_study_is_restricted_to_val():
    study = _load(RERANK)
    assert study['split'] == 'val'
    assert study['test_evaluated'] is False
    assert study['training_performed'] is False
    assert study['threshold_tuning_performed'] is False
    assert study['n'] == 999
    assert set(study['methods']) == {'bm25', 'dense', 'hybrid'}
    for result in study['methods'].values():
        assert '@5' in result['retrieval']['baseline']
        assert result['pipeline']['baseline']['n'] == 999
        assert result['pipeline']['candidate']['n'] == 999


@pytest.mark.skipif(not CONTEXT.exists(), reason='Context study not generated')
def test_context_study_is_restricted_to_val():
    study = _load(CONTEXT)
    assert study['split'] == 'val'
    assert study['test_evaluated'] is False
    assert study['training_performed'] is False
    assert study['n'] == 999
    assert set(study['methods']) == {'bm25', 'dense', 'hybrid'}
    for result in study['methods'].values():
        assert result['pipeline']['candidate']['n'] == 999
