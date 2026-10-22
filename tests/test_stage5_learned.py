"""Regression checks on the committed Stage 5 artifacts.

These assert the *shipped* claims, not a fresh fit: the learned combiner must beat
the frozen rule, the protocol must record a single test evaluation, and the
recorded cache fingerprints must match the committed retrieval caches.
"""
import json

import pytest

from src.utils.config import PROJECT_ROOT
from src.utils.provenance import matches_text_sha256

OUT = PROJECT_ROOT / 'experiments/stage5_learned'
METRICS = OUT / 'metrics.json'
ABLATIONS = OUT / 'metrics_ablations.json'
METHODS = ('bm25', 'dense', 'hybrid')


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.skipif(not METRICS.exists(), reason='Stage 5 artifacts not generated')
def test_learned_combiner_beats_the_frozen_rule():
    report = _load(METRICS)
    for method in METHODS:
        result = report['methods'][method]
        learned = result['test']['macro_f1']
        fixed = result['fixed_rule']['test']['macro_f1']
        assert learned > fixed + 0.02, f'{method}: {learned:.4f} vs {fixed:.4f}'
        assert result['test']['n'] == 1998


@pytest.mark.skipif(not METRICS.exists(), reason='Stage 5 artifacts not generated')
def test_protocol_records_a_single_test_evaluation():
    report = _load(METRICS)
    protocol = report['protocol']
    assert protocol['training_performed'] is True
    assert protocol['test_evaluated'] is True
    assert _load(OUT / 'protocol.json')['test_evaluated'] is True
    assert protocol['fit_split'] == 'train' and protocol['report_split'] == 'test'


@pytest.mark.skipif(not METRICS.exists(), reason='Stage 5 artifacts not generated')
def test_protocol_cache_hashes_still_match():
    protocol = _load(METRICS)['protocol']
    for split, expected in protocol['cache_sha256'].items():
        path = PROJECT_ROOT / 'data/processed/retrieval' / f'fever_{split}.jsonl'
        assert matches_text_sha256(path, expected)


@pytest.mark.skipif(not ABLATIONS.exists(), reason='Stage 5 artifacts not generated')
def test_ablation_grid_is_complete():
    ablations = _load(ABLATIONS)['ablations']
    expected = {'full', 'no_nli', 'no_retrieval', 'no_similarity', 'no_claim_features',
                'nli_only', 'retrieval_only', 'similarity_only'}
    for method in METHODS:
        assert set(ablations[method]) == expected
        assert ablations[method]['full']['test']['macro_f1'] > 0.6


@pytest.mark.skipif(not METRICS.exists(), reason='Stage 5 artifacts not generated')
def test_predictions_are_probability_distributions():
    for method in METHODS:
        rows = [json.loads(line) for line in
                (OUT / f'predictions_{method}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 1998
        for row in rows[:50]:
            probabilities = list(row['probabilities'].values())
            assert abs(sum(probabilities) - 1) < 1e-6
            assert row['prediction'] in row['probabilities']
