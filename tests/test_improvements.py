from unittest.mock import Mock
import hashlib

import numpy as np
import pytest

from src.evaluation.reliability import reliability_metrics
from src.verification.evidence import acceptance, select_evidence
from src.verification.features import (FEATURE_NAMES, LEGACY_SCHEMA, RANK_CORRECTED_SCHEMA,
                                       build_matrix, build_rank_corrected_matrix)
from src.utils.provenance import matches_text_sha256


def test_reliability_counts_nei_false_assertions_in_predicted_precision():
    result = reliability_metrics(
        ['SUPPORTED', 'INSUFFICIENT', 'INSUFFICIENT', 'CONTRADICTED'],
        ['SUPPORTED', 'SUPPORTED', 'INSUFFICIENT', 'CONTRADICTED'],
        [True, True, False, True])
    assert result['per_predicted_class']['SUPPORTED']['precision'] == .5
    assert result['per_gold_class']['SUPPORTED']['accepted_accuracy'] == 1
    assert result['nei_false_assertions'] == 1
    assert result['nei_false_assertion_rate'] == .5
    assert result['nei_share_of_accepted_errors'] == 1
    assert result['per_gold_class']['INSUFFICIENT']['correct_rejected'] == 1


def test_reliability_keeps_insufficient_verdict_distinct_from_abstention():
    result = reliability_metrics(['INSUFFICIENT'], ['INSUFFICIENT'], [True])
    assert result['n_abstained'] == 0
    assert result['per_predicted_class']['INSUFFICIENT']['precision'] == 1
    assert result['per_predicted_class']['SUPPORTED']['precision'] is None
    assert reliability_metrics([], [], [])['accepted_accuracy'] is None
    with pytest.raises(ValueError):
        reliability_metrics(['SUPPORTED'], [], [])


def test_feature_correction_is_versioned_and_preserves_legacy():
    row = {'claim': 'Example claim', 'hybrid': [
        {'doc': 0, 'page_id': 'P', 'score': 2, 'nli': [.1, .1, .8]},
        {'doc': 1, 'page_id': 'P', 'score': 1, 'nli': [.9, .05, .05]}]}
    sims = np.array([[.5, .9]])
    legacy = build_matrix([row], 'hybrid', sims)
    corrected = build_rank_corrected_matrix([row], 'hybrid', sims)
    e = FEATURE_NAMES.index('nli_e_rank')
    c = FEATURE_NAMES.index('nli_c_rank')
    assert legacy[0, e] == legacy[0, c] == 1
    assert corrected.values[0, e] == 2 and corrected.values[0, c] == 1
    assert corrected.names[e] == 'nli_e_retrieval_rank'
    with pytest.raises(ValueError, match='refit required'):
        corrected.for_schema(LEGACY_SCHEMA)
    np.testing.assert_array_equal(corrected.for_schema(RANK_CORRECTED_SCHEMA), corrected.values)


def test_evidence_policy_respects_original_rank_and_page():
    hits = [{'page_id': 'A'}, {'page_id': 'B'}, {'page_id': 'A'}]
    assert select_evidence(hits, 'top1') == hits[:1]
    assert select_evidence(hits, 'same_page') == [hits[0], hits[2]]
    with pytest.raises(ValueError):
        select_evidence([{'text': 'source'}], 'same_page')
    assert acceptance('CONTRADICTED', has_evidence=True, conflict=True)['status'] == 'abstain'


def test_cross_platform_hash_check_still_rejects_changed_data(tmp_path):
    original = b'{"id": 1}\r\n'
    expected = hashlib.sha256(original).hexdigest()
    path = tmp_path / 'cache.jsonl'
    path.write_bytes(original.replace(b'\r\n', b'\n'))
    assert matches_text_sha256(path, expected)
    path.write_bytes(b'{"id": 2}\n')
    assert not matches_text_sha256(path, expected)


def test_cached_study_and_serving_use_the_same_evidence_and_acceptance_rules():
    from scripts.evaluate_evidence_policies import evaluate
    from src.verification.detector import Detector
    hits = [{'text': 'support', 'page_id': 'A', 'sent_id': 0, 'nli': [.99, .009, .001]},
            {'text': 'distractor', 'page_id': 'B', 'sent_id': 0, 'nli': [.001, .001, .998]}]
    row = {'id': 'x', 'claim': 'claim', 'label': 'SUPPORTED', 'hybrid': hits}
    example = {'x': {'meta': {'evidence_sets': [[['A', 0]]]}}}
    _, saved = evaluate([row], example, 'hybrid', 'top1', .7, .95)
    nli = Mock()
    nli.predict.return_value = np.array([h['nli'] for h in hits])
    detector = Detector(extractor=Mock(), nli=nli)
    live = detector.verify('claim', hits, 'top1')
    assert saved[0]['prediction'] == live['label'] == 'SUPPORTED'
    assert saved[0]['decision'] == live['decision']


def test_reranker_fails_fast_without_starting_a_download(monkeypatch):
    from src.retrieval.rerank import Reranker, RerankerUnavailable
    monkeypatch.setattr('src.retrieval.rerank.reranker_available', lambda model: False)
    with pytest.raises(RerankerUnavailable, match='weights are not installed'):
        Reranker(local_files_only=True)
