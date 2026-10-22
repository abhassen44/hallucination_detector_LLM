from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_detector, get_extractor
from src.generation.claim_extraction import ExtractionResult
from src.verification.detector import Detector, aggregate_probabilities, decide


@pytest.fixture
def detector():
    extractor, retriever, nli = Mock(), Mock(), Mock()
    extractor.llm.model = 'test-model'
    extractor.extract_with_metadata.return_value = ExtractionResult(['Paris is in France.'])
    retriever.search.return_value = [{'text': 'Paris is the capital of France.', 'page_id': 'Paris', 'sent_id': 0, 'score': .02}]
    nli.predict.return_value = np.array([[.99, .005, .005]])
    return Detector(extractor, retriever, nli)


@pytest.fixture
def client(detector):
    app.dependency_overrides[get_detector] = lambda: detector
    app.dependency_overrides[get_extractor] = lambda: detector.extractor
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_decision_boundaries_and_no_evidence(detector):
    assert decide([.7, .3, 0]) == 'SUPPORTED'
    assert decide([0, .05, .95]) == 'CONTRADICTED'
    assert decide([.6, .4, 0]) == 'INSUFFICIENT'
    result = detector.verify('claim', [])
    assert result['label'] == 'INSUFFICIENT'
    detector.nli.predict.assert_not_called()
    np.testing.assert_allclose(aggregate_probabilities([[.9, .05, .05], [.01, .01, .98]]), [.02, 0, .98])


def test_detect_and_answer_aggregation(detector):
    assert detector.detect(None, 'answer')['label'] == 'SUPPORTED'
    detector.nli.predict.return_value = np.array([[.01, .01, .98]])
    assert detector.detect(None, 'answer')['label'] == 'CONTRADICTED'
    detector.extractor.extract_with_metadata.return_value = ExtractionResult([], True)
    result = detector.detect(None, 'answer')
    assert result['label'] == 'INSUFFICIENT' and result['warnings']


def test_api_endpoints(client):
    assert client.get('/health').json()['mode'] == 'inference_only'
    result = client.post('/detect', json={'answer': 'Paris is in France.'})
    assert result.status_code == 200
    assert result.json()['claims'][0]['evidence'][0]['text']
    assert client.post('/claims', json={'answer': 'answer'}).json()['claims']
    assert client.post('/retrieve', json={'claim': 'claim'}).json()['evidence']
    assert client.post('/verify', json={'claim': 'claim', 'evidence': []}).json()['label'] == 'INSUFFICIENT'
    assert client.get('/metrics').json()['results']
    served = client.get('/metrics').json()['results']
    assert any(key.startswith('baseline_review/') for key in served)
    assert all(key.startswith(('stage', 'baseline_review', 'improvements', 'grounded_pilot_')) for key in served)


@pytest.mark.parametrize('payload', [{'answer': '   '}, {'answer': 'a', 'method': 'invalid'},
                                    {'answer': 'a', 'top_k': 21}, {'answer': 'a', 'unexpected': 1}])
def test_api_validation(client, payload):
    assert client.post('/detect', json=payload).status_code == 422


def test_api_inference_failure_is_actionable(client, detector):
    detector.nli.predict.side_effect = RuntimeError('private details')
    response = client.post('/detect', json={'answer': 'answer'})
    assert response.status_code == 503
    assert 'private details' not in response.text


def test_conflicting_evidence_is_visible_without_changing_baseline(detector):
    detector.nli.predict.return_value = np.array([[.989, .01, .001], [.001, .003, .996]])
    result = detector.verify('claim', [{'text': 'support'}, {'text': 'unrelated passage'}])
    assert result['evidence_conflict'] is True
    assert result['label'] == 'CONTRADICTED'
    assert result['decision']['status'] == 'abstain'
    assert result['evidence'][0]['nli']['entailment'] == .989


def test_provided_source_bypasses_retrieval(detector):
    result = detector.detect(None,'Paris is in France.',evidence=['Paris is the capital of France.'])
    assert result['label'] == 'SUPPORTED'
    assert result['evidence_source'] == 'provided' and result['method'] == 'provided'
    assert result['claims'][0]['evidence'][0]['source'] == 'provided'
    detector.retriever.search.assert_not_called()


def test_explicit_empty_source_does_not_fall_back_to_search(detector):
    result = detector.detect(None,'answer',evidence=[])
    assert result['label'] == 'INSUFFICIENT'
    detector.retriever.search.assert_not_called()
    detector.nli.predict.assert_not_called()


def test_api_accepts_source_and_bounds_evidence_count(client,detector):
    result = client.post('/detect',json={'answer':'Paris is in France.','evidence':['Paris is the capital of France.']})
    assert result.status_code == 200 and result.json()['evidence_source'] == 'provided'
    detector.retriever.search.assert_not_called()
    assert client.post('/detect',json={'answer':'a','evidence':['x']*21}).status_code == 422


def test_source_verification_never_initializes_a_wikipedia_retriever(detector):
    source_only = Detector(extractor=detector.extractor, nli=detector.nli, cfg=detector.cfg)
    assert source_only._retriever is None
    result = source_only.detect(None,'answer',evidence=['evidence'])
    assert result['label'] == 'SUPPORTED'
    assert source_only._retriever is None


def test_focused_evidence_keeps_distractor_visible_and_conflict_unaccepted(detector):
    detector.nli.predict.return_value = np.array([[.99, .009, .001], [.001, .001, .998]])
    result = detector.verify('claim', [{'text': 'support'}, {'text': 'distractor'}], 'top1')
    assert result['label'] == 'SUPPORTED'
    assert result['decision']['status'] == 'abstain'
    assert result['evidence'][0]['used_for_verdict'] is True
    assert result['evidence'][1]['used_for_verdict'] is False


def test_no_evidence_is_operational_abstention(detector):
    result = detector.verify('claim', [])
    assert result['label'] == 'INSUFFICIENT'
    assert result['decision']['reasons'] == ['no_evidence']


def test_optional_reranking_preserves_provenance_and_retrieval_scores(detector):
    detector.retriever.search.return_value = [
        {'text': 'irrelevant', 'page_id': 'A', 'score': 10},
        {'text': 'relevant', 'page_id': 'B', 'score': 5}]
    reranker = Mock()
    reranker.score.return_value = np.array([-.5, 2])
    detector._reranker = reranker
    result = detector.retrieve('claim', top_k=1, rerank=True)
    assert result[0]['page_id'] == 'B'
    assert result[0]['score'] == 5 and result[0]['rerank_score'] == 2
    detector.retriever.search.assert_called_once_with('claim', method='hybrid', k=50)


def test_provided_source_ignores_search_reranking(client, detector):
    result = client.post('/detect', json={'answer': 'claim', 'evidence': ['source'], 'rerank': True})
    assert result.status_code == 200
    assert result.json()['reranked'] is False
    detector.retriever.search.assert_not_called()


def test_fallback_needs_review_even_for_supported_claims(detector):
    detector.extractor.extract_with_metadata.return_value = ExtractionResult(['claim'], True)
    result = detector.detect(None, 'answer', evidence=['source'])
    assert result['label'] == 'SUPPORTED'
    assert result['decision']['status'] == 'abstain'
    assert 'extraction_fallback' in result['decision']['reasons']


def test_missing_optional_reranker_has_actionable_error(client, detector, monkeypatch):
    from src.retrieval.rerank import RerankerUnavailable
    monkeypatch.setattr('src.api.main.reranker_available', lambda: False)
    assert client.get('/health').json()['reranker_available'] is False
    detector._reranker = Mock()
    detector._reranker.score.side_effect = RerankerUnavailable('Optional reranker weights are not installed.')
    response = client.post('/retrieve', json={'claim': 'claim', 'rerank': True})
    assert response.status_code == 503
    assert response.json()['detail'] == 'Optional reranker weights are not installed.'


def test_api_grounded_review_is_explicit_and_preserves_original_verdict(client, detector):
    detector.nli.predict.return_value = np.array([[.001, .001, .998]])
    secondary = Mock()
    secondary.review.return_value = {'status': 'ok', 'verdict': 'INSUFFICIENT',
                                    'reason': 'The passage does not establish the claim.',
                                    'citations': [], 'addresses_claim': False}
    detector._reviewer = secondary
    response = client.post('/verify', json={'claim': 'claim', 'evidence': ['source'],
                                           'grounded_review': True})
    assert response.status_code == 200
    assert response.json()['baseline_label'] == 'CONTRADICTED'
    assert response.json()['label'] == 'INSUFFICIENT'
    assert response.json()['decision']['status'] == 'abstain'
    assert secondary.review.call_count == 1
