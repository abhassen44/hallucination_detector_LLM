from unittest.mock import Mock

import numpy as np
import pytest

from src.generation.llm import ProviderUnavailableError
from src.verification.detector import Detector
from src.verification.grounded import (GroundedVerifier, apply_review, needs_review,
                                      validate_review)


def review(verdict='CONTRADICTED', quote='Paris is in France.'):
    return {'verdict': verdict, 'addresses_claim': True,
            'citations': [{'evidence_id': 1, 'quote': quote}], 'reason': 'The passage addresses the claim.'}


def test_verdict_requires_aligned_verbatim_evidence():
    evidence = [{'text': 'Paris is in France.'}]
    assert validate_review(review(), evidence)['verdict'] == 'CONTRADICTED'
    for bad in [dict(review(), citations=[]), dict(review(), addresses_claim='false'),
                dict(review(), addresses_claim=False), review(quote='Paris is in Italy.'),
                dict(review(), citations=[{'evidence_id': True, 'quote': 'Paris'}]),
                dict(review(), citations=[{'evidence_id': 2, 'quote': 'Paris'}])]:
        with pytest.raises(ValueError):
            validate_review(bad, evidence)


def test_parse_failure_defers_and_provider_auth_failure_propagates():
    llm = Mock(model='gemma-test')
    verifier = GroundedVerifier(llm)
    llm.generate_json.return_value = review(quote='invented quote')
    result = verifier.review('claim', [{'text': 'actual source'}])
    assert result['status'] == 'invalid'
    llm.generate_json.side_effect = ProviderUnavailableError('unauthorized')
    with pytest.raises(ProviderUnavailableError):
        verifier.review('claim', [{'text': 'source'}])


def test_prompt_contains_only_claim_and_passages_not_baseline_or_gold():
    import json
    llm = Mock(model='gemma-test')
    llm.generate_json.return_value = review('INSUFFICIENT', 'source')
    verifier = GroundedVerifier(llm)
    verifier.review('A claim.', [{'text': 'source', 'label': 'CONTRADICTED'}])
    payload = json.loads(llm.generate_json.call_args.args[0])
    assert payload == {'claim': 'A claim.', 'passages': [{'evidence_id': 1, 'text': 'source'}]}


def test_second_verifier_can_defer_but_never_flip_a_baseline_verdict():
    base = {'label': 'CONTRADICTED', 'decision': {'status': 'answer', 'reasons': []}}
    output = apply_review(base, dict(review('SUPPORTED'), status='ok'))
    assert output['label'] == 'INSUFFICIENT'
    assert output['baseline_label'] == 'CONTRADICTED'
    assert output['decision']['status'] == 'abstain'
    assert base['label'] == 'CONTRADICTED'
    assert apply_review(base, {'status': 'invalid'})['decision']['status'] == 'abstain'


def test_matching_second_verdict_can_resolve_review_without_hiding_conflict():
    base = {'label': 'SUPPORTED', 'evidence_conflict': True,
            'decision': {'status': 'abstain', 'reasons': ['conflicting_evidence']}}
    output = apply_review(base, dict(review('SUPPORTED'), status='ok'))
    assert output['evidence_conflict'] is True
    assert output['decision']['status'] == 'answer'
    assert output['baseline_decision']['status'] == 'abstain'


def test_serving_review_is_opt_in_and_only_runs_on_triggered_claims():
    nli, secondary = Mock(), Mock()
    nli.predict.return_value = np.array([[.001, .001, .998]])
    secondary.review.return_value = dict(review('INSUFFICIENT'), status='ok')
    detector = Detector(extractor=Mock(), nli=nli, reviewer=secondary)
    evidence = [{'text': 'Paris is in France.'}]
    assert detector.verify('claim', evidence)['label'] == 'CONTRADICTED'
    secondary.review.assert_not_called()
    result = detector.verify('claim', evidence, grounded_review=True)
    assert result['label'] == 'INSUFFICIENT'
    secondary.review.assert_called_once_with('claim', evidence)
    secondary.reset_mock()
    nli.predict.return_value = np.array([[.99, .009, .001]])
    assert detector.verify('claim', evidence, grounded_review=True)['grounded_review']['status'] == 'not_required'
    secondary.review.assert_not_called()


def test_context_limit_prevents_silent_evidence_truncation():
    llm = Mock(model='gemma-test')
    with pytest.raises(ValueError, match='48,000'):
        GroundedVerifier(llm).review('claim', [{'text': 'x' * 48000}])
    llm.generate_json.assert_not_called()
