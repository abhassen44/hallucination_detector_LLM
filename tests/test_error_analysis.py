"""Schema guards for the Gemma-assisted error analysis."""
from scripts.shared.review.analyze_errors import _validate_extraction, _validate_verification


def test_pass_forces_the_none_category():
    result = _validate_extraction({'verdict': 'Pass', 'category': 'nli_failure',
                                   'rationale': 'Claims match the answer.', 'confidence': 0.9})
    assert result['verdict'] == 'Pass' and result['category'] == 'none'


def test_fail_requires_a_known_category():
    assert _validate_extraction({'verdict': 'Fail', 'category': 'made_up',
                                 'rationale': 'x', 'confidence': 0.5}) is None
    valid = _validate_extraction({'verdict': 'Fail', 'category': 'claim_extraction_failure',
                                  'rationale': 'Adds an unsupported entity.', 'confidence': 2})
    assert valid['category'] == 'claim_extraction_failure'
    assert valid['confidence'] == 1.0


def test_verification_rejects_unknown_categories_and_bad_confidence():
    assert _validate_verification({'category': 'nli_failure', 'rationale': 'x',
                                   'confidence': 'high'}) is None
    assert _validate_verification({'category': 'other', 'rationale': 'x',
                                   'confidence': 0.4}) is None
    assert _validate_verification({'category': 'retrieval_failure', 'rationale': 'x',
                                   'confidence': -3})['confidence'] == 0.0


def test_rationale_is_required_and_bounded():
    assert _validate_verification({'category': 'nli_failure', 'rationale': '   ',
                                   'confidence': 0.5}) is None
    long = _validate_verification({'category': 'nli_failure', 'rationale': 'a' * 900,
                                   'confidence': 0.5})
    assert len(long['rationale']) == 400
