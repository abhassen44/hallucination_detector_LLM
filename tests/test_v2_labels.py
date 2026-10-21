import pytest

from scripts.materialize_v2_labels import agreement, materialize

TAXONOMY = {
    'version': 'v2',
    'verification': {'categories': ['retrieval_failure', 'verifier_error',
                                    'ambiguous_evidence']},
}


def _queue(identifier, condition=None, verdict=None, category=None, reviewer=None):
    row = {'id': identifier,
           'human_review': {'open_coding': '', 'pass_fail_assessment': verdict,
                            'category': category}}
    if condition:
        row['condition'] = condition
    if reviewer:
        row['review_meta'] = {'reviewer': reviewer}
    return row


def _proposal(identifier, condition=None, verdict=None, category=None):
    row = {'id': identifier, 'condition': condition, 'valid': True,
           'model': 'test-model', 'verdict': verdict, 'category': category}
    return row


def test_human_rows_keep_their_labels():
    rows = [_queue('x', verdict='Pass', category='ambiguous_evidence', reviewer='Abhas')]
    proposals = {('x', None): _proposal('x', verdict='Fail', category='none')}

    labels = materialize(rows, proposals, 'extraction', TAXONOMY)

    assert labels[0]['verdict'] == 'Pass'
    assert labels[0]['label_source'] == 'human_queue'
    assert labels[0]['human_reviewed'] is True
    assert labels[0]['verdict_source'] == 'human'
    assert labels[0]['legacy_category'] == 'ambiguous_evidence'
    assert labels[0]['category'] is None


def test_extraction_proposals_fill_the_verdict_only():
    rows = [_queue('x')]
    proposals = {('x', None): _proposal('x', verdict='Fail', category='claim_extraction_failure')}

    labels = materialize(rows, proposals, 'extraction', TAXONOMY)

    assert labels[0]['verdict'] == 'Fail'
    assert labels[0]['category'] is None
    assert labels[0]['proposal_category'] == 'claim_extraction_failure'
    assert labels[0]['label_source'] == 'llm_proposal'
    assert labels[0]['human_reviewed'] is False


def test_verification_proposals_are_fail_by_construction():
    rows = [_queue('x', 'bm25')]
    proposals = {('x', 'bm25'): _proposal('x', 'bm25', category='retrieval_failure')}

    labels = materialize(rows, proposals, 'verification', TAXONOMY)

    assert labels[0]['verdict'] == 'Fail'
    assert labels[0]['category'] == 'retrieval_failure'
    assert labels[0]['verdict_source'] == 'queue_construction'
    assert labels[0]['category_source'] == 'proposal'


def test_verification_matches_on_id_and_condition():
    rows = [_queue('x', 'bm25'), _queue('x', 'dense')]
    proposals = {('x', 'bm25'): _proposal('x', 'bm25', category='retrieval_failure'),
                 ('x', 'dense'): _proposal('x', 'dense', category='verifier_error')}

    labels = materialize(rows, proposals, 'verification', TAXONOMY)

    assert [label['category'] for label in labels] == ['retrieval_failure', 'verifier_error']


def test_missing_proposal_raises():
    with pytest.raises(KeyError, match='no usable proposal'):
        materialize([_queue('x')], {}, 'extraction', TAXONOMY)


def test_verification_proposal_outside_taxonomy_raises():
    proposals = {('x', 'bm25'): _proposal('x', 'bm25', category='nli_failure')}
    with pytest.raises(ValueError, match='outside taxonomy v2'):
        materialize([_queue('x', 'bm25')], proposals, 'verification', TAXONOMY)


def test_agreement_compares_proposals_with_human_rows():
    rows = [_queue('a', verdict='Pass'), _queue('b', verdict='Fail'),
            _queue('c', verdict='Pass')]
    proposals = {('a', None): _proposal('a', verdict='Pass'),
                 ('b', None): _proposal('b', verdict='Pass'),
                 ('c', None): _proposal('c', verdict='Fail')}

    result = agreement(rows, proposals, 'extraction')

    assert result['n'] == 3
    assert result['agreement'] == 1 / 3
    assert 'kappa' in result
