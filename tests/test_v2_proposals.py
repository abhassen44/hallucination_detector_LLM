import pytest

from scripts.propose_v2 import build_prompt, copy_extraction, load_taxonomy, validate


@pytest.fixture(scope='module')
def taxonomy():
    return load_taxonomy()


def test_taxonomy_renames_the_catch_all(taxonomy):
    categories = taxonomy['verification']['categories']
    assert 'nli_failure' not in categories
    assert {'verifier_error', 'ambiguous_evidence'} <= set(categories)
    assert set(categories) == set(taxonomy['verification']['definitions'])


def test_validate_rejects_the_v1_catch_all(taxonomy):
    payload = {'category': 'nli_failure', 'rationale': 'old label', 'confidence': 0.9}
    assert validate(payload, taxonomy) is None


def test_validate_accepts_v2_categories(taxonomy):
    payload = {'category': 'verifier_error', 'rationale': 'Evidence span was clear.',
               'confidence': 1.7}
    result = validate(payload, taxonomy)
    assert result == {'category': 'verifier_error', 'rationale': 'Evidence span was clear.',
                      'confidence': 1.0}


def test_validate_rejects_missing_rationale(taxonomy):
    assert validate({'category': 'ambiguous_evidence', 'rationale': '  ',
                     'confidence': 0.5}, taxonomy) is None


def test_prompt_carries_definitions_and_precedence(taxonomy):
    row = {'id': 'x', 'condition': 'bm25_extracted_claims', 'question': 'Q?', 'answer': 'A',
           'answer_gold_label': 'supported', 'answer_prediction': 'hallucinated',
           'claims': [{'claim': 'A is true.', 'probabilities': [0.2, 0.7, 0.1]}],
           'provided_context': ['Some evidence.']}
    prompt = build_prompt(row, taxonomy)
    assert 'verifier_error' in prompt
    assert 'ambiguous_evidence' in prompt
    assert 'Precedence rules' in prompt
    assert '0.2' in prompt and 'Some evidence.' in prompt


def test_extraction_copy_covers_the_v2_queue_and_is_stamped():
    rows = copy_extraction()
    assert len(rows) == 171
    assert len({row['id'] for row in rows}) == 171
    assert all(row['taxonomy_version'] == 'v2' for row in rows)
    assert all('source' in row for row in rows)
