import json

import pytest

from scripts.review import (BOOL_FIELDS, CATEGORIES, apply_review, boolean_options, category_options,
                            is_complete, load_queue, pending, queue_kind, review_interactively,
                            save_queue, summarise)
from src.datasets.schema import write_jsonl


def _row(identifier, complete=False):
    row = {'id': identifier,
           'human_review': {'open_coding': '', 'pass_fail_assessment': None,
                            'category': None}}
    if complete:
        row['human_review']['pass_fail_assessment'] = 'Pass'
    return row


def test_queue_round_trip_and_progress(tmp_path):
    path = tmp_path / 'queue.jsonl'
    write_jsonl(path, [_row('a'), _row('b', complete=True)])

    rows = load_queue(path)
    assert [row['id'] for row in pending(rows)] == ['a']
    assert is_complete(rows[1]) and not is_complete(rows[0])

    apply_review(rows[0], {'open_coding': 'Faithful but not atomic.',
                           'pass_fail_assessment': 'Fail', 'category': 'atomicity'})
    rows[0]['review_meta'] = {'suggestion_used': True}
    save_queue(path, rows)

    reloaded = load_queue(path)
    assert is_complete(reloaded[0]) and not pending(reloaded)
    assert (tmp_path / 'queue.jsonl.bak').exists()

    summary = summarise(reloaded)
    assert summary['completed'] == 2 and summary['pending'] == 0
    assert summary['pass_fail'] == {'Pass': 1, 'Fail': 1}
    assert summary['categories'] == {'atomicity': 1}
    assert summary['suggestion_used'] == 1


def test_require_verdict_keeps_open_coding_rows_pending():
    row = _row('a')
    row['human_review']['open_coding'] = 'Observation without a verdict.'
    assert is_complete(row)
    assert not is_complete(row, require_verdict=True)
    assert pending([row], require_verdict=True) == [row]
    assert summarise([row], require_verdict=True)['pending'] == 1


def test_completed_rows_are_never_part_of_the_pending_queue(tmp_path):
    path = tmp_path / 'queue.jsonl'
    write_jsonl(path, [_row('a', complete=True), _row('b')])
    rows = load_queue(path)
    pending_rows = pending(rows)
    assert len(pending_rows) == 1 and pending_rows[0]['id'] == 'b'


def test_unknown_review_field_is_rejected():
    with pytest.raises(KeyError):
        apply_review(_row('a'), {'invented_field': 'x'})


TAXONOMY = {
    'verification': {'categories': ['retrieval_failure', 'verifier_error']},
    'extraction': {'category_enabled': False, 'booleans_enabled': False},
}


def test_queue_kind_uses_the_condition_field():
    assert queue_kind([{'id': 'a'}, {'id': 'b'}]) == 'extraction'
    assert queue_kind([{'id': 'a'}, {'id': 'b', 'condition': 'bm25'}]) == 'verification'


def test_category_options_read_the_verification_taxonomy():
    assert category_options('verification', TAXONOMY) == ['retrieval_failure', 'verifier_error']


def test_category_options_disable_extraction_causes():
    assert category_options('extraction', TAXONOMY) == []


def test_category_options_fall_back_to_v1_categories():
    assert category_options('extraction', None) == CATEGORIES
    assert category_options('verification', {}) == CATEGORIES


def test_boolean_options_retire_extraction_diagnostics_only():
    assert boolean_options('extraction', TAXONOMY) == []
    assert boolean_options('verification', TAXONOMY) == list(BOOL_FIELDS)
    assert boolean_options('extraction', None) == list(BOOL_FIELDS)


def test_extraction_under_taxonomy_asks_only_for_verdict(tmp_path, monkeypatch):
    path = tmp_path / 'queue.jsonl'
    row = {'id': 'x', 'human_review': {'open_coding': '', 'pass_fail_assessment': None,
                                       'category': None, 'fidelity': None}}
    write_jsonl(path, [row])
    answers = iter(['', 'p'])
    monkeypatch.setattr('builtins.input', lambda *args: next(answers))
    monkeypatch.setattr('scripts.review._show', lambda *args: None)

    rows = review_interactively(path, None, None, 'test', taxonomy=TAXONOMY)

    assert rows[0]['human_review']['pass_fail_assessment'] == 'Pass'
    assert rows[0]['human_review']['category'] is None
    assert rows[0]['human_review']['fidelity'] is None


def test_verification_under_taxonomy_asks_for_a_category(tmp_path, monkeypatch):
    path = tmp_path / 'queue.jsonl'
    row = {'id': 'x', 'condition': 'bm25', 'human_review': {'open_coding': '',
                                                            'pass_fail_assessment': None,
                                                            'category': None}}
    write_jsonl(path, [row])
    answers = iter(['', 'f', '2'])
    monkeypatch.setattr('builtins.input', lambda *args: next(answers))
    monkeypatch.setattr('scripts.review._show', lambda *args: None)

    rows = review_interactively(path, None, None, 'test', taxonomy=TAXONOMY)

    assert rows[0]['human_review']['pass_fail_assessment'] == 'Fail'
    assert rows[0]['human_review']['category'] == 'verifier_error'


def test_saving_is_atomic_and_preserves_other_data(tmp_path):
    path = tmp_path / 'queue.jsonl'
    original = _row('a')
    original['quality_details'] = {'answer_word_count': 3}
    write_jsonl(path, [original])
    rows = load_queue(path)
    apply_review(rows[0], {'pass_fail_assessment': 'Pass'})
    save_queue(path, rows, backup=False)
    payload = json.loads(path.read_text(encoding='utf-8').strip())
    assert payload['quality_details'] == {'answer_word_count': 3}
    assert payload['human_review']['pass_fail_assessment'] == 'Pass'
