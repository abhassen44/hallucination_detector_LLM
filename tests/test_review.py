import json

import pytest

from scripts.review import (apply_review, is_complete, load_queue, pending, save_queue,
                            summarise)
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


def test_completed_rows_are_never_part_of_the_pending_queue(tmp_path):
    path = tmp_path / 'queue.jsonl'
    write_jsonl(path, [_row('a', complete=True), _row('b')])
    rows = load_queue(path)
    pending_rows = pending(rows)
    assert len(pending_rows) == 1 and pending_rows[0]['id'] == 'b'


def test_unknown_review_field_is_rejected():
    with pytest.raises(KeyError):
        apply_review(_row('a'), {'invented_field': 'x'})


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
