import pytest

from scripts.v2.seed_v2_queues import review_key, seed_queue


def _row(identifier, condition=None, verdict=None, category=None, coding=''):
    row = {
        'id': identifier,
        'human_review': {'open_coding': coding, 'pass_fail_assessment': verdict,
                         'category': category},
    }
    if condition:
        row['condition'] = condition
    return row


def test_review_key_separates_repeated_verification_ids():
    assert review_key({'id': 'x', 'condition': 'bm25'}) != review_key(
        {'id': 'x', 'condition': 'dense'})
    assert review_key({'id': 'y'}) == ('y', None)


def test_seed_copies_verdict_by_id_and_condition():
    rows = [_row('x'), _row('x', 'bm25'), _row('x', 'dense')]
    pilot = [_row('x', 'bm25', 'Fail', 'retrieval_failure')]

    rows, result = seed_queue(rows, pilot)

    assert [row['human_review']['pass_fail_assessment'] for row in rows] == [None, 'Fail', None]
    assert rows[1]['human_review']['category'] == 'retrieval_failure'
    assert result['seeded'] == [{'id': 'x', 'condition': 'bm25', 'verdict': 'Fail',
                                 'category': 'retrieval_failure'}]
    assert result['skipped'] == []


def test_seed_preserves_target_fields_not_covered_by_the_pilot():
    row = _row('x')
    row['human_review']['fidelity'] = None
    row['quality_details'] = {'answer_word_count': 3}
    pilot = _row('x', verdict='Pass', coding='Faithful.')

    rows, _ = seed_queue([row], [pilot])

    assert rows[0]['human_review']['fidelity'] is None
    assert rows[0]['human_review']['open_coding'] == 'Faithful.'
    assert rows[0]['quality_details'] == {'answer_word_count': 3}
    assert rows[0]['review_meta'] == {}


def test_seed_skips_rows_without_a_verdict_and_records_why():
    pilot = [_row('x', verdict=None, category='ambiguous_evidence', coding='Needs a decision.')]

    rows, result = seed_queue([_row('x')], pilot)

    assert rows[0]['human_review']['pass_fail_assessment'] is None
    assert rows[0]['human_review']['category'] is None
    assert result['seeded'] == []
    assert result['skipped'] == [{'id': 'x', 'condition': None, 'reason': 'no Pass/Fail verdict',
                                  'had_open_coding': True, 'had_category': True}]


def test_seed_translates_a_renamed_legacy_category():
    pilot = [_row('x', 'bm25', 'Fail', 'nli_failure')]

    rows, result = seed_queue([_row('x', 'bm25')], pilot,
                              category_map={'nli_failure': 'verifier_error'})

    assert rows[0]['human_review']['category'] == 'verifier_error'
    assert result['migrated'] == [{'id': 'x', 'condition': 'bm25', 'from': 'nli_failure',
                                   'to': 'verifier_error'}]


def test_seed_keeps_categories_without_a_mapping():
    rows, result = seed_queue([_row('x', 'bm25')],
                              [_row('x', 'bm25', 'Fail', 'ambiguous_evidence')])
    assert rows[0]['human_review']['category'] == 'ambiguous_evidence'
    assert result['migrated'] == []


def test_seed_rejects_a_conflicting_verdict():
    row = _row('x', verdict='Pass')
    with pytest.raises(ValueError, match='conflicting verdict'):
        seed_queue([row], [_row('x', verdict='Fail')])


def test_seed_rejects_a_pilot_row_missing_from_the_queue():
    with pytest.raises(KeyError, match='not in the full queue'):
        seed_queue([_row('x')], [_row('y', verdict='Pass')])
