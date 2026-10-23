import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.shared.review.make_pilot import (
    EXTRACTION_REVIEW, _prepare, allocate, sample_extraction, sample_verification,
)
from scripts.shared.review.pilot_agreement import gate, queue_kind, score, timing
from scripts.shared.review.review import load_suggestions, suggestion_key


def test_allocate_respects_strata_and_total():
    counts = {'retrieval_failure': 56, 'nli_failure': 38, 'entity_confusion': 4,
              'multi_hop_reasoning': 4, 'numerical_reasoning': 1}
    allocation = allocate(counts, 20)
    assert sum(allocation.values()) == 20
    assert all(allocation[k] >= 1 for k in counts)
    assert all(allocation[k] <= counts[k] for k in counts)


def test_allocate_smaller_than_strata_picks_largest():
    allocation = allocate({'a': 5, 'b': 3, 'c': 1}, 2)
    assert allocation == {'a': 1, 'b': 1, 'c': 0}


def test_allocate_never_exceeds_available_rows():
    allocation = allocate({'a': 1, 'b': 1}, 5)
    assert sum(allocation.values()) == 2


def test_extraction_pilot_is_balanced_unique_and_deterministic():
    first, selection = sample_extraction(random.Random(42), 20)
    second, _ = sample_extraction(random.Random(42), 20)
    assert [row['id'] for row in first] == [row['id'] for row in second]
    assert len(first) == 20
    assert len({row['id'] for row in first}) == 20
    assert selection['achieved'] == {'Pass': 10, 'Fail': 10}


def test_verification_pilot_has_one_row_per_id_and_covers_categories():
    rows, selection = sample_verification(random.Random(42), 20)
    assert len(rows) == 20
    assert len({row['id'] for row in rows}) == 20
    assert all(row.get('condition') for row in rows)
    assert set(selection['achieved']) == {'retrieval_failure', 'nli_failure',
                                          'entity_confusion', 'multi_hop_reasoning',
                                          'numerical_reasoning'}


def test_prepare_strips_meta_and_clears_review():
    row = {'id': 'x', 'human_review': {'pass_fail_assessment': 'Pass', 'category': 'nli_failure'},
           'review_meta': {'reviewer': 'old'}, 'keep': True}
    prepared = _prepare([row], EXTRACTION_REVIEW)[0]
    assert 'review_meta' not in prepared
    assert prepared['keep'] is True
    assert prepared['human_review'] == EXTRACTION_REVIEW
    assert row['human_review']['pass_fail_assessment'] == 'Pass'  # source untouched


def test_generated_queues_have_no_proposal_leakage():
    root = Path(__file__).resolve().parents[1] / 'experiments/pilot_v1'
    for name in ('extraction_queue.jsonl', 'verification_queue.jsonl'):
        path = root / name
        if not path.exists():
            pytest.skip(f'{path} not generated')
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            for leaked in ('rationale', 'verdict', 'confidence', 'provisional_llm_assisted',
                           'suggestion'):
                assert leaked not in row
            if 'review_meta' in row:
                # The pilot is annotated; review metadata is expected, proposals are not.
                assert {'reviewer', 'reviewed_utc'} <= set(row['review_meta'])


def test_suggestion_key_separates_repeated_ids():
    a = {'id': 'x', 'condition': 'bm25_extracted_claims'}
    b = {'id': 'x', 'condition': 'hybrid_extracted_claims'}
    assert suggestion_key(a) != suggestion_key(b)
    assert suggestion_key({'id': 'y'}) == ('y', None)


def test_load_suggestions_keeps_both_conditions(tmp_path):
    path = tmp_path / 'suggestions.jsonl'
    rows = [{'id': 'x', 'condition': 'bm25', 'valid': True, 'category': 'nli_failure'},
            {'id': 'x', 'condition': 'dense', 'valid': True, 'category': 'retrieval_failure'}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    suggestions = load_suggestions(path)
    assert suggestions[('x', 'bm25')]['category'] == 'nli_failure'
    assert suggestions[('x', 'dense')]['category'] == 'retrieval_failure'


def _row(i, verdict=None, category=None, condition=None, minutes=0):
    stamp = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes)).isoformat()
    row = {'id': f'x{i}', 'human_review': {'open_coding': '', 'pass_fail_assessment': verdict,
                                           'category': category},
           'review_meta': {'reviewer': 'test', 'reviewed_utc': stamp}}
    if condition:
        row['condition'] = condition
    return row


def _proposal(i, verdict=None, category=None, condition=None):
    row = {'id': f'x{i}', 'valid': True, 'queue': 'verification' if condition else 'extraction'}
    if verdict:
        row['verdict'] = verdict
    if category:
        row['category'] = category
    if condition:
        row['condition'] = condition
    return row


def test_score_extraction_reports_kappa_and_disagreements(tmp_path):
    queue = tmp_path / 'extraction_queue.jsonl'
    provisional = tmp_path / 'provisional_extraction.jsonl'
    rows = [_row(1, 'Pass', None), _row(2, 'Pass', None), _row(3, 'Fail', 'claim_extraction_failure'),
            _row(4, 'Pass', None, minutes=1)]
    proposals = [_proposal(1, 'Pass', 'claim_extraction_failure'), _proposal(2, 'Pass'),
                 _proposal(3, 'Fail', 'claim_extraction_failure'),
                 _proposal(4, 'Fail', 'claim_extraction_failure')]
    queue.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    provisional.write_text(''.join(json.dumps(p) + '\n' for p in proposals), encoding='utf-8')

    result = score(queue, provisional)
    assert result['kind'] == 'extraction'
    assert result['n_completed'] == 4
    assert result['pass_fail']['agreement'] == 0.75
    assert result['category']['agreement'] == 1.0
    assert result['pass_fail']['kappa'] is not None


def test_score_verification_matches_on_id_and_condition(tmp_path):
    queue = tmp_path / 'verification_queue.jsonl'
    provisional = tmp_path / 'provisional_verification.jsonl'
    rows = [_row(1, 'Fail', 'nli_failure', condition='bm25'),
            _row(2, 'Fail', 'entity_confusion', condition='dense')]
    proposals = [_proposal(1, category='retrieval_failure', condition='dense'),  # wrong condition
                 _proposal(2, category='entity_confusion', condition='dense')]
    queue.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    provisional.write_text(''.join(json.dumps(p) + '\n' for p in proposals), encoding='utf-8')

    result = score(queue, provisional)
    assert result['kind'] == 'verification'
    assert result['unmatched'] == ['x1']
    assert result['category']['agreement'] == 1.0


def test_timing_excludes_long_breaks():
    rows = [_row(1, 'Pass', minutes=0), _row(2, 'Pass', minutes=1),
            _row(3, 'Pass', minutes=200)]
    result = timing(rows)
    assert result['active_cases'] == 1
    assert result['seconds_per_case'] == 60.0
    assert result['excluded_breaks'] == 1


def test_score_reports_kappa_sensitivity_for_missing_verdicts(tmp_path):
    queue = tmp_path / 'extraction_queue.jsonl'
    provisional = tmp_path / 'provisional_extraction.jsonl'
    rows = [_row(1, None, 'claim_extraction_failure'), _row(2, 'Pass', None),
            _row(3, 'Fail', None), _row(4, 'Pass', None)]
    proposals = [_proposal(1, 'Pass', 'claim_extraction_failure'), _proposal(2, 'Pass'),
                 _proposal(3, 'Fail'), _proposal(4, 'Fail')]
    queue.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    provisional.write_text(''.join(json.dumps(p) + '\n' for p in proposals), encoding='utf-8')

    result = score(queue, provisional)
    sensitivity = result['pass_fail']['sensitivity']
    assert result['n_completed'] == 4
    assert sensitivity['unlabeled'] == ['x1']
    assert sensitivity['assignments'] == 2
    assert sensitivity['min_kappa'] <= sensitivity['max_kappa']
    assert sensitivity['all_meet_min'] is False


def test_gate_allow_incomplete_requires_worst_case():
    extraction = {'n_completed': 19, 'n_total': 20,
                  'pass_fail': {'kappa': 0.88,
                                'sensitivity': {'min_kappa': 0.60, 'max_kappa': 0.90,
                                                'all_meet_min': True, 'unlabeled': ['x']}}}
    verification = {'n_completed': 20, 'n_total': 20,
                    'category': {'agreement': 0.80}}
    assert gate(extraction, verification)['status'] == 'incomplete'

    verdict = gate(extraction, verification, allow_incomplete=True)
    assert verdict['status'] == 'pass_with_deviation'
    assert verdict['checks']['kappa_pass_fail']['conservative_missing_verdicts'] == 0.60
    assert verdict['sensitivity']['min_kappa'] == 0.60

    extraction['pass_fail']['sensitivity']['min_kappa'] = 0.55
    assert gate(extraction, verification, allow_incomplete=True)['status'] == 'fail'


def test_gate_pass_fail_and_incomplete():
    extraction = {'n_completed': 20, 'n_total': 20,
                  'pass_fail': {'kappa': 0.72}}
    verification = {'n_completed': 20, 'n_total': 20,
                    'category': {'agreement': 0.65}}
    assert gate(extraction, verification)['status'] == 'pass'

    extraction['pass_fail']['kappa'] = 0.41
    verdict = gate(extraction, verification)
    assert verdict['status'] == 'fail'
    assert 'revise' in verdict['decision']

    extraction['n_completed'] = 19
    assert gate(extraction, verification)['status'] == 'incomplete'
    assert gate(None, verification)['status'] == 'incomplete'


def test_queue_kind_detects_condition():
    assert queue_kind([{'id': 'a'}]) == 'extraction'
    assert queue_kind([{'id': 'a', 'condition': 'bm25'}]) == 'verification'
