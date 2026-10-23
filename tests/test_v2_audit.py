from scripts.v2.make_v2_audit import allocate, sample_rows
from scripts.v2.score_v2_audit import _category, _verdict


def _queue(identifier, condition=None):
    row = {'id': identifier, 'question': f'Q{identifier}', 'answer': 'A',
           'claims': [f'claim {identifier}'], 'provided_context': ['evidence']}
    if condition:
        row['condition'] = condition
    return row


def _label(identifier, condition=None, verdict='Pass', category=None, human=False):
    return {'id': identifier, 'condition': condition, 'verdict': verdict,
            'category': category, 'human_reviewed': human}


def test_allocate_respects_minimum_and_total():
    allocation = allocate({'a': 50, 'b': 10, 'c': 1}, 20, minimum=2)
    assert sum(allocation.values()) == 20
    assert allocation['c'] == 1
    assert allocation['a'] > allocation['b']


def test_sample_rows_excludes_human_labels_and_is_deterministic():
    import random

    queue = [_queue(f'x{i}') for i in range(30)]
    labels = [_label(f'x{i}', verdict='Pass' if i < 15 else 'Fail') for i in range(30)]
    labels[0] = _label('x0', human=True)

    first = sample_rows(queue, labels, 'extraction', random.Random(42))
    second = sample_rows(queue, labels, 'extraction', random.Random(42))

    assert [row['id'] for row in first] == [row['id'] for row in second]
    assert len(first) == 20
    assert 'x0' not in {row['id'] for row in first}


def test_sample_rows_stratifies_verification_categories():
    import random

    queue = [_queue(f'v{i}', 'bm25') for i in range(20)]
    labels = [_label(f'v{i}', 'bm25', verdict='Fail',
                     category='retrieval_failure' if i < 15 else 'verifier_error')
              for i in range(20)]

    rows = sample_rows(queue, labels, 'verification', random.Random(42))

    assert len(rows) == 20
    assert sum(int(row['id'][1:]) < 15 for row in rows) == 15
    assert sum(int(row['id'][1:]) >= 15 for row in rows) == 5


def test_audit_value_parsers_are_forgiving_but_not_inventive():
    assert _verdict('p') == 'Pass'
    assert _verdict(' FAIL ') == 'Fail'
    assert _verdict('') is None
    assert _verdict('maybe') is None
    assert _category('Verifier_Error', {'verifier_error'}) == 'verifier_error'
    assert _category('nli_failure', {'verifier_error'}) is None
