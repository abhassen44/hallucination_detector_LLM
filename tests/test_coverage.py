import numpy as np

from src.evaluation.coverage import (apply_flag, gold_evidence, select_threshold,
                                     strict_failure_labels)


class _Example:
    def __init__(self, identifier, evidence_sets):
        self.id = identifier
        self.meta = {'evidence_sets': evidence_sets}


def _row(identifier, label, hits, method='hybrid'):
    return {'id': identifier, 'label': label,
            method: [{'page_id': page, 'sent_id': sent} for page, sent in hits]}


def test_gold_evidence_normalises_page_sentence_pairs():
    examples = [_Example('x', [[['Page_A', 1], ['Page_B', 2]]])]
    assert gold_evidence(examples) == {'x': [{('Page_A', 1), ('Page_B', 2)}]}


def test_strict_failure_requires_the_complete_gold_set():
    gold = {'x': [{('Page_A', 1), ('Page_B', 2)}], 'y': [{('Page_C', 0)}],
            'z': [{('Page_D', 0)}]}
    rows = [_row('x', 'SUPPORTED', [('Page_A', 1)]),          # partial set -> failure
            _row('y', 'CONTRADICTED', [('Page_C', 0)]),        # complete -> success
            _row('z', 'INSUFFICIENT', [])]                     # excluded
    labels, kept = strict_failure_labels(rows, gold)

    assert labels.tolist() == [1, 0]
    assert [row['id'] for row in kept] == ['x', 'y']


def test_rows_without_gold_evidence_are_excluded():
    rows = [_row('x', 'SUPPORTED', [('Page_A', 0)])]
    labels, kept = strict_failure_labels(rows, {})
    assert labels.size == 0 and kept == []


def test_threshold_selection_respects_the_flag_budget():
    rng = np.random.default_rng(42)
    scores = rng.random(500)
    failures = (scores > 0.8).astype(int)
    errors = (scores > 0.7).astype(int)

    chosen = select_threshold(scores, failures, errors, n_bootstrap=200)

    assert 0.05 <= chosen['flag_rate'] <= 0.25
    assert chosen['error_enrichment'] >= 1.0
    assert chosen['error_enrichment_lower_95'] is not None


def test_apply_flag_uses_the_saved_parameters():
    parameters = {'mean': [0.0, 0.0], 'scale': [1.0, 1.0],
                  'coef': [1.0, 0.0], 'intercept': 0.0}
    scores = apply_flag(parameters, np.array([[0.0, 9.0], [2.0, 9.0]]))
    assert np.allclose(scores, [0.5, 1.0 / (1.0 + np.exp(-2.0))])
