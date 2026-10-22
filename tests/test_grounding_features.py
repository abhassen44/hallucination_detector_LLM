import numpy as np
import pytest

from src.verification.features import FEATURE_NAMES
from src.verification.grounding import summarize_scores
from src.verification.grounding_features import (build_grounding_features,
                                                 feature_names, grounding_path,
                                                 variant_columns)


def _row(claim='Paris is in France.', nli_e=(.9, .2), nli_c=(.01, .7)):
    hits = [{'doc': i, 'page_id': f'P{i}', 'sent_id': 0, 'score': .1,
             'nli': [e, .2, c]} for i, (e, c) in enumerate(zip(nli_e, nli_c))]
    return {'id': 'x', 'claim': claim, 'label': 'SUPPORTED', 'rerank': hits}


def _base(rows, value=0.0):
    matrix = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    matrix[:, FEATURE_NAMES.index('nli_e_max')] = value
    matrix[:, FEATURE_NAMES.index('nli_c_max')] = value
    return matrix


def test_summarize_scores_handles_empty_and_ordered_values():
    empty = summarize_scores([])
    assert empty.maximum == 0.0 and empty.count_above_half == 0

    summary = summarize_scores([.1, .7, .9])
    assert summary.maximum == pytest.approx(.9)
    assert summary.top2_mean == pytest.approx(.8)
    assert summary.margin == pytest.approx(.2)
    assert summary.count_above_half == 2
    assert summary.count_below_twenty == 1


def test_grounding_features_join_by_claim_id():
    rows = [_row(), _row('Berlin is in Germany.', nli_e=(.1, .05), nli_c=(.1, .8))]
    base = _base(rows, value=.6)
    scores = {'mc': {'x': [.1, .9]}, 'hhem': {'x': [.2, .8]}}
    matrix, names = build_grounding_features(rows, scores, base)

    assert matrix.shape == (2, len(names))
    assert np.isfinite(matrix).all()
    index = {name: i for i, name in enumerate(names)}
    assert matrix[0, index['mc_max']] == pytest.approx(.9)
    assert matrix[0, index['mc_mean']] == pytest.approx(.5)
    assert matrix[0, index['mc_minus_nli_e']] == pytest.approx(.3)
    assert matrix[0, index['mc_at_nli_e']] == pytest.approx(.1)
    assert matrix[0, index['both_high']] == 1.0


def test_grounding_paths_map_public_names():
    assert grounding_path('mc', 'val').name == 'minicheck_val.jsonl'
    assert grounding_path('mc', 'train', 3000).name == 'minicheck_train3k.jsonl'
    assert grounding_path('hhem', 'train', 9000).name == 'hhem_train.jsonl'


def test_missing_grounding_id_is_rejected():
    with pytest.raises(KeyError):
        build_grounding_features([_row()], {'mc': {}}, _base([_row()]))


def test_variant_columns_are_unique_and_known():
    names = feature_names('mc') + feature_names('hhem')
    for variant in ('mc', 'hhem', 'compact', 'both'):
        columns = variant_columns(names, variant)
        assert len(columns) == len(set(columns))
        assert all(0 <= column < len(names) for column in columns)
    assert variant_columns(names, 'none') == []
    assert len(variant_columns(names, 'compact')) < len(names)
    with pytest.raises(ValueError):
        variant_columns(names, 'missing')
