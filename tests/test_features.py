import numpy as np
import pytest

from src.verification.features import (FEATURE_GROUPS, FEATURE_NAMES, build_matrix,
                                       row_features)


def _hit(entail, neutral, contradiction, score, page='P'):
    return {'doc': 0, 'page_id': page, 'sent_id': 0, 'score': score,
            'nli': [entail, neutral, contradiction]}


def _row(claim='Paris is in France.'):
    return {'id': 'x', 'claim': claim, 'label': 'SUPPORTED',
            'hybrid': [_hit(.9, .09, .01, .03), _hit(.2, .7, .1, .02),
                       _hit(.1, .3, .6, .01), _hit(.05, .9, .05, .009),
                       _hit(.01, .09, .9, .001)]}


def test_feature_vector_is_complete_and_finite():
    vector = row_features(_row(), 'hybrid', np.array([.8, .5, .4, .3, .2]))
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    values = dict(zip(FEATURE_NAMES, vector))
    assert values['nli_e_max'] == pytest.approx(.9)
    assert values['nli_c_max'] == pytest.approx(.9)
    assert values['nli_e_count_70'] == 1
    assert values['nli_c_count_95'] == 0
    assert values['ret_top1_score'] == pytest.approx(.03)
    assert values['ret_distinct_pages'] == 1
    assert values['sim_top1'] == pytest.approx(.8)
    assert values['claim_words'] == 4


def test_empty_evidence_yields_zeros_and_no_error():
    row = {'id': 'x', 'claim': 'claim', 'label': 'INSUFFICIENT', 'hybrid': []}
    vector = row_features(row, 'hybrid', np.zeros(0, dtype=np.float32))
    assert not vector.any()


def test_all_groups_have_unique_feature_names():
    names = [name for group in FEATURE_GROUPS.values() for name in group]
    assert names == FEATURE_NAMES
    assert len(set(names)) == len(names)


def test_matrix_shape_matches_rows():
    rows = [_row(), _row('Berlin is in Germany.')]
    matrix = build_matrix(rows, 'hybrid', np.ones((2, 5), dtype=np.float32))
    assert matrix.shape == (2, len(FEATURE_NAMES))


def test_similarity_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_matrix([_row()], 'hybrid', np.ones((2, 5), dtype=np.float32))
