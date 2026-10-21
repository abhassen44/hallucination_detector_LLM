from src.evaluation.weighted_rrf import weighted_hits, with_weighted_hits


def _hit(doc, score=1.0, page='p', sent=0):
    return {'doc': doc, 'score': score, 'page_id': page, 'sent_id': sent,
            'nli': [0.1, 0.2, 0.7]}


def _row():
    return {'bm25': [_hit(1), _hit(2), _hit(3)],
            'dense': [_hit(2), _hit(4), _hit(1)]}


def test_zero_bm25_weight_returns_dense_order():
    hits = weighted_hits(_row(), 0.0)
    assert [hit['doc'] for hit in hits] == [2, 4, 1]


def test_unit_bm25_weight_returns_bm25_order():
    hits = weighted_hits(_row(), 1.0)
    assert [hit['doc'] for hit in hits] == [1, 2, 3]


def test_equal_weight_fuses_both_rankings():
    hits = weighted_hits(_row(), 0.5)
    # Doc 2 is rank 2 in BM25 and rank 1 in dense, so it wins the fusion.
    assert hits[0]['doc'] == 2
    assert {hit['doc'] for hit in hits} == {1, 2, 3, 4}


def test_hits_carry_cached_nli_and_page_ids():
    hits = weighted_hits(_row(), 0.5)
    assert all(hit['nli'] == [0.1, 0.2, 0.7] for hit in hits)
    assert all('page_id' in hit and 'sent_id' in hit for hit in hits)


def test_invalid_weight_is_rejected():
    try:
        weighted_hits(_row(), 1.5)
    except ValueError as exc:
        assert 'bm25_weight' in str(exc)
    else:
        raise AssertionError('expected ValueError')


def test_with_weighted_hits_does_not_mutate_the_input():
    rows = [_row()]
    output = with_weighted_hits(rows, 0.5)
    assert 'weighted' in output[0]
    assert 'weighted' not in rows[0]
