import numpy as np
import pytest

from src.retrieval import dense


class Encoder:
    model_name = 'test-encoder'
    vectors = {'a': [1, 0], 'b': [0, 1], 'c': [-1, 0]}

    def encode(self, texts, **kwargs):
        return np.array([self.vectors[t] for t in texts], dtype=np.float32)


def test_numpy_backend_matches_exact_inner_products_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(dense, 'USE_NUMPY', True)
    index = dense.DenseIndex(Encoder()).fit(['a', 'b', 'c'])
    assert index.size == 3
    assert index.search('a', 2) == [(0, 1.0), (1, 0.0)]
    assert index.search('b', 20)[0] == (1, 1.0)
    # Cross the internal query chunk boundary.
    results = index.search_batch(['c'] * 17, 1)
    assert results == [[(2, 1.0)]] * 17
    assert index.search_batch([], 2) == []
    with pytest.raises(ValueError):
        index.search('a', 0)
    path = tmp_path / 'dense.npy'
    index.save(path)
    loaded = dense.DenseIndex.load(path, Encoder())
    assert loaded.search('a', 2) == index.search('a', 2)
