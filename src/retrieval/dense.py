"""Exact cosine retrieval with FAISS, or NumPy on macOS (OpenMP isolation)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    from src.verification.similarity import SimilarityModel

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# The macOS wheels of torch and faiss can load incompatible OpenMP runtimes.
# NumPy computes the same inner products without importing faiss into this process.
USE_NUMPY = sys.platform == 'darwin'
INDEX_FILENAME = 'dense.npy' if USE_NUMPY else 'dense.faiss'


class DenseIndex:
    def __init__(self, encoder: SimilarityModel | None = None):
        if encoder is None:
            from src.verification.similarity import SimilarityModel
            encoder = SimilarityModel()
        self.encoder = encoder
        self.index = None
        self.embeddings = None
        self.query_prefix = BGE_QUERY_PREFIX if 'bge' in self.encoder.model_name.lower() else ''

    @property
    def size(self):
        return len(self.embeddings) if USE_NUMPY else self.index.ntotal

    def fit(self, texts: list[str]) -> 'DenseIndex':
        emb = self.encoder.encode(texts, show_progress=True).astype(np.float32)
        if USE_NUMPY:
            self.embeddings = emb
        else:
            import faiss
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        return self

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        if k <= 0:
            raise ValueError('k must be positive')
        if not queries:
            return []
        q = self.encoder.encode([self.query_prefix + x for x in queries]).astype(np.float32)
        if self.size == 0:
            return [[] for _ in queries]
        k = min(k, self.size)
        if not USE_NUMPY:
            D, I = self.index.search(q, k)
            return [[(int(i), float(d)) for i, d in zip(ri, rd) if i >= 0] for ri, rd in zip(I, D)]
        results = []
        # Bound temporary score matrices for large evaluation batches.
        for start in range(0, len(q), 16):
            scores = q[start:start+16] @ self.embeddings.T
            for row in scores:
                top = np.argpartition(-row, k-1)[:k]
                top = top[np.argsort(-row[top])]
                results.append([(int(i), float(row[i])) for i in top])
        return results

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.search_batch([query], k)[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if USE_NUMPY:
            with path.open('wb') as f:
                np.save(f, self.embeddings, allow_pickle=False)
        else:
            import faiss
            faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, encoder: SimilarityModel | None = None) -> 'DenseIndex':
        obj = cls(encoder)
        if USE_NUMPY:
            obj.embeddings = np.load(path, mmap_mode='r', allow_pickle=False)
        else:
            import faiss
            obj.index = faiss.read_index(str(path))
        return obj
