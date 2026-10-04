"""Dense retrieval: bge-small embeddings + exact FAISS inner-product index."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.verification.similarity import SimilarityModel

# bge-*-v1.5 recommends this instruction for short queries -> passages
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseIndex:
    def __init__(self, encoder: SimilarityModel | None = None):
        self.encoder = encoder or SimilarityModel()
        self.index: faiss.Index | None = None
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in self.encoder.model_name.lower() else ""

    def fit(self, texts: list[str]) -> "DenseIndex":
        emb = self.encoder.encode(texts, show_progress=True).astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])   # cosine (embeddings are normalised)
        self.index.add(emb)
        return self

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = self.encoder.encode([self.query_prefix + x for x in queries]).astype(np.float32)
        D, I = self.index.search(q, k)
        return [[(int(i), float(d)) for i, d in zip(ri, rd) if i >= 0] for ri, rd in zip(I, D)]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.search_batch([query], k)[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, encoder: SimilarityModel | None = None) -> "DenseIndex":
        obj = cls(encoder)
        obj.index = faiss.read_index(str(path))
        return obj
