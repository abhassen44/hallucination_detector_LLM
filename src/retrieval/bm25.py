"""Sparse BM25 (Okapi) on scipy matrices.

``rank_bm25`` loops over every document in Python per query term, which is far too slow
for ~250k sentences; this vectorised version scores a query in a few milliseconds.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english",
                                          token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        self.W: sp.csc_matrix | None = None   # docs x vocab, BM25 term weights

    def fit(self, texts: list[str]) -> "BM25Index":
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)   # docs x vocab
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # w = idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)), applied on non-zeros only
        norm = self.k1 * (1 - self.b + self.b * dl / avgdl)
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data
        tf.data = idf[tf.indices] * data * (self.k1 + 1) / (data + norm[rows])
        self.W = tf.tocsc()
        return self

    def query_terms(self, query: str) -> np.ndarray:
        return self.vectorizer.transform([query]).indices

    def scores(self, query: str) -> np.ndarray:
        terms = self.query_terms(query)
        if len(terms) == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, terms].sum(axis=1)).ravel()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
