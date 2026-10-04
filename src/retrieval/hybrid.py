"""Hybrid retrieval via Reciprocal Rank Fusion + a cached `Retriever` facade.

RRF(d) = sum_m 1 / (rrf_k + rank_m(d)), rank starting at 1. Score-scale free, so BM25
and cosine scores never need calibration against each other.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from src.retrieval.bm25 import BM25Index
from src.retrieval.corpus import SentenceCorpus
from src.retrieval.dense import DenseIndex
from src.utils.config import PROJECT_ROOT, load_config

Hits = list[tuple[int, float]]
METHODS = ("bm25", "dense", "hybrid")


def rrf(rankings: list[Hits], k: int, rrf_k: int = 60) -> Hits:
    score: dict[int, float] = defaultdict(float)
    for hits in rankings:
        for rank, (doc, _) in enumerate(hits, start=1):
            score[doc] += 1.0 / (rrf_k + rank)
    return sorted(score.items(), key=lambda x: -x[1])[:k]


class Retriever:
    """Builds (once) and loads BM25 + FAISS indexes over the FEVER sentence corpus."""

    def __init__(self, cfg: dict | None = None, index_dir: Path | None = None, rebuild: bool = False):
        cfg = cfg or load_config()
        self.rcfg = cfg["retrieval"]
        self.index_dir = index_dir or PROJECT_ROOT / cfg["paths"]["processed"] / "indexes"
        self.corpus = SentenceCorpus.from_fever()

        bm25_p, dense_p = self.index_dir / "bm25.pkl", self.index_dir / "dense.faiss"
        if bm25_p.exists() and not rebuild:
            self.bm25 = BM25Index.load(bm25_p)
        else:
            t = time.time()
            self.bm25 = BM25Index().fit(self.corpus.texts)
            self.bm25.save(bm25_p)
            print(f"[retrieval] BM25 built in {time.time() - t:.0f}s")

        if dense_p.exists() and not rebuild:
            self.dense = DenseIndex.load(dense_p)
        else:
            t = time.time()
            self.dense = DenseIndex().fit(self.corpus.texts)
            self.dense.save(dense_p)
            print(f"[retrieval] FAISS built in {time.time() - t:.0f}s")

        n = self.dense.index.ntotal
        if n != len(self.corpus) or self.bm25.W.shape[0] != len(self.corpus):
            raise RuntimeError("Index/corpus size mismatch - rerun with rebuild=True")

    def search_all(self, queries: list[str], k: int | None = None) -> dict[str, list[Hits]]:
        """Returns {method: [hits per query]} for bm25, dense and hybrid (RRF)."""
        k = k or self.rcfg["top_k"]
        bm = [self.bm25.search(q, self.rcfg["bm25_k"]) for q in queries]
        de = self.dense.search_batch(queries, self.rcfg["dense_k"])
        hy = [rrf([b, d], max(k, self.rcfg["top_k"]), self.rcfg["rrf_k"]) for b, d in zip(bm, de)]
        return {"bm25": bm, "dense": de, "hybrid": hy}

    def search(self, query: str, method: str = "hybrid", k: int | None = None) -> list[dict]:
        k = k or self.rcfg["top_k"]
        hits = self.search_all([query], k)[method][0][:k]
        return [{"text": self.corpus.texts[i], "page_id": self.corpus.keys[i][0],
                 "sent_id": self.corpus.keys[i][1], "score": s} for i, s in hits]
