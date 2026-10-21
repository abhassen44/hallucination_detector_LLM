"""Weighted Reciprocal Rank Fusion for the P6 retrieval study.

The v1 hybrid fuses BM25 and dense rankings with equal weight. This module
recomputes the fusion from the cached candidate lists so a BM25 weight can be
tuned on val without re-running retrieval.

Only the cached top-50 BM25/dense candidates are considered; the NLI probabilities
and corpus sentence ids are the cached ones. Gold labels are never read here.
"""
from __future__ import annotations

import numpy as np


def weighted_hits(row: dict, bm25_weight: float, rrf_k: int = 60,
                  top_k: int = 20) -> list[dict]:
    """Fuse cached BM25/dense rankings; returns hits with a weighted RRF score."""
    if not 0.0 <= bm25_weight <= 1.0:
        raise ValueError('bm25_weight must be in [0, 1]')
    scores: dict[int, float] = {}
    meta: dict[int, dict] = {}
    for method, weight in (('bm25', bm25_weight), ('dense', 1.0 - bm25_weight)):
        if weight == 0.0:
            continue
        for rank, hit in enumerate(row[method], start=1):
            doc = int(hit['doc'])
            scores[doc] = scores.get(doc, 0.0) + weight / (rrf_k + rank)
            meta.setdefault(doc, hit)
    # Stable sort mirrors ``src.retrieval.hybrid.rrf``: ties keep insertion order
    # (BM25 docs first, then dense-only docs).
    ordered = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
    return [{
        'doc': doc,
        'page_id': meta[doc]['page_id'],
        'sent_id': int(meta[doc]['sent_id']),
        'score': float(score),
        'nli': meta[doc].get('nli'),
    } for doc, score in ordered]


def with_weighted_hits(rows: list[dict], bm25_weight: float, rrf_k: int = 60,
                       top_k: int = 20) -> list[dict]:
    """Copy rows and attach the fused ranking under the ``weighted`` key."""
    output = []
    for row in rows:
        copied = dict(row)
        copied['weighted'] = weighted_hits(row, bm25_weight, rrf_k=rrf_k, top_k=top_k)
        output.append(copied)
    return output


def similarity_matrices(rows: list[dict], sim_model, top: int = 5) -> np.ndarray:
    """Cosine similarities for the weighted top-5, batched over the split."""
    claims = [row['claim'] for row in rows]
    claim_vectors = sim_model.encode(claims)
    evid_docs, evid_texts = [], []
    seen = set()
    for row in rows:
        for hit in row['weighted'][:top]:
            doc = hit['doc']
            if doc not in seen:
                seen.add(doc)
                evid_docs.append(doc)
                evid_texts.append(hit.get('text'))
    # Text lookup is supplied by the caller through a private key; fall back to
    # embedding the doc ids only if texts are missing.
    if any(text is None for text in evid_texts):
        raise ValueError('weighted hits need evidence text for similarity features')
    evidence_vectors = sim_model.encode(evid_texts) if evid_texts else np.zeros(
        (0, claim_vectors.shape[1]), dtype=np.float32)
    position = {doc: index for index, doc in enumerate(evid_docs)}
    out = np.zeros((len(rows), top), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, hit in enumerate(row['weighted'][:top]):
            out[i, j] = float(claim_vectors[i] @ evidence_vectors[position[hit['doc']]])
    return out


def attach_texts(rows: list[dict], corpus) -> None:
    """Add the corpus sentence text to each weighted hit, in place."""
    for row in rows:
        for hit in row['weighted']:
            hit['text'] = corpus.texts[hit['doc']]
