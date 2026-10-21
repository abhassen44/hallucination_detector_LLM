"""Deterministic claim features for the Stage 5 learned combiner.

Everything is computed from the Stage 3 retrieval cache: the top-5 evidence
sentences per retrieval method, their BM25 / dense / RRF scores and the cached
NLI probabilities. Gold labels are never read while building features.

Similarity features use the pretrained BGE encoder only (no fitting).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.datasets.schema import read_jsonl
from src.utils.config import PROJECT_ROOT
from src.verification.detector import aggregate_probabilities

TOP_K = 5
RETRIEVAL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'retrieval'

FEATURE_GROUPS = {
    'nli': [
        'nli_e_max', 'nli_c_max', 'nli_n_max',
        'nli_e_mean', 'nli_c_mean',
        'nli_e_top2', 'nli_c_top2',
        'nli_margin_max',
        'nli_e_count_70', 'nli_c_count_95', 'nli_n_count_50',
        'nli_e_rank', 'nli_c_rank',
        'nli_agg_c', 'nli_agg_n',
    ],
    'retrieval': [
        'ret_top1_score', 'ret_top2_margin', 'ret_mean_score', 'ret_score_std',
        'ret_distinct_pages', 'ret_top1_entail_agree', 'ret_cross_page_conflict',
        'ret_top1_page_share',
    ],
    'similarity': ['sim_top1', 'sim_mean', 'sim_max'],
    'claim': ['claim_words', 'claim_has_digit', 'claim_has_year', 'claim_has_negation'],
}

FEATURE_NAMES = [name for group in FEATURE_GROUPS.values() for name in group]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

_YEAR = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')
_NEGATION = re.compile(r"\b(not|no|never|none|isn't|wasn't|aren't|weren't|didn't|doesn't|don't)\b", re.I)


def load_split(split: str, path: Path | None = None) -> list[dict]:
    path = path or RETRIEVAL_DIR / f'fever_{split}.jsonl'
    return list(read_jsonl(path))


def similarity_matrix(rows: list[dict], method: str, corpus, model, top: int = TOP_K) -> np.ndarray:
    """Cosine similarity between each claim and each of its top-k evidence sentences."""
    evidence: dict[str, int] = {}
    flat: list[str] = []
    for row in rows:
        for hit in row[method][:top]:
            text = corpus.texts[hit['doc']]
            if text not in evidence:
                evidence[text] = len(flat)
                flat.append(text)
    claim_vectors = model.encode([r['claim'] for r in rows])
    evidence_vectors = model.encode(flat) if flat else np.zeros((0, claim_vectors.shape[1]), np.float32)
    out = np.zeros((len(rows), top), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, hit in enumerate(row[method][:top]):
            out[i, j] = float(claim_vectors[i] @ evidence_vectors[evidence[corpus.texts[hit['doc']]]])
    return out


def _rank(values: np.ndarray, index: int) -> int:
    order = np.argsort(-values)
    return int(np.where(order == index)[0][0]) + 1


def row_features(row: dict, method: str, similarities: np.ndarray, top: int = TOP_K) -> np.ndarray:
    hits = row[method][:top]
    n = len(hits)
    vector = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if not n:
        return vector

    probs = np.asarray([h['nli'] for h in hits], dtype=np.float32)
    scores = np.asarray([float(h['score']) for h in hits], dtype=np.float32)
    pages = [h['page_id'] for h in hits]
    e, c, neu = probs[:, 0], probs[:, 2], probs[:, 1]

    aggregated = aggregate_probabilities(probs)
    e_idx, c_idx = int(np.argmax(e)), int(np.argmax(c))

    values = {
        'nli_e_max': e.max(), 'nli_c_max': c.max(), 'nli_n_max': neu.max(),
        'nli_e_mean': e.mean(), 'nli_c_mean': c.mean(),
        'nli_e_top2': np.sort(e)[-2:].mean(), 'nli_c_top2': np.sort(c)[-2:].mean(),
        'nli_margin_max': float(e.max() - c.max()),
        'nli_e_count_70': float((e >= .7).sum()), 'nli_c_count_95': float((c >= .95).sum()),
        'nli_n_count_50': float((neu >= .5).sum()),
        'nli_e_rank': float(_rank(e, e_idx)), 'nli_c_rank': float(_rank(c, c_idx)),
        'nli_agg_c': aggregated[2], 'nli_agg_n': aggregated[1],

        'ret_top1_score': scores[0],
        'ret_top2_margin': float(scores[0] - scores[1]) if n > 1 else 0.,
        'ret_mean_score': scores.mean(), 'ret_score_std': scores.std(),
        'ret_distinct_pages': float(len(set(pages))),
        'ret_top1_entail_agree': float(pages[e_idx] == pages[0]),
        'ret_cross_page_conflict': float(pages[e_idx] != pages[c_idx] and
                                         (e >= .7).any() and (c >= .95).any()),
        'ret_top1_page_share': float(sum(p == pages[0] for p in pages) / n),

        'sim_top1': similarities[0] if len(similarities) else 0.,
        'sim_mean': similarities.mean() if len(similarities) else 0.,
        'sim_max': similarities.max() if len(similarities) else 0.,

        'claim_words': float(len(row['claim'].split())),
        'claim_has_digit': float(any(ch.isdigit() for ch in row['claim'])),
        'claim_has_year': float(bool(_YEAR.search(row['claim']))),
        'claim_has_negation': float(bool(_NEGATION.search(row['claim']))),
    }
    for name, value in values.items():
        vector[FEATURE_INDEX[name]] = value
    return vector


def build_matrix(rows: list[dict], method: str, similarities: np.ndarray,
                 top: int = TOP_K) -> np.ndarray:
    if len(similarities) != len(rows):
        raise ValueError('Similarity rows do not match claims')
    matrix = np.stack([row_features(r, method, similarities[i], top) for i, r in enumerate(rows)])
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite feature values')
    return matrix


def labels(rows: list[dict]) -> list[str]:
    return [row['label'] for row in rows]
