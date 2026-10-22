"""Feature construction for the v5 pretrained grounding scores.

The scorers in :mod:`src.verification.grounding` produce one support probability
per (claim, evidence sentence) pair. This module turns those scores into a small,
deterministic feature block that can be joined to the v4 claim features.

Only cached scores and the v4 NLI/retrieval features are used. Gold labels are
never read here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.config import PROJECT_ROOT
from src.verification.features import FEATURE_INDEX, TOP_K
from src.verification.grounding import summarize_scores

GROUNDING_DIR = PROJECT_ROOT / 'data' / 'processed' / 'grounding'
SCORE_FEATURES = (
    'max', 'mean', 'top2', 'margin', 'minimum', 'std',
    'count_ge_50', 'count_lt_20',
)
INTERACTION_FEATURES = (
    'minus_nli_e', 'times_nli_e', 'minus_nli_c', 'times_nli_c',
    'at_nli_e', 'at_top1', 'minus_other',
)
PAIR_FEATURES = ('both_high', 'both_low')


def load_cache(path: Path) -> dict[str, list[float]]:
    """Load a grounding JSONL cache into ``id -> scores``."""
    scores: dict[str, list[float]] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[row['id']] = [float(value) for value in row['scores']]
    return scores


def grounding_path(model: str, split: str, train_size: int = 9000) -> Path:
    filename = {'mc': 'minicheck', 'hhem': 'hhem'}.get(model, model)
    if split == 'train' and train_size == 3000:
        return GROUNDING_DIR / f'{filename}_train3k.jsonl'
    return GROUNDING_DIR / f'{filename}_{split}.jsonl'


def feature_names(model: str) -> list[str]:
    return ([f'{model}_{name}' for name in SCORE_FEATURES]
            + [f'{model}_{name}' for name in INTERACTION_FEATURES])


def _score_block(scores: list[float], model: str) -> tuple[dict[str, float], np.ndarray]:
    values = np.asarray(scores, dtype=np.float32)
    summary = summarize_scores(values)
    std = float(values.std()) if values.size else 0.0
    block = {
        f'{model}_max': summary.maximum,
        f'{model}_mean': summary.mean,
        f'{model}_top2': summary.top2_mean,
        f'{model}_margin': summary.margin,
        f'{model}_minimum': float(values.min()) if values.size else 0.0,
        f'{model}_std': std,
        f'{model}_count_ge_50': float(summary.count_above_half),
        f'{model}_count_lt_20': float(summary.count_below_twenty),
    }
    return block, values


def build_grounding_features(rows: list[dict],
                             scores_by_model: dict[str, dict[str, list[float]]],
                             base_matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build the grounding feature block for rows in the v4 feature order.

    ``rows`` must contain the ``rerank`` top-5 hits with the cached NLI
    probabilities. ``scores_by_model`` maps a model name to its id-keyed cache.
    """
    if len(rows) != len(base_matrix):
        raise ValueError('rows and base_matrix must have the same length')
    models = list(scores_by_model)
    columns: list[str] = []
    for model in models:
        columns.extend(feature_names(model))
    if {'mc', 'hhem'} <= set(models):
        columns.extend(PAIR_FEATURES)
    index = {name: i for i, name in enumerate(columns)}
    matrix = np.zeros((len(rows), len(columns)), dtype=np.float32)

    for row_number, row in enumerate(rows):
        blocks = {}
        score_arrays = {}
        for model in models:
            scores = scores_by_model[model].get(row['id'])
            if scores is None:
                raise KeyError(f'missing {model} grounding scores for {row["id"]}')
            block, values = _score_block(scores, model)
            blocks[model] = block
            score_arrays[model] = values
        nli_hits = row['rerank'][:TOP_K]
        nli_e = np.asarray([hit['nli'][0] for hit in nli_hits], dtype=np.float32)
        nli_c = np.asarray([hit['nli'][2] for hit in nli_hits], dtype=np.float32)
        e_max = float(base_matrix[row_number, FEATURE_INDEX['nli_e_max']])
        c_max = float(base_matrix[row_number, FEATURE_INDEX['nli_c_max']])
        e_idx = int(np.argmax(nli_e)) if nli_e.size else 0

        for model in models:
            values = score_arrays[model]
            at_nli = float(values[e_idx]) if values.size > e_idx else 0.0
            at_top1 = float(values[0]) if values.size else 0.0
            score_max = float(values.max()) if values.size else 0.0
            blocks[model][f'{model}_minus_nli_e'] = float(score_max - e_max)
            blocks[model][f'{model}_times_nli_e'] = float(score_max * e_max)
            blocks[model][f'{model}_minus_nli_c'] = float(score_max - c_max)
            blocks[model][f'{model}_times_nli_c'] = float(score_max * c_max)
            blocks[model][f'{model}_at_nli_e'] = at_nli
            blocks[model][f'{model}_at_top1'] = at_top1

        if 'mc' in blocks and 'hhem' in blocks:
            blocks['mc']['mc_minus_other'] = float(
                score_arrays['mc'].max() - score_arrays['hhem'].max())
            blocks['hhem']['hhem_minus_other'] = float(
                score_arrays['hhem'].max() - score_arrays['mc'].max())
            mc_max = float(score_arrays['mc'].max()) if score_arrays['mc'].size else 0.0
            hhem_max = float(score_arrays['hhem'].max()) if score_arrays['hhem'].size else 0.0
            pair = {
                'both_high': float(mc_max >= 0.5 and hhem_max >= 0.5),
                'both_low': float(mc_max < 0.5 and hhem_max < 0.5),
            }
        else:
            pair = {}

        for name, value in {**{k: v for block in blocks.values() for k, v in block.items()},
                            **pair}.items():
            if name not in index:
                raise KeyError(f'unknown grounding feature {name}')
            matrix[row_number, index[name]] = value

    if not np.isfinite(matrix).all():
        raise ValueError('non-finite grounding feature values')
    return matrix, columns


def variant_columns(columns: list[str], variant: str) -> list[int]:
    """Select a named feature variant from a grounding block."""
    if variant == 'both':
        return list(range(len(columns)))
    if variant == 'mc':
        return [i for i, name in enumerate(columns) if name.startswith('mc_')]
    if variant == 'hhem':
        return [i for i, name in enumerate(columns) if name.startswith('hhem_')]
    if variant == 'compact':
        keep = {
            'mc_max', 'mc_mean', 'mc_top2', 'mc_minus_nli_e', 'mc_minus_nli_c',
            'mc_at_nli_e', 'mc_at_top1',
            'hhem_max', 'hhem_mean', 'hhem_top2', 'hhem_minus_nli_e',
            'hhem_minus_nli_c', 'hhem_at_nli_e', 'hhem_at_top1',
            'mc_minus_other', 'hhem_minus_other', 'both_high', 'both_low',
        }
        return [i for i, name in enumerate(columns) if name in keep]
    if variant == 'none':
        return []
    raise ValueError(f'unknown grounding variant: {variant}')
