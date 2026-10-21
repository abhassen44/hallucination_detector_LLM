"""P6 feasibility: tune weighted RRF on val only.

This does **not** spend the retrieval-variant test pass. It checks whether a
non-uniform BM25/dense fusion beats the v1 hybrid on val; a test pass would only
follow under a separate pre-registration.

    python -m scripts.val_study_weighted_rrf
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import LABELS, load_examples
from src.evaluation.fever_retrieval import retrieval_metrics
from src.evaluation.fit_detector import C_GRID, fit_model, fixed_rule_predictions
from src.evaluation.metrics import multiclass_metrics
from src.evaluation.weighted_rrf import (attach_texts, similarity_matrices,
                                         with_weighted_hits)
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide
from src.verification.features import FEATURE_NAMES, build_matrix, load_split

OUT = PROJECT_ROOT / 'experiments/v2'
V1_VAL_MACRO_F1 = 0.6763644683287465


def _encode(rows: list[dict]) -> np.ndarray:
    return np.array([LABELS.index(row['label']) for row in rows], dtype=np.int64)


def fill_missing_nli(rows: list[dict], nli_model, cache: dict[tuple[str, str], list[float]],
                     top: int = 5) -> int:
    """Compute NLI only for fused top-5 hits whose cached probabilities are absent."""
    pairs = {}
    for row in rows:
        for hit in row['weighted'][:top]:
            if hit.get('nli') is None:
                key = (hit['text'], row['claim'])
                pairs.setdefault(key, None)
    pending = [key for key in pairs if key not in cache]
    if pending:
        probabilities = nli_model.predict([key[0] for key in pending],
                                          [key[1] for key in pending], show_progress=False)
        for key, probability in zip(pending, probabilities):
            cache[key] = probability.tolist()
    for row in rows:
        for hit in row['weighted'][:top]:
            if hit.get('nli') is None:
                hit['nli'] = cache[(hit['text'], row['claim'])]
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='0,0.25,0.5,0.75,1.0',
                        help='BM25 weights; dense gets 1 - w.')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'val_study_weighted_rrf.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    weights = [float(value) for value in args.weights.split(',')]
    cfg = load_config()
    entail = cfg['detector']['entail_threshold']
    contra = cfg['detector']['contra_threshold']

    from src.retrieval.corpus import SentenceCorpus
    from src.verification.nli import NLIModel
    from src.verification.similarity import SimilarityModel

    corpus = SentenceCorpus.from_fever()
    sim_model = SimilarityModel()
    nli_model = NLIModel()
    nli_cache: dict[tuple[str, str], list[float]] = {}
    train_rows = load_split('train')
    val_rows = load_split('val')
    val_examples = load_examples(PROJECT_ROOT / 'data/splits/fever_val.jsonl')
    train_y, val_y = _encode(train_rows), _encode(val_rows)

    results = []
    for weight in weights:
        train_w = with_weighted_hits(train_rows, weight)
        val_w = with_weighted_hits(val_rows, weight)
        attach_texts(train_w, corpus)
        attach_texts(val_w, corpus)
        train_nli = fill_missing_nli(train_w, nli_model, nli_cache)
        val_nli = fill_missing_nli(val_w, nli_model, nli_cache)
        print(f'w_bm25={weight:.2f}: computed {train_nli} train + {val_nli} val NLI pairs',
              flush=True)
        train_x = build_matrix(train_w, 'weighted', similarity_matrices(train_w, sim_model))
        val_x = build_matrix(val_w, 'weighted', similarity_matrices(val_w, sim_model))
        model, chosen_c, val_score = fit_model(train_x, train_y, val_x, val_y,
                                               list(range(len(FEATURE_NAMES))))
        fixed = fixed_rule_predictions(val_w, 'weighted', entail, contra)
        fixed_metrics = multiclass_metrics([row['label'] for row in val_rows], fixed)
        hits = [[(h['doc'], h['score']) for h in row['weighted']] for row in val_w]
        retrieval = retrieval_metrics(val_examples, hits, corpus.keys)['@5']
        results.append({
            'bm25_weight': weight, 'dense_weight': 1.0 - weight,
            'selected_C': chosen_c, 'val_macro_f1': float(val_score),
            'fixed_rule_val_macro_f1': float(fixed_metrics['macro_f1']),
            'fixed_rule_accuracy': float(fixed_metrics['accuracy']),
            'strict_recall_at_5': retrieval['strict_recall'],
            'hit_at_5': retrieval['hit'],
        })
        print(f'w_bm25={weight:.2f} C={chosen_c} val macro-F1={val_score:.4f} '
              f'strict R@5={retrieval["strict_recall"]:.4f}', flush=True)

    best = max(results, key=lambda row: (row['val_macro_f1'], row['strict_recall_at_5']))
    report = {
        'study': 'weighted_rrf',
        'split': 'val',
        'test_evaluated': False,
        'training_performed': True,
        'selection_split': 'val',
        'n_train': len(train_rows), 'n_val': len(val_rows),
        'c_grid': list(C_GRID),
        'weights': weights,
        'baseline_v1_hybrid_val_macro_f1': V1_VAL_MACRO_F1,
        'results': results,
        'best': best,
        'delta_vs_v1_hybrid': best['val_macro_f1'] - V1_VAL_MACRO_F1,
        'note': ('Val-only feasibility. The test split was not read; a retrieval-variant '
                 'test pass requires a separate pre-registration.'),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'best': best, 'delta_vs_v1': report['delta_vs_v1_hybrid']}, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
