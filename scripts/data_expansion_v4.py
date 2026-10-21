"""P2: refit and evaluate the expanded v4 train set.

Combines the existing reranked train cache with the extra v4 claims, refits the
base combiner, calibrates on val, and re-checks the answerability/NEI trade-off.
Val only; no test data is read.

    python -m scripts.data_expansion_v4 --train-size 6000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.answerability_v3 import (CLASS_THRESHOLDS, answerability_policy,
                                      cross_fit_base_probabilities, fit_answerability,
                                      meets_constraints, meets_verifiable)
from scripts.calibrate_v2 import fit_base
from src.datasets.schema import LABELS, load_examples
from src.evaluation.calibration import (apply_vector_scaling, brier_score,
                                        class_calibration_error,
                                        cross_fit_vector_scaling_brier,
                                        expected_calibration_error)
from src.evaluation.fit_detector import C_GRID
from src.evaluation.metrics import multiclass_metrics
from src.evaluation.selective import metrics_for_answered
from src.utils.config import PROJECT_ROOT
from src.verification.detector import aggregate_probabilities, decide
from src.verification.features import (FEATURE_NAMES, build_matrix, labels, load_split,
                                       similarity_matrix)

OUT = PROJECT_ROOT / 'experiments/v4'
CACHE = PROJECT_ROOT / 'data/processed/rerank'
RERANKED_TRAIN = CACHE / 'train_hybrid_top8.jsonl'
EXTRA = CACHE / 'train_extra_hybrid_top5.jsonl'
RERANKED_VAL = CACHE / 'val_hybrid_top8.jsonl'
V3_ANSWERABILITY_AUROC = 0.765405044684324
V3_NEI_FLOOR = 0.483
V1_VAL_MACRO_F1 = 0.6763644683287465


def load_cache(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def enriched(rows: list[dict], cached: list[dict], k: int = 5) -> list[dict]:
    by_id = {row['id']: row for row in rows}
    out = []
    for item in cached:
        row = dict(by_id[item['id']])
        row['rerank'] = item['hits'][:k]
        out.append(row)
    return out


def policy_search(p_answer: np.ndarray, calibrated: np.ndarray, gold: np.ndarray) -> dict:
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        answered = answerability_policy(calibrated, p_answer, float(tau))
        metrics = metrics_for_answered(gold, calibrated, answered)
        candidates.append({'tau': float(tau), **metrics})
    feasible = [candidate for candidate in candidates if meets_constraints(candidate)]
    verifiable = [candidate for candidate in candidates if meets_verifiable(candidate)]
    best = (max(feasible, key=lambda candidate: candidate['coverage']) if feasible else
            (min(verifiable, key=lambda candidate: candidate['nei_coverage'] or 1.0)
             if verifiable else None))
    return {'feasible': bool(feasible), 'best': best,
            'n_feasible': len(feasible)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-size', type=int, default=6000, choices=[6000, 9000])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / f'data_expansion_{args.train_size}.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    train_rows = load_split('train')
    val_rows = load_split('val')
    base_cached = load_cache(RERANKED_TRAIN)
    extra_path = EXTRA if args.train_size == 6000 else CACHE / 'train_extra_hybrid_top5_6000.jsonl'
    if not extra_path.exists():
        raise SystemExit(f'Missing expanded train cache: {extra_path}')
    extra_cached = load_cache(extra_path)
    train_enriched = enriched(train_rows, base_cached)
    extra_rows = [{'id': row['id'], 'claim': row['claim'], 'label': row['label']}
                  for row in extra_cached]
    train_enriched = enriched(train_rows, base_cached) + enriched(extra_rows, extra_cached)
    val_enriched = enriched(val_rows, load_cache(RERANKED_VAL))

    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    corpus = SentenceCorpus.from_fever()
    sim_model = SimilarityModel()
    train_x = build_matrix(train_enriched, 'rerank',
                           similarity_matrix(train_enriched, 'rerank', corpus, sim_model, top=5),
                           top=5)
    val_x = build_matrix(val_enriched, 'rerank',
                         similarity_matrix(val_enriched, 'rerank', corpus, sim_model, top=5),
                         top=5)
    train_y = np.array([LABELS.index(row['label']) for row in train_enriched], dtype=np.int64)
    val_y = np.array([LABELS.index(row['label']) for row in val_enriched], dtype=np.int64)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    best = None
    for c_value in C_GRID:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=c_value, max_iter=5000, random_state=42))
        model.fit(train_x, train_y)
        score = f1_score(val_y, model.predict(val_x), average='macro', zero_division=0)
        if best is None or score > best[0]:
            best = (score, c_value, model)
    _, c_value, base_model = best
    val_metrics = multiclass_metrics([LABELS[i] for i in val_y],
                                     [LABELS[i] for i in base_model.predict(val_x)])
    logits = base_model.decision_function(val_x)
    # The v4 P1 calibrator is vector scaling fitted on Brier; keep it fixed here.
    calibrated, _, final_vector = cross_fit_vector_scaling_brier(logits, val_y, n_splits=2)

    train_oof = cross_fit_base_probabilities(train_x, train_y, c_value=c_value, folds=5)
    train_features = np.hstack([train_x, train_oof])
    val_features = np.hstack([val_x, base_model.predict_proba(val_x)])
    train_answerable = (train_y != 2).astype(np.int64)
    val_answerable = (val_y != 2).astype(np.int64)
    answer_model, answer_info, answer_comparison = fit_answerability(
        train_features, train_answerable, val_features, val_answerable)
    p_answer = answer_model.predict_proba(val_features)[:, 1]
    policy = policy_search(p_answer, calibrated, val_y)

    report = {
        'version': 'v4-data-expansion',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val',
        'test_evaluated': False,
        'train_size': args.train_size,
        'n_train': len(train_y), 'n_val': len(val_y),
        'base_model': {'selected_C': c_value, 'val': val_metrics},
        'calibration': {
            'method': 'vector_scaling_brier',
            'final': final_vector,
            'brier': brier_score(val_y, calibrated),
            'ece': expected_calibration_error(val_y, calibrated),
            'per_class_ece': {label: class_calibration_error(val_y, calibrated, index)
                              for index, label in enumerate(LABELS)},
        },
        'answerability': {
            'model': answer_info, 'comparison': answer_comparison,
            'val_auroc': float(roc_auc_score(val_answerable, p_answer)),
            'val_average_precision': float(average_precision_score(val_answerable, p_answer)),
            'policy': policy,
        },
        'baseline_v3': {'answerability_auroc': V3_ANSWERABILITY_AUROC,
                        'nei_floor': V3_NEI_FLOOR},
        'delta': {'answerability_auroc': float(roc_auc_score(val_answerable, p_answer))
                  - V3_ANSWERABILITY_AUROC,
                  'nei_floor': ((policy['best'] or {}).get('nei_coverage', 1.0)
                                - V3_NEI_FLOOR)},
        'notes': ['Val only; no test data was read.', 'Extra train claims use the reranker top-5 cache.'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'train_size': args.train_size,
        'val_macro_f1': val_metrics['macro_f1'],
        'selected_C': c_value,
        'answerability_auroc': report['answerability']['val_auroc'],
        'policy_feasible': policy['feasible'],
        'best_nei': ((policy['best'] or {}).get('nei_coverage')),
        'best_overall_precision': ((policy['best'] or {}).get('answered_precision')),
        'delta': report['delta'],
    }, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
