"""v4 P3 diagnostic: tune answerability at 9k and re-check the NEI gate.

Val only. Reuses the 9k reranker caches, compares HGB configurations and a
logistic/HGB ensemble, then searches the answerability threshold with the frozen
class thresholds. No test data is read.

    python -m scripts.answerability_tuning_v4
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.answerability_v3 import (CLASS_THRESHOLDS, answerability_policy,
                                      cross_fit_base_probabilities, meets_constraints,
                                      meets_verifiable)
from src.datasets.schema import LABELS
from src.evaluation.calibration import (apply_vector_scaling, brier_score,
                                        cross_fit_vector_scaling_brier,
                                        expected_calibration_error)
from src.evaluation.selective import metrics_for_answered
from src.utils.config import PROJECT_ROOT
from src.verification.features import build_matrix, load_split, similarity_matrix

OUT = PROJECT_ROOT / 'experiments/v4'
CACHE = PROJECT_ROOT / 'data/processed/rerank'


def load_cache(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def enriched(rows: list[dict], cached: list[dict]) -> list[dict]:
    by_id = {row['id']: row for row in rows}
    return [{**by_id[item['id']], 'rerank': item['hits'][:5]} for item in cached]


def policy_search(p_answer: np.ndarray, calibrated: np.ndarray, gold: np.ndarray) -> dict:
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        answered = answerability_policy(calibrated, p_answer, float(tau))
        candidates.append({'tau': float(tau),
                           **metrics_for_answered(gold, calibrated, answered)})
    feasible = [candidate for candidate in candidates if meets_constraints(candidate)]
    verifiable = [candidate for candidate in candidates if meets_verifiable(candidate)]
    best = (max(feasible, key=lambda candidate: candidate['coverage']) if feasible else
            (min(verifiable, key=lambda candidate: candidate['nei_coverage'] or 1.0)
             if verifiable else None))
    return {'feasible': bool(feasible), 'best': best, 'n_feasible': len(feasible)}


def joint_policy_search(p_answer: np.ndarray, calibrated: np.ndarray,
                        gold: np.ndarray) -> dict:
    predicted = calibrated.argmax(axis=1)
    confidence = calibrated.max(axis=1)
    feasible, verifiable = [], []
    for tau in np.arange(0.05, 0.81, 0.05):
        for supported in np.arange(0.45, 0.96, 0.05):
            for contradicted in np.arange(0.35, 0.96, 0.05):
                thresholds = np.array([supported, contradicted, 0.84])
                answered = (p_answer >= tau) & (confidence >= thresholds[predicted])
                metrics = metrics_for_answered(gold, calibrated, answered)
                candidate = {'tau': float(tau), 'thresholds': thresholds.tolist(),
                             **metrics}
                if meets_constraints(metrics):
                    feasible.append(candidate)
                if meets_verifiable(metrics):
                    verifiable.append(candidate)
    best = (max(feasible, key=lambda candidate: candidate['coverage']) if feasible else
            (min(verifiable, key=lambda candidate: candidate['nei_coverage'] or 1.0)
             if verifiable else None))
    return {'feasible': bool(feasible), 'n_feasible': len(feasible), 'best': best}


def main() -> None:
    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    base_cached = load_cache(CACHE / 'train_hybrid_top8.jsonl')
    extra_cached = load_cache(CACHE / 'train_extra_hybrid_top5_6000.jsonl')
    val_cached = load_cache(CACHE / 'val_hybrid_top8.jsonl')
    train_rows = load_split('train')
    val_rows = load_split('val')
    extra_rows = [{'id': row['id'], 'claim': row['claim'], 'label': row['label']}
                  for row in extra_cached]
    train_enriched = enriched(train_rows, base_cached) + enriched(extra_rows, extra_cached)
    val_enriched = enriched(val_rows, val_cached)

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

    # Base model: 9k report's selected C.
    from src.evaluation.fit_detector import fit_model
    base_model, selected_c, _ = fit_model(train_x, train_y, val_x, val_y,
                                          list(range(train_x.shape[1])))
    logits = base_model.decision_function(val_x)
    calibrated, _, calibrator = cross_fit_vector_scaling_brier(logits, val_y, n_splits=2)
    train_oof = cross_fit_base_probabilities(train_x, train_y, c_value=selected_c, folds=5)
    train_features = np.hstack([train_x, train_oof])
    val_features = np.hstack([val_x, base_model.predict_proba(val_x)])
    train_answerable = (train_y != 2).astype(np.int64)
    val_answerable = (val_y != 2).astype(np.int64)

    configs = []
    logistic_best = None
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=c_value, max_iter=5000, random_state=42,
                                                 class_weight='balanced'))
        model.fit(train_features, train_answerable)
        probability = model.predict_proba(val_features)[:, 1]
        auroc = float(roc_auc_score(val_answerable, probability))
        configs.append({'name': f'logistic_C{c_value}', 'model': model, 'probability': probability,
                        'auroc': auroc})
        if logistic_best is None or auroc > logistic_best['auroc']:
            logistic_best = configs[-1]

    for max_iter in (300, 600):
        for learning_rate in (0.03, 0.05):
            for leaves in (7, 15, 31):
                model = HistGradientBoostingClassifier(
                    max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=leaves,
                    l2_regularization=1.0, class_weight='balanced', random_state=42)
                model.fit(train_features, train_answerable)
                probability = model.predict_proba(val_features)[:, 1]
                configs.append({
                    'name': f'hgb_{max_iter}_{learning_rate}_{leaves}', 'model': model,
                    'probability': probability,
                    'auroc': float(roc_auc_score(val_answerable, probability)),
                })

    for nei_weight in (2, 3, 4, 5, 6, 8):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight={0: 1.0, 1: float(nei_weight)},
            random_state=42)
        model.fit(train_features, train_answerable)
        probability = model.predict_proba(val_features)[:, 1]
        configs.append({
            'name': f'hgb_nei_weight_{nei_weight}', 'model': model,
            'probability': probability,
            'auroc': float(roc_auc_score(val_answerable, probability)),
        })

    best_hgb = max((config for config in configs if config['name'].startswith('hgb')),
                   key=lambda config: config['auroc'])
    ensemble_probability = (best_hgb['probability'] + logistic_best['probability']) / 2
    configs.append({'name': 'ensemble_logistic_hgb', 'model': None,
                    'probability': ensemble_probability,
                    'auroc': float(roc_auc_score(val_answerable, ensemble_probability))})

    results = []
    for config in configs:
        policy = policy_search(config['probability'], calibrated, val_y)
        results.append({
            'name': config['name'], 'auroc': config['auroc'],
            'average_precision': float(average_precision_score(val_answerable,
                                                               config['probability'])),
            'policy': policy,
        })
    selected = min(results, key=lambda result: (
        (result['policy']['best'] or {}).get('nei_coverage', 1.0),
        -result['auroc']))
    selected_config = next(config for config in configs if config['name'] == selected['name'])
    joint = joint_policy_search(selected_config['probability'], calibrated, val_y)
    report = {
        'version': 'v4-answerability-tuning',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val', 'test_evaluated': False,
        'train_size': len(train_y),
        'calibration': {'brier': brier_score(val_y, calibrated),
                        'ece': expected_calibration_error(val_y, calibrated)},
        'results': results,
        'selected': selected,
        'feasible': bool(selected['policy']['feasible']),
        'joint_threshold_search': joint,
        'notes': ['Val-only tuning; no test data was read.',
                  'The primary search keeps the frozen v2 class thresholds; the joint search '
                  'is a diagnostic for whether any class-threshold change could meet the gates.'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'answerability_tuning_9000.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'best_auroc': max(results, key=lambda result: result['auroc'])['auroc'],
        'selected': selected['name'],
        'auroc': selected['auroc'],
        'feasible': selected['policy']['feasible'],
        'nei': ((selected['policy']['best'] or {}).get('nei_coverage')),
        'overall_precision': ((selected['policy']['best'] or {}).get('answered_precision')),
        'verifiable_coverage': ((selected['policy']['best'] or {}).get('verifiable_coverage')),
        'joint_feasible': joint['feasible'],
        'joint_best': joint['best'],
    }, indent=2))
    print(f'Saved -> {OUT / "answerability_tuning_9000.json"}')


if __name__ == '__main__':
    main()
