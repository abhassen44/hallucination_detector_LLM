"""v3.1 diagnostic: can extra serving features make the NEI gate reachable?

Val only. Adds coverage-flag scores and uncertainty features to the answerability
model, then re-checks the NEI/coverage trade-off under the original gates. No test
data is read and no test pass is spent.

    python -m scripts.v3.answerability_diagnostic_v3
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v3.answerability_v3 import (answerability_policy, cross_fit_base_probabilities,
                                      fit_answerability, load_calibrated_val,
                                      meets_constraints, meets_verifiable)
from scripts.v2.calibrate_v2 import fit_base, load_matrix
from src.datasets.schema import load_examples
from src.evaluation.coverage import gold_evidence, strict_failure_labels
from src.evaluation.selective import metrics_for_answered
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/v3'
SPLITS = PROJECT_ROOT / 'data/splits'
SEED = 42
COVERAGE_C = 0.01


def uncertainty_features(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ordered = np.sort(probabilities, axis=1)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    entropy = entropy / np.log(probabilities.shape[1])
    return np.column_stack([
        probabilities.max(axis=1),
        ordered[:, -1] - ordered[:, -2],
        entropy,
        probabilities[:, 2],
        probabilities[:, 0] + probabilities[:, 1],
    ])


def coverage_scores(train_x: np.ndarray, val_x: np.ndarray,
                    train_rows: list[dict], val_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    train_gold = gold_evidence(load_examples(SPLITS / 'fever_train.jsonl'))
    val_gold = gold_evidence(load_examples(SPLITS / 'fever_val.jsonl'))
    train_fail, train_kept = strict_failure_labels(train_rows, train_gold, 'hybrid')
    _, val_kept = strict_failure_labels(val_rows, val_gold, 'hybrid')
    train_index = {row['id']: index for index, row in enumerate(train_rows)}
    val_index = {row['id']: index for index, row in enumerate(val_rows)}
    kept_index = np.array([train_index[row['id']] for row in train_kept])
    fail = np.asarray(train_fail, dtype=np.int64)

    oof = np.zeros(len(train_rows), dtype=np.float64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_part, score_part in splitter.split(np.arange(len(train_rows))):
        fit_index = np.intersect1d(fit_part, kept_index)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=COVERAGE_C, max_iter=5000, random_state=SEED,
                               class_weight='balanced'))
        labels = {index: label for index, label in zip(kept_index, fail)}
        model.fit(train_x[fit_index], np.array([labels[index] for index in fit_index]))
        oof[score_part] = model.predict_proba(train_x[score_part])[:, 1]

    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=COVERAGE_C, max_iter=5000, random_state=SEED,
                           class_weight='balanced'))
    final.fit(train_x[kept_index], fail)
    return oof, final.predict_proba(val_x)[:, 1]


def policy_search(p_answer: np.ndarray, calibrated: np.ndarray,
                  gold: np.ndarray) -> dict:
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        answered = answerability_policy(calibrated, p_answer, float(tau))
        metrics = metrics_for_answered(gold, calibrated, answered)
        candidates.append({'tau': float(tau), **metrics})
    feasible = [candidate for candidate in candidates if meets_constraints(candidate)]
    verifiable = [candidate for candidate in candidates if meets_verifiable(candidate)]
    best_nei = (min(verifiable, key=lambda candidate: candidate['nei_coverage'] or 1.0)
                if verifiable else None)
    best_cov = (max(feasible, key=lambda candidate: candidate['coverage'])
                if feasible else None)
    return {
        'feasible': bool(feasible),
        'n_feasible': len(feasible),
        'best_feasible': best_cov,
        'best_nei_under_verifiable': best_nei,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'answerability_diagnostic.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    train_rows, train_x, train_y = load_matrix('train', 'hybrid')
    val_rows, val_x, val_y = load_matrix('val', 'hybrid')
    base_model = fit_base(train_x, train_y, 1.0)
    train_probs = cross_fit_base_probabilities(train_x, train_y)
    val_probs = base_model.predict_proba(val_x)
    train_coverage, val_coverage = coverage_scores(train_x, val_x, train_rows, val_rows)

    base_train = np.hstack([train_x, train_probs])
    base_val = np.hstack([val_x, val_probs])
    train_uncertainty = uncertainty_features(train_probs)
    val_uncertainty = uncertainty_features(val_probs)
    train_coverage_col = train_coverage[:, None]
    val_coverage_col = val_coverage[:, None]
    train_interactions = np.column_stack([
        train_coverage * train_probs.max(axis=1),
        train_coverage * train_probs[:, 2],
    ])
    val_interactions = np.column_stack([
        val_coverage * val_probs.max(axis=1),
        val_coverage * val_probs[:, 2],
    ])

    variants = {
        'base': (base_train, base_val),
        'base_uncertainty': (np.hstack([base_train, train_uncertainty]),
                             np.hstack([base_val, val_uncertainty])),
        'base_coverage': (np.hstack([base_train, train_coverage_col]),
                          np.hstack([base_val, val_coverage_col])),
        'base_uncertainty_coverage': (
            np.hstack([base_train, train_uncertainty, train_coverage_col]),
            np.hstack([base_val, val_uncertainty, val_coverage_col])),
        'full_interactions': (
            np.hstack([base_train, train_uncertainty, train_coverage_col, train_interactions]),
            np.hstack([base_val, val_uncertainty, val_coverage_col, val_interactions])),
    }

    _, calibrated, gold = load_calibrated_val()
    if len(calibrated) != len(val_rows):
        raise ValueError('calibrated val rows do not match the feature matrix')

    train_answerable = (train_y != 2).astype(np.int64)
    val_answerable = (val_y != 2).astype(np.int64)
    results = {}
    for name, (variant_train, variant_val) in variants.items():
        model, model_info, comparison = fit_answerability(
            variant_train, train_answerable, variant_val, val_answerable)
        p_answer = model.predict_proba(variant_val)[:, 1]
        policy = policy_search(p_answer, calibrated, gold)
        results[name] = {
            'model': model_info,
            'model_comparison': comparison,
            'val_metrics': {
                'auroc': float(roc_auc_score(val_answerable, p_answer)),
                'average_precision': float(average_precision_score(val_answerable, p_answer)),
            },
            'policy': policy,
            'n_features': variant_train.shape[1],
        }
        print(f'{name}: features={variant_train.shape[1]} '
              f'AUROC={results[name]["val_metrics"]["auroc"]:.4f} '
              f'feasible={policy["feasible"]} '
              f'best_nei={((policy["best_nei_under_verifiable"] or {}).get("nei_coverage"))}',
              flush=True)

    feasible_names = [name for name, result in results.items()
                      if result['policy']['feasible']]
    best_name = (max(feasible_names, key=lambda name: results[name]['val_metrics']['auroc'])
                 if feasible_names
                 else max(results, key=lambda name: results[name]['val_metrics']['auroc']))
    report = {
        'version': 'v3.1-answerability-diagnostic',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val',
        'test_evaluated': False,
        'test_pass_spent': False,
        'coverage_feature': {
            'model': 'logistic on strict retrieval failure',
            'C': COVERAGE_C,
            'train_oof': True,
        },
        'variants': results,
        'feasible_variants': feasible_names,
        'best_by_auroc': max(results, key=lambda name: results[name]['val_metrics']['auroc']),
        'best': results[best_name],
        'conclusion': ('NEI gate reachable with the expanded features'
                       if feasible_names else 'NEI gate still unreachable on val'),
        'notes': [
            'Val-only diagnostic. No test data was read and no test pass was spent.',
            'The coverage score is cross-fitted on train and fitted on full train for val.',
            'The class thresholds are the frozen v2 values; only the answerability gate is searched.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'best_by_auroc': best_name,
                      'best': results[best_name]['val_metrics'],
                      'policy': results[best_name]['policy']['best_nei_under_verifiable'],
                      'feasible': results[best_name]['policy']['feasible']}, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
