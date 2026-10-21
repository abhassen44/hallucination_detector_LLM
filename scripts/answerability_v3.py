"""P2: answerability / NEI model and the v3 selective policy.

Train a binary answerability model (gold != INSUFFICIENT) on train, select on val,
then search a single answerability threshold around the frozen v2 class thresholds.
No test data is read.

    python -m scripts.answerability_v3
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.calibrate_v2 import fit_base, load_matrix
from src.datasets.schema import LABELS
from src.evaluation.selective import metrics_for_answered, policy_metrics
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/v3'
CALIBRATION = OUT / 'calibration.json'
VAL_PREDICTIONS = OUT / 'val_predictions_v3.jsonl'
SEED = 42
CLASS_THRESHOLDS = np.array([0.50, 0.38, 0.84])  # frozen v2 policy
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def cross_fit_base_probabilities(matrix: np.ndarray, labels: np.ndarray,
                                 c_value: float = 1.0, folds: int = 5) -> np.ndarray:
    """Out-of-fold class probabilities, so the answerability model sees realistic train scores."""
    oof = np.zeros((len(labels), len(LABELS)), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train_index, test_index in splitter.split(matrix, labels):
        model = fit_base(matrix[train_index], labels[train_index], c_value)
        oof[test_index] = model.predict_proba(matrix[test_index])
    return oof


def fit_answerability(train_x: np.ndarray, train_y: np.ndarray,
                      val_x: np.ndarray, val_y: np.ndarray) -> tuple[object, dict, dict]:
    best = None
    for c_value in C_GRID:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=5000, random_state=SEED,
                               class_weight='balanced'))
        model.fit(train_x, train_y)
        score = roc_auc_score(val_y, model.predict_proba(val_x)[:, 1])
        if best is None or score > best[0]:
            best = (score, 'logistic', {'C': float(c_value)}, model)
    auc, name, params, model = best

    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, class_weight='balanced', random_state=SEED)
    hgb.fit(train_x, train_y)
    hgb_auc = roc_auc_score(val_y, hgb.predict_proba(val_x)[:, 1])
    if hgb_auc > auc:
        model, name, auc = hgb, 'hist_gradient_boosting', hgb_auc
        params = {'max_iter': 300, 'learning_rate': 0.05, 'max_leaf_nodes': 15,
                  'l2_regularization': 1.0, 'class_weight': 'balanced'}
    return model, {'type': name, 'params': params, 'val_auroc': float(auc)}, {
        'logistic_best_auroc': float(best[0]),
        'hgb_auroc': float(hgb_auc),
    }


def answerability_policy(probabilities: np.ndarray, p_answer: np.ndarray,
                         tau: float, thresholds: np.ndarray = CLASS_THRESHOLDS) -> np.ndarray:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    return (p_answer >= tau) & (confidence >= thresholds[predicted])


def load_calibrated_val() -> tuple[dict, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in VAL_PREDICTIONS.read_text(
        encoding='utf-8').splitlines() if line.strip()]
    probabilities = np.array([[row['calibrated_probabilities_oof'][label] for label in LABELS]
                              for row in rows])
    gold = np.array([LABELS.index(row['gold']) for row in rows], dtype=np.int64)
    return {row['id']: row for row in rows}, probabilities, gold


def meets_constraints(metrics: dict) -> bool:
    return bool(
        (metrics['verifiable_precision'] or 0) >= 0.94
        and (metrics['verifiable_precision_lower_95'] or 0) >= 0.90
        and (metrics['verifiable_coverage'] or 0) >= 0.80
        and metrics['coverage'] >= 0.55
        and (metrics['answered_precision'] or 0) >= 0.75
        and (metrics['nei_coverage'] or 1) <= 0.40)


def meets_verifiable(metrics: dict) -> bool:
    return bool(
        (metrics['verifiable_precision'] or 0) >= 0.94
        and (metrics['verifiable_precision_lower_95'] or 0) >= 0.90
        and (metrics['verifiable_coverage'] or 0) >= 0.80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'answerability.json'
    policy_output = OUT / 'policy.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    if calibration['selection']['status'] != 'selected':
        raise SystemExit('P1 did not select a calibrator; v3 stops before P2.')

    train_rows, train_x, train_y = load_matrix('train', 'hybrid')
    val_rows, val_x, val_y = load_matrix('val', 'hybrid')
    base = fit_base(train_x, train_y, 1.0)
    train_base_probs = cross_fit_base_probabilities(train_x, train_y)
    val_base_probs = base.predict_proba(val_x)

    train_features = np.hstack([train_x, train_base_probs])
    val_features = np.hstack([val_x, val_base_probs])
    train_answerable = (train_y != 2).astype(np.int64)
    val_answerable = (val_y != 2).astype(np.int64)

    model, model_info, comparison = fit_answerability(
        train_features, train_answerable, val_features, val_answerable)
    p_answer = model.predict_proba(val_features)[:, 1]
    auprc = float(average_precision_score(val_answerable, p_answer))
    brier = float(np.mean((p_answer - val_answerable) ** 2))

    _, calibrated, gold = load_calibrated_val()
    if len(calibrated) != len(val_rows):
        raise ValueError('calibrated val rows do not match the feature matrix')

    baseline = policy_metrics(gold, calibrated, CLASS_THRESHOLDS)
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        answered = answerability_policy(calibrated, p_answer, float(tau))
        metrics = metrics_for_answered(gold, calibrated, answered)
        candidates.append({'tau': float(tau), **metrics})

    feasible = [candidate for candidate in candidates if meets_constraints(candidate)]
    fallback = False
    if feasible:
        chosen = max(feasible, key=lambda candidate: (
            candidate['coverage'], candidate['verifiable_precision_lower_95'],
            candidate['answered_precision'] or 0.0))
        mode = 'all_constraints'
    else:
        fallback = True
        verifiable = [candidate for candidate in candidates if meets_verifiable(candidate)]
        if not verifiable:
            raise ValueError('no answerability threshold meets the verifiable constraints')
        chosen = min(verifiable, key=lambda candidate: (
            candidate['nei_coverage'] or 1.0, -(candidate['coverage'])))
        mode = 'fallback_verifiable_only'

    report = {
        'version': 'v3-answerability',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'target': 'gold != INSUFFICIENT',
        'model': model_info,
        'model_comparison': comparison,
        'features': '30 cached features + 3 base-model class probabilities',
        'training': {
            'fit_split': 'train', 'selection_split': 'val',
            'train_n': len(train_y), 'val_n': len(val_y),
            'train_oof_probabilities': True,
            'class_counts': {'answerable': int(train_answerable.sum()),
                             'nei': int(len(train_answerable) - train_answerable.sum())},
        },
        'val_metrics': {'auroc': float(roc_auc_score(val_answerable, p_answer)),
                        'average_precision': auprc, 'brier': brier},
        'class_thresholds': dict(zip(LABELS, CLASS_THRESHOLDS.tolist())),
        'selection': {
            'mode': mode,
            'fallback_triggered': fallback,
            'constraints': {
                'verifiable_precision_min': 0.94,
                'verifiable_precision_cp_lower_min': 0.90,
                'verifiable_coverage_min': 0.80,
                'overall_coverage_min': 0.55,
                'overall_precision_min': 0.75,
                'nei_coverage_max': 0.40,
            },
            'tau_answerable': chosen['tau'],
        },
        'val_policy': chosen,
        'v2_policy_baseline': baseline,
        'delta': {
            'nei_coverage': chosen['nei_coverage'] - baseline['nei_coverage'],
            'verifiable_coverage': chosen['verifiable_coverage'] - baseline['verifiable_coverage'],
            'verifiable_precision': chosen['verifiable_precision'] - baseline['verifiable_precision'],
            'overall_precision': chosen['answered_precision'] - baseline['answered_precision'],
        },
        'notes': [
            'No test data was read.',
            'The class thresholds are the frozen v2 policy; only the answerability threshold is selected.',
            'If fallback_triggered is true, the NEI improvement is not claimed; see the protocol.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    policy = {
        'version': 'v3-policy',
        'created_utc': report['created_utc'],
        'policy': {
            'type': 'answerability_gate + class_conditional_thresholds',
            'rule': 'abstain if P(answerable) < tau; otherwise answer argmax if calibrated p[argmax] >= threshold[argmax]',
            'answerability_threshold': chosen['tau'],
            'class_thresholds': dict(zip(LABELS, CLASS_THRESHOLDS.tolist())),
        },
        'selection': report['selection'],
        'val_metrics': chosen,
        'v2_policy_baseline': baseline,
        'calibration': {
            'source': 'experiments/v3/calibration.json',
            'method': calibration['selection']['selected'],
        },
        'test_gate': {
            'verifiable_precision_min': 0.94,
            'verifiable_precision_cp_lower_min': 0.90,
            'verifiable_coverage_min': 0.80,
            'overall_coverage_min': 0.55,
            'overall_precision_min': 0.75,
            'nei_coverage_max': 0.40,
        },
    }
    policy_output.write_text(json.dumps(policy, indent=2) + '\n', encoding='utf-8')

    with (OUT / 'val_policy_v3.jsonl').open('w', encoding='utf-8') as handle:
        for index, row in enumerate(val_rows):
            answered = bool(answerability_policy(calibrated, p_answer,
                                                 chosen['tau'])[index])
            handle.write(json.dumps({
                'id': row['id'], 'claim': row['claim'],
                'gold': LABELS[int(val_y[index])],
                'prediction': LABELS[int(calibrated[index].argmax())],
                'calibrated_probabilities': {label: float(calibrated[index, number])
                                             for number, label in enumerate(LABELS)},
                'p_answerable': float(p_answer[index]),
                'answered': answered,
                'abstained_by_answerability': bool(p_answer[index] < chosen['tau']),
            }, ensure_ascii=False) + '\n')

    print(json.dumps({
        'model': model_info,
        'model_comparison': comparison,
        'val_metrics': report['val_metrics'],
        'selection': report['selection'],
        'val_policy': {key: chosen[key] for key in
                       ('coverage', 'answered_precision', 'verifiable_coverage',
                        'verifiable_precision', 'verifiable_precision_lower_95',
                        'nei_coverage', 'nei_precision')},
        'v2_baseline': {key: baseline[key] for key in
                        ('coverage', 'answered_precision', 'verifiable_coverage',
                         'verifiable_precision', 'nei_coverage')},
        'delta': report['delta'],
    }, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}, {policy_output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
