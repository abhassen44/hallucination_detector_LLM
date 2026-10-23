"""P4: the single v2 test pass.

Evaluates the frozen v2 system in one read of the test split:

* the v1 hybrid verdicts (unchanged) with temperature-calibrated confidence;
* the frozen class-conditional selective policy;
* the frozen evidence-coverage flag.

Thresholds come from `experiments/v2/policy.json` and `coverage_flag.json`; they
are not tuned here. A second run requires an explicit acknowledgement and is
recorded as an extra pass.

    python -m scripts.v2.test_pass_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from scripts.v2.calibrate_v2 import load_matrix
from src.datasets.schema import load_examples
from src.evaluation.calibration import (brier_score, expected_calibration_error,
                                        negative_log_likelihood, softmax)
from src.evaluation.coverage import apply_flag, gold_evidence, strict_failure_labels
from src.evaluation.selective import LABELS, policy_metrics
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/v2'
PREDICTIONS = PROJECT_ROOT / 'experiments/stage5_learned/predictions_hybrid.jsonl'
POLICY = OUT / 'policy.json'
CALIBRATION = OUT / 'calibration.json'
COVERAGE = OUT / 'coverage_flag.json'
TEST_SPLIT = PROJECT_ROOT / 'data/splits/fever_test.jsonl'
OUTPUT = OUT / 'metrics_v2.json'
BOOTSTRAP = 2000
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _macro_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(y, prediction, average='macro', zero_division=0))


def _bootstrap_macro_f1(y: np.ndarray, prediction: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, len(y), len(y))
        values.append(_macro_f1(y[index], prediction[index]))
    return {'mean': float(np.mean(values)), 'lower_95': float(np.percentile(values, 2.5)),
            'upper_95': float(np.percentile(values, 97.5)), 'n_resamples': BOOTSTRAP}


def _bootstrap_rate(correct: np.ndarray, selected: np.ndarray) -> dict:
    """Bootstrap a precision-style rate over the selected rows."""
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, len(correct), len(correct))
        chosen = selected[index]
        if not chosen.any():
            continue
        values.append(float(correct[index][chosen].mean()))
    if not values:
        return {'mean': None, 'lower_95': None, 'upper_95': None, 'n_resamples': 0}
    return {'mean': float(np.mean(values)), 'lower_95': float(np.percentile(values, 2.5)),
            'upper_95': float(np.percentile(values, 97.5)), 'n_resamples': len(values)}


def _bootstrap_coverage(numerator: np.ndarray, denominator: np.ndarray) -> dict:
    """Bootstrap a ratio of row counts (coverage within the denominator)."""
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP):
        index = rng.integers(0, len(numerator), len(numerator))
        selected = denominator[index].sum()
        if selected:
            values.append(float(numerator[index].sum() / selected))
    return {'mean': float(np.mean(values)), 'lower_95': float(np.percentile(values, 2.5)),
            'upper_95': float(np.percentile(values, 97.5)), 'n_resamples': len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--acknowledge-second-pass', action='store_true')
    args = parser.parse_args()

    previous_pass = 0
    if OUTPUT.exists():
        if not args.force or not args.acknowledge_second_pass:
            raise SystemExit('The v2 test pass already exists. Re-running requires '
                             '--force --acknowledge-second-pass; it will be recorded as '
                             'an extra pass.')
        previous = json.loads(OUTPUT.read_text(encoding='utf-8'))
        previous_pass = int(previous.get('test_pass', 1))

    predictions_raw = [json.loads(line) for line in PREDICTIONS.read_text(
        encoding='utf-8').splitlines() if line.strip()]
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    coverage = json.loads(COVERAGE.read_text(encoding='utf-8'))
    temperature = float(calibration['calibrator']['final_temperature'])
    thresholds = np.array([policy['policy']['thresholds'][label] for label in LABELS])

    y_true = np.array([LABELS.index(row['gold']) for row in predictions_raw], dtype=np.int64)
    raw = np.array([[row['probabilities'][label] for label in LABELS]
                    for row in predictions_raw])
    calibrated = softmax(np.log(np.clip(raw, 1e-12, 1.0)), temperature)
    prediction = calibrated.argmax(axis=1)

    raw_summary = {
        'accuracy': float((raw.argmax(axis=1) == y_true).mean()),
        'macro_f1': _macro_f1(y_true, raw.argmax(axis=1)),
        'ece_10bin': expected_calibration_error(y_true, raw),
        'brier': brier_score(y_true, raw),
        'nll': negative_log_likelihood(y_true, raw),
    }
    calibrated_summary = {
        'accuracy': float((prediction == y_true).mean()),
        'macro_f1': _macro_f1(y_true, prediction),
        'ece_10bin': expected_calibration_error(y_true, calibrated),
        'brier': brier_score(y_true, calibrated),
        'nll': negative_log_likelihood(y_true, calibrated),
    }
    calibrated_summary['macro_f1_bootstrap'] = _bootstrap_macro_f1(y_true, prediction)

    selective = policy_metrics(y_true, calibrated, thresholds)
    answered = np.array([calibrated[index, prediction[index]] >= thresholds[prediction[index]]
                         for index in range(len(y_true))])
    correct = (prediction == y_true)
    verifiable = np.isin(y_true, (0, 1))
    selective['bootstrap'] = {
        'answered_precision': _bootstrap_rate(correct, answered),
        'verifiable_precision': _bootstrap_rate(correct, answered & verifiable),
        'coverage': _bootstrap_coverage(answered, np.ones(len(y_true))),
        'verifiable_coverage': _bootstrap_coverage(answered & verifiable, verifiable),
    }

    # Coverage flag on test: features are serving-available; gold is label-only.
    test_rows, test_x, _ = load_matrix('test', 'hybrid')
    test_gold = gold_evidence(load_examples(TEST_SPLIT))
    test_fail, test_kept = strict_failure_labels(test_rows, test_gold, 'hybrid')
    row_index = {row['id']: index for index, row in enumerate(test_rows)}
    prediction_by_id = {row['id']: row for row in predictions_raw}
    kept_x = test_x[[row_index[row['id']] for row in test_kept]]
    scores = apply_flag(coverage['model']['parameters'], kept_x)
    flag_threshold = float(coverage['metrics']['operating_point']['threshold'])
    flag = scores >= flag_threshold
    kept_prediction = np.array([prediction_by_id[row['id']]['prediction'] for row in test_kept])
    kept_gold = np.array([row['label'] for row in test_kept])
    flag_errors = (kept_prediction != kept_gold).astype(np.int64)
    flag_abstentions = (kept_prediction == 'INSUFFICIENT').astype(np.int64)
    flag_metrics = {
        'n': int(len(test_fail)),
        'n_flagged': int(flag.sum()),
        'flag_rate': float(flag.mean()),
        'failure_rate': float(test_fail.mean()),
        'flagged_failure_rate': float(test_fail[flag].mean()) if flag.any() else None,
        'base_error_rate': float(flag_errors.mean()),
        'flagged_error_rate': float(flag_errors[flag].mean()) if flag.any() else None,
        'error_enrichment': (float(flag_errors[flag].mean() / flag_errors.mean())
                             if flag.any() and flag_errors.mean() > 0 else None),
        'base_false_abstention_rate': float(flag_abstentions.mean()),
        'flagged_false_abstention_rate': (float(flag_abstentions[flag].mean())
                                          if flag.any() else None),
        'false_abstention_enrichment': (
            float(flag_abstentions[flag].mean() / flag_abstentions.mean())
            if flag.any() and flag_abstentions.mean() > 0 else None),
    }
    flag_metrics['auroc'] = float(roc_auc_score(test_fail, scores))
    flag_metrics['average_precision'] = float(average_precision_score(test_fail, scores))
    flag_gate = {
        'min_auroc': 0.75,
        'min_error_enrichment': 2.0,
        'max_flag_rate': 0.25,
        'passed': bool(flag_metrics['auroc'] >= 0.75
                       and (flag_metrics['error_enrichment'] or 0) >= 2.0
                       and flag_metrics['flag_rate'] <= 0.25),
    }
    if not flag_gate['passed']:
        flag_gate['consequence'] = 'omit the flag from serving; report it as a diagnostic'

    macro_gate = {'min': 0.697, 'value': calibrated_summary['macro_f1'],
                  'passed': calibrated_summary['macro_f1'] >= 0.697}
    calibration_gate = {
        'max_ece': 0.05,
        'ece': calibrated_summary['ece_10bin'],
        'brier_not_worse': calibrated_summary['brier'] <= raw_summary['brier'] + 1e-12,
        'passed': bool(calibrated_summary['ece_10bin'] <= 0.05
                       and calibrated_summary['brier'] <= raw_summary['brier'] + 1e-12),
    }
    policy_gate = {
        'verifiable_precision_min': 0.94,
        'verifiable_precision_lower_95_min': 0.90,
        'coverage_min': 0.50,
        'verifiable_coverage_min': 0.75,
        'answered_precision_min': 0.70,
        'passed': bool(
            (selective['verifiable_precision'] or 0) >= 0.94
            and (selective['verifiable_precision_lower_95'] or 0) >= 0.90
            and selective['coverage'] >= 0.50
            and (selective['verifiable_coverage'] or 0) >= 0.75
            and (selective['answered_precision'] or 0) >= 0.70),
    }
    gates = {'macro_f1': macro_gate, 'calibration': calibration_gate,
             'selective_policy': policy_gate, 'coverage_flag': flag_gate}
    gates['promotion'] = bool(macro_gate['passed'] and calibration_gate['passed']
                              and policy_gate['passed'])

    report = {
        'version': 'v2-test-pass',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'test_pass': previous_pass + 1,
        'inputs': {
            'predictions': {'path': str(PREDICTIONS.relative_to(PROJECT_ROOT)),
                            'sha256': sha256(PREDICTIONS)},
            'policy': {'path': str(POLICY.relative_to(PROJECT_ROOT)), 'sha256': sha256(POLICY)},
            'calibration': {'path': str(CALIBRATION.relative_to(PROJECT_ROOT)),
                            'sha256': sha256(CALIBRATION)},
            'coverage_flag': {'path': str(COVERAGE.relative_to(PROJECT_ROOT)),
                              'sha256': sha256(COVERAGE)},
            'test_split': {'path': str(TEST_SPLIT.relative_to(PROJECT_ROOT)),
                           'sha256': sha256(TEST_SPLIT)},
        },
        'system': {'temperature': temperature,
                   'thresholds': dict(zip(LABELS, thresholds.tolist())),
                   'coverage_flag_threshold': flag_threshold},
        'overall': {'n': int(len(y_true)), 'uncalibrated': raw_summary,
                    'calibrated': calibrated_summary},
        'selective': selective,
        'coverage_flag': {'metrics': flag_metrics, 'gate': flag_gate},
        'gates': gates,
        'notes': [
            'Macro-F1 uses the classifier verdicts; abstention is a serving layer and '
            'does not rewrite the verdict path, so the frozen model cannot regress it.',
            'The coverage flag is optional for promotion; if its gate fails it is omitted.',
            'Thresholds and temperature were frozen before this pass; no test-driven tuning.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    with (OUT / 'test_predictions_v2.jsonl').open('w', encoding='utf-8') as handle:
        flag_by_id = {row['id']: bool(value) for row, value in zip(test_kept, flag)}
        for index, row in enumerate(predictions_raw):
            key = row['id']
            handle.write(json.dumps({
                'id': key, 'claim': row['claim'], 'gold': row['gold'],
                'prediction': LABELS[int(prediction[index])],
                'probabilities': {label: float(raw[index, number])
                                  for number, label in enumerate(LABELS)},
                'calibrated_probabilities': {label: float(calibrated[index, number])
                                             for number, label in enumerate(LABELS)},
                'confidence': float(calibrated[index].max()),
                'answered': bool(answered[index]),
                'coverage_flag': flag_by_id.get(key),
            }, ensure_ascii=False) + '\n')

    print(json.dumps({'gates': gates,
                      'selective': {key: selective[key] for key in
                                    ('coverage', 'answered_precision', 'verifiable_coverage',
                                     'verifiable_precision', 'verifiable_precision_lower_95',
                                     'nei_coverage', 'nei_precision')},
                      'coverage_flag': flag_metrics}, indent=2))
    print(f'Saved -> {OUTPUT.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
