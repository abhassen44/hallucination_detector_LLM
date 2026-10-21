"""P2: calibration and training-variance study on the FEVER val split.

Fits the v1-selected hybrid configuration on train (frozen C), then:

* reports uncalibrated val ECE/Brier/NLL;
* fits temperature scaling with 2-fold cross-fitting inside val and reports the
  out-of-fold metrics;
* refits the base model on bootstrap samples of train to estimate training
  variance on val.

The test split is never loaded. The output is ``experiments/v2/calibration.json``
plus val probabilities for the selective-policy step.

    python -m scripts.calibrate_v2
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import LABELS
from src.evaluation.calibration import (brier_score, calibration_bins,
                                        class_calibration_error, cross_fit_temperature,
                                        expected_calibration_error, summary, softmax)
from src.evaluation.fit_detector import SEED, _sha256
from src.utils.config import PROJECT_ROOT
from src.verification.features import build_matrix, load_split

OUT = PROJECT_ROOT / 'experiments/v2'
SIM_DIR = PROJECT_ROOT / 'data/processed/features'
V1_METRICS = PROJECT_ROOT / 'experiments/stage5_learned/metrics.json'


def load_matrix(split: str, method: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = load_split(split)
    path = SIM_DIR / f'sim_{split}_{method}.npz'
    cached = np.load(path, allow_pickle=False)
    if list(cached['ids']) != [row['id'] for row in rows]:
        raise ValueError(f'{path}: cached similarity ids do not match the {split} split')
    matrix = build_matrix(rows, method, cached['sims'])
    labels = np.array([LABELS.index(row['label']) for row in rows], dtype=np.int64)
    return rows, matrix, labels


def fit_base(matrix: np.ndarray, labels: np.ndarray, c_value: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=c_value, max_iter=5000, random_state=SEED))
    model.fit(matrix, labels)
    return model


def macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(labels, probabilities.argmax(axis=1), average='macro',
                          zero_division=0))


def bootstrap_refits(matrix: np.ndarray, labels: np.ndarray, val_matrix: np.ndarray,
                     val_labels: np.ndarray, c_value: float, n_resamples: int) -> dict:
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(labels), len(labels))
        model = fit_base(matrix[index], labels[index], c_value)
        probabilities = model.predict_proba(val_matrix)
        rows.append({
            'macro_f1': macro_f1(val_labels, probabilities),
            'accuracy': float((probabilities.argmax(axis=1) == val_labels).mean()),
            'ece_10bin': expected_calibration_error(val_labels, probabilities),
            'brier': brier_score(val_labels, probabilities),
        })

    def spread(name: str) -> dict:
        values = np.array([row[name] for row in rows])
        return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)),
                'lower_95': float(np.percentile(values, 2.5)),
                'upper_95': float(np.percentile(values, 97.5))}

    return {'n_resamples': n_resamples, 'resample': 'train rows, with replacement',
            'evaluated_on': 'val', 'macro_f1': spread('macro_f1'),
            'accuracy': spread('accuracy'), 'ece_10bin': spread('ece_10bin'),
            'brier': spread('brier')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='hybrid', choices=['bm25', 'dense', 'hybrid'])
    parser.add_argument('--c', type=float, default=None,
                        help='Frozen regularisation strength; defaults to the v1 selected C.')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'calibration.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    v1 = json.loads(V1_METRICS.read_text(encoding='utf-8'))['methods'][args.method]
    c_value = args.c if args.c is not None else float(v1['selected_C'])

    train_rows, train_x, train_y = load_matrix('train', args.method)
    val_rows, val_x, val_y = load_matrix('val', args.method)
    if {row['id'] for row in train_rows} & {row['id'] for row in val_rows}:
        raise ValueError('train and val ids overlap')

    model = fit_base(train_x, train_y, c_value)
    logits = model.decision_function(val_x)
    raw = softmax(logits, 1.0)
    raw_summary = summary(val_y, raw)
    raw_summary['macro_f1'] = macro_f1(val_y, raw)

    calibrated, fold_temperatures, final_temperature = cross_fit_temperature(
        logits, val_y, n_splits=2, seed=SEED)
    calibrated_summary = summary(val_y, calibrated)
    calibrated_summary['macro_f1'] = macro_f1(val_y, calibrated)

    raw_ece = raw_summary['ece_10bin']
    calibrated_ece = calibrated_summary['ece_10bin']
    relative_reduction = (1.0 - calibrated_ece / raw_ece) if raw_ece > 0 else None
    gate = {
        'min_relative_ece_reduction': 0.25,
        'relative_ece_reduction': relative_reduction,
        'brier_not_worse': calibrated_summary['brier'] <= raw_summary['brier'] + 1e-12,
        'passed': bool(relative_reduction is not None and relative_reduction >= 0.25
                       and calibrated_summary['brier'] <= raw_summary['brier'] + 1e-12),
    }

    reliability = {
        'uncalibrated': calibration_bins(val_y, raw),
        'calibrated_oof': calibration_bins(val_y, calibrated),
    }
    per_class = {
        name: {LABELS[index]: class_calibration_error(val_y, probabilities, index)
               for index in range(len(LABELS))}
        for name, probabilities in (('uncalibrated', raw), ('calibrated_oof', calibrated))
    }

    report = {
        'version': 'v2-calibration',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': args.method,
        'base_model': {
            'fit_split': 'train',
            'selection_split': 'val (v1 selected C, frozen here)',
            'C': c_value,
            'seed': SEED,
            'n_train': len(train_y),
            'n_val': len(val_y),
            'val_macro_f1': raw_summary['macro_f1'],
            'v1_val_macro_f1': v1['val_macro_f1_at_selection'],
            'train_ids_sha256': _sha256(PROJECT_ROOT / 'data/splits/fever_train.jsonl'),
            'val_ids_sha256': _sha256(PROJECT_ROOT / 'data/splits/fever_val.jsonl'),
        },
        'calibrator': {
            'method': 'temperature scaling',
            'cross_fit_folds': 2,
            'fold_temperatures': fold_temperatures,
            'final_temperature': final_temperature,
            'fit_split': 'val',
            'note': 'Reported metrics are out-of-fold; the full-val temperature is for serving.',
        },
        'uncalibrated': raw_summary,
        'calibrated_oof': calibrated_summary,
        'per_class_ece': per_class,
        'reliability': reliability,
        'gate': gate,
        'bootstrap_refits': bootstrap_refits(train_x, train_y, val_x, val_y, c_value,
                                             args.bootstrap),
        'notes': [
            'No test data was read; the test split is untouched.',
            'C was selected on val in v1, so val is not fully pristine for calibration. '
            'The calibrator is one-dimensional and cross-fitted; report the caveat.',
            'Bootstrap refits vary training rows only; they do not vary the val sample.',
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    predictions = OUT / 'val_predictions_v2.jsonl'
    with predictions.open('w', encoding='utf-8') as handle:
        for row, gold, raw_probs, calibrated_probs in zip(
                val_rows, val_y, raw, calibrated):
            predicted = LABELS[int(calibrated_probs.argmax())]
            handle.write(json.dumps({
                'id': row['id'], 'claim': row['claim'], 'gold': LABELS[int(gold)],
                'prediction': predicted,
                'probabilities': {label: float(prob) for label, prob in zip(LABELS, raw_probs)},
                'calibrated_probabilities': {label: float(prob)
                                             for label, prob in zip(LABELS, calibrated_probs)},
                'confidence': float(calibrated_probs.max()),
                'correct': predicted == LABELS[int(gold)],
            }, ensure_ascii=False) + '\n')

    print(json.dumps({
        'val_macro_f1': raw_summary['macro_f1'],
        'v1_val_macro_f1': v1['val_macro_f1_at_selection'],
        'ece_uncalibrated': raw_summary['ece_10bin'],
        'ece_calibrated_oof': calibrated_summary['ece_10bin'],
        'temperature': final_temperature,
        'gate': gate,
    }, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}, {predictions.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
