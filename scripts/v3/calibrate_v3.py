"""P1: calibration bake-off for v3.

Compares temperature scaling, vector scaling, and multiclass isotonic on the val
split with 2-fold cross-fitting. Selection is by out-of-fold Brier, subject to
ECE <= 0.05. No test data is read. If no calibrator strictly beats the uncalibrated
Brier, the script records a stop result instead of selecting one.

    python -m scripts.v3.calibrate_v3
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.v2.calibrate_v2 import fit_base, load_matrix, macro_f1
from src.datasets.schema import LABELS
from src.evaluation.calibration import (apply_isotonic_parameters, apply_vector_scaling,
                                        calibration_bins, class_calibration_error,
                                        cross_fit_isotonic, cross_fit_temperature,
                                        cross_fit_temperature_brier,
                                        cross_fit_vector_scaling,
                                        cross_fit_vector_scaling_brier,
                                        isotonic_to_parameters, softmax, summary)
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/v3'
SEED = 42
MAX_ECE = 0.05


def method_payload(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    result = summary(y_true, probabilities)
    result['macro_f1'] = macro_f1(y_true, probabilities)
    result['per_class_ece'] = {
        label: class_calibration_error(y_true, probabilities, index)
        for index, label in enumerate(LABELS)
    }
    result['reliability'] = calibration_bins(y_true, probabilities)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='hybrid', choices=['bm25', 'dense', 'hybrid'])
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'calibration.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    train_rows, train_x, train_y = load_matrix('train', args.method)
    val_rows, val_x, val_y = load_matrix('val', args.method)
    model = fit_base(train_x, train_y, args.c)
    logits = model.decision_function(val_x)
    raw = softmax(logits)
    raw_payload = method_payload(val_y, raw)

    methods: dict[str, dict] = {
        'temperature_nll': {'oof': None, 'final': None, 'fold_parameters': None},
        'temperature_brier': {'oof': None, 'final': None, 'fold_parameters': None},
        'vector_scaling_nll': {'oof': None, 'final': None, 'fold_parameters': None},
        'vector_scaling_brier': {'oof': None, 'final': None, 'fold_parameters': None},
        'isotonic_multiclass': {'oof': None, 'final': None, 'fold_parameters': None},
    }

    oof, temperatures, final_temperature = cross_fit_temperature(logits, val_y,
                                                                n_splits=2, seed=SEED)
    methods['temperature_nll'].update({
        'oof': oof, 'final': {'temperature': final_temperature},
        'fold_parameters': temperatures,
    })

    oof, temperatures, final_temperature = cross_fit_temperature_brier(
        logits, val_y, n_splits=2, seed=SEED)
    methods['temperature_brier'].update({
        'oof': oof, 'final': {'temperature': final_temperature},
        'fold_parameters': temperatures,
    })

    oof, parameters, final_parameters = cross_fit_vector_scaling(logits, val_y,
                                                                n_splits=2, seed=SEED)
    methods['vector_scaling_nll'].update({
        'oof': oof, 'final': final_parameters, 'fold_parameters': parameters,
    })

    oof, parameters, final_parameters = cross_fit_vector_scaling_brier(
        logits, val_y, n_splits=2, seed=SEED)
    methods['vector_scaling_brier'].update({
        'oof': oof, 'final': final_parameters, 'fold_parameters': parameters,
    })

    oof, fold_models, final_models = cross_fit_isotonic(raw, val_y, n_splits=2, seed=SEED)
    methods['isotonic_multiclass'].update({
        'oof': oof, 'final': isotonic_to_parameters(final_models),
        'fold_parameters': [isotonic_to_parameters(models) for models in fold_models],
    })

    report_methods = {}
    for name, payload in methods.items():
        calibrated = payload['oof']
        metrics = method_payload(val_y, calibrated)
        metrics['brier_improvement_vs_raw'] = raw_payload['brier'] - metrics['brier']
        metrics['strictly_better_brier'] = metrics['brier'] < raw_payload['brier'] - 1e-12
        metrics['ece_ok'] = metrics['ece_10bin'] <= MAX_ECE
        report_methods[name] = {
            'metrics': metrics, 'final': payload['final'],
            'fold_parameters': payload['fold_parameters'],
        }

    eligible = [name for name, payload in report_methods.items()
                if payload['metrics']['strictly_better_brier']
                and payload['metrics']['ece_ok']]
    selected = (min(eligible, key=lambda name: report_methods[name]['metrics']['brier'])
                if eligible else None)

    final_probabilities = None
    if selected in ('temperature_nll', 'temperature_brier'):
        final_probabilities = softmax(logits, report_methods[selected]['final']['temperature'])
    elif selected in ('vector_scaling_nll', 'vector_scaling_brier'):
        final_probabilities = apply_vector_scaling(logits,
                                                   report_methods[selected]['final'])
    elif selected == 'isotonic_multiclass':
        final_probabilities = apply_isotonic_parameters(
            raw, report_methods[selected]['final'])

    report = {
        'version': 'v3-calibration',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': args.method,
        'base_model': {
            'fit_split': 'train', 'selection_split': 'val',
            'C': args.c, 'seed': SEED,
            'n_train': len(train_y), 'n_val': len(val_y),
            'val_macro_f1': raw_payload['macro_f1'],
        },
        'raw': raw_payload,
        'methods': report_methods,
        'selection': {
            'rule': 'lowest out-of-fold Brier with ECE <= 0.05',
            'eligible': eligible,
            'selected': selected,
            'status': 'selected' if selected else 'stop_no_calibrator_beats_brier',
        },
        'final_calibrator': report_methods[selected]['final'] if selected else None,
        'notes': [
            'OOF metrics are the selection metrics; the final calibrator is fitted on all val for serving.',
            'No test data was read.',
            'If status is stop_no_calibrator_beats_brier, v3 does not proceed to P2/P4.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    predictions = OUT / 'val_predictions_v3.jsonl'
    oof_selected = methods[selected]['oof'] if selected else raw
    with predictions.open('w', encoding='utf-8') as handle:
        for index, row in enumerate(val_rows):
            handle.write(json.dumps({
                'id': row['id'], 'claim': row['claim'],
                'gold': LABELS[int(val_y[index])],
                'prediction': LABELS[int(oof_selected[index].argmax())],
                'probabilities': {label: float(raw[index, number])
                                  for number, label in enumerate(LABELS)},
                'calibrated_probabilities_oof': {
                    label: float(oof_selected[index, number])
                    for number, label in enumerate(LABELS)},
                'calibrated_probabilities_final': ({
                    label: float(final_probabilities[index, number])
                    for number, label in enumerate(LABELS)}
                    if final_probabilities is not None else None),
                'calibrator': selected,
            }, ensure_ascii=False) + '\n')

    print(json.dumps({
        'selected': selected,
        'status': report['selection']['status'],
        'raw_brier': raw_payload['brier'],
        'methods': {name: {'brier': payload['metrics']['brier'],
                           'ece': payload['metrics']['ece_10bin'],
                           'better': payload['metrics']['strictly_better_brier']}
                    for name, payload in report_methods.items()},
    }, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
