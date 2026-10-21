"""P3: fit and select the evidence-coverage flag on FEVER train/val.

The flag predicts strict retrieval failure (no complete gold evidence set in the
top-5). It is fitted on train, selected on val, and frozen before the test pass.
No test data is read.

    python -m scripts.coverage_flag_v2
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.calibrate_v2 import load_matrix
from src.datasets.schema import LABELS, load_examples
from src.evaluation.coverage import (apply_flag, fit_flag, gold_evidence, select_threshold,
                                     strict_failure_labels)
from src.utils.config import PROJECT_ROOT
from src.verification.features import FEATURE_NAMES

OUT = PROJECT_ROOT / 'experiments/v2'
VAL_PREDICTIONS = OUT / 'val_predictions_v2.jsonl'
SPLITS = PROJECT_ROOT / 'data/splits'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', default='hybrid', choices=['bm25', 'dense', 'hybrid'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'coverage_flag.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    train_rows, train_x, _ = load_matrix('train', args.method)
    val_rows, val_x, _ = load_matrix('val', args.method)
    train_gold = gold_evidence(load_examples(SPLITS / 'fever_train.jsonl'))
    val_gold = gold_evidence(load_examples(SPLITS / 'fever_val.jsonl'))

    train_fail, train_kept = strict_failure_labels(train_rows, train_gold, args.method)
    val_fail, val_kept = strict_failure_labels(val_rows, val_gold, args.method)
    train_index = {row['id']: index for index, row in enumerate(train_rows)}
    val_index = {row['id']: index for index, row in enumerate(val_rows)}
    train_kept_x = train_x[[train_index[row['id']] for row in train_kept]]
    val_kept_x = val_x[[val_index[row['id']] for row in val_kept]]

    predictions = {row['id']: row for row in
                   (json.loads(line) for line in VAL_PREDICTIONS.read_text(
                       encoding='utf-8').splitlines() if line.strip())}
    missing = [row['id'] for row in val_kept if row['id'] not in predictions]
    if missing:
        raise ValueError(f'{len(missing)} val rows lack P2 predictions; run calibrate_v2 first')

    prediction = np.array([predictions[row['id']]['prediction'] for row in val_kept])
    gold = np.array([row['label'] for row in val_kept])
    errors = (prediction != gold).astype(np.int64)
    false_abstentions = (prediction == 'INSUFFICIENT').astype(np.int64)

    model, parameters = fit_flag(train_kept_x, train_fail, val_kept_x, val_fail)
    scores = apply_flag(parameters, val_kept_x)
    auroc = float(roc_auc_score(val_fail, scores))
    average_precision = float(average_precision_score(val_fail, scores))
    operating = select_threshold(scores, val_fail, errors)

    flag = scores >= operating['threshold']
    base_abstention = float(false_abstentions.mean())
    flagged_abstention = float(false_abstentions[flag].mean())
    operating['base_false_abstention_rate'] = base_abstention
    operating['flagged_false_abstention_rate'] = flagged_abstention
    operating['false_abstention_enrichment'] = (
        flagged_abstention / base_abstention if base_abstention > 0 else None)

    gate = {
        'min_val_auroc': 0.75,
        'min_error_enrichment': 2.0,
        'max_flag_rate': 0.25,
        'auroc': auroc,
        'error_enrichment': operating['error_enrichment'],
        'flag_rate': operating['flag_rate'],
        'passed': bool(auroc >= 0.75 and operating['error_enrichment'] >= 2.0
                       and operating['flag_rate'] <= 0.25),
    }
    if not gate['passed']:
        gate['consequence'] = 'omit the flag from serving; report it as a diagnostic only'

    report = {
        'version': 'v2-coverage-flag',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': args.method,
        'target': 'strict retrieval failure (no complete gold evidence set in top-5)',
        'splits': {
            'fit': 'train', 'selection': 'val',
            'train_rows': len(train_rows), 'val_rows': len(val_rows),
            'train_labelled': len(train_fail), 'val_labelled': len(val_fail),
            'train_failures': int(train_fail.sum()), 'val_failures': int(val_fail.sum()),
            'train_failure_rate': float(train_fail.mean()),
            'val_failure_rate': float(val_fail.mean()),
            'excluded': 'INSUFFICIENT claims and claims without a gold evidence set',
        },
        'features': FEATURE_NAMES,
        'model': {'type': 'StandardScaler + class-balanced LogisticRegression',
                  'parameters': parameters},
        'metrics': {'auroc': auroc, 'average_precision': average_precision,
                    'operating_point': operating},
        'gate': gate,
        'notes': [
            'Gold evidence is used only to create the label, never as a feature.',
            'The threshold is selected on val and frozen; the val operating point is '
            'optimistic because selection and evaluation use the same split.',
            'No test data was read.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    with (OUT / 'val_coverage_flags.jsonl').open('w', encoding='utf-8') as handle:
        for row, failure, score, is_flagged, pred, true, error, abstain in zip(
                val_kept, val_fail, scores, flag, prediction, gold, errors, false_abstentions):
            handle.write(json.dumps({
                'id': row['id'], 'claim': row['claim'], 'gold': true,
                'prediction': pred, 'strict_failure': bool(failure),
                'score': float(score), 'flagged': bool(is_flagged),
                'error': bool(error), 'false_abstention': bool(abstain),
            }, ensure_ascii=False) + '\n')

    print(json.dumps({'auroc': auroc, 'average_precision': average_precision,
                      'operating_point': operating, 'gate': gate}, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
