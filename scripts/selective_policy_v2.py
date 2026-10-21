"""P3: select and freeze the v2 selective policy on val.

The policy answers a class only when its calibrated probability clears a
class-specific threshold. Thresholds are chosen on val under the pre-registered
constraints and written to ``experiments/v2/policy.json``; the test split is not
read here.

    python -m scripts.selective_policy_v2
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation.selective import LABELS, risk_coverage_curve, select_policy
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/v2'
PREDICTIONS = OUT / 'val_predictions_v2.jsonl'
CALIBRATION = OUT / 'calibration.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'policy.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    predictions = [json.loads(line) for line in PREDICTIONS.read_text(
        encoding='utf-8').splitlines() if line.strip()]
    probabilities = np.array([[row['calibrated_probabilities'][label] for label in LABELS]
                              for row in predictions])
    y_true = np.array([LABELS.index(row['gold']) for row in predictions], dtype=np.int64)
    calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))

    chosen = select_policy(y_true, probabilities)
    curve = risk_coverage_curve(y_true, probabilities)

    test_gate = {
        'verifiable_precision_min': 0.94,
        'verifiable_precision_lower_95_min': 0.90,
        'coverage_min': 0.50,
        'verifiable_coverage_min': 0.75,
        'answered_precision_min': 0.70,
        'note': ('The overall 90% precision target was infeasible at >=50% coverage on '
                 'val; the fallback selects max coverage subject to the verifiable '
                 'target. Test must meet these frozen minimums, not reproduce val exactly.'),
    }
    report = {
        'version': 'v2-selective-policy',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'policy': {
            'type': 'class_conditional_threshold',
            'rule': 'answer argmax if calibrated p[argmax] >= threshold[argmax], else abstain',
            'thresholds': dict(zip(LABELS, chosen['thresholds'])),
        },
        'calibration': {
            'temperature': calibration['calibrator']['final_temperature'],
            'source': 'experiments/v2/calibration.json',
        },
        'selection': {
            'split': 'val',
            'mode': chosen['selection_mode'],
            'feasible_both_at_min_coverage': chosen['feasible_both_at_min_coverage'],
            'constraints': chosen['constraints'],
            'search_grid': {'start': 0.30, 'stop': 0.95, 'step': 0.01},
        },
        'val_metrics': {key: value for key, value in chosen.items()
                        if key not in ('thresholds', 'selection_mode',
                                       'feasible_both_at_min_coverage', 'constraints')},
        'risk_coverage': curve,
        'test_gate': test_gate,
        'notes': [
            'No test data was read; thresholds are frozen for the single P4 pass.',
            'The selected thresholds maximise coverage on val under the fallback; '
            'class-specific thresholds are not a per-class precision guarantee.',
            'NEI coverage is reported separately: a confident wrong verdict on an NEI '
            'claim is still an error even when the verifiable subset looks strong.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'policy': report['policy'], 'selection': report['selection'],
                      'val_metrics': report['val_metrics'], 'test_gate': test_gate},
                     indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
