"""Summarize frozen second-verifier outputs; bootstrap only, never fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.datasets.schema import LABELS, read_jsonl
from src.evaluation.metrics import multiclass_metrics
from src.utils.config import PROJECT_ROOT


def f1(encoded):
    matrix = np.bincount(encoded, minlength=9).reshape(3, 3)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2 * matrix.diagonal(), denominator,
                           out=np.zeros(3, dtype=float), where=denominator > 0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT_ROOT / 'experiments/grounded_pilot_999')
    args = parser.parse_args()
    metrics = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
    rows = list(read_jsonl(args.run / 'predictions.jsonl'))
    assert len(rows) == metrics['n'] == len({r['id'] for r in rows})
    assert {r['id'] for r in rows} == set(metrics['protocol']['ids'])
    codes = {label: i for i, label in enumerate(LABELS)}
    gold = np.array([codes[r['gold']] for r in rows])
    old = gold * 3 + np.array([codes[r['baseline_label']] for r in rows])
    new = gold * 3 + np.array([codes[r['prediction']] for r in rows])
    rng = np.random.default_rng(20261022)
    deltas = []
    for _ in range(2000):
        index = rng.integers(0, len(rows), size=len(rows))
        deltas.append(f1(new[index]) - f1(old[index]))
    invalid = [r for r in rows if r['review']['status'] == 'invalid']
    valid = [r for r in rows if r['review']['status'] != 'invalid']
    summary = {
        'scope': 'conditional bootstrap on reused validation; not a new held-out estimate',
        'training_performed': False, 'test_evaluated': False,
        'paired_macro_f1_delta': f1(new) - f1(old),
        'bootstrap_95': [float(np.percentile(deltas, p)) for p in (2.5, 97.5)],
        'n_resamples': 2000, 'invalid_responses': len(invalid),
        'invalid_response_gains': sum(r['prediction'] == r['gold'] and r['baseline_label'] != r['gold'] for r in invalid),
        'invalid_response_losses': sum(r['prediction'] != r['gold'] and r['baseline_label'] == r['gold'] for r in invalid),
        'excluding_invalid': {
            key: multiclass_metrics([r['gold'] for r in valid], [r[field] for r in valid])
            for key, field in [('baseline', 'baseline_label'), ('candidate', 'prediction')]
        },
    }
    (args.run / 'paired_analysis.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
