"""Compare deterministic evidence policies on cached FEVER validation only.

No model loading, training, new NLI, learned-model changes, or test evaluation.
Run: python -m scripts.evaluate_evidence_policies
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.datasets.schema import read_jsonl
from src.evaluation.metrics import multiclass_metrics
from src.evaluation.reliability import reliability_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide
from src.verification.evidence import MODES, acceptance, select_evidence


def evaluate(rows, examples, method, mode, entail, contra):
    predictions, accepted, diagnostics = [], [], []
    for row in rows:
        pool = row[method][:5]
        selected = select_evidence(pool, mode)
        probs = np.asarray([hit['nli'] for hit in selected])
        label = decide(aggregate_probabilities(probs), entail, contra)
        # Preserve conflicts seen in the original pool, even if selection hides one side.
        conflict = bool(pool and max(h['nli'][0] for h in pool) >= entail and
                        max(h['nli'][2] for h in pool) >= contra)
        decision = acceptance(label, has_evidence=bool(selected), conflict=conflict)
        predictions.append(label)
        accepted.append(decision['status'] == 'answer')
        groups = examples[row['id']]['meta'].get('evidence_sets', [])
        top = {(h['page_id'], int(h['sent_id'])) for h in selected}
        complete = (any({(p, int(s)) for p, s in group} <= top for group in groups)
                    if groups else None)
        diagnostics.append({'id': row['id'], 'gold': row['label'], 'prediction': label,
                            'decision': decision, 'gold_complete_in_selected': complete,
                            'selected_evidence': [{'page_id': h['page_id'], 'sent_id': h['sent_id']}
                                                  for h in selected]})
    gold = [row['label'] for row in rows]
    result = {'classification': multiclass_metrics(gold, predictions),
              'reliability': reliability_metrics(gold, predictions, accepted)}
    verifiable = [r for r in diagnostics if r['gold_complete_in_selected'] is not None]
    result['evidence'] = {
        'n_with_gold_sets': len(verifiable),
        'complete_gold_sets_selected': sum(r['gold_complete_in_selected'] for r in verifiable),
        'false_insufficient_with_complete_gold': sum(
            r['gold_complete_in_selected'] and r['prediction'] == 'INSUFFICIENT'
            for r in verifiable),
    }
    return result, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'experiments/improvements')
    args = parser.parse_args()
    if (args.out / 'metrics_evidence_policies.json').exists():
        raise SystemExit('Results already exist; use a separate --out directory.')
    cache = PROJECT_ROOT / 'data/processed/retrieval/fever_val.jsonl'
    split = PROJECT_ROOT / 'data/splits/fever_val.jsonl'
    rows = list(read_jsonl(cache))
    examples = {r['id']: r for r in read_jsonl(split)}
    if len(rows) != len(examples) or {r['id'] for r in rows} != set(examples):
        raise ValueError('Validation cache/split IDs differ')
    if any(r['label'] != examples[r['id']]['label'] or
           r['claim'] != examples[r['id']]['answer'] for r in rows):
        raise ValueError('Validation cache/split content differs')
    cfg = load_config()['detector']
    te, tc = cfg['entail_threshold'], cfg['contra_threshold']
    report = {'study': 'cached-evidence-selection-v1', 'split': 'val',
              'training_performed': False, 'inference_performed': False,
              'threshold_selection_performed': False, 'test_evaluated': False,
              'thresholds': {'entail': te, 'contra': tc},
              'modes': list(MODES), 'n': len(rows),
              'input_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (cache, split)},
              'promotion': False, 'methods': {}}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'validation_predictions.jsonl').open('w', encoding='utf-8') as handle:
        for method in ('bm25', 'dense', 'hybrid'):
            report['methods'][method] = {}
            for mode in MODES:
                result, diagnostics = evaluate(rows, examples, method, mode, te, tc)
                report['methods'][method][mode] = result
                for row in diagnostics:
                    handle.write(json.dumps(dict(row, method=method, evidence_policy=mode)) + '\n')
                print(f'{method}/{mode}: F1={result["classification"]["macro_f1"]:.4f}, '
                      f'accepted accuracy={result["reliability"]["accepted_accuracy"]}, '
                      f'acceptance={result["reliability"]["acceptance_rate"]:.3f}')
    (args.out / 'metrics_evidence_policies.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
