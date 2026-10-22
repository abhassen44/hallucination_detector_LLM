"""Audit saved v2 results only: no inference, fitting, or threshold selection.

Run: python -m scripts.review_current_results
Historical artifacts are read-only. Output is a separate diagnostic, not a test pass.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ('SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT')


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len({row['id'] for row in rows}) == len(rows), path
    return rows


def ratio(a, b):
    return a / b if b else None


def main():
    sources = {}

    def source(name):
        path = ROOT / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    protocol = json.loads(source('experiments/stage5_learned/protocol.json').read_text(encoding='utf-8'))
    policy = json.loads(source('experiments/v2/policy.json').read_text(encoding='utf-8'))
    thresholds = policy['policy']['thresholds']
    audit = {'scope': 'saved-prediction diagnostic; no new test pass',
             'training_performed': False, 'inference_performed': False,
             'threshold_selection_performed': False, 'splits': {}}
    for split in ('val', 'test'):
        rows = read_rows(source(f'experiments/v2/{split}_predictions_v2.jsonl'))
        for row in rows:
            answered = row['confidence'] >= thresholds[row['prediction']]
            if 'answered' in row:
                assert row['answered'] == answered, row['id']
            row['answered'] = answered
        assert len(rows) == protocol['row_counts'][split]
        confusion = [[sum(r['gold'] == g and r['prediction'] == p for r in rows)
                      for p in LABELS] for g in LABELS]
        selected = [r for r in rows if r['answered']]
        selected_confusion = [[sum(r['gold'] == g and r['prediction'] == p
                                  for r in selected) for p in LABELS] for g in LABELS]
        per_prediction = {}
        per_gold = {}
        for label in LABELS:
            predicted = [r for r in selected if r['prediction'] == label]
            gold = [r for r in rows if r['gold'] == label]
            correct = [r for r in gold if r['prediction'] == label]
            answered_gold = [r for r in gold if r['answered']]
            per_prediction[label] = {
                'selected': len(predicted),
                'correct': sum(r['gold'] == label for r in predicted),
                'precision': ratio(sum(r['gold'] == label for r in predicted), len(predicted)),
            }
            per_gold[label] = {
                'n': len(gold), 'correct_before_policy': len(correct),
                'selected': len(answered_gold),
                'correct_selected': sum(r['answered'] for r in correct),
                'correct_rejected': sum(not r['answered'] for r in correct),
                'selected_accuracy': ratio(sum(r['prediction'] == label for r in answered_gold),
                                           len(answered_gold)),
            }
        result = {'n': len(rows), 'label_order': LABELS, 'confusion': confusion,
                  'selected_confusion': selected_confusion,
                  'selected_n': len(selected),
                  'selected_correct': sum(r['prediction'] == r['gold'] for r in selected),
                  'per_predicted_class': per_prediction, 'per_gold_class': per_gold}
        cache_name = f'data/processed/retrieval/fever_{split}.jsonl'
        cache_path = source(cache_name)
        expected = protocol['cache_sha256'][split]
        crlf_hash = hashlib.sha256(cache_path.read_bytes().replace(b'\r\n', b'\n')
                                   .replace(b'\n', b'\r\n')).hexdigest()
        assert expected in (sources[cache_name], crlf_hash), 'cache differs from frozen protocol'
        result['cache_protocol_match'] = ('exact' if sources[cache_name] == expected
                                          else 'exact after LF-to-CRLF conversion')
        cached = {r['id']: r for r in read_rows(cache_path)}
        examples = {r['id']: r for r in read_rows(source(f'data/splits/fever_{split}.jsonl'))}
        assert set(cached) == set(examples) == {r['id'] for r in rows}
        buckets = Counter()
        ranks = {'entailment': Counter(), 'contradiction': Counter()}
        for row in rows:
            hits = cached[row['id']]['hybrid'][:5]
            for name, index in [('entailment', 0), ('contradiction', 2)]:
                values = [h['nli'][index] for h in hits]
                if values:
                    # Reproduce _rank(values, argmax(values)) on non-tied maxima.
                    if values.count(max(values)) == 1:
                        maximum = values.index(max(values))
                        rank = sorted(range(len(values)), key=lambda i: -values[i]).index(maximum) + 1
                        ranks[name][rank] += 1
                    else:
                        ranks[name]['tied_maximum'] += 1
            if row['gold'] == 'INSUFFICIENT':
                continue
            groups = examples[row['id']]['meta'].get('evidence_sets', [])
            if not groups:
                continue
            top = {(h['page_id'], int(h['sent_id'])) for h in hits}
            complete = any({(p, int(s)) for p, s in group} <= top for group in groups)
            evidence = 'gold_complete' if complete else 'gold_incomplete'
            outcome = ('correct' if row['prediction'] == row['gold'] else
                       'predicted_insufficient' if row['prediction'] == 'INSUFFICIENT' else
                       'wrong_supported_or_contradicted')
            buckets[f'{evidence}/{outcome}'] += 1
        result['retrieval_vs_verdict'] = dict(sorted(buckets.items()))
        result['current_nli_rank_features'] = {k: dict(v) for k, v in ranks.items()}
        audit['splits'][split] = result
    recorded = json.loads(source('experiments/v2/metrics_v2.json').read_text(encoding='utf-8'))
    test = audit['splits']['test']
    matrix = test['confusion']
    accuracy = sum(matrix[i][i] for i in range(3)) / test['n']
    macro_f1 = sum(2 * matrix[i][i] / (sum(matrix[i]) +
                   sum(matrix[j][i] for j in range(3))) for i in range(3)) / 3
    assert abs(accuracy - recorded['overall']['calibrated']['accuracy']) < 1e-12
    assert abs(macro_f1 - recorded['overall']['calibrated']['macro_f1']) < 1e-12
    assert test['selected_n'] == recorded['selective']['n_answered']
    assert abs(test['selected_correct'] / test['selected_n'] -
               recorded['selective']['answered_precision']) < 1e-12
    audit['sources_sha256'] = sources
    output = ROOT / 'experiments/results_review/saved_prediction_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit['splits'], indent=2))
    print(f'Saved {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
