"""Saved validation diagnostics only; no training, inference, or test data."""
import hashlib
import json
from collections import Counter

from src.datasets.schema import read_jsonl
from src.evaluation.reliability import reliability_metrics
from src.evaluation.metrics import multiclass_metrics
from src.utils.config import PROJECT_ROOT
from src.verification.detector import aggregate_probabilities, decide


def main():
    root = PROJECT_ROOT
    paths = [root / 'experiments/improvements/validation_predictions.jsonl',
             root / 'data/processed/retrieval/fever_val.jsonl',
             root / 'data/splits/fever_val.jsonl',
             root / 'experiments/improvements/metrics_evidence_policies.json']
    recorded = json.loads(paths[3].read_text(encoding='utf-8'))
    for path in paths[1:3]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded['input_sha256'][
            str(path.relative_to(root))], 'Input changed since the evidence study'
    saved = list(read_jsonl(paths[0]))
    cached = {r['id']: r for r in read_jsonl(paths[1])}
    examples = {r['id']: r for r in read_jsonl(paths[2])}
    policies = {(method, mode): {r['id']: r for r in saved
                 if r['method'] == method and r['evidence_policy'] == mode}
                for method in ('bm25', 'dense', 'hybrid')
                for mode in ('all', 'top1', 'top3', 'same_page')}
    for rows in policies.values():
        assert len(rows) == recorded['n'] and set(rows) == set(cached) == set(examples)
    result = {'scope': 'exploratory validation diagnostics', 'training_performed': False,
              'inference_performed': False, 'test_evaluated': False, 'methods': {}}
    for method in ('bm25', 'dense', 'hybrid'):
        focused, full = policies[method, 'top1'], policies[method, 'all']
        counts, transitions, gold_shape = Counter(), Counter(), Counter()
        for id_, row in focused.items():
            correct = row['prediction'] == row['gold']
            old_correct = full[id_]['prediction'] == row['gold']
            counts['correct' if correct else 'errors'] += 1
            counts['gained' if correct and not old_correct else
                   'lost' if old_correct and not correct else 'unchanged_correctness'] += 1
            transitions[f'{row["gold"]}->{row["prediction"]}'] += 1
            if correct and row['prediction'] != 'INSUFFICIENT' and row['decision']['status'] == 'abstain':
                counts['correct_factual_verdict_rejected_for_pool_conflict'] += 1
            if not correct and row['gold'] != 'INSUFFICIENT':
                hits = cached[id_][method][:5]
                if any(decide(aggregate_probabilities([h['nli']]),
                              recorded['thresholds']['entail'], recorded['thresholds']['contra'])
                       == row['gold'] for h in hits[1:]):
                    counts['verifiable_errors_with_correct_single_sentence_verdict_in_lower_hits'] += 1
                if full[id_]['gold_complete_in_selected']:
                    counts['verifiable_errors_with_complete_gold_in_pool'] += 1
            groups = examples[id_]['meta'].get('evidence_sets', [])
            if groups:
                category = 'single_sentence_gold_available' if any(len(g) == 1 for g in groups) else 'multi_sentence_gold_required'
                gold_shape[f'{category}/n'] += 1
                gold_shape[f'{category}/errors'] += not correct
        result['methods'][method] = {'counts': dict(counts),
                                     'confusion_counts': dict(sorted(transitions.items())),
                                     'gold_evidence_shape': dict(gold_shape)}
    # Predefined acceptance-only checks, never select a factual label using gold.
    # Verdicts remain hybrid top1; disagreement causes operational abstention.
    hybrid = policies['hybrid', 'top1']
    result['consensus_acceptance'] = {}
    for name, methods in [('hybrid_only', ('hybrid',)),
                          ('dense_hybrid', ('dense', 'hybrid')),
                          ('all_three', ('bm25', 'dense', 'hybrid'))]:
        accepted = [all(policies[m, 'top1'][id_]['prediction'] == row['prediction'] and
                        policies[m, 'top1'][id_]['decision']['status'] == 'answer'
                        for m in methods) for id_, row in hybrid.items()]
        result['consensus_acceptance'][name] = reliability_metrics(
            [r['gold'] for r in hybrid.values()],
            [r['prediction'] for r in hybrid.values()], accepted)
    # Two fixed fallback hypotheses: expand only if the first passage is neutral.
    # These are development comparisons, not calibrated or promoted policies.
    result['neutral_fallback'] = {}
    for mode in ('top3', 'same_page'):
        selected = [policies['hybrid', mode][id_] if r['prediction'] == 'INSUFFICIENT' else r
                    for id_, r in hybrid.items()]
        result['neutral_fallback'][mode] = {
            'classification': multiclass_metrics([r['gold'] for r in selected],
                                                [r['prediction'] for r in selected]),
            'reliability': reliability_metrics([r['gold'] for r in selected],
                                               [r['prediction'] for r in selected],
                                               [r['decision']['status'] == 'answer' for r in selected]),
        }
    result['caveats'] = [
        'Gold-based recovery counts are oracle diagnostics, not implementable policies or gains.',
        'Consensus shares one NLI verifier and corpus; agreement need not mean correctness.',
        'Validation is repeatedly reused; no serving change or test generalization claim.',
    ]
    result['input_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}
    out = root / 'experiments/improvements/remaining_error_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
