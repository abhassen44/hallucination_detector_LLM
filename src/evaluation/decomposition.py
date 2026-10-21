"""Where the remaining error lives on verifiable claims.

The Stage 3/4 pipeline is scored on all 1,998 FEVER test claims, including NEI claims
that have no gold evidence. The Stage 2 oracle answers only the 1,324 claims that do
have gold evidence. Comparing the two headline numbers therefore mixes two questions:

  1. does the pipeline find and use evidence for claims that have it? (coverage)
  2. when it does answer, is its verdict right? (verifier quality)

This script restricts every system to the oracle's 1,324 claims and splits the error
into four exhaustive, mutually exclusive buckets:

  abstained  + oracle correct   -> evidence/retrieval loss (answer was recoverable)
  abstained  + oracle wrong     -> hard claim (oracle evidence does not settle it)
  confident  + oracle wrong     -> shared verifier difficulty
  confident  + oracle correct   -> error introduced by retrieved evidence/pipeline

The oracle verdicts are recomputed once with the frozen Stage 2 model (NLI-concat) and
cached; the recount must reproduce the committed Stage 2 2-way accuracy exactly.

    python -m src.evaluation.decomposition            # uses cache, computes if missing
    python -m src.evaluation.decomposition --recompute
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import CONTRADICTED, INSUFFICIENT, SUPPORTED, load_examples
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide

METHODS = ('bm25', 'dense', 'hybrid')
OUT = PROJECT_ROOT / 'experiments/stage5_learned'
SPLIT = PROJECT_ROOT / 'data/splits/fever_test.jsonl'
RETRIEVAL_CACHE = PROJECT_ROOT / 'data/processed/retrieval/fever_test.jsonl'
STAGE2 = PROJECT_ROOT / 'experiments/stage2_oracle'
ORACLE_CACHE = OUT / 'oracle_predictions_test.jsonl'


def oracle_subset():
    return [e for e in load_examples(SPLIT) if e.label != INSUFFICIENT and e.context]


def oracle_predictions(exs, recompute: bool = False) -> dict[str, dict]:
    """Frozen Stage 2 NLI-concat verdicts for the verifiable claims, cached."""
    if ORACLE_CACHE.exists() and not recompute:
        rows = [json.loads(line) for line in ORACLE_CACHE.read_text(encoding='utf-8').splitlines()]
        if len(rows) == len(exs) and {r['id'] for r in rows} == {e.id for e in exs}:
            return {r['id']: r for r in rows}

    from src.verification.nli import NLIModel

    nli = NLIModel()
    probabilities = nli.predict([' '.join(e.context) for e in exs], [e.answer for e in exs])
    rows = []
    for example, p in zip(exs, probabilities):
        rows.append({'id': example.id, 'gold': example.label, 'nli_model': nli.model_name,
                     'probabilities': [float(x) for x in p],
                     'prediction': SUPPORTED if p[0] >= p[2] else CONTRADICTED})
    ORACLE_CACHE.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return {r['id']: r for r in rows}


def fixed_predictions(ids, method: str, entail: float, contra: float, cache: dict) -> list[str]:
    return [decide(aggregate_probabilities([h['nli'] for h in cache[i][method][:5]]), entail, contra)
            for i in ids]


def decompose(gold: list[str], predictions: list[str], oracle: list[str]) -> dict:
    abstained = [i for i, p in enumerate(predictions) if p == INSUFFICIENT]
    answered = [i for i, p in enumerate(predictions) if p != INSUFFICIENT]
    errors = [i for i, (g, p) in enumerate(zip(gold, predictions)) if p != g]
    oracle_right = [o == g for g, o in zip(gold, oracle)]
    buckets = {
        'abstained_oracle_correct': sum(1 for i in abstained if oracle_right[i]),
        'abstained_oracle_wrong': sum(1 for i in abstained if not oracle_right[i]),
        'confident_oracle_correct': sum(1 for i in errors
                                        if predictions[i] != INSUFFICIENT and oracle_right[i]),
        'confident_oracle_wrong': sum(1 for i in errors
                                      if predictions[i] != INSUFFICIENT and not oracle_right[i]),
        'pipeline_wins_over_oracle': sum(1 for i in answered
                                         if predictions[i] == gold[i] and not oracle_right[i]),
    }
    n = len(gold)
    oracle_errors = sum(1 for right in oracle_right if not right)
    answered_correct = sum(1 for i in answered if predictions[i] == gold[i])
    return {
        'n': n,
        'answered': len(answered), 'abstained': len(abstained),
        'coverage': len(answered) / n,
        'accuracy': sum(1 for g, p in zip(gold, predictions) if g == p) / n,
        'pipeline_accuracy_answered': (answered_correct / len(answered) if answered else None),
        'selective_accuracy': (answered_correct / len(answered) if answered else None),
        'oracle_accuracy': 1 - oracle_errors / n,
        'oracle_accuracy_answered': (1 - (buckets['confident_oracle_wrong']
                                          + buckets['pipeline_wins_over_oracle']) / len(answered)
                                     if answered else None),
        'oracle_error': oracle_errors,
        'pipeline_error': len(errors),
        'error_buckets': buckets,
        'error_buckets_sum': sum(v for k, v in buckets.items() if k != 'pipeline_wins_over_oracle'),
        'net_incremental_error_vs_oracle': len(errors) - oracle_errors,
        'accounting': 'pipeline_error = four error buckets; net incremental error = '
                      'abstained_oracle_correct + confident_oracle_correct - pipeline_wins',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'decomposition.json')
    parser.add_argument('--recompute', action='store_true',
                        help='ignore the cached oracle predictions and rerun NLI-concat')
    args = parser.parse_args()

    cfg = load_config()
    entail = cfg['detector']['entail_threshold']
    contra = cfg['detector']['contra_threshold']

    exs = oracle_subset()
    ids = [e.id for e in exs]
    gold = [e.label for e in exs]
    oracle_rows = oracle_predictions(exs, args.recompute)
    oracle = [oracle_rows[i]['prediction'] for i in ids]

    stage2_path = Path(sorted(glob.glob(str(STAGE2 / 'metrics_*.json')))[-1])
    stage2 = json.loads(stage2_path.read_text(encoding='utf-8'))
    committed = stage2['fever']['nli_concat/test']['2way']
    expected = committed['accuracy']
    observed = sum(1 for g, o in zip(gold, oracle) if g == o) / len(gold)
    order = [SUPPORTED, CONTRADICTED]
    confusion = [[sum(1 for g, o in zip(gold, oracle) if g == row and o == column)
                  for column in order] for row in order]
    if abs(observed - expected) > 1e-9 or confusion != committed['confusion_matrix']:
        raise SystemExit('oracle recount does not reproduce the committed Stage 2 2-way '
                         f'confusion matrix {confusion} vs {committed["confusion_matrix"]}')

    cache = {row['id']: row for row in
             (json.loads(line) for line in RETRIEVAL_CACHE.read_text(encoding='utf-8').splitlines())}
    results = {}
    for method in METHODS:
        learned = {row['id']: row['prediction'] for row in
                   (json.loads(line) for line in
                    (OUT / f'predictions_{method}.jsonl').read_text(encoding='utf-8').splitlines())}
        fixed = fixed_predictions(ids, method, entail, contra, cache)
        results[method] = {
            'fixed_rule': decompose(gold, fixed, oracle),
            'learned': decompose(gold, [learned[i] for i in ids], oracle),
        }
        for system, stats in results[method].items():
            buckets = stats['error_buckets']
            print(f'[{method}/{system}] coverage {stats["coverage"]:.3f} '
                  f'selective acc {stats["selective_accuracy"]:.3f} '
                  f'errors {stats["pipeline_error"]:3d} = '
                  f'{buckets["abstained_oracle_correct"]} abstain/evid + '
                  f'{buckets["abstained_oracle_wrong"]} abstain/hard + '
                  f'{buckets["confident_oracle_wrong"]} shared + '
                  f'{buckets["confident_oracle_correct"]} introduced', flush=True)

    report = {
        'analysis': 'verifiable_subset_error_decomposition',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'subset': 'FEVER test claims with gold evidence (Stage 2 oracle subset)',
        'n': len(gold),
        'sample_note': 'balanced/stratified evaluation set; not a prevalence sample',
        'oracle': {'model': 'NLI-concat', 'accuracy': observed,
                   'confusion_matrix': confusion,
                   'committed_metrics': str(stage2_path.relative_to(PROJECT_ROOT))},
        'fixed_rule': {'entail': entail, 'contra': contra},
        'methods': results,
        'caveats': [
            'abstention buckets include claims where retrieval did not surface the gold evidence',
            'confident errors are jointly retrieval and verifier errors; the buckets bound, not isolate, blame',
            'selective accuracy is computed on the self-selected answered subset; compare it with '
            'oracle_accuracy_answered on the same subset, not with the oracle overall',
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
