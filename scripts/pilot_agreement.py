"""Score the blind v1 pilot against the provisional LLM proposals (B3).

The pilot is a *validation* sample, not a prevalence sample: extraction is
deliberately balanced 10 Pass / 10 Fail, verification is stratified across error
categories. Numbers here describe agreement with the proposals, not how often the
pipeline is right.

Gate (declared before annotation):

  * Cohen's kappa on extraction Pass/Fail          >= 0.60
  * exact category agreement on verification errors >= 0.50

Passing means the taxonomy is usable as-is and the remaining 274 rows can be
annotated. Failing means the taxonomy is revised before spending that time.

    python -m scripts.pilot_agreement --queue experiments/pilot_v1/extraction_queue.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl
from src.utils.config import PROJECT_ROOT

PROVISIONALS = {
    'extraction': PROJECT_ROOT / 'experiments/stage6_errors/provisional_extraction.jsonl',
    'verification': PROJECT_ROOT / 'experiments/stage6_errors/provisional_verification.jsonl',
}
KAPPA_MIN = 0.60
CATEGORY_AGREEMENT_MIN = 0.50
BREAK_SECONDS = 600


def queue_kind(rows: list[dict]) -> str:
    return 'verification' if any('condition' in row for row in rows) else 'extraction'


def _key(row: dict):
    return row['id'], row.get('condition')


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({label for pair in pairs for label in pair})
    if len(labels) < 2:
        return None
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for a, b in pairs:
        matrix[index[a]][index[b]] += 1
    n = len(pairs)
    observed = sum(matrix[i][i] for i in range(len(labels))) / n
    expected = sum(sum(matrix[i]) * sum(matrix[j][i] for j in range(len(labels)))
                   for i in range(len(labels))) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def _agreement(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {'n': 0, 'matches': 0, 'agreement': None, 'kappa': None,
                'confusion': {}, 'disagreements': []}
    matches = sum(a == b for a, b in pairs)
    confusion = Counter(f'{a} -> {b}' for a, b in pairs if a != b)
    return {'n': len(pairs), 'matches': matches, 'agreement': matches / len(pairs),
            'kappa': _cohen_kappa(pairs), 'confusion': dict(confusion.most_common()),
            'disagreements': []}


def timing(rows: list[dict]) -> dict:
    stamps = sorted(datetime.fromisoformat(row['review_meta']['reviewed_utc'])
                    for row in rows if (row.get('review_meta') or {}).get('reviewed_utc'))
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    active = [gap for gap in gaps if gap <= BREAK_SECONDS]
    if not active:
        return {'stamped': len(stamps), 'active_cases': 0, 'minutes_total': 0.0,
                'seconds_per_case': None, 'excluded_breaks': 0}
    return {'stamped': len(stamps), 'active_cases': len(active),
            'minutes_total': round(sum(active) / 60, 1),
            'seconds_per_case': round(sum(active) / len(active), 1),
            'excluded_breaks': len(gaps) - len(active)}


def score(queue_path: Path, provisional_path: Path) -> dict:
    rows = list(read_jsonl(queue_path))
    kind = queue_kind(rows)
    proposals = {_key(row): row for row in read_jsonl(provisional_path) if row.get('valid')}
    completed = [row for row in rows if any(v not in (None, '') for v in row['human_review'].values())]
    unmatched = [row['id'] for row in completed if _key(row) not in proposals]

    verdicts = [(proposals[_key(row)]['verdict'], row['human_review']['pass_fail_assessment'])
                for row in completed
                if _key(row) in proposals and proposals[_key(row)].get('verdict')
                and row['human_review'].get('pass_fail_assessment')]
    categories = [(proposals[_key(row)]['category'], row['human_review']['category'])
                  for row in completed
                  if _key(row) in proposals and proposals[_key(row)].get('category')
                  and row['human_review'].get('category')]
    category_result = _agreement(categories)
    category_result['disagreements'] = [row['id'] for row in completed
                                        if _key(row) in proposals
                                        and proposals[_key(row)].get('category')
                                        and row['human_review'].get('category')
                                        and proposals[_key(row)]['category']
                                        != row['human_review']['category']]
    result = {
        'queue': str(queue_path.relative_to(PROJECT_ROOT)) if queue_path.is_relative_to(PROJECT_ROOT)
        else str(queue_path),
        'kind': kind, 'n_total': len(rows), 'n_completed': len(completed),
        'n_completed_expected': len(rows), 'unmatched': unmatched,
        'pass_fail': _agreement(verdicts),
        'category': category_result,
        'timing': timing(completed),
        'provisional_model': next((p.get('model') for p in proposals.values()), None),
        'sample_note': 'balanced/stratified validation sample, not a prevalence sample',
    }
    return result


def gate(extraction: dict | None, verification: dict | None) -> dict:
    if not extraction or not verification:
        return {'status': 'incomplete', 'reason': 'both queues are required'}
    if extraction['n_completed'] < extraction['n_total'] or \
            verification['n_completed'] < verification['n_total']:
        return {'status': 'incomplete',
                'reason': f'{extraction["n_completed"]}/{extraction["n_total"]} extraction, '
                          f'{verification["n_completed"]}/{verification["n_total"]} verification'}
    kappa = extraction['pass_fail']['kappa']
    agreement = verification['category']['agreement']
    checks = {'kappa_pass_fail': {'observed': kappa, 'min': KAPPA_MIN},
              'category_agreement': {'observed': agreement, 'min': CATEGORY_AGREEMENT_MIN}}
    passed = kappa is not None and agreement is not None and \
        kappa >= KAPPA_MIN and agreement >= CATEGORY_AGREEMENT_MIN
    return {'status': 'pass' if passed else 'fail', 'checks': checks,
            'decision': 'annotate the remaining 274 rows' if passed
            else 'revise the taxonomy before annotating all 274',
            'note': 'kappa is extraction Pass/Fail; category agreement is verification errors '
                    '(the only queue with more than one proposed category)'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, action='append', required=True,
                        help='pilot queue (repeat for extraction and verification)')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    results = {}
    for path in args.queue:
        kind = queue_kind(list(read_jsonl(path)))
        results[kind] = score(path, PROVISIONALS[kind])
        payload = results[kind]
        print(f'[{kind}] {payload["n_completed"]}/{payload["n_total"]} completed; '
              f'pass/fail kappa {payload["pass_fail"]["kappa"]}; '
              f'category agreement {payload["category"]["agreement"]}; '
              f'{payload["timing"]["seconds_per_case"]}s/case', flush=True)

    verdict = gate(results.get('extraction'), results.get('verification'))
    report = {'pilot': 'v1', 'scored_utc': datetime.now(timezone.utc).isoformat(),
              'gate': verdict, 'queues': results}
    out = args.out or Path(args.queue[0]).parent / 'pilot_agreement.json'
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verdict, indent=2))
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
