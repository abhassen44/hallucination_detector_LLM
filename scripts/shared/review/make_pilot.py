"""Build the blind v1 pilot queues (B1).

Two 20-case queues, sampled deterministically from the 171 extraction and 103
verification review rows:

  extraction    balanced 10 Pass / 10 Fail across the provisional proposals, so the
                pilot can validate both the "extraction is fine" and "extraction
                failed" poles. Deliberately balanced: this is a validation sample,
                not a prevalence estimate.
  verification  stratified across provisional categories, proportional with a
                minimum of one per category.

Blind by construction: rows carry no proposal, no rationale, and no
``review_meta``. Suggestions are only allowed in the later full annotation pass.
Verification rows repeat example ids across evidence conditions, so the pilot
keeps at most one row per id and writes the queue into its own directory (and
therefore its own ``review_progress.json``).

    python -m scripts.shared.review.make_pilot --force
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl, write_jsonl
from src.utils.config import PROJECT_ROOT

EXTRACTION_QUEUE = PROJECT_ROOT / 'experiments/stage4_claims/human_review.jsonl'
VERIFICATION_QUEUE = PROJECT_ROOT / 'experiments/stage5_inference/error_review.jsonl'
EXTRACTION_PROPOSALS = PROJECT_ROOT / 'experiments/stage6_errors/provisional_extraction.jsonl'
VERIFICATION_PROPOSALS = PROJECT_ROOT / 'experiments/stage6_errors/provisional_verification.jsonl'
OUT_DIR = PROJECT_ROOT / 'experiments/pilot_v1'

EXTRACTION_REVIEW = {'open_coding': '', 'pass_fail_assessment': None, 'category': None}
VERIFICATION_REVIEW = {'open_coding': '', 'pass_fail_assessment': None, 'category': None}


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with at least one per available stratum."""
    available = {k: v for k, v in counts.items() if v > 0}
    ordered = sorted(available, key=lambda k: (-available[k], k))
    allocation = dict.fromkeys(available, 0)
    if n <= len(available):
        for k in ordered[:n]:
            allocation[k] = 1
        return allocation
    for k in available:
        allocation[k] = 1
    remaining = n - len(available)
    total = sum(available.values())
    desired = {k: n * available[k] / total for k in available}
    while remaining > 0:
        candidates = [k for k in available if allocation[k] < available[k]]
        if not candidates:
            break
        k = min(candidates, key=lambda k: (allocation[k] - desired[k], k))
        allocation[k] += 1
        remaining -= 1
    return allocation


def _sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]


def sample_extraction(rng: random.Random, n: int = 20) -> tuple[list[dict], dict]:
    rows = list(read_jsonl(EXTRACTION_QUEUE))
    proposals = {row['id']: row for row in read_jsonl(EXTRACTION_PROPOSALS)}
    strata: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        strata[proposals[row['id']]['verdict']].append(row)
    share = {label: n // 2 for label in ('Pass', 'Fail')}
    selected = []
    for label in ('Pass', 'Fail'):
        pool = strata[label]
        if len(pool) < share[label]:
            raise SystemExit(f'Not enough {label} rows: {len(pool)} < {share[label]}')
        selected += _sample(pool, share[label], rng)
    order = {row['id']: i for i, row in enumerate(rows)}
    selected.sort(key=lambda row: order[row['id']])
    selection = {'frame': {k: len(v) for k, v in sorted(strata.items())},
                 'target': share, 'achieved': dict(Counter(proposals[r['id']]['verdict']
                                                           for r in selected))}
    return selected, selection


def sample_verification(rng: random.Random, n: int = 20) -> tuple[list[dict], dict]:
    rows = list(read_jsonl(VERIFICATION_QUEUE))
    proposals = {(row['id'], row['condition']): row for row in read_jsonl(VERIFICATION_PROPOSALS)}
    # One condition per example id: repeated ids are not independent pilot cases.
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[row['id']].append(row)
    unique = [rng.choice(sorted(by_id[i], key=lambda r: r['condition'])) for i in sorted(by_id)]

    strata: dict[str, list[dict]] = defaultdict(list)
    for row in unique:
        strata[proposals[(row['id'], row['condition'])]['category']].append(row)
    allocation = allocate({k: len(v) for k, v in strata.items()}, n)
    selected = []
    for category in sorted(allocation):
        selected += _sample(strata[category], allocation[category], rng)
    order = {row['id']: i for i, row in enumerate(rows)}
    selected.sort(key=lambda row: order[row['id']])
    selection = {'frame_rows': len(rows), 'frame_unique_ids': len(unique),
                 'conditions_frame': dict(Counter(row['condition'] for row in rows)),
                 'allocation': allocation,
                 'achieved': dict(Counter(proposals[(r['id'], r['condition'])]['category']
                                          for r in selected))}
    return selected, selection


def _prepare(rows: list[dict], review_template: dict) -> list[dict]:
    prepared = []
    for row in rows:
        clean = dict(row)
        clean['human_review'] = dict(review_template)
        clean.pop('review_meta', None)
        prepared.append(clean)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--n', type=int, default=20, help='cases per queue')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true', help='overwrite existing queues')
    args = parser.parse_args()

    out = args.out
    extraction_path = out / 'extraction_queue.jsonl'
    verification_path = out / 'verification_queue.jsonl'
    if not args.force and (extraction_path.exists() or verification_path.exists()):
        raise SystemExit(f'{out} already holds a pilot; pass --force to rebuild (loses judgments)')

    rng = random.Random(args.seed)
    extraction, extraction_selection = sample_extraction(rng, args.n)
    verification, verification_selection = sample_verification(rng, args.n)
    if len({r['id'] for r in verification}) != len(verification):
        raise SystemExit('verification pilot must have one row per example id')

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(extraction_path, _prepare(extraction, EXTRACTION_REVIEW))
    write_jsonl(verification_path, _prepare(verification, VERIFICATION_REVIEW))

    manifest = {
        'pilot': 'v1', 'blind': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed, 'cases_per_queue': args.n,
        'sampling_note': 'balanced/stratified validation sample, not a prevalence sample',
        'extraction': {**extraction_selection, 'source': str(EXTRACTION_QUEUE.relative_to(PROJECT_ROOT)),
                       'proposals': str(EXTRACTION_PROPOSALS.relative_to(PROJECT_ROOT)),
                       'review_fields': list(EXTRACTION_REVIEW)},
        'verification': {**verification_selection,
                         'source': str(VERIFICATION_QUEUE.relative_to(PROJECT_ROOT)),
                         'proposals': str(VERIFICATION_PROPOSALS.relative_to(PROJECT_ROOT)),
                         'review_fields': list(VERIFICATION_REVIEW)},
        'gate': {'cohen_kappa_pass_fail_min': 0.60, 'category_agreement_min': 0.50,
                 'if_failed': 'revise taxonomy before annotating all 274'},
        'files': {'extraction': str(extraction_path.relative_to(PROJECT_ROOT)),
                  'verification': str(verification_path.relative_to(PROJECT_ROOT))},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({'extraction': extraction_selection, 'verification': verification_selection},
                     indent=2))
    print(f'Queues -> {out}')


if __name__ == '__main__':
    main()
