"""Seed the v2 review queues with the v1 pilot judgments.

The v1 pilot covered 40 rows (20 extraction, 20 verification) that also appear in
the full 274-row queues. This copies every completed pilot judgment into fresh
v2 queues, keyed by ``(id, condition)``, so the full pass does not re-annotate
those rows.

A pilot row without a Pass/Fail verdict is *not* seeded: the review tool treats
any non-empty review field as complete, so copying an open-coding note alone
would hide a missing verdict from the pending queue. The skipped rows are
recorded in the seed manifest instead.

    python -m scripts.seed_v2_queues
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl, write_jsonl
from src.utils.config import PROJECT_ROOT

SOURCES = {
    'extraction': PROJECT_ROOT / 'experiments/stage4_claims/human_review.jsonl',
    'verification': PROJECT_ROOT / 'experiments/stage5_inference/error_review.jsonl',
}
PILOT = {
    'extraction': PROJECT_ROOT / 'experiments/pilot_v1/extraction_queue.jsonl',
    'verification': PROJECT_ROOT / 'experiments/pilot_v1/verification_queue.jsonl',
}
OUTPUT_DIR = PROJECT_ROOT / 'experiments/v2/queues'
MANIFEST = PROJECT_ROOT / 'experiments/v2/queue_seed.json'


def review_key(row: dict) -> tuple[str, str | None]:
    """Extraction rows have no condition; verification repeats ids per condition."""
    return row['id'], row.get('condition')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_queue(rows: list[dict], pilot_rows: list[dict]) -> tuple[list[dict], dict]:
    """Copy pilot verdicts into ``rows`` in place and report what happened."""
    index = {review_key(row): row for row in rows}
    seeded: list[dict] = []
    skipped: list[dict] = []
    for pilot in pilot_rows:
        key = review_key(pilot)
        if key not in index:
            raise KeyError(f'pilot row {key} is not in the full queue')
        target = index[key]
        review = pilot.get('human_review') or {}
        verdict = review.get('pass_fail_assessment')
        if not verdict:
            skipped.append({
                'id': key[0], 'condition': key[1],
                'reason': 'no Pass/Fail verdict',
                'had_open_coding': bool((review.get('open_coding') or '').strip()),
                'had_category': bool(review.get('category')),
            })
            continue
        current = target.get('human_review') or {}
        existing = current.get('pass_fail_assessment')
        if existing not in (None, '', verdict):
            raise ValueError(f'conflicting verdict for {key}: {existing!r} vs {verdict!r}')
        target['human_review'] = {**current, **review}
        target['review_meta'] = dict(pilot.get('review_meta') or {})
        seeded.append({'id': key[0], 'condition': key[1], 'verdict': verdict,
                       'category': review.get('category')})
    return rows, {'seeded': seeded, 'skipped': skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing seed manifest and output queues.')
    args = parser.parse_args()

    if MANIFEST.exists() and not args.force:
        raise SystemExit(f'{MANIFEST.relative_to(PROJECT_ROOT)} exists; pass --force to reseed.')

    report: dict = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'key': '(id, condition)',
        'queues': {},
        'outputs': {},
    }
    for kind in ('extraction', 'verification'):
        rows = list(read_jsonl(SOURCES[kind]))
        pilot_rows = list(read_jsonl(PILOT[kind]))
        rows, result = seed_queue(rows, pilot_rows)
        output = OUTPUT_DIR / f'{kind}.jsonl'
        write_jsonl(output, rows)
        keys = [review_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f'{kind}: duplicate (id, condition) keys after seeding')

        report['queues'][kind] = {
            'source': str(SOURCES[kind].relative_to(PROJECT_ROOT)),
            'pilot': str(PILOT[kind].relative_to(PROJECT_ROOT)),
            'source_sha256': sha256(SOURCES[kind]),
            'pilot_sha256': sha256(PILOT[kind]),
            'n_rows': len(rows),
            'n_unique_keys': len(set(keys)),
            'n_pilot_rows': len(pilot_rows),
            'n_seeded': len(result['seeded']),
            'n_skipped': len(result['skipped']),
            'seeded': result['seeded'],
            'skipped': result['skipped'],
        }
        report['outputs'][str(output.relative_to(PROJECT_ROOT))] = {
            'n_rows': len(rows), 'sha256': sha256(output),
        }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({kind: {field: value for field, value in queue.items()
                             if field not in ('seeded', 'skipped')}
                      for kind, queue in report['queues'].items()}, indent=2))
    print(f'Saved -> {MANIFEST.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
