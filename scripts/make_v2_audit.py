"""Build a blind 40-row audit sheet for the LLM-proposed v2 labels.

The audit is optional but cheap: it supports "LLM-proposed labels, human-audited
on a stratified sample" instead of "LLM-proposed labels" with no human check.
The sheet hides the proposals; fill `verdict` (and `category` for verification)
in Excel, then score it with ``scripts.score_v2_audit``.

    python -m scripts.make_v2_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl
from src.utils.config import PROJECT_ROOT

QUEUES = {
    'extraction': PROJECT_ROOT / 'experiments/v2/queues/extraction.jsonl',
    'verification': PROJECT_ROOT / 'experiments/v2/queues/verification.jsonl',
}
LABELS = {
    'extraction': PROJECT_ROOT / 'experiments/v2/labels_llm_proposed_extraction.jsonl',
    'verification': PROJECT_ROOT / 'experiments/v2/labels_llm_proposed_verification.jsonl',
}
SHEET = PROJECT_ROOT / 'experiments/v2/audit_sheet.csv'
MANIFEST = PROJECT_ROOT / 'experiments/v2/audit_manifest.json'
SEED = 42
N_EXTRACTION = 20
N_VERIFICATION = 20

FIELDS = ['queue', 'key', 'question', 'answer', 'gold', 'predicted',
          'claims', 'evidence', 'verdict', 'category', 'notes']


def key(row: dict) -> tuple[str, str | None]:
    return row['id'], row.get('condition')


def sheet_key(row: dict) -> str:
    return row['id'] if row.get('condition') is None else f"{row['id']}::{row['condition']}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocate(counts: dict[str, int], total: int, minimum: int = 2) -> dict[str, int]:
    """Minimum per stratum, then largest remaining count until the quota is met."""
    allocation = {name: min(minimum, count) for name, count in counts.items()}
    while sum(allocation.values()) < total:
        open_strata = [name for name in counts if allocation[name] < counts[name]]
        if not open_strata:
            break
        name = max(open_strata, key=lambda item: (counts[item] - allocation[item], item))
        allocation[name] += 1
    return allocation


def _clean(text, limit=1200) -> str:
    text = '' if text is None else str(text)
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit] + ' ...[truncated]'


def _claims(row: dict, kind: str) -> str:
    if kind == 'extraction':
        return _clean('; '.join(str(claim) for claim in row.get('claims', [])))
    parts = []
    for claim in row.get('claims', []):
        probabilities = claim.get('probabilities') or []
        scores = ', '.join(f'{float(p):.2f}' for p in probabilities)
        parts.append(f"{claim.get('claim')} [E,N,C={scores}]")
    return _clean('; '.join(parts))


def _evidence(row: dict) -> str:
    evidence = row.get('provided_context') or []
    if not evidence:
        evidence = [hit.get('text', '') for claim in row.get('claims', [])
                    if isinstance(claim, dict)
                    for hit in (claim.get('evidence') or [])][:6]
    return _clean(' || '.join(str(item) for item in evidence))


def sample_rows(queue_rows: list[dict], labels: list[dict], kind: str,
                rng: random.Random) -> list[dict]:
    by_key = {key(label): label for label in labels if not label['human_reviewed']}
    pool = [row for row in queue_rows if key(row) in by_key]
    if kind == 'extraction':
        strata = {'Pass': [row for row in pool if by_key[key(row)]['verdict'] == 'Pass'],
                  'Fail': [row for row in pool if by_key[key(row)]['verdict'] == 'Fail']}
        quota = {name: min(N_EXTRACTION // 2, len(rows)) for name, rows in strata.items()}
    else:
        strata = {}
        for row in pool:
            strata.setdefault(by_key[key(row)]['category'], []).append(row)
        quota = allocate({name: len(rows) for name, rows in strata.items()}, N_VERIFICATION)

    selected = []
    for name in sorted(strata):
        rows = sorted(strata[name], key=sheet_key)
        selected.extend(rng.sample(rows, min(quota.get(name, 0), len(rows))))
    return sorted(selected, key=sheet_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if SHEET.exists() and not args.force:
        raise SystemExit(f'{SHEET.relative_to(PROJECT_ROOT)} exists; pass --force.')
    if MANIFEST.exists() and not args.force:
        raise SystemExit(f'{MANIFEST.relative_to(PROJECT_ROOT)} exists; pass --force.')

    rng = random.Random(SEED)
    records, manifest = [], {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED,
        'design': '20 extraction (10 Pass / 10 Fail proposals) + 20 verification '
                  '(stratified by proposal category); human-labeled pilot rows excluded',
        'queues': {},
    }
    for kind in ('extraction', 'verification'):
        queue_rows = list(read_jsonl(QUEUES[kind]))
        labels = list(read_jsonl(LABELS[kind]))
        by_key = {key(label): label for label in labels}
        rows = sample_rows(queue_rows, labels, kind, rng)
        if not rows:
            raise ValueError(f'{kind}: no rows to audit')
        for row in rows:
            k = key(row)
            label = by_key[k]
            records.append({
                'queue': kind,
                'key': sheet_key(row),
                'question': _clean(row.get('question'), 500),
                'answer': _clean(row.get('answer'), 500),
                'gold': row.get('answer_gold_label') or '',
                'predicted': row.get('answer_prediction') or '',
                'claims': _claims(row, kind),
                'evidence': _evidence(row),
                'verdict': '',
                'category': '',
                'notes': '',
            })
            manifest['queues'].setdefault(kind, []).append({
                'key': sheet_key(row),
                'proposal_verdict': label['verdict'],
                'proposal_category': label['category'],
            })

    with SHEET.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    manifest['sheet'] = {'path': str(SHEET.relative_to(PROJECT_ROOT)),
                         'n_rows': len(records), 'sha256': sha256(SHEET)}
    manifest['inputs'] = {kind: {'queue_sha256': sha256(QUEUES[kind]),
                                 'labels_sha256': sha256(LABELS[kind])}
                          for kind in QUEUES}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({kind: dict(Counter(entry['proposal_verdict'] for entry in entries))
                      for kind, entries in manifest['queues'].items()}, indent=2))
    print(f'Rows: {len(records)} -> {SHEET.relative_to(PROJECT_ROOT)}')
    print(f'Saved -> {MANIFEST.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
