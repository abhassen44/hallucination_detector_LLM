"""Score the filled 40-row v2 audit sheet against the LLM proposals.

Fill `verdict` (Pass/Fail) for every row and `category` for verification rows in
``experiments/v2/audit_sheet.csv``, then run this. Blank rows are reported as
unscored rather than silently treated as agreement.

    python -m scripts.score_v2_audit
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.pilot_agreement import _cohen_kappa
from src.utils.config import PROJECT_ROOT

SHEET = PROJECT_ROOT / 'experiments/v2/audit_sheet.csv'
MANIFEST = PROJECT_ROOT / 'experiments/v2/audit_manifest.json'
TAXONOMY = PROJECT_ROOT / 'experiments/v2/taxonomy.json'
RESULT = PROJECT_ROOT / 'experiments/v2/audit_results.json'


def _verdict(value: str) -> str | None:
    lowered = (value or '').strip().lower()
    if lowered in ('p', 'pass'):
        return 'Pass'
    if lowered in ('f', 'fail'):
        return 'Fail'
    return None


def _category(value: str, allowed: set[str]) -> str | None:
    lowered = (value or '').strip().lower()
    return lowered if lowered in allowed else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sheet', type=Path, default=SHEET)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--out', type=Path, default=RESULT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    taxonomy = json.loads(TAXONOMY.read_text(encoding='utf-8'))
    categories = set(taxonomy['verification']['categories'])
    proposals = {(entry['key'], kind): entry
                 for kind, entries in manifest['queues'].items() for entry in entries}

    with args.sheet.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))

    extraction_pairs, verification_pairs = [], []
    unscored = []
    for row in rows:
        key = (row['key'], row['queue'])
        proposal = proposals.get(key)
        if proposal is None:
            raise KeyError(f'{key}: not in the audit manifest')
        verdict = _verdict(row.get('verdict', ''))
        if verdict is None:
            unscored.append({'queue': row['queue'], 'key': row['key'],
                             'reason': 'no verdict'})
            continue
        if row['queue'] == 'extraction':
            extraction_pairs.append((proposal['proposal_verdict'], verdict))
        else:
            category = _category(row.get('category', ''), categories)
            if category is None:
                unscored.append({'queue': row['queue'], 'key': row['key'],
                                 'reason': 'no valid category'})
                continue
            verification_pairs.append((proposal['proposal_category'], category))

    def summarise(pairs: list[tuple[str, str]], with_kappa: bool) -> dict:
        result = {'n': len(pairs),
                  'agreement': (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
                  'disagreements': [{'proposal': a, 'audit': b} for a, b in pairs if a != b],
                  'confusion': dict(Counter(f'{a} -> {b}' for a, b in pairs if a != b))}
        if with_kappa:
            result['kappa'] = _cohen_kappa(pairs)
        return result

    payload = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sheet': str(args.sheet.relative_to(PROJECT_ROOT)),
        'n_sheet_rows': len(rows),
        'n_unscored': len(unscored),
        'unscored': unscored,
        'extraction_verdict': summarise(extraction_pairs, with_kappa=True),
        'verification_category': summarise(verification_pairs, with_kappa=False),
        'note': ('This audits LLM-proposed labels; it does not turn them into human '
                 'labels. Report the audited agreement alongside the pilot agreement.'),
    }
    args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: payload[key] for key in ('n_sheet_rows', 'n_unscored',
                                                    'extraction_verdict',
                                                    'verification_category')}, indent=2))
    print(f'Saved -> {args.out.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
