"""Materialize v2 labels from the review proposals.

These are **not human labels**. The 37 seeded pilot judgments are copied as human
labels; every other row takes the LLM proposal (extraction: verdict; verification:
category, with ``Fail`` by queue construction because the queue contains only
mistaken decisions). Every record records its source so downstream reports cannot
mistake one for the other.

The human review queues are left untouched, so a human pass can still replace these
labels later.

    python -m scripts.materialize_v2_labels
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.pilot_agreement import _cohen_kappa
from src.datasets.schema import read_jsonl, write_jsonl
from src.utils.config import PROJECT_ROOT

QUEUES = {
    'extraction': PROJECT_ROOT / 'experiments/v2/queues/extraction.jsonl',
    'verification': PROJECT_ROOT / 'experiments/v2/queues/verification.jsonl',
}
PROPOSALS = {
    'extraction': PROJECT_ROOT / 'experiments/v2/proposals/extraction.jsonl',
    'verification': PROJECT_ROOT / 'experiments/v2/proposals/verification.jsonl',
}
TAXONOMY = PROJECT_ROOT / 'experiments/v2/taxonomy.json'
OUTPUTS = {
    'extraction': PROJECT_ROOT / 'experiments/v2/labels_llm_proposed_extraction.jsonl',
    'verification': PROJECT_ROOT / 'experiments/v2/labels_llm_proposed_verification.jsonl',
}
MANIFEST = PROJECT_ROOT / 'experiments/v2/label_manifest.json'


def key(row: dict) -> tuple[str, str | None]:
    return row['id'], row.get('condition')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_proposals(path: Path) -> dict[tuple, dict]:
    return {key(row): row for row in read_jsonl(path) if row.get('valid')}


def materialize(queue_rows: list[dict], proposals: dict[tuple, dict],
                kind: str, taxonomy: dict) -> list[dict]:
    labels = []
    for row in queue_rows:
        k = key(row)
        review = row.get('human_review') or {}
        human_verdict = review.get('pass_fail_assessment')
        proposal = proposals.get(k)
        if human_verdict:
            if kind == 'verification' and review.get('category') not in (
                    taxonomy['verification']['categories']):
                raise ValueError(f'{k}: human category is outside taxonomy v2')
            labels.append({
                'id': k[0], 'condition': k[1], 'queue': kind,
                'verdict': human_verdict,
                'category': review.get('category') if kind == 'verification' else None,
                'label_source': 'human_queue',
                'human_reviewed': True,
                'verdict_source': 'human',
                'category_source': ('human' if kind == 'verification'
                                    and review.get('category') else None),
                'legacy_category': (review.get('category') if kind == 'extraction'
                                    else None),
                'proposal_category': (proposal or {}).get('category'),
                'model': (proposal or {}).get('model'),
                'notes': 'seeded v1 pilot judgment' if (row.get('review_meta')
                                                         or {}).get('reviewer') else None,
            })
            continue

        if proposal is None:
            raise KeyError(f'{k}: no usable proposal and no human verdict')
        if kind == 'extraction':
            labels.append({
                'id': k[0], 'condition': k[1], 'queue': kind,
                'verdict': proposal['verdict'], 'category': None,
                'label_source': 'llm_proposal', 'human_reviewed': False,
                'verdict_source': 'proposal', 'category_source': None,
                'legacy_category': None, 'proposal_category': proposal.get('category'),
                'model': proposal.get('model'),
                'notes': 'extraction category not collected in v2',
            })
        else:
            if proposal.get('category') not in taxonomy['verification']['categories']:
                raise ValueError(f'{k}: proposal category is outside taxonomy v2')
            labels.append({
                'id': k[0], 'condition': k[1], 'queue': kind,
                'verdict': 'Fail', 'category': proposal['category'],
                'label_source': 'llm_proposal', 'human_reviewed': False,
                'verdict_source': 'queue_construction', 'category_source': 'proposal',
                'legacy_category': None, 'proposal_category': proposal['category'],
                'model': proposal.get('model'),
                'notes': 'error-review queue; every row is a mistaken decision',
            })
    return labels


def agreement(queue_rows: list[dict], proposals: dict[tuple, dict], kind: str) -> dict:
    """Compare proposals with the human pilot rows only."""
    pairs = []
    for row in queue_rows:
        human = (row.get('human_review') or {}).get('pass_fail_assessment')
        proposal = proposals.get(key(row))
        if not human or proposal is None:
            continue
        if kind == 'extraction':
            if proposal.get('verdict'):
                pairs.append((proposal['verdict'], human))
        elif proposal.get('category'):
            human_category = (row.get('human_review') or {}).get('category')
            if human_category:
                pairs.append((proposal['category'], human_category))
    result = {'n': len(pairs),
              'agreement': (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
              'confusion': dict(Counter(f'{a} -> {b}' for a, b in pairs if a != b))}
    if kind == 'extraction':
        result['kappa'] = _cohen_kappa(pairs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if MANIFEST.exists() and not args.force:
        raise SystemExit(f'{MANIFEST.relative_to(PROJECT_ROOT)} exists; pass --force.')

    taxonomy = json.loads(TAXONOMY.read_text(encoding='utf-8'))
    report: dict = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'policy': ('human_queue rows are the seeded pilot; llm_proposal rows take the '
                   'proposal verdict/category and are not human-validated'),
        'queues': {},
        'outputs': {},
    }
    for kind in ('extraction', 'verification'):
        queue_rows = list(read_jsonl(QUEUES[kind]))
        proposals = load_proposals(PROPOSALS[kind])
        labels = materialize(queue_rows, proposals, kind, taxonomy)
        if len(labels) != len(queue_rows):
            raise ValueError(f'{kind}: label count does not match the queue')
        if len({key(label) for label in labels}) != len(labels):
            raise ValueError(f'{kind}: duplicate label keys')
        write_jsonl(OUTPUTS[kind], labels)
        report['queues'][kind] = {
            'n_rows': len(queue_rows),
            'n_labels': len(labels),
            'n_human': sum(label['human_reviewed'] for label in labels),
            'n_llm': sum(not label['human_reviewed'] for label in labels),
            'verdicts': dict(Counter(label['verdict'] for label in labels)),
            'categories': dict(Counter(label['category'] for label in labels
                                       if label['category'])),
            'pilot_agreement': agreement(queue_rows, proposals, kind),
        }
        report['outputs'][str(OUTPUTS[kind].relative_to(PROJECT_ROOT))] = {
            'sha256': sha256(OUTPUTS[kind]), 'n_rows': len(labels)}
    report['inputs'] = {
        'queues': {kind: {'path': str(path.relative_to(PROJECT_ROOT)),
                          'sha256': sha256(path)} for kind, path in QUEUES.items()},
        'proposals': {kind: {'path': str(path.relative_to(PROJECT_ROOT)),
                             'sha256': sha256(path)} for kind, path in PROPOSALS.items()},
        'taxonomy': {'path': str(TAXONOMY.relative_to(PROJECT_ROOT)),
                     'sha256': sha256(TAXONOMY)},
    }

    MANIFEST.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['queues'], indent=2))
    print(f'Saved -> {MANIFEST.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
