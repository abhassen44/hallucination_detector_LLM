"""Record a decision on the three JSON syntax-repair proposals.

The repairs never change the dataset here: they only resolve the review queue.
Applying them would alter three HaluEval examples and require regenerating the
committed HaluEval predictions, so that is a separate, explicit version.

    python -m scripts.shared.review.resolve_recovery --accept all --reviewer "Abhas"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from src.utils.config import PROJECT_ROOT

PATH = PROJECT_ROOT / 'experiments/stage4_claims/json_recovery_proposals.json'

FIDELITY_NOTES = {
    'halueval-qa-698-hallucinated':
        'Both recovered claims appear verbatim in the raw response and decompose the '
        'question-aware answer without adding content. The answer remains hallucinated; '
        'extraction fidelity is independent of answer truth.',
    'halueval-qa-8136-right':
        'Removing one extra array closer and one stray escape yields the single claim the '
        'model produced; the added production wording comes from the question, as the '
        'extraction prompt requires.',
    'halueval-qa-2590-right':
        'The raw escape sequence left the final string unterminated; the repair closes it '
        'without altering any model-produced word. Both claims are faithful.',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept', nargs='*', default=[])
    parser.add_argument('--reject', nargs='*', default=[])
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    proposals = json.loads(PATH.read_text(encoding='utf-8'))
    ids = {row['id'] for row in proposals}
    accept = set(args.accept)
    reject = set(args.reject)
    if 'all' in accept:
        accept.discard('all')
        accept |= ids
    if 'all' in reject:
        reject.discard('all')
        reject |= ids
    decision = {identifier: True for identifier in accept} | {identifier: False for identifier in reject}
    unknown = set(decision) - ids
    if unknown:
        raise SystemExit(f'Unknown ids: {sorted(unknown)}')
    if set(accept) & set(reject):
        raise SystemExit(f'Accept and reject overlap: {sorted(set(accept) & set(reject))}')

    now = datetime.now(timezone.utc).isoformat()
    for row in proposals:
        if row['id'] not in decision:
            continue
        if row['human_accepted'] is not None and not args.force:
            raise SystemExit(f'{row["id"]} already reviewed; pass --force to overwrite')
        if row['syntax_proposal'] is None:
            raise SystemExit(f'{row["id"]} has no syntax proposal to decide')
        row['human_accepted'] = decision[row['id']]
        row['reviewed_by'] = args.reviewer
        row['reviewed_utc'] = now
        row['fidelity_check'] = FIDELITY_NOTES.get(row['id'])
        row['applied_to_dataset'] = False

    PATH.write_text(json.dumps(proposals, indent=2) + '\n', encoding='utf-8')
    accepted = sum(bool(row['human_accepted']) for row in proposals)
    print(json.dumps({'accepted': accepted, 'rejected': sum(row['human_accepted'] is False
                                                            for row in proposals),
                      'applied_to_dataset': 0, 'file': str(PATH)}, indent=2))


if __name__ == '__main__':
    main()
