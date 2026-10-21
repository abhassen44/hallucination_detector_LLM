"""Interactive human review of the prepared queues.

Only fills empty ``human_review`` fields, never overwrites completed judgments,
and backs the queue up before the first write. Optional ``--suggestions`` shows
the provisional LLM proposal (``scripts.analyze_errors``); pressing Enter accepts
it explicitly, typing overrides it, and the choice is recorded in ``review_meta``.

    python -m scripts.review --queue experiments/stage4_claims/human_review.jsonl \
        --suggestions experiments/stage6_errors/provisional_extraction.jsonl
    python -m scripts.review --queue experiments/v2/queues/verification.jsonl \
        --taxonomy experiments/v2/taxonomy.json --summary
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl, write_jsonl
from src.evaluation.cached_baseline import CATEGORIES
from src.utils.config import PROJECT_ROOT

DEFAULT_QUEUE = PROJECT_ROOT / 'experiments/stage4_claims/human_review.jsonl'
BOOL_FIELDS = ('fidelity', 'atomicity', 'coverage', 'self_contained')


def load_queue(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def suggestion_key(row_or_proposal: dict):
    """Verification rows repeat ids across evidence conditions, so key on both."""
    return row_or_proposal['id'], row_or_proposal.get('condition')


def load_taxonomy(path: Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding='utf-8'))


def queue_kind(rows: list[dict]) -> str:
    """Verification rows carry a condition; extraction rows do not."""
    return 'verification' if any('condition' in row for row in rows) else 'extraction'


def category_options(kind: str, taxonomy: dict | None) -> list[str]:
    """Categories to offer, or an empty list when the kind does not collect them."""
    if not taxonomy:
        return list(CATEGORIES)
    if kind == 'verification':
        return list(taxonomy.get('verification', {}).get('categories') or CATEGORIES)
    if not taxonomy.get('extraction', {}).get('category_enabled', True):
        return []
    return list(taxonomy.get('extraction', {}).get('categories') or CATEGORIES)


def boolean_options(kind: str, taxonomy: dict | None) -> list[str]:
    """Diagnostic extraction Booleans, unless the taxonomy retires them."""
    if (taxonomy and kind == 'extraction'
            and not taxonomy.get('extraction', {}).get('booleans_enabled', True)):
        return []
    return list(BOOL_FIELDS)


def load_suggestions(path: Path | None) -> dict[tuple, dict]:
    if not path or not Path(path).exists():
        return {}
    return {suggestion_key(row): row for row in read_jsonl(path) if row.get('valid')}


def is_complete(row: dict, require_verdict: bool = False) -> bool:
    review = row.get('human_review') or {}
    if require_verdict:
        return review.get('pass_fail_assessment') not in (None, '')
    return any(value is not None and value != '' for value in review.values())


def pending(rows: list[dict], require_verdict: bool = False) -> list[dict]:
    return [row for row in rows if not is_complete(row, require_verdict)]


def apply_review(row: dict, answers: dict) -> dict:
    review = row.setdefault('human_review', {})
    for field, value in answers.items():
        if field not in review:
            raise KeyError(f'Unknown review field: {field}')
        review[field] = value
    return row


def save_queue(path: Path, rows: list[dict], backup: bool = True) -> None:
    path = Path(path)
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + '.bak'))
    write_jsonl(path, rows)


def summarise(rows: list[dict], require_verdict: bool = False) -> dict:
    completed = [row for row in rows if is_complete(row, require_verdict)]
    categories, verdicts, booleans = Counter(), Counter(), {f: Counter() for f in BOOL_FIELDS}
    suggestion_used = 0
    for row in completed:
        review = row.get('human_review', {})
        if review.get('category'):
            categories[review['category']] += 1
        if review.get('pass_fail_assessment'):
            verdicts[review['pass_fail_assessment']] += 1
        for field in BOOL_FIELDS:
            if field in review:
                booleans[field][str(review[field])] += 1
        suggestion_used += bool((row.get('review_meta') or {}).get('suggestion_used'))
    return {'total': len(rows), 'completed': len(completed),
            'pending': len(rows) - len(completed),
            'pass_fail': dict(verdicts), 'categories': dict(categories),
            'booleans': {field: dict(counts) for field, counts in booleans.items()},
            'suggestion_used': suggestion_used,
            'human_annotations_completed': len(completed)}


def _prompt(question: str, default: str | None = None) -> str | None:
    suffix = f' [{default}]' if default is not None else ''
    value = input(f'{question}{suffix}: ').strip()
    if value == '':
        return default
    return value


def _ask_verdict(suggestion: dict | None) -> str | None:
    default = suggestion.get('verdict') if suggestion else None
    while True:
        value = _prompt('Pass/Fail [p/f/q=quit]', default)
        if value is None:
            return None
        lowered = value.lower()
        if lowered in ('p', 'pass'):
            return 'Pass'
        if lowered in ('f', 'fail'):
            return 'Fail'
        if lowered == 'q':
            raise KeyboardInterrupt
        print('  Enter "p", "f", or press Enter to skip.')


def _ask_category(suggestion: dict | None, categories: list[str]) -> str | None:
    print('  Categories:')
    for number, category in enumerate(categories, 1):
        print(f'    {number:2d}. {category}')
    default = suggestion.get('category') if suggestion else None
    if default not in categories:
        default = None
    while True:
        value = _prompt('Category [number/name/s=skip]', default)
        if value is None:
            return None
        lowered = value.lower()
        if lowered in ('s', 'skip', 'none'):
            return None
        if lowered.isdigit() and 1 <= int(lowered) <= len(categories):
            return categories[int(lowered) - 1]
        if lowered in categories:
            return lowered
        print('  Unknown category.')


def _ask_boolean(field: str) -> bool | None:
    while True:
        value = _prompt(f'{field} [y/n/Enter=unset]')
        if value is None:
            return None
        if value.lower() in ('y', 'yes', 'true'):
            return True
        if value.lower() in ('n', 'no', 'false'):
            return False
        print('  Enter "y", "n", or press Enter.')


def _show(row: dict, suggestion: dict | None, position: int, total: int) -> None:
    print('\n' + '=' * 78)
    print(f'[{position}/{total}] {row.get("id")}  ({row.get("split", row.get("condition", ""))})')
    print('=' * 78)
    if row.get('question'):
        print(f'Question : {row["question"]}')
    if row.get('answer'):
        print(f'Answer   : {row["answer"]}')
    if row.get('answer_gold_label'):
        print(f'Gold     : {row["answer_gold_label"]}  predicted: {row.get("answer_prediction", "-")}')
    print(f'Claims   : {json.dumps(row.get("claims"), ensure_ascii=False)[:600]}')
    if row.get('heuristic_flags'):
        print(f'Flags    : {row["heuristic_flags"]}')
    if row.get('raw_model_response'):
        print(f'Raw      : {row["raw_model_response"][:400]}')
    if suggestion:
        print(f'-- LLM proposal ({suggestion.get("confidence", 0):.2f}): '
              f'{suggestion.get("verdict", "")} {suggestion.get("category", "")} '
              f'| {suggestion.get("rationale", "")}')


def review_interactively(path: Path, suggestions_path: Path | None, limit: int | None,
                         reviewer: str | None, taxonomy: dict | None = None,
                         categories: list[str] | None = None) -> list[dict]:
    rows = load_queue(path)
    suggestions = load_suggestions(suggestions_path)
    kind = queue_kind(rows)
    if categories is None:
        categories = category_options(kind, taxonomy)
    booleans = boolean_options(kind, taxonomy)
    require_verdict = taxonomy is not None
    outstanding = pending(rows, require_verdict)
    reviewed = len(rows) - len(outstanding)
    queue = outstanding[:limit] if limit else outstanding
    if not queue:
        print('Nothing pending; every row already has a human judgment.')
        return rows

    if path.exists():
        shutil.copy2(path, path.with_name(path.name + '.bak'))
    print(f'{len(queue)} of {len(rows)} rows pending; {reviewed} already reviewed. '
          'Ctrl+C saves and exits.')
    definitions = (taxonomy or {}).get('verification', {}).get('definitions', {})
    if kind == 'verification' and definitions:
        print('\nCategory definitions:')
        for name in categories:
            print(f'  {name}: {definitions.get(name, "")}')
    for position, row in enumerate(queue, 1):
        suggestion = suggestions.get(suggestion_key(row))
        _show(row, suggestion, position, len(queue))
        answers: dict = {}
        try:
            answers['open_coding'] = _prompt('Open coding (observation, empty=skip)') or ''
            verdict = _ask_verdict(suggestion)
            if verdict is not None:
                answers['pass_fail_assessment'] = verdict
            if categories and 'category' in (row.get('human_review') or {}):
                category = _ask_category(suggestion, categories)
                if category is not None:
                    answers['category'] = category
            for field in booleans:
                if field in (row.get('human_review') or {}):
                    value = _ask_boolean(field)
                    if value is not None:
                        answers[field] = value
        except KeyboardInterrupt:
            print('\nStopped.')
            break
        except EOFError:
            print('\nEnd of input.')
            break

        if not any(value not in (None, '') for value in answers.values()):
            print('  Skipped (no fields filled).')
            continue
        apply_review(row, answers)
        row['review_meta'] = {
            'reviewer': reviewer, 'reviewed_utc': datetime.now(timezone.utc).isoformat(),
            'suggestion_used': bool(suggestion) and (
                answers.get('pass_fail_assessment') == suggestion.get('verdict') or
                answers.get('category') == suggestion.get('category')),
        }
        save_queue(path, rows, backup=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=DEFAULT_QUEUE)
    parser.add_argument('--suggestions', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--reviewer', default=None)
    parser.add_argument('--taxonomy', type=Path, default=None,
                        help='JSON taxonomy; controls verification categories and whether '
                             'extraction collects causes.')
    parser.add_argument('--summary', action='store_true',
                        help='Print progress and write review_progress.json without prompting.')
    args = parser.parse_args()

    rows = load_queue(args.queue)
    taxonomy = load_taxonomy(args.taxonomy)
    if not args.summary:
        rows = review_interactively(args.queue, args.suggestions, args.limit, args.reviewer,
                                    taxonomy=taxonomy)

    summary = summarise(rows, require_verdict=taxonomy is not None)
    target = args.queue.with_name('review_progress.json')
    target.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('\n' + json.dumps(summary, indent=2))
    print(f'Saved -> {target}')


if __name__ == '__main__':
    main()
