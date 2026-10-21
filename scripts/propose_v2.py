"""Generate the v2 human-review proposals.

Extraction suggestions are copied from the frozen v1 proposals: taxonomy v2
collects Pass/Fail only, and the v1 extraction verdicts are unchanged. The copy
is stamped with its provenance so it cannot be mistaken for a fresh v2 proposal.

Verification suggestions are regenerated. v2 splits the v1 ``nli_failure``
catch-all into ``verifier_error`` (the evidence was there and clear) and
``ambiguous_evidence`` (the shown evidence cannot settle the claim). Mapping the
old labels mechanically would bake in the very confusion the pilot found.

    python -m scripts.propose_v2 --queue all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl, write_jsonl
from src.generation.llm import OllamaLLM, ProviderUnavailableError
from src.utils.config import PROJECT_ROOT

TAXONOMY = PROJECT_ROOT / 'experiments/v2/taxonomy.json'
EXTRACTION_QUEUE = PROJECT_ROOT / 'experiments/v2/queues/extraction.jsonl'
VERIFICATION_QUEUE = PROJECT_ROOT / 'experiments/v2/queues/verification.jsonl'
V1_EXTRACTION = PROJECT_ROOT / 'experiments/stage6_errors/provisional_extraction.jsonl'
OUT_DIR = PROJECT_ROOT / 'experiments/v2/proposals'

SYSTEM = ('You assist a human reviewer of an NLP hallucination-detection pipeline. '
          'You never decide the final label; you propose one short, checkable cause. '
          'Reply with a single JSON object and nothing else.')

PROMPT = """Task: propose the most likely cause of a mistaken verification decision.
Gold answer label: {gold}
Predicted answer label: {prediction}

Question: {question}
Answer: {answer}
Per-claim NLI probabilities [entailment, neutral, contradiction]:
{claims}
Evidence shown to the verifier (may be truncated):
{evidence}

Category definitions:
{definitions}

Precedence rules:
{precedence}

Choose exactly one category. Use the most specific category that fits; use
verifier_error when the needed evidence is present and clear but no more
specific category applies.

Reply with JSON:
{{"category": one of {categories},
  "rationale": "at most 30 words citing the specific claim or evidence span",
  "confidence": number between 0 and 1}}"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip(text, limit=700):
    text = '' if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + ' ...[truncated]'


def load_taxonomy(path: Path = TAXONOMY) -> dict:
    taxonomy = json.loads(path.read_text(encoding='utf-8'))
    categories = taxonomy.get('verification', {}).get('categories')
    definitions = taxonomy.get('verification', {}).get('definitions')
    if not categories or not definitions or set(categories) != set(definitions):
        raise ValueError('verification taxonomy needs a category for every definition')
    return taxonomy


def build_prompt(row: dict, taxonomy: dict) -> str:
    claims = []
    for claim in row.get('claims', []):
        probabilities = claim.get('probabilities') or []
        claims.append({'claim': claim.get('claim'),
                       'probabilities': [round(float(p), 3) for p in probabilities]})
    evidence = row.get('provided_context') or []
    if not evidence:
        evidence = [hit.get('text', '') for claim in row.get('claims', [])
                    for hit in (claim.get('evidence') or [])][:6]
    definitions = taxonomy['verification']['definitions']
    return PROMPT.format(
        gold=row.get('answer_gold_label'), prediction=row.get('answer_prediction'),
        question=_clip(row.get('question'), 400), answer=_clip(row.get('answer'), 300),
        claims=json.dumps(claims, ensure_ascii=False),
        evidence=json.dumps([_clip(e, 280) for e in evidence], ensure_ascii=False),
        definitions='\n'.join(f'- {name}: {text}' for name, text in definitions.items()),
        precedence='\n'.join(f'- {rule}' for rule in taxonomy['verification'].get('precedence', [])),
        categories=json.dumps(taxonomy['verification']['categories']))


def validate(payload, taxonomy: dict) -> dict | None:
    categories = set(taxonomy['verification']['categories'])
    if not isinstance(payload, dict) or payload.get('category') not in categories:
        return None
    try:
        confidence = min(1., max(0., float(payload.get('confidence', 0.))))
    except (TypeError, ValueError):
        return None
    rationale = str(payload.get('rationale', '')).strip()
    if not rationale:
        return None
    return {'category': payload['category'], 'rationale': rationale[:400],
            'confidence': confidence}


def propose(row: dict, llm, taxonomy: dict) -> dict:
    try:
        payload = llm.generate_json(build_prompt(row, taxonomy), system=SYSTEM)
    except ProviderUnavailableError:
        raise
    except Exception as exc:  # parse failure -> reviewer decides without a suggestion
        return {'id': row['id'], 'condition': row.get('condition'), 'queue': 'verification',
                'provisional_llm_assisted': True, 'model': llm.model, 'valid': False,
                'error': f'{type(exc).__name__}: {exc}'[:200]}
    suggestion = validate(payload, taxonomy)
    record = {'id': row['id'], 'condition': row.get('condition'), 'queue': 'verification',
              'provisional_llm_assisted': True, 'model': llm.model,
              'taxonomy_version': taxonomy.get('version'), 'valid': suggestion is not None}
    if suggestion is None:
        record['error'] = 'Model replied outside the allowed schema'
        record['raw_response'] = _clip(json.dumps(payload, ensure_ascii=False), 600)
    else:
        record.update(suggestion)
    return record


def run(rows: list[dict], llm, taxonomy: dict, workers: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda row: propose(row, llm, taxonomy), rows))


def copy_extraction() -> list[dict]:
    rows = []
    for row in read_jsonl(V1_EXTRACTION):
        copied = dict(row)
        copied['taxonomy_version'] = 'v2'
        copied['source'] = 'experiments/stage6_errors/provisional_extraction.jsonl'
        copied['source_note'] = 'v1 verdicts reused; v2 collects Pass/Fail only'
        rows.append(copied)
    queue_ids = {row['id'] for row in read_jsonl(EXTRACTION_QUEUE)}
    copied_ids = {row['id'] for row in rows}
    if queue_ids != copied_ids:
        raise ValueError(f'extraction proposal ids do not match the queue: '
                         f'{len(queue_ids - copied_ids)} missing, {len(copied_ids - queue_ids)} extra')
    return rows


def summarise(verification: list[dict], taxonomy: dict, model: str | None) -> str:
    valid = [row for row in verification if row.get('valid')]
    categories = Counter(row['category'] for row in valid)
    lines = ['# v2 provisional verification proposals', '',
             f"Taxonomy: `{taxonomy['version']}`. Generated by `{model}`.",
             '**Proposals, not human labels.** A human accepts or corrects each one.', '',
             f'- records: {len(verification)}', f'- usable: {len(valid)}',
             f'- unparseable: {len(verification) - len(valid)}', '',
             '| Category | Count |', '|---|---:|']
    for category, count in categories.most_common():
        lines.append(f'| {category} | {count} |')
    low = sum(1 for row in valid if row['confidence'] < 0.5)
    lines += ['', f'Proposals with confidence below 0.5: {low}', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', choices=['extraction', 'verification', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verification_path = OUT_DIR / 'verification.jsonl'
    extraction_path = OUT_DIR / 'extraction.jsonl'
    if not args.force and verification_path.exists() and args.queue in ('verification', 'all'):
        raise SystemExit(f'{verification_path.relative_to(PROJECT_ROOT)} exists; pass --force.')

    taxonomy = load_taxonomy()
    extraction, verification, llm = [], [], None
    if args.queue in ('extraction', 'all'):
        extraction = copy_extraction()
        write_jsonl(extraction_path, extraction)
        print(f'extraction proposals copied: {len(extraction)}', flush=True)
    if args.queue in ('verification', 'all'):
        rows = list(read_jsonl(VERIFICATION_QUEUE))
        rows = rows[:args.limit] if args.limit else rows
        llm = OllamaLLM()
        verification = run(rows, llm, taxonomy, args.workers)
        write_jsonl(verification_path, verification)
        print(f'verification proposals: {len(verification)}', flush=True)
        (OUT_DIR / 'SUMMARY.md').write_text(summarise(verification, taxonomy, llm.model),
                                               encoding='utf-8')

    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'taxonomy': str(TAXONOMY.relative_to(PROJECT_ROOT)),
        'taxonomy_sha256': sha256(TAXONOMY),
        'model': getattr(llm, 'model', None),
        'provisional_llm_assisted': True,
        'human_labels_modified': False,
        'extraction_records': len(extraction),
        'verification_records': len(verification),
        'verification_categories': taxonomy['verification']['categories'],
    }
    (OUT_DIR / 'proposal_summary.json').write_text(json.dumps(report, indent=2) + '\n',
                                                   encoding='utf-8')
    print(f'Saved -> {OUT_DIR.relative_to(PROJECT_ROOT)}', flush=True)


if __name__ == '__main__':
    main()
