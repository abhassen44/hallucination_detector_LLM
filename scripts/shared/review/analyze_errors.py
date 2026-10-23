"""Gemma-assisted first pass over the prepared human-review queues.

Writes *provisional* proposals only. Human review fields are never modified, and
every output row is stamped ``provisional_llm_assisted``. The reviewer accepts or
corrects these suggestions with ``scripts/shared/review/review.py``.

    python -m scripts.shared.review.analyze_errors --queue all
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.schema import read_jsonl, write_jsonl
from src.evaluation.cached_baseline import CATEGORIES
from src.generation.llm import OllamaLLM, ProviderUnavailableError
from src.utils.config import PROJECT_ROOT

EXTRACTION_QUEUE = PROJECT_ROOT / 'experiments/stage4_claims/human_review.jsonl'
VERIFICATION_QUEUE = PROJECT_ROOT / 'experiments/stage5_inference/error_review.jsonl'
OUT_DIR = PROJECT_ROOT / 'experiments/stage6_errors'

SYSTEM = ('You assist a human reviewer of an NLP hallucination-detection pipeline. '
          'You never decide the final label; you propose one short, checkable reason. '
          'Reply with a single JSON object and nothing else.')

EXTRACTION_PROMPT = """Task: judge whether the CLAIM EXTRACTION faithfully decomposed the answer.
Do not judge whether the answer is true. A deliberately false answer should be
extracted faithfully, without being corrected.

Question: {question}
Answer: {answer}
Extracted claims: {claims}
Heuristic flags: {flags}
Raw model response (only for fallbacks): {raw}

Category list: {categories}

Reply with JSON:
{{"verdict": "Pass" or "Fail",
  "category": one of the category list, or "none" when the verdict is Pass,
  "rationale": "at most 30 words describing an observable property of the claims",
  "confidence": number between 0 and 1}}"""

VERIFICATION_PROMPT = """Task: propose the most likely cause of a mistaken verification decision.
Gold answer label: {gold}
Predicted answer label: {prediction}

Question: {question}
Answer: {answer}
Per-claim NLI probabilities [entailment, neutral, contradiction]:
{claims}
Evidence shown to the verifier (may be truncated):
{evidence}

Category list: {categories}

Reply with JSON:
{{"category": one of the category list,
  "rationale": "at most 30 words citing the specific claim or evidence span",
  "confidence": number between 0 and 1}}"""


def _clip(text, limit=700):
    text = '' if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + ' ...[truncated]'


def _extraction_prompt(row: dict) -> str:
    return EXTRACTION_PROMPT.format(
        question=_clip(row.get('question'), 400), answer=_clip(row.get('answer'), 500),
        claims=json.dumps(row.get('claims', []), ensure_ascii=False),
        flags=row.get('heuristic_flags', []), raw=_clip(row.get('raw_model_response')),
        categories=json.dumps(CATEGORIES))


def _verification_prompt(row: dict) -> str:
    claims = []
    for claim in row.get('claims', []):
        probabilities = claim.get('probabilities') or []
        claims.append({'claim': claim.get('claim'),
                       'probabilities': [round(float(p), 3) for p in probabilities]})
    evidence = row.get('provided_context') or []
    if not evidence:
        evidence = [hit.get('text', '') for claim in row.get('claims', [])
                    for hit in (claim.get('evidence') or [])][:6]
    return VERIFICATION_PROMPT.format(
        gold=row.get('answer_gold_label'), prediction=row.get('answer_prediction'),
        question=_clip(row.get('question'), 400), answer=_clip(row.get('answer'), 300),
        claims=json.dumps(claims, ensure_ascii=False),
        evidence=json.dumps([_clip(e, 280) for e in evidence], ensure_ascii=False),
        categories=json.dumps(CATEGORIES))


def _validate_extraction(payload) -> dict | None:
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get('verdict', '')).strip().capitalize()
    category = payload.get('category')
    if verdict not in ('Pass', 'Fail'):
        return None
    if verdict == 'Pass':
        category = 'none'
    elif category not in CATEGORIES:
        return None
    try:
        confidence = min(1., max(0., float(payload.get('confidence', 0.))))
    except (TypeError, ValueError):
        return None
    rationale = str(payload.get('rationale', '')).strip()
    if not rationale:
        return None
    return {'verdict': verdict, 'category': category, 'rationale': rationale[:400],
            'confidence': confidence}


def _validate_verification(payload) -> dict | None:
    if not isinstance(payload, dict) or payload.get('category') not in CATEGORIES:
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


def _propose(llm, row, queue, validate, build_prompt):
    condition = row.get('condition')
    try:
        payload = llm.generate_json(build_prompt(row), system=SYSTEM)
    except ProviderUnavailableError:
        raise
    except Exception as exc:  # parse failure -> reviewer decides without a suggestion
        return {'id': row['id'], 'condition': condition, 'queue': queue,
                'provisional_llm_assisted': True, 'model': llm.model, 'valid': False,
                'error': f'{type(exc).__name__}: {exc}'[:200]}
    proposal = validate(payload)
    record = {'id': row['id'], 'condition': condition, 'queue': queue,
              'provisional_llm_assisted': True, 'model': llm.model,
              'valid': proposal is not None}
    if proposal is None:
        record['error'] = 'Model replied outside the allowed schema'
        record['raw_response'] = _clip(json.dumps(payload, ensure_ascii=False), 600)
    else:
        record.update(proposal)
    return record


def run(queue_name: str, rows: list[dict], llm, workers: int) -> list[dict]:
    if queue_name == 'extraction':
        validate, build = _validate_extraction, _extraction_prompt
    else:
        validate, build = _validate_verification, _verification_prompt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda row: _propose(llm, row, queue_name, validate, build), rows))
    return results


def summarise(extraction: list[dict], verification: list[dict]) -> str:
    lines = ['# Provisional error analysis', '',
             'Generated by an LLM (`gemma4:31b-cloud`) as a first pass over the prepared',
             'review queues. **These are proposals, not human labels.** Every record carries',
             '`provisional_llm_assisted: true`. A human must accept or correct each one.', '']
    for name, records in (('Extraction review', extraction), ('Verification errors', verification)):
        valid = [r for r in records if r.get('valid')]
        invalid = len(records) - len(valid)
        lines += [f'## {name}', '', f'- records: {len(records)}',
                  f'- usable proposals: {len(valid)}', f'- unparseable: {invalid}']
        if name.startswith('Extraction'):
            verdicts = Counter(r['verdict'] for r in valid)
            lines.append(f"- proposed pass/fail: {dict(verdicts)}")
            categories = Counter(r['category'] for r in valid if r['verdict'] == 'Fail')
        else:
            categories = Counter(r['category'] for r in valid)
        lines += ['', '| Category | Count |', '|---|---:|']
        for category, count in categories.most_common():
            lines.append(f'| {category} | {count} |')
        low = sum(1 for r in valid if r['confidence'] < 0.5)
        lines += [f'', f'Proposals with confidence below 0.5: {low}', '']
        for category, _ in categories.most_common(1):
            examples = [r for r in valid if r['category'] == category][:2]
            for example in examples:
                lines.append(f'- **{category}** `{example["id"]}`: {example["rationale"]}')
            lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', choices=['extraction', 'verification', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    llm = OllamaLLM()
    extraction, verification = [], []
    if args.queue in ('extraction', 'all'):
        rows = list(read_jsonl(EXTRACTION_QUEUE))
        rows = rows[:args.limit] if args.limit else rows
        extraction = run('extraction', rows, llm, args.workers)
        write_jsonl(args.out / 'provisional_extraction.jsonl', extraction)
        print(f'extraction proposals: {len(extraction)}', flush=True)
    if args.queue in ('verification', 'all'):
        rows = list(read_jsonl(VERIFICATION_QUEUE))
        rows = rows[:args.limit] if args.limit else rows
        verification = run('verification', rows, llm, args.workers)
        write_jsonl(args.out / 'provisional_verification.jsonl', verification)
        print(f'verification proposals: {len(verification)}', flush=True)

    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'model': llm.model,
              'provisional_llm_assisted': True, 'human_labels_modified': False,
              'extraction_records': len(extraction), 'verification_records': len(verification),
              'categories': CATEGORIES}
    (args.out / 'provisional_summary.json').write_text(json.dumps(report, indent=2) + '\n',
                                                       encoding='utf-8')
    (args.out / 'SUMMARY.md').write_text(summarise(extraction, verification), encoding='utf-8')
    print(f'Saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
