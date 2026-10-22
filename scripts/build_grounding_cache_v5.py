"""Build v5 MiniCheck/HHEM grounding scores for train, validation, and test.

The test split is refused until the final freeze exists. The cache is keyed by
claim ID and the ordered top-5 evidence list, so it can be joined back to the
v4 feature matrix.

    python -m scripts.build_grounding_cache_v5 --model minicheck --split val
    python -m scripts.build_grounding_cache_v5 --model hhem --split test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.retrieval.corpus import SentenceCorpus
from src.utils.config import PROJECT_ROOT
from src.verification.features import TOP_K
from src.verification.grounding import HHEMScorer, MiniCheckScorer

OUT = PROJECT_ROOT / 'data/processed/grounding'
FREEZE = PROJECT_ROOT / 'experiments/v5/freeze.json'
SPLITS = {
    'train': PROJECT_ROOT / 'data/splits/fever_train_v4_9000.jsonl',
    'val': PROJECT_ROOT / 'data/splits/fever_val.jsonl',
    'test': PROJECT_ROOT / 'data/splits/fever_test.jsonl',
}
RERANK = {
    'train': [
        PROJECT_ROOT / 'data/processed/rerank/train_hybrid_top8.jsonl',
        PROJECT_ROOT / 'data/processed/rerank/train_extra_hybrid_top5_6000.jsonl',
    ],
    'val': [PROJECT_ROOT / 'data/processed/rerank/val_hybrid_top8.jsonl'],
    'test': [PROJECT_ROOT / 'data/processed/rerank/test_hybrid_top8.jsonl'],
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _load_rows(split: str) -> list[dict]:
    rows = []
    for row in _load_jsonl(SPLITS[split]):
        item = {'id': row['id'],
                'claim': row.get('claim') or (row.get('claims') or [''])[0]}
        if split != 'test':
            item['label'] = row['label']
        rows.append(item)
    return rows


def _load_hits(split: str) -> dict[str, list[dict]]:
    hits: dict[str, list[dict]] = {}
    for path in RERANK[split]:
        for row in _load_jsonl(path):
            hits.setdefault(row['id'], row['hits'])
    return hits


def _scorer(name: str, device: str | None, batch_size: int | None):
    if name == 'minicheck':
        return MiniCheckScorer(device=device, batch_size=batch_size or 4)
    if name == 'hhem':
        return HHEMScorer(device=device, batch_size=batch_size or 8)
    raise ValueError(f'Unknown grounding model: {name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('minicheck', 'hhem'), required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--claim-batch', type=int, default=32)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.split == 'test' and not FREEZE.exists():
        raise SystemExit(f'{FREEZE.relative_to(PROJECT_ROOT)} does not exist; freeze first.')
    output = args.output or OUT / f'{args.model}_{args.split}.jsonl'
    if output.exists() and not args.overwrite:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --overwrite.')

    rows = _load_rows(args.split)
    hits = _load_hits(args.split)
    missing = [row['id'] for row in rows if row['id'] not in hits]
    if missing:
        raise SystemExit(f'Missing reranker hits for {len(missing)} rows, e.g. {missing[:5]}')
    if args.limit:
        rows = rows[:args.limit]

    corpus = SentenceCorpus.from_fever()
    scorer = _scorer(args.model, args.device, args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    written = 0
    with output.open('w', encoding='utf-8') as handle:
        for chunk_start in range(0, len(rows), args.claim_batch):
            chunk = rows[chunk_start:chunk_start + args.claim_batch]
            documents: list[str] = []
            claims: list[str] = []
            lengths: list[int] = []
            evidence_by_row: list[list[dict]] = []
            for row in chunk:
                evidence_rows = hits[row['id']][:TOP_K]
                evidence = [corpus.texts[hit['doc']] for hit in evidence_rows]
                documents.extend(evidence)
                claims.extend([row['claim']] * len(evidence))
                lengths.append(len(evidence))
                evidence_by_row.append(evidence_rows)

            scores = scorer.score_pairs(documents, claims) if documents else []
            scores = np.asarray(scores, dtype=np.float32)
            if scores.size and not np.isfinite(scores).all():
                bad = int((~np.isfinite(scores)).sum())
                print(f'{args.model}/{args.split}: replacing {bad} non-finite '
                      'grounding scores with 0.0', flush=True)
                scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            offset = 0
            for row, evidence_rows, length in zip(chunk, evidence_by_row, lengths):
                row_scores = scores[offset:offset + length]
                offset += length
                record = {
                    'id': row['id'],
                    'split': args.split,
                    'model': args.model,
                    'top_k': len(evidence_rows),
                    'claim': row['claim'],
                    'label': row.get('label'),
                    'scores': [float(value) for value in row_scores],
                    'evidence': [
                        {'page_id': hit['page_id'], 'sent_id': int(hit['sent_id'])}
                        for hit in evidence_rows
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                written += 1
            if written % 100 == 0 or written == len(rows):
                elapsed = time.time() - started
                rate = written / elapsed if elapsed else 0.0
                print(f'{args.model}/{args.split}: {written}/{len(rows)} '
                      f'({rate:.1f} claims/s)', flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        display_path = output.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        display_path = output
    print(f'Saved {written} rows -> {display_path}')


if __name__ == '__main__':
    main()
