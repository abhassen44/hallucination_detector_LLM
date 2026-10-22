"""Build the v5 reranker + NLI cache for a split.

The validation split is used only for smoke tests. The test split is refused
until ``experiments/v5/freeze.json`` exists, so feature construction cannot
become an excuse to inspect test labels before the final pass.

    python -m scripts.build_test_rerank_v5 --split val --limit 2 --out data/processed/rerank/val_smoke.jsonl
    python -m scripts.build_test_rerank_v5 --split test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.retrieval.corpus import SentenceCorpus
from src.retrieval.rerank import Reranker
from src.utils.config import PROJECT_ROOT
from src.verification.nli import NLIModel

RETRIEVAL = PROJECT_ROOT / 'data' / 'processed' / 'retrieval'
CACHE = PROJECT_ROOT / 'data' / 'processed' / 'rerank'
FREEZE = PROJECT_ROOT / 'experiments' / 'v5' / 'freeze.json'
SAVE_EVERY = 100


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_features(split: str) -> list[dict]:
    path = RETRIEVAL / f'fever_{split}.jsonl'
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        # Deliberately drop the label: only the claim and cached candidates are
        # needed to build retrieval features.
        rows.append({'id': source['id'], 'claim': source['claim'],
                     'hybrid': source['hybrid'][:50]})
    return rows


def _load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                    encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.split == 'test' and not FREEZE.exists():
        raise SystemExit(f'{_relative(FREEZE)} does not exist; freeze first.')
    output = args.out or CACHE / f'{args.split}_hybrid_top{args.top_k}.jsonl'
    checkpoint = output.with_suffix('.partial.jsonl')
    if output.exists() and not args.force:
        print(f'Cache exists: {_relative(output)}')
        return

    rows = _load_features(args.split)
    if args.limit:
        rows = rows[:args.limit]
    expected_ids = [row['id'] for row in rows]
    done = _load_done(checkpoint) if not args.force else []
    done_ids = {row['id'] for row in done}
    todo = [row for row in rows if row['id'] not in done_ids]
    if not todo:
        _save(output, done)
        checkpoint.unlink(missing_ok=True)
        print(f'Cache complete: {_relative(output)}')
        return
    print(f'{args.split}: {len(done)} done, {len(todo)} remaining', flush=True)

    corpus = SentenceCorpus.from_fever()
    reranker = Reranker(batch_size=16)
    nli_model = NLIModel()
    started = time.time()
    for number, row in enumerate(todo, start=1):
        candidates = [(int(hit['doc']), float(hit['score'])) for hit in row['hybrid']]
        reranked = reranker.rerank(row['claim'], candidates, corpus, k=args.top_k)
        docs = [doc for doc, _ in reranked]
        probabilities = nli_model.predict([corpus.texts[doc] for doc in docs],
                                          [row['claim']] * len(docs), show_progress=False)
        hits = []
        for (doc, score), probability in zip(reranked, probabilities):
            page_id, sent_id = corpus.keys[doc]
            hits.append({'doc': int(doc), 'page_id': page_id, 'sent_id': int(sent_id),
                         'score': float(score), 'nli': probability.tolist()})
        done.append({'id': row['id'], 'claim': row['claim'], 'hits': hits})
        if number % SAVE_EVERY == 0:
            _save(checkpoint, done)
            elapsed = time.time() - started
            print(f'{len(done)}/{len(rows)} done ({elapsed:.0f}s)', flush=True)
    _save(checkpoint, done)
    if [row['id'] for row in done] != expected_ids:
        raise SystemExit('Cache row ids do not match the requested split.')
    _save(output, done)
    checkpoint.unlink(missing_ok=True)
    print(f'Saved {len(done)} rows -> {_relative(output)}')


if __name__ == '__main__':
    main()
