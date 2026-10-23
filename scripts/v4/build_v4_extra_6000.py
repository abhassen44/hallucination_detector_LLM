"""Build the second 3,000 extra-claim cache (9k train stage).

Reuses the existing 3k extra cache and processes only the remaining claims from
`fever_train_v4_extra_6000.jsonl`. Checkpointed for CUDA faults.

    python -m scripts.v4.build_v4_extra_6000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.datasets.schema import load_examples
from src.retrieval.corpus import SentenceCorpus
from src.retrieval.hybrid import Retriever
from src.retrieval.rerank import Reranker
from src.utils.config import PROJECT_ROOT
from src.verification.nli import NLIModel

SPLIT = PROJECT_ROOT / 'data/splits/fever_train_v4_extra_6000.jsonl'
FIRST = PROJECT_ROOT / 'data/processed/rerank/train_extra_hybrid_top5.jsonl'
OUT = PROJECT_ROOT / 'data/processed/rerank/train_extra_hybrid_top5_6000.jsonl'
CHECKPOINT = OUT.with_suffix('.partial.jsonl')
SAVE_EVERY = 100


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def save(rows: list[dict]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                  for row in rows), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--max-new', type=int, default=None,
                        help='Process at most this many new rows, then checkpoint and exit.')
    args = parser.parse_args()

    if OUT.exists():
        print(f'Cache exists: {OUT}')
        return
    examples = load_examples(SPLIT)
    first = read_jsonl(FIRST)
    first_ids = {row['id'] for row in first}
    done = [row for row in read_jsonl(CHECKPOINT) if row['id'] not in first_ids]
    done_ids = first_ids | {row['id'] for row in done}
    todo = [example for example in examples if example.id not in done_ids]
    if not todo:
        save(first + done)
        OUT.write_text(CHECKPOINT.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'Cache complete: {OUT}')
        return
    print(f'{len(first)} first-stage rows, {len(todo)} remaining', flush=True)

    corpus = SentenceCorpus.from_fever()
    retriever = Retriever()
    retrieved = retriever.search_all([example.claims[0] for example in todo], k=50)
    rows = []
    for index, example in enumerate(todo):
        row = {'id': example.id, 'claim': example.claims[0], 'label': example.label}
        for method in ('bm25', 'dense', 'hybrid'):
            hits = []
            for doc, score in retrieved[method][index]:
                page_id, sent_id = corpus.keys[doc]
                hits.append({'doc': int(doc), 'page_id': page_id,
                             'sent_id': int(sent_id), 'score': float(score)})
            row[method] = hits
        rows.append(row)

    import torch

    reranker = Reranker(batch_size=args.batch_size, max_length=args.max_length,
                        device=args.device)
    nli_model = NLIModel()
    started = time.time()
    for number, row in enumerate(rows, 1):
        candidates = [(int(hit['doc']), float(hit['score'])) for hit in row['hybrid'][:50]]
        reranked = reranker.rerank(row['claim'], candidates, corpus, k=5)
        docs = [doc for doc, _ in reranked]
        probabilities = nli_model.predict([corpus.texts[doc] for doc in docs],
                                          [row['claim']] * len(docs), show_progress=False)
        hits = []
        for (doc, score), probability in zip(reranked, probabilities):
            page_id, sent_id = corpus.keys[doc]
            hits.append({'doc': int(doc), 'page_id': page_id, 'sent_id': int(sent_id),
                         'score': float(score), 'nli': probability.tolist()})
        done.append({'id': row['id'], 'claim': row['claim'], 'label': row['label'],
                     'hits': hits})
        if number % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if number % SAVE_EVERY == 0:
            save(first + done)
            print(f'{len(done)}/{len(rows)} new done ({time.time() - started:.0f}s)',
                  flush=True)
        if args.max_new and number >= args.max_new:
            save(first + done)
            print(f'chunk complete at {len(done)} rows', flush=True)
            return
    save(first + done)
    OUT.write_text(CHECKPOINT.read_text(encoding='utf-8'), encoding='utf-8')
    CHECKPOINT.unlink(missing_ok=True)
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
