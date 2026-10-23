"""Build reranker + NLI cache for the 3,000 extra v4 train claims.

Checkpointed: progress is saved every 100 claims and a CUDA fault can be resumed
without redoing the finished rows. Val/test are untouched.

    python -m scripts.v4.build_v4_extra_rerank
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.datasets.schema import load_examples
from src.retrieval.corpus import SentenceCorpus
from src.retrieval.hybrid import Retriever
from src.retrieval.rerank import Reranker
from src.utils.config import PROJECT_ROOT
from src.verification.nli import NLIModel

EXTRA = PROJECT_ROOT / 'data/splits/fever_train_v4_extra_3000.jsonl'
OUT = PROJECT_ROOT / 'data/processed/rerank/train_extra_hybrid_top5.jsonl'
CHECKPOINT = OUT.with_suffix('.partial.jsonl')
SAVE_EVERY = 100


def load_done() -> list[dict]:
    path = CHECKPOINT if CHECKPOINT.exists() else OUT
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def save(rows: list[dict]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                  for row in rows), encoding='utf-8')


def main() -> None:
    if OUT.exists():
        print(f'Cache exists: {OUT}')
        return
    examples = load_examples(EXTRA)
    corpus = SentenceCorpus.from_fever()
    retriever = Retriever()
    retrieved = retriever.search_all([example.claims[0] for example in examples], k=50)
    rows = []
    for index, example in enumerate(examples):
        row = {'id': example.id, 'claim': example.claims[0], 'label': example.label}
        for method in ('bm25', 'dense', 'hybrid'):
            hits = []
            for doc, score in retrieved[method][index]:
                page_id, sent_id = corpus.keys[doc]
                hits.append({'doc': int(doc), 'page_id': page_id,
                             'sent_id': int(sent_id), 'score': float(score)})
            row[method] = hits
        rows.append(row)

    done = load_done()
    done_ids = {row['id'] for row in done}
    todo = [row for row in rows if row['id'] not in done_ids]
    if not todo:
        OUT.write_text(CHECKPOINT.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'Cache complete: {OUT}')
        return
    print(f'Resuming with {len(done)} done, {len(todo)} remaining', flush=True)

    import torch

    reranker = Reranker(batch_size=16)
    nli_model = NLIModel()
    started = time.time()
    for number, row in enumerate(todo, 1):
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
        if number % SAVE_EVERY == 0:
            save(done)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elapsed = time.time() - started
            print(f'{len(done)}/{len(rows)} done ({elapsed:.0f}s)', flush=True)
    save(done)
    OUT.write_text(CHECKPOINT.read_text(encoding='utf-8'), encoding='utf-8')
    CHECKPOINT.unlink(missing_ok=True)
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
