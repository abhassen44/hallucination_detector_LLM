"""P1: reranker + top-8 retrieval stack on val.

Reranks the cached top-50 hybrid candidates with bge-reranker-base, keeps top-8,
computes NLI for the reranked hits, then refits the logistic combiner and compares
against v1 on val. No test data is read.

    python -m scripts.retrieval_stack_v4
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import LABELS, load_examples
from src.evaluation.fever_retrieval import retrieval_metrics
from src.evaluation.fit_detector import C_GRID, fit_model
from src.evaluation.metrics import multiclass_metrics
from src.utils.config import PROJECT_ROOT
from src.verification.detector import aggregate_probabilities, decide
from src.verification.features import (FEATURE_NAMES, build_matrix, labels, load_split,
                                       similarity_matrix)

OUT = PROJECT_ROOT / 'experiments/v4'
CACHE = PROJECT_ROOT / 'data/processed/rerank'
V1_VAL_MACRO_F1 = 0.6763644683287465
V1_VAL_INSUFFICIENT_F1 = 0.5207296849087893


def cache_path(split: str, top_k: int, limit: int | None = None) -> Path:
    suffix = f'_limit{limit}' if limit else ''
    return CACHE / f'{split}_hybrid_top{top_k}{suffix}.jsonl'


def load_cache(path: Path, rows: list[dict]) -> list[dict] | None:
    if not path.exists():
        return None
    cached = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
              if line.strip()]
    if [row['id'] for row in cached] != [row['id'] for row in rows]:
        return None
    return cached


def build_cache(rows: list[dict], split: str, top_k: int, depth: int,
                reranker, nli_model, corpus) -> list[dict]:
    cached = []
    for number, row in enumerate(rows, 1):
        candidates = [(int(hit['doc']), float(hit['score'])) for hit in row['hybrid'][:depth]]
        reranked = reranker.rerank(row['claim'], candidates, corpus, k=top_k)
        docs = [doc for doc, _ in reranked]
        probabilities = nli_model.predict([corpus.texts[doc] for doc in docs],
                                          [row['claim']] * len(docs), show_progress=False)
        hits = []
        for (doc, score), probability in zip(reranked, probabilities):
            page_id, sent_id = corpus.keys[doc]
            hits.append({'doc': int(doc), 'page_id': page_id, 'sent_id': int(sent_id),
                         'score': float(score), 'nli': probability.tolist()})
        cached.append({'id': row['id'], 'claim': row['claim'], 'label': row['label'],
                       'hits': hits})
        if number % 500 == 0:
            print(f'{split}: reranked {number}/{len(rows)}', flush=True)
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output = OUT / 'retrieval_stack.json'
    if output.exists() and not args.force:
        raise SystemExit(f'{output.relative_to(PROJECT_ROOT)} exists; pass --force.')

    splits = {}
    for split in ('train', 'val'):
        rows = load_split(split)
        if args.limit:
            rows = rows[:args.limit]
        splits[split] = rows

    from src.retrieval.corpus import SentenceCorpus

    corpus = SentenceCorpus.from_fever()
    need_build = any(load_cache(cache_path(split, args.top_k, args.limit), rows) is None
                     for split, rows in splits.items())
    reranker = nli_model = None
    if need_build:
        from src.retrieval.rerank import Reranker
        from src.verification.nli import NLIModel

        reranker = Reranker()
        nli_model = NLIModel()

    cached = {}
    for split, rows in splits.items():
        path = cache_path(split, args.top_k, args.limit)
        cached[split] = load_cache(path, rows)
        if cached[split] is None:
            print(f'{split}: building rerank cache at {path}', flush=True)
            cached[split] = build_cache(rows, split, args.top_k, args.depth,
                                        reranker, nli_model, corpus)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                    for row in cached[split]), encoding='utf-8')

    from src.verification.similarity import SimilarityModel

    sim_model = SimilarityModel()
    matrices, golds = {}, {}
    for split, rows in splits.items():
        enriched = []
        by_id = {row['id']: row for row in rows}
        for cached_row in cached[split]:
            row = dict(by_id[cached_row['id']])
            row['rerank'] = cached_row['hits']
            enriched.append(row)
        sims = similarity_matrix(enriched, 'rerank', corpus, sim_model, top=args.top_k)
        matrices[split] = build_matrix(enriched, 'rerank', sims, top=args.top_k)
        golds[split] = labels(enriched)
    print('Features ready:', {split: matrix.shape for split, matrix in matrices.items()},
          flush=True)

    train_y = np.array([LABELS.index(label) for label in golds['train']], dtype=np.int64)
    val_y = np.array([LABELS.index(label) for label in golds['val']], dtype=np.int64)
    model, chosen_c, val_score = fit_model(
        matrices['train'], train_y, matrices['val'], val_y,
        list(range(len(FEATURE_NAMES))))
    predictions = model.predict(matrices['val'])
    predicted_labels = [LABELS[i] for i in predictions]
    val_metrics = multiclass_metrics(golds['val'], predicted_labels)
    fixed = [decide(aggregate_probabilities([hit['nli'] for hit in row['hits'][:args.top_k]]),
                    0.7, 0.95) for row in cached['val']]
    fixed_metrics = multiclass_metrics(golds['val'], fixed)
    examples = load_examples(PROJECT_ROOT / 'data/splits/fever_val.jsonl')[:len(golds['val'])]
    hits = [[(hit['doc'], hit['score']) for hit in row['hits']] for row in cached['val']]
    retrieval = retrieval_metrics(examples, hits, corpus.keys)

    keep = {
        'val_macro_f1_min': V1_VAL_MACRO_F1 + 0.005,
        'insufficient_f1_min': V1_VAL_INSUFFICIENT_F1 - 0.01,
        'macro_f1': val_metrics['macro_f1'],
        'insufficient_f1': val_metrics['per_class']['INSUFFICIENT']['f1'],
    }
    keep['passed'] = bool(keep['macro_f1'] >= keep['val_macro_f1_min']
                          and keep['insufficient_f1'] >= keep['insufficient_f1_min'])
    report = {
        'version': 'v4-retrieval-stack',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val',
        'test_evaluated': False,
        'config': {'top_k': args.top_k, 'depth': args.depth,
                   'reranker': 'BAAI/bge-reranker-base',
                   'nli_model': 'MoritzLaurer/DeBERTa-v3-base-mnli'},
        'selected_C': chosen_c,
        'val': val_metrics,
        'fixed_rule': fixed_metrics,
        'retrieval': retrieval,
        'v1_baseline': {'macro_f1': V1_VAL_MACRO_F1,
                        'insufficient_f1': V1_VAL_INSUFFICIENT_F1},
        'delta': {'macro_f1': val_metrics['macro_f1'] - V1_VAL_MACRO_F1,
                  'insufficient_f1': val_metrics['per_class']['INSUFFICIENT']['f1']
                  - V1_VAL_INSUFFICIENT_F1},
        'keep_rule': keep,
        'c_grid': list(C_GRID),
        'notes': [
            'Val only; no test data was read.',
            'The reranker changes the retrieval cache; if kept, all later v4 feature builds must use it.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'val_macro_f1': val_metrics['macro_f1'],
        'v1_val_macro_f1': V1_VAL_MACRO_F1,
        'insufficient_f1': val_metrics['per_class']['INSUFFICIENT']['f1'],
        'v1_insufficient_f1': V1_VAL_INSUFFICIENT_F1,
        'strict_r5': retrieval['@5']['strict_recall'],
        'strict_r8_effective': retrieval['@10']['strict_recall'],
        'keep': keep,
    }, indent=2))
    print(f'Saved -> {output.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
