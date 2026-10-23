"""Evaluate the reranked top-5 vs top-8 arms from the P1 cache.

The reranker cache stores top-8 hits; this script slices it to evaluate both arms
without re-running the cross-encoder. Val only.

    python -m scripts.v4.evaluate_rerank_arms_v4
"""
from __future__ import annotations

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


def load_cache(split: str, top_k: int) -> list[dict]:
    path = CACHE / f'{split}_hybrid_top{top_k}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def main() -> None:
    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    corpus = SentenceCorpus.from_fever()
    sim_model = SimilarityModel()
    examples = load_examples(PROJECT_ROOT / 'data/splits/fever_val.jsonl')
    rows = {'train': load_split('train'), 'val': load_split('val')}
    cached = {'train': load_cache('train', 8), 'val': load_cache('val', 8)}
    for split in rows:
        if [row['id'] for row in cached[split]] != [row['id'] for row in rows[split]]:
            raise ValueError(f'{split}: reranker cache ids do not match the split')

    arms = {}
    for k in (5, 8):
        enriched = {}
        for split in rows:
            by_id = {row['id']: row for row in rows[split]}
            enriched[split] = []
            for item in cached[split]:
                row = dict(by_id[item['id']])
                row['rerank'] = item['hits'][:k]
                enriched[split].append(row)
        matrices = {split: build_matrix(enriched[split], 'rerank',
                                        similarity_matrix(enriched[split], 'rerank', corpus,
                                                          sim_model, top=k), top=k)
                    for split in rows}
        golds = {split: labels(enriched[split]) for split in rows}
        train_y = np.array([LABELS.index(label) for label in golds['train']], dtype=np.int64)
        val_y = np.array([LABELS.index(label) for label in golds['val']], dtype=np.int64)
        model, chosen_c, _ = fit_model(matrices['train'], train_y, matrices['val'], val_y,
                                       list(range(len(FEATURE_NAMES))))
        predicted = [LABELS[i] for i in model.predict(matrices['val'])]
        metrics = multiclass_metrics(golds['val'], predicted)
        fixed = [decide(aggregate_probabilities([hit['nli'] for hit in row['rerank'][:k]]),
                        0.7, 0.95) for row in enriched['val']]
        fixed_metrics = multiclass_metrics(golds['val'], fixed)
        hits = [[(hit['doc'], hit['score']) for hit in row['rerank']] for row in enriched['val']]
        retrieval = retrieval_metrics(examples, hits, corpus.keys)
        keep = {
            'val_macro_f1_min': V1_VAL_MACRO_F1 + 0.005,
            'insufficient_f1_min': V1_VAL_INSUFFICIENT_F1 - 0.01,
            'macro_f1': metrics['macro_f1'],
            'insufficient_f1': metrics['per_class']['INSUFFICIENT']['f1'],
        }
        keep['passed'] = bool(keep['macro_f1'] >= keep['val_macro_f1_min']
                              and keep['insufficient_f1'] >= keep['insufficient_f1_min'])
        arms[k] = {
            'k': k, 'selected_C': chosen_c, 'val': metrics,
            'fixed_rule': fixed_metrics,
            'retrieval': retrieval,
            'delta_vs_v1': {
                'macro_f1': metrics['macro_f1'] - V1_VAL_MACRO_F1,
                'insufficient_f1': metrics['per_class']['INSUFFICIENT']['f1']
                - V1_VAL_INSUFFICIENT_F1,
            },
            'keep_rule': keep,
        }
        print(f'k={k}: val macro-F1={metrics["macro_f1"]:.4f} '
              f'INSUFFICIENT F1={metrics["per_class"]["INSUFFICIENT"]["f1"]:.4f} '
              f'C={chosen_c} keep={keep["passed"]}', flush=True)

    passing = [arm for arm in arms.values() if arm['keep_rule']['passed']]
    selected = max(passing, key=lambda arm: arm['val']['macro_f1']) if passing else None
    report = {
        'version': 'v4-retrieval-stack',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val',
        'test_evaluated': False,
        'config': {'depth': 50, 'reranker': 'BAAI/bge-reranker-base',
                   'nli_model': 'MoritzLaurer/DeBERTa-v3-base-mnli',
                   'arms': [5, 8]},
        'arms': {str(k): arm for k, arm in arms.items()},
        'selected_k': selected['k'] if selected else None,
        'v1_baseline': {'macro_f1': V1_VAL_MACRO_F1,
                        'insufficient_f1': V1_VAL_INSUFFICIENT_F1},
        'c_grid': list(C_GRID),
        'notes': [
            'Val only; no test data was read.',
            'Top-5 is the arm to keep if selected; the reranker cache can still store top-8.',
            'The cache is under data/processed/rerank and is not committed.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'retrieval_stack.json').write_text(json.dumps(report, indent=2) + '\n',
                                              encoding='utf-8')
    print(json.dumps({'selected_k': report['selected_k'],
                      'arms': {k: {'macro_f1': arm['val']['macro_f1'],
                                   'insufficient_f1': arm['val']['per_class']['INSUFFICIENT']['f1'],
                                   'delta': arm['delta_vs_v1']}
                               for k, arm in arms.items()}}, indent=2))


if __name__ == '__main__':
    main()
