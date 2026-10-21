"""Val-only improvement studies: cross-encoder reranking and neighbour context.

Both studies are deliberately restricted to the FEVER *val* split. Nothing is
fit, no threshold is tuned, and the test split is never touched. Results are
evidence for (or against) promoting a change into a future version.

    python -m src.evaluation.val_studies --study rerank
    python -m src.evaluation.val_studies --study context
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import LABELS, load_examples
from src.evaluation.fever_retrieval import fever_score, retrieval_metrics
from src.evaluation.halueval_claims import PairCache
from src.evaluation.metrics import multiclass_metrics
from src.retrieval.corpus import SentenceCorpus
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide
from src.verification.features import load_split

METHODS = ('bm25', 'dense', 'hybrid')
TOP_K = 5


def _setup(limit: int | None):
    examples = load_examples(PROJECT_ROOT / 'data/splits/fever_val.jsonl')
    rows = load_split('val')
    if [e.id for e in examples] != [r['id'] for r in rows]:
        raise ValueError('Val split and retrieval cache are out of order')
    if limit:
        examples, rows = examples[:limit], rows[:limit]
    return examples, rows, SentenceCorpus.from_fever()


def _baseline(rows, method, entail, contra):
    predictions = [decide(aggregate_probabilities([h['nli'] for h in row[method][:TOP_K]]),
                          entail, contra) for row in rows]
    return predictions


def _score(examples, predictions, gold_labels):
    return multiclass_metrics(gold_labels, predictions)


def _strip_title(text: str) -> str:
    head, sep, tail = text.partition(': ')
    return tail if sep and len(head) < 60 else text


def _expanded_premise(corpus, index, page_id, sent_id):
    parts = [_strip_title(corpus.texts[index[key]])
             for key in ((page_id, sent_id - 1), (page_id, sent_id), (page_id, sent_id + 1))
             if key in index]
    return ' '.join(parts)


def _summarise(baseline, candidate):
    return {'baseline': baseline, 'candidate': candidate,
            'delta': {key: candidate[key] - baseline[key]
                      for key in ('accuracy', 'macro_f1')},
            'per_class_f1_delta': {label: candidate['per_class'][label]['f1'] -
                                   baseline['per_class'][label]['f1'] for label in LABELS}}


def rerank_study(methods, limit, depth, out, device='auto'):
    import torch

    from src.retrieval.rerank import Reranker
    from src.verification.nli import NLIModel

    cfg = load_config()
    entail = cfg['detector']['entail_threshold']
    contra = cfg['detector']['contra_threshold']
    examples, rows, corpus = _setup(limit)
    labels = [e.label for e in examples]
    reranker = Reranker(batch_size=32, device=device)
    cache = PairCache(NLIModel(), PROJECT_ROOT / 'data/cache/val_study_rerank_nli.jsonl')

    existing = _load(out / 'val_study_rerank.json') or {}
    results = existing.get('methods', {})
    for method in methods:
        cached_hits = [[(h['doc'], float(h['score'])) for h in row[method]] for row in rows]
        baseline_retrieval = retrieval_metrics(examples, cached_hits, corpus.keys)

        reranked = [reranker.rerank(row['claim'],
                                    [(h['doc'], float(h['score'])) for h in row[method][:depth]],
                                    corpus, k=depth)
                    for row in rows]
        reranked_retrieval = retrieval_metrics(examples, reranked, corpus.keys)

        pairs = [(corpus.texts[doc], row['claim'])
                 for row, hits in zip(rows, reranked) for doc, _ in hits[:TOP_K]]
        probs = cache.predict(pairs).reshape((len(rows), -1, 3))
        baseline_predictions = _baseline(rows, method, entail, contra)
        reranked_predictions = [decide(aggregate_probabilities(p), entail, contra)
                                for p in probs]
        baseline_metrics = _score(examples, baseline_predictions, labels)
        reranked_metrics = _score(examples, reranked_predictions, labels)
        results[method] = {
            'retrieval': {'baseline': baseline_retrieval, 'reranked': reranked_retrieval},
            'pipeline': _summarise(baseline_metrics, reranked_metrics),
            'reranked_fever_score': fever_score(examples, reranked_predictions, reranked,
                                                corpus.keys, TOP_K),
        }
        _save(out, 'rerank', {
            'study': 'cross_encoder_rerank', 'split': 'val', 'test_evaluated': False,
            'reranker': reranker.model_name, 'nli_model': cache.fingerprint[0],
            'device': reranker.device,
            'depth': depth, 'top_k': TOP_K, 'n': len(examples),
            'thresholds': {'entail': entail, 'contra': contra},
            'training_performed': False, 'threshold_tuning_performed': False,
            'methods': results,
        })
        print(f'[{method}] strict R@5 {baseline_retrieval["@5"]["strict_recall"]:.4f} -> '
              f'{reranked_retrieval["@5"]["strict_recall"]:.4f} | '
              f'macro-F1 {baseline_metrics["macro_f1"]:.4f} -> '
              f'{reranked_metrics["macro_f1"]:.4f}', flush=True)
        del reranked, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def context_study(methods, limit, out):
    from src.verification.nli import NLIModel

    cfg = load_config()
    entail = cfg['detector']['entail_threshold']
    contra = cfg['detector']['contra_threshold']
    examples, rows, corpus = _setup(limit)
    labels = [e.label for e in examples]
    cache = PairCache(NLIModel(), PROJECT_ROOT / 'data/cache/val_study_context_nli.jsonl')
    index = {key: i for i, key in enumerate(corpus.keys)}

    results = {}
    for method in methods:
        expanded, flat = [], []
        for row in rows:
            premises = []
            for hit in row[method][:TOP_K]:
                premises.append(_expanded_premise(corpus, index, hit['page_id'], hit['sent_id']))
            expanded.append(premises)
            flat.extend((premise, row['claim']) for premise in premises)
        probs = cache.predict(flat).reshape((len(rows), -1, 3))

        baseline_predictions = _baseline(rows, method, entail, contra)
        context_predictions = [decide(aggregate_probabilities(p), entail, contra)
                               for p in probs]
        baseline_metrics = _score(examples, baseline_predictions, labels)
        context_metrics = _score(examples, context_predictions, labels)
        changed = sum(a != b for a, b in zip(baseline_predictions, context_predictions))
        results[method] = {
            'pipeline': _summarise(baseline_metrics, context_metrics),
            'changed_predictions': changed,
            'premise': 'previous + current + next sentence from the same page',
        }
        print(f'[{method}] macro-F1 {baseline_metrics["macro_f1"]:.4f} -> '
              f'{context_metrics["macro_f1"]:.4f} ({changed} predictions changed)', flush=True)

    _save(out, 'context', {
        'study': 'neighbour_sentence_context', 'split': 'val', 'test_evaluated': False,
        'nli_model': cache.fingerprint[0], 'top_k': TOP_K, 'n': len(examples),
        'thresholds': {'entail': entail, 'contra': contra},
        'training_performed': False, 'threshold_tuning_performed': False,
        'methods': results,
    })
    return results


def _save(out: Path, name: str, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report['timestamp'] = datetime.now(timezone.utc).isoformat()
    (out / f'val_study_{name}.json').write_text(json.dumps(report, indent=2) + '\n',
                                                encoding='utf-8')
    print(f'Saved -> {out / f"val_study_{name}.json"}', flush=True)


def _load(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['rerank', 'context', 'all'], default='all')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--limit', type=int, default=None,
                        help='Development smoke test only: first N val rows.')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Reranker device. Use cpu if the CUDA cross-encoder is unstable.')
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'experiments/stage5_learned')
    args = parser.parse_args()
    methods = tuple(m.strip() for m in args.methods.split(',') if m.strip())
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f'Unknown methods: {sorted(unknown)}')

    if args.study in ('rerank', 'all'):
        rerank_study(methods, args.limit, args.depth, args.out, args.device)
    if args.study in ('context', 'all'):
        context_study(methods, args.limit, args.out)


if __name__ == '__main__':
    main()
