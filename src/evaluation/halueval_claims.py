"""Frozen-rule HaluEval answer evaluation. No training, fitting or threshold search.

Provided context is an evidence upper-bound condition, not Wikipedia retrieval.
All claim predictions are scored ONLY after answer aggregation: HaluEval has no
claim-level gold labels. Existing extraction outputs are read without LLM calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import BIN_HALLUCINATED, BIN_SUPPORTED, load_examples, read_jsonl, write_jsonl
from src.evaluation.metrics import binary_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide

PROTOCOL = {
    'dataset': 'halueval', 'split': 'test', 'expected_examples': 600,
    'extraction_model': 'gemma4:31b-cloud', 'top_k': 5,
    'rules': ['argmax', 'fixed_threshold'],
    'entail_threshold': .7, 'contra_threshold': .95,
    'threshold_source': 'existing Stage 3 configuration; frozen before this evaluation',
    'answer_aggregation': 'supported iff nonempty and every claim is supported',
    'provided_context': 'all supplied knowledge joined; evidence upper bound, no retrieval',
    'retrieved_context': 'FEVER Wikipedia corpus, BM25/dense/hybrid top-5 per claim',
    'training_performed': False, 'threshold_tuning_performed': False,
    'gold_label_unit': 'answer only; no claim-level accuracy is computed',
}


def answer_prediction(probabilities, rule):
    if not len(probabilities):
        return BIN_HALLUCINATED
    if rule == 'argmax':
        supported = all(int(np.argmax(p)) == 0 for p in probabilities)
    elif rule == 'fixed_threshold':
        supported = all(decide(p, PROTOCOL['entail_threshold'], PROTOCOL['contra_threshold']) == 'SUPPORTED'
                        for p in probabilities)
    else:
        raise ValueError(f'Unknown decision rule: {rule}')
    return BIN_SUPPORTED if supported else BIN_HALLUCINATED


def cached_revision(model_name):
    home = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))
    hub = Path(os.environ.get('HF_HUB_CACHE', str(home/'hub')))
    ref = hub/('models--'+model_name.replace('/','--'))/'refs/main'
    return ref.read_text(encoding='utf-8').strip() if ref.exists() else None


class PairCache:
    """Append-only cached pretrained predictions, keyed by input and model config."""
    def __init__(self, model, path):
        self.model, self.path = model, Path(path)
        self.fingerprint = [model.model_name, getattr(model.model.config, '_commit_hash', None),
                            model.max_length, model.use_fp16]
        self.rows = {}
        versions = {}
        for package in ('torch','transformers','numpy'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = 'not-installed'
        provenance = {'versions': versions, 'unresolved_snapshot':
                      cached_revision(model.model_name) if self.fingerprint[1] is None else None}
        manifest = self.path.with_suffix('.metadata.json')
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != provenance:
                raise ValueError('Cache runtime versions differ; preserve this cache and use a separate one')
        elif self.path.exists() and self.path.stat().st_size:
            raise ValueError('Cache lacks runtime provenance; finalize the original run before reuse')
        else:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
        if self.path.exists():
            lines = self.path.read_text(encoding='utf-8').splitlines(keepends=True)
            valid = []
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if i != len(lines)-1 or line.endswith('\n'):
                        raise
                    # Only discard a visibly interrupted final record. Completed
                    # records remain intact; the missing prediction is recomputed.
                    write_jsonl(self.path, valid)
                    break
                probs = row['probs']
                if (len(probs) != 3 or not np.isfinite(probs).all() or
                    min(probs) < 0 or max(probs) > 1 or abs(sum(probs)-1) > 1e-4):
                    raise ValueError('Invalid cached NLI probabilities')
                self.rows[row['key']] = probs
                valid.append(row)
            else:
                if lines and not lines[-1].endswith('\n'):
                    with self.path.open('a', encoding='utf-8') as f:
                        f.write('\n')

    def key(self, premise, hypothesis):
        return hashlib.sha256(json.dumps([self.fingerprint, premise, hypothesis],
                                         ensure_ascii=False).encode()).hexdigest()

    def predict(self, pairs):
        keys = [self.key(p, h) for p, h in pairs]
        missing = dict((k, pair) for k, pair in zip(keys, pairs) if k not in self.rows)
        entries = list(missing.items())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(entries), 128):
            batch = entries[start:start+128]
            probs = self.model.predict([p for _, (p, h) in batch], [h for _, (p, h) in batch],
                                       show_progress=False)
            with self.path.open('a', encoding='utf-8') as out:
                for (k, _), p in zip(batch, probs):
                    self.rows[k] = p.tolist()
                    out.write(json.dumps({'key': k, 'probs': self.rows[k]})+'\n')
            print(f'NLI pairs: {min(start+128,len(entries))}/{len(entries)} new; cached checkpoints saved', flush=True)
        return np.asarray([self.rows[k] for k in keys], dtype=np.float32).reshape((-1, 3))


def provided_rows(examples, cache):
    pairs, spans = [], []
    for e in examples:
        premise = ' '.join(e.context)
        whole = len(pairs)
        pairs.append((premise, f'{e.question or ""} {e.answer}'.strip()))
        start = len(pairs)
        pairs.extend((premise, c) for c in e.claims)
        spans.append((whole, start, len(pairs)))
    probs = cache.predict(pairs)
    rows = []
    for e, (whole, a, b) in zip(examples, spans):
        rows.append({'id': e.id, 'binary_gold': e.binary_label,
                     'extraction_fallback': e.meta['claim_extraction']['is_fallback'],
                     'whole_probabilities': probs[whole].tolist(),
                     'claims': [{'claim': c, 'probabilities': p.tolist()}
                                for c, p in zip(e.claims, probs[a:b])],
                     'context': e.context})
    return rows


def retrieved_rows(examples, cache, retriever):
    claims = [c for e in examples for c in e.claims]
    all_hits = retriever.search_all(claims, k=5) if claims else {m: [] for m in ('bm25','dense','hybrid')}
    print(f'Retrieval ready: {len(claims)} stored claims', flush=True)
    pairs = []
    for ci, claim in enumerate(claims):
        for method in ('bm25','dense','hybrid'):
            pairs.extend((retriever.corpus.texts[d], claim) for d, score in all_hits[method][ci][:5])
    cache.predict(pairs)  # De-duplicated by PairCache across all three retrievers.
    results = {}
    for method in ('bm25','dense','hybrid'):
        rows, ci = [], 0
        for e in examples:
            out_claims = []
            for claim in e.claims:
                hits = all_hits[method][ci][:5]
                ev = [{'text': retriever.corpus.texts[d], 'page_id': retriever.corpus.keys[d][0],
                       'sent_id': retriever.corpus.keys[d][1], 'score': float(score),
                       'probabilities': cache.rows[cache.key(retriever.corpus.texts[d], claim)]}
                      for d, score in hits]
                p = aggregate_probabilities([x['probabilities'] for x in ev])
                out_claims.append({'claim': claim, 'probabilities': p.tolist(), 'evidence': ev})
                ci += 1
            rows.append({'id': e.id, 'binary_gold': e.binary_label,
                         'extraction_fallback': e.meta['claim_extraction']['is_fallback'], 'claims': out_claims})
        results[method] = rows
    return results


def score_rows(rows, whole=False):
    scores = {}
    y = [r['binary_gold'] for r in rows]
    for rule in PROTOCOL['rules']:
        pred = [answer_prediction([r['whole_probabilities']] if whole else
                                  [c['probabilities'] for c in r['claims']], rule) for r in rows]
        for row, p in zip(rows, pred):
            row.setdefault('whole_predictions' if whole else 'predictions', {})[rule] = p
        scores[rule] = binary_metrics(y, pred)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence', choices=['provided','retrieved','all'], default='all')
    ap.add_argument('--output', type=Path, default=PROJECT_ROOT/'experiments/stage5_inference')
    ap.add_argument('--cache', type=Path, default=PROJECT_ROOT/'data/cache/halueval_nli_pairs.jsonl')
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != PROTOCOL:
        raise ValueError('Existing evaluation protocol differs; use a separate experiment directory')
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2)+'\n', encoding='utf-8')
    dataset = PROJECT_ROOT / 'data/splits/halueval_test.jsonl'
    exs = load_examples(dataset)
    if len(exs) != PROTOCOL['expected_examples'] or any(
        e.meta.get('claim_extraction', {}).get('model') != PROTOCOL['extraction_model'] for e in exs):
        raise ValueError('Expected 600 fully processed Gemma test examples')
    from src.verification.nli import NLIModel
    import torch
    torch.set_num_threads(1)
    nli = NLIModel()
    cache = PairCache(nli, args.cache)
    metrics_path = out / 'metrics.json'
    metrics = {'protocol': PROTOCOL, 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
               'nli_fingerprint': cache.fingerprint, 'fallback_examples': sum(e.meta['claim_extraction']['is_fallback'] for e in exs),
               'empty_claim_examples': sum(not e.claims for e in exs),
               'versions': {p: importlib.metadata.version(p) for p in ('torch','transformers','numpy')}, 'results': {}}
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding='utf-8'))
        if any(previous.get(k) != metrics.get(k) for k in ('protocol','dataset_sha256','nli_fingerprint','versions')):
            raise ValueError('Stored metrics have different inputs/model; preserve them as a separate run')
        metrics['results'] = previous['results']
    if args.evidence in ('provided','all'):
        rows = provided_rows(exs, cache)
        metrics['results']['provided_whole_answer'] = score_rows(rows, whole=True)
        metrics['results']['provided_extracted_claims'] = score_rows(rows)
        write_jsonl(out / 'predictions_provided.jsonl', rows)
        metrics['timestamp'] = datetime.now(timezone.utc).isoformat()
        metrics_path.write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
        print('Provided-context evaluation saved', flush=True)
    if args.evidence in ('retrieved','all'):
        from src.retrieval.hybrid import Retriever
        retriever = Retriever()
        for method, rows in retrieved_rows(exs, cache, retriever).items():
            metrics['results'][f'{method}_extracted_claims'] = score_rows(rows)
            write_jsonl(out / f'predictions_{method}.jsonl', rows)
    metrics['timestamp'] = datetime.now(timezone.utc).isoformat()
    metrics_path.write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
