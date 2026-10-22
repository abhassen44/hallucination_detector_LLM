"""Fixed balanced validation pilot for the optional grounded second verifier.

Only pretrained cloud inference is used. No training, fitting, threshold search,
or test access. Checkpoints and the frozen protocol allow exact-input resumption.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import time

from src.datasets.schema import LABELS, read_jsonl
from src.evaluation.metrics import multiclass_metrics
from src.evaluation.reliability import reliability_metrics
from src.generation.llm import ProviderUnavailableError
from src.retrieval.corpus import SentenceCorpus
from src.utils.config import PROJECT_ROOT
from src.verification.grounded import (GroundedVerifier, VERSION, PROMPT_SHA256,
                                      apply_review, needs_review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 999 or args.limit % 3:
        raise SystemExit('--limit must be a positive multiple of three, at most 999')
    if not 1 <= args.workers <= 4:
        raise SystemExit('--workers must be between one and four')
    root = PROJECT_ROOT
    source = root / 'experiments/improvements/validation_predictions.jsonl'
    cache_path = root / 'data/processed/retrieval/fever_val.jsonl'
    corpus_path = root / 'data/processed/fever_corpus.jsonl'
    baseline = [r for r in read_jsonl(source)
                if r['method'] == 'hybrid' and r['evidence_policy'] == 'top1']
    assert len(baseline) == 999 and len({r['id'] for r in baseline}) == 999
    rng = random.Random(20261022)
    sample = []
    for label in LABELS:
        bucket = sorted([r for r in baseline if r['gold'] == label], key=lambda r: r['id'])
        rng.shuffle(bucket)
        sample.extend(bucket[:args.limit // 3])
    rng.shuffle(sample)
    verifier = GroundedVerifier()
    out = root / f'experiments/grounded_pilot_{args.limit}'
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'version': VERSION, 'prompt_sha256': PROMPT_SHA256, 'model': verifier.llm.model,
        'split': 'val', 'seed': 20261022, 'n': args.limit,
        'sample': 'balanced random claims, not selected by model correctness',
        'ids': [r['id'] for r in sample],
        'trigger': 'baseline CONTRADICTED or conflicting NLI evidence',
        'policy': 'confirm same factual verdict or defer to INSUFFICIENT; never flip verdict',
        'training_performed': False, 'test_evaluated': False,
        'cloud_revision': 'provider alias; immutable weights revision unavailable',
        'input_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (source, cache_path, corpus_path)},
    }
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol:
            raise SystemExit('Protocol changed; preserve these results and use a new experiment version.')
    else:
        path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    cached = {r['id']: r for r in read_jsonl(cache_path)}
    corpus = SentenceCorpus.from_fever(corpus_path)
    checkpoint = out / 'predictions.jsonl'
    completed = list(read_jsonl(checkpoint)) if checkpoint.exists() else []
    if len({r['id'] for r in completed}) != len(completed) or not {r['id'] for r in completed} <= set(protocol['ids']):
        raise SystemExit('Invalid checkpoint IDs')
    done = {r['id']: r for r in completed}

    def task(row):
        evidence = [dict(h, text=corpus.texts[h['doc']]) for h in cached[row['id']]['hybrid'][:5]]
        original = {'claim': cached[row['id']]['claim'], 'label': row['prediction'],
                    'decision': row['decision'], 'evidence': evidence,
                    'evidence_conflict': 'conflicting_evidence' in row['decision']['reasons']}
        start = time.monotonic()
        if needs_review(original):
            review = verifier.review(original['claim'], evidence)
            if review['status'] == 'unavailable':
                raise ProviderUnavailableError('Second verifier unavailable; resume this checkpoint later.')
            revised = apply_review(original, review)
        else:
            revised = dict(original, grounded_review={'status': 'not_required'})
        return {'id': row['id'], 'gold': row['gold'], 'claim': original['claim'],
                'baseline_label': original['label'], 'baseline_decision': original['decision'],
                'prediction': revised['label'], 'decision': revised['decision'],
                'review': revised['grounded_review'], 'seconds': time.monotonic() - start}

    with checkpoint.open('a', encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(task, r) for r in sample if r['id'] not in done]
        try:
            for future in as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                done[row['id']] = row
                print(f'{len(done)}/{args.limit} saved; review={row["review"]["status"]}', flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    rows = [done[id_] for id_ in protocol['ids']]
    gold = [r['gold'] for r in rows]
    report = {'protocol': protocol, 'promotion': False, 'n': len(rows),
              'review_status': dict(Counter(r['review']['status'] for r in rows)),
              'baseline': {}, 'candidate': {}}
    for system, label, decision in [('baseline', 'baseline_label', 'baseline_decision'),
                                     ('candidate', 'prediction', 'decision')]:
        report[system] = {
            'classification': multiclass_metrics(gold, [r[label] for r in rows]),
            'reliability': reliability_metrics(gold, [r[label] for r in rows],
                                               [r[decision]['status'] == 'answer' for r in rows]),
        }
    report['paired_changes'] = {
        'gained': sum(r['prediction'] == r['gold'] and r['baseline_label'] != r['gold'] for r in rows),
        'lost': sum(r['prediction'] != r['gold'] and r['baseline_label'] == r['gold'] for r in rows),
    }
    report['review_latency_seconds'] = {
        'mean': sum(r['seconds'] for r in rows if r['review']['status'] != 'not_required') /
                max(1, sum(r['review']['status'] != 'not_required' for r in rows)),
        'note': 'includes API/cache time, excludes unchanged rows; not serving P95',
    }
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
