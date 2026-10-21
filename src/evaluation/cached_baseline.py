"""Replay fixed Stage 3 decisions without training, model downloads or LLM calls.

Creates a deterministic, class-balanced failure review sample. Review categories
are deliberately left empty for human annotation, rather than guessed.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.datasets.schema import LABELS, read_jsonl
from src.retrieval.corpus import SentenceCorpus
from src.utils.config import PROJECT_ROOT
from src.verification.detector import aggregate_probabilities, decide

CATEGORIES = ['retrieval_failure', 'claim_extraction_failure', 'nli_failure',
              'ambiguous_evidence', 'numerical_reasoning', 'temporal_information',
              'entity_confusion', 'multi_hop_reasoning']


def evaluate(method='hybrid', output=None):
    cache = PROJECT_ROOT / 'data/processed/retrieval/fever_test.jsonl'
    examples = {r['id']: r for r in read_jsonl(PROJECT_ROOT / 'data/splits/fever_test.jsonl')}
    confusion = {a: Counter() for a in LABELS}
    failures = []
    for row in read_jsonl(cache):
        if row['id'] not in examples or row['label'] != examples[row['id']]['label']:
            raise ValueError('Retrieval cache does not match the test split')
        hits = row[method][:5]
        if any('nli' not in hit for hit in hits):
            raise ValueError('Cache is missing top-5 NLI signals')
        prediction = decide(aggregate_probabilities([h['nli'] for h in hits]))
        confusion[row['label']][prediction] += 1
        if prediction != row['label']:
            failures.append({'id': row['id'], 'claim': row['claim'], 'gold': row['label'],
                             'prediction': prediction, 'evidence': hits,
                             'gold_context': examples[row['id']]['context'],
                             'category': None, 'review_notes': ''})
    n = sum(sum(c.values()) for c in confusion.values())
    if n != len(examples):
        raise ValueError('Incomplete retrieval cache')
    f1 = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    result = {'method': method, 'mode': 'fixed_threshold_cache_replay', 'n': n,
              'accuracy': sum(confusion[l][l] for l in LABELS)/n, 'macro_f1': sum(f1)/len(f1),
              'labels': list(LABELS), 'confusion_matrix': [[confusion[a][b] for b in LABELS] for a in LABELS],
              'thresholds': {'entail': .7, 'contra': .95}, 'training_performed': False}
    rng = random.Random(42)
    sample = []
    for label in LABELS:
        candidates = [f for f in failures if f['gold'] == label]
        sample.extend(rng.sample(candidates, min(20, len(candidates))))
    corpus = SentenceCorpus.from_fever()
    for failure in sample:
        for hit in failure['evidence']:
            doc = hit['doc']
            if corpus.keys[doc] != (hit['page_id'], hit['sent_id']):
                raise ValueError('Retrieval cache and corpus ordering differ')
            hit['text'] = corpus.texts[doc]
    out = Path(output) if output else PROJECT_ROOT / 'experiments/baseline_review'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{method}_metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (out / f'{method}_review.json').write_text(
        json.dumps({'categories': CATEGORIES, 'examples': sample}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['bm25', 'dense', 'hybrid'], default='hybrid')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    evaluate(args.method, args.output)
