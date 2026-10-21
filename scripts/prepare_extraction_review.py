"""Prepare a local human-review queue; does not assign human quality labels."""
import json
import random
from collections import Counter
from dataclasses import asdict

from src.claims.extractor import assess_claims
from src.datasets.schema import load_examples, write_jsonl
from src.generation.claim_extraction import ClaimExtractor, EXTRACTION_SYSTEM_PROMPT
from src.generation.llm import OllamaLLM
from src.utils.config import PROJECT_ROOT, load_config


def main():
    cfg = load_config().get('claims', {})
    llm, rng = OllamaLLM(), random.Random(42)
    extractor = ClaimExtractor(llm)
    flagged, clean = [], []
    counts = Counter()
    for split in ('train','val','test'):
        for e in load_examples(PROJECT_ROOT / f'data/splits/halueval_{split}.jsonl'):
            quality = assess_claims(e.id, e.answer, e.claims,
                is_fallback=e.meta['claim_extraction']['is_fallback'],
                short_threshold=cfg.get('short_claim_threshold',5),
                long_threshold=cfg.get('long_claim_threshold',40),
                dup_threshold=cfg.get('duplicate_jaccard',.85))
            flags = [name for name, active in {
                'fallback': quality.is_fallback, 'empty': not e.claims,
                'short': bool(quality.short_claims), 'long': bool(quality.long_claims),
                'duplicates': bool(quality.duplicate_pairs)}.items() if active]
            counts.update(flags)
            row = {'id': e.id, 'split': split, 'answer_gold_label': e.binary_label,
                   'question': e.question, 'answer': e.answer, 'claims': e.claims,
                   'model': e.meta['claim_extraction']['model'], 'heuristic_flags': flags,
                   'quality_details': asdict(quality), 'selection': 'flagged' if flags else 'comparison_sample',
                   'human_review': {'open_coding': '', 'pass_fail_assessment': None,
                                    'fidelity': None, 'atomicity': None, 'coverage': None,
                                    'self_contained': None, 'category': None}}
            if quality.is_fallback:
                row['raw_model_response'] = llm.cache.get(llm._key(
                    extractor._build_prompt(e.answer,e.question), EXTRACTION_SYSTEM_PROMPT,None))
            (flagged if flags else clean).append(row)
    sample = []
    for split in ('train','val','test'):
        for label in ('supported','hallucinated'):
            candidates = [r for r in clean if r['split']==split and r['answer_gold_label']==label]
            sample.extend(rng.sample(candidates,min(8,len(candidates))))
    selected = {r['id'] for r in sample}
    sample.extend(rng.sample([r for r in clean if r['id'] not in selected], 50-len(sample)))
    flagged.sort(key=lambda r: ('fallback' not in r['heuristic_flags'],r['split'],r['id']))
    out = PROJECT_ROOT / 'experiments/stage4_claims'
    queue = out/'human_review.jsonl'
    if queue.exists():
        for line in queue.read_text(encoding='utf-8').splitlines():
            review = json.loads(line).get('human_review', {})
            if any(v is not None and v != '' for v in review.values()):
                raise ValueError('Refusing to overwrite completed human annotations; save a separate queue')
    write_jsonl(queue,flagged+sample)
    summary={'flagged_examples':len(flagged),'comparison_examples':len(sample),
             'queue_size':len(flagged)+len(sample),'flag_counts_overlap':dict(counts),
             'human_annotations_completed':0,'training_performed':False}
    (out/'review_summary.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
