"""Render the frozen-rule results and deterministic error-review sample."""
import json
import random
from collections import Counter

from src.datasets.schema import load_examples, read_jsonl, write_jsonl
from src.utils.config import PROJECT_ROOT


def main():
    out=PROJECT_ROOT/'experiments/stage5_inference'
    report=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    lines=['# Frozen-rule HaluEval inference evaluation','',
        'No training, classifier fitting, threshold search, or new LLM calls were performed.',
        'All results use the same 600 test answers and stored Gemma extractions.',
        'Positive class: hallucinated. Scores are answer-level; no claim-level gold labels exist.','',
        '| Evidence / text unit | Decision rule | Accuracy | Precision | Recall | F1 |',
        '|---|---|---:|---:|---:|---:|']
    for condition,rules in report['results'].items():
        for rule,m in rules.items():
            lines.append(f"| {condition} | {rule} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ['', 'Provided-context results use supplied knowledge, not Wikipedia search.',
              'Wikipedia results use the limited FEVER corpus and should be interpreted separately.',
              'Rules and thresholds were frozen before prediction. The existing 0.7/0.95 thresholds',
              'were inherited from FEVER; they were not optimized for HaluEval.',
              'Any result differences are descriptive comparisons, not statistical significance claims.',
              'The two test fallback extractions and two examples without claims are included,',
              'so counts stay at 600 for every method. Predicted probabilities are uncalibrated.', '',
              '## Human error review','',
              'The sample contains up to 20 false positives and 20 false negatives per evidence',
              'condition using the fixed-threshold rule, with original inputs and evidence attached.',
              'Fields for observations, pass/fail and category remain unassigned. This enriched',
              'failure sample does not estimate population error rates.','']
    examples={e.id:e for e in load_examples(PROJECT_ROOT/'data/splits/halueval_test.jsonl')}
    rng=random.Random(42)
    review=[]
    counts={}
    for condition,filename in [('provided_extracted_claims','provided'),
        ('bm25_extracted_claims','bm25'),('dense_extracted_claims','dense'),('hybrid_extracted_claims','hybrid')]:
        path=out/f'predictions_{filename}.jsonl'
        if not path.exists():
            continue
        rows=list(read_jsonl(path))
        errors=[r for r in rows if r['predictions']['fixed_threshold']!=r['binary_gold']]
        counts[condition]=dict(Counter('false_positive' if r['predictions']['fixed_threshold']=='hallucinated'
                                       else 'false_negative' for r in errors))
        for pred in ('hallucinated','supported'):
            candidates=[r for r in errors if r['predictions']['fixed_threshold']==pred]
            for r in rng.sample(candidates,min(20,len(candidates))):
                e=examples[r['id']]
                review.append({'condition':condition,'id':r['id'],'question':e.question,
                    'answer':e.answer,'answer_gold_label':r['binary_gold'],
                    'answer_prediction':r['predictions']['fixed_threshold'],'claims':r['claims'],
                    'provided_context':e.context if filename=='provided' else None,
                    'human_review':{'open_coding':'','pass_fail_assessment':None,'category':None}})
    hybrid = list(read_jsonl(out/'predictions_hybrid.jsonl'))
    flagged = sum(r['predictions']['fixed_threshold']=='hallucinated' for r in hybrid)
    lines += ['## Interpretation', '',
        'An always-hallucinated predictor scores 50% accuracy and F1 0.667 on this balanced set.',
        f'The hybrid fixed rule flags {flagged} of {len(hybrid)} answers. Its F1 near 0.668',
        'is consequently close to this trivial reference, not evidence of useful discrimination.',
        'The experiment demonstrates the importance of evidence quality. It does not by itself',
        'separate corpus coverage, ranking relevance and NLI errors into causal categories.', '']
    queue=out/'error_review.jsonl'
    if queue.exists() and any(any(v is not None and v!='' for v in r.get('human_review',{}).values())
                              for r in read_jsonl(queue)):
        raise ValueError('Refusing to overwrite human annotations')
    write_jsonl(queue,review)
    (out/'error_counts.json').write_text(json.dumps({'counts':counts,'review_examples':len(review),
            'human_categories_assigned':0},indent=2)+'\n', encoding='utf-8')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Saved report and',len(review),'unlabelled review examples')


if __name__ == '__main__':
    main()
