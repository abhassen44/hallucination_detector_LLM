"""v5 P0.5: validation error analysis and retrieval/answerability decomposition.

Reads only the frozen FEVER validation split, the v4 validation reranker cache,
and the v4 validation policy predictions. Never reads test data.

    python -m scripts.v5.error_analysis_v5
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.utils.config import PROJECT_ROOT

VAL_SPLIT = PROJECT_ROOT / 'data/splits/fever_val.jsonl'
VAL_RERANK = PROJECT_ROOT / 'data/processed/rerank/val_hybrid_top8.jsonl'
VAL_POLICY = PROJECT_ROOT / 'experiments/v4/val_policy_9000.jsonl'
V4_TUNING = PROJECT_ROOT / 'experiments/v4/answerability_tuning_9000.json'
OUT = PROJECT_ROOT / 'experiments/v5'

LABELS = ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT']
TOP_K = 5


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_metrics(gold: list[str], predicted: list[str]) -> dict:
    """Macro-F1, accuracy, per-class metrics, and confusion matrix."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, zero_division=0)
    return {
        'n': len(gold),
        'accuracy': float(accuracy_score(gold, predicted)),
        'macro_f1': float(f1_score(gold, predicted, labels=LABELS, average='macro',
                                   zero_division=0)),
        'per_class': {
            label: {
                'precision': float(precision[index]),
                'recall': float(recall[index]),
                'f1': float(f1[index]),
                'support': int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        'confusion_matrix': confusion_matrix(gold, predicted, labels=LABELS).tolist(),
    }


def _gold_sets(row: dict) -> list[set[tuple[str, int]]]:
    sets = []
    for evidence_set in row.get('meta', {}).get('evidence_sets', []) or []:
        pairs = {(str(page), int(sent)) for page, sent in evidence_set}
        if pairs:
            sets.append(pairs)
    return sets


def _retrieval_stats(row: dict, hits: list[dict], k: int = TOP_K) -> dict:
    retrieved = [(str(hit['page_id']), int(hit['sent_id'])) for hit in hits[:k]]
    positions = {pair: index + 1 for index, pair in enumerate(retrieved)}
    gold_sets = _gold_sets(row)
    if not gold_sets:
        return {
            'status': 'not_applicable',
            'n_gold_sets': 0,
            'strict_hit': None,
            'hit_any': None,
            'page_hit': None,
            'complete_rank': None,
            'first_gold_rank': None,
            'retrieved_pairs': retrieved,
        }

    complete_ranks = []
    for gold_set in gold_sets:
        if all(pair in positions for pair in gold_set):
            complete_ranks.append(max(positions[pair] for pair in gold_set))
    all_gold_pairs = set().union(*gold_sets)
    first_gold_rank = min(
        (positions[pair] for pair in all_gold_pairs if pair in positions),
        default=None,
    )
    retrieved_pages = {page for page, _ in retrieved}
    gold_pages = {page for page, _ in all_gold_pairs}
    return {
        'status': 'verifiable',
        'n_gold_sets': len(gold_sets),
        'strict_hit': bool(complete_ranks),
        'hit_any': first_gold_rank is not None,
        'page_hit': bool(retrieved_pages & gold_pages),
        'complete_rank': min(complete_ranks) if complete_ranks else None,
        'first_gold_rank': first_gold_rank,
        'retrieved_pairs': retrieved,
    }


def _bucket(gold: str, final_prediction: str, argmax: str, answered: bool,
            p_answerable: float, tau: float | None, retrieval: dict) -> str:
    if gold == final_prediction:
        return 'correct'
    if gold == 'INSUFFICIENT':
        return 'false_answer_nei'
    if final_prediction == 'INSUFFICIENT':
        if retrieval['strict_hit'] is False:
            return 'retrieval_miss_abstention'
        if tau is not None and p_answerable < tau:
            return 'answerability_abstention'
        if argmax == 'INSUFFICIENT':
            return 'verifier_abstention'
        return 'class_threshold_abstention'
    if retrieval['strict_hit'] is False:
        return 'retrieval_miss_verifier'
    return 'verifier_error'


def _oracle_metrics(rows: list[dict]) -> dict:
    gold = [row['gold'] for row in rows]
    current = [row['prediction'] for row in rows]
    oracle_answerability = [
        row['prediction'] if row['gold'] != 'INSUFFICIENT' else 'INSUFFICIENT'
        for row in rows
    ]
    oracle_verdict = [
        row['gold'] if row['gold'] != 'INSUFFICIENT' else row['prediction']
        for row in rows
    ]
    return {
        'current': _classification_metrics(gold, current),
        'oracle_answerability': _classification_metrics(gold, oracle_answerability),
        'oracle_verdict': _classification_metrics(gold, oracle_verdict),
        'oracle_both': _classification_metrics(gold, gold),
    }


def _summary_markdown(report: dict) -> str:
    metrics = report['metrics']
    retrieval = report['retrieval']
    buckets = report['buckets']
    lines = [
        '# v5 P0.5 — validation error analysis',
        '',
        f"Created: {report['created_utc']}",
        f"Split: {report['split']} ({report['n']} claims); test_evaluated: false",
        '',
        'Pre-P1 snapshot. The final v5 validation/test report is `VALIDATION.md`.',
        '',
        '## Headline',
        '',
        f"- Current val macro-F1: **{metrics['current']['macro_f1']:.5f}**",
        f"- Current val accuracy: **{metrics['current']['accuracy']:.5f}**",
        f"- Verifiable strict recall@{TOP_K}: **{retrieval['strict_recall']:.4f}**",
        f"- Verifiable hit recall@{TOP_K}: **{retrieval['hit_recall']:.4f}**",
        f"- Verifiable page recall@{TOP_K}: **{retrieval['page_recall']:.4f}**",
        '',
        '## Error buckets',
        '',
        '| Bucket | Count | Share of errors |',
        '|---|---:|---:|',
    ]
    total_errors = sum(count for name, count in buckets.items() if name != 'correct')
    for name, count in sorted(
        ((name, count) for name, count in buckets.items() if name != 'correct'),
        key=lambda item: -item[1],
    ):
        share = count / total_errors if total_errors else 0.0
        lines.append(f'| {name} | {count} | {share:.1%} |')
    lines.extend([
        '',
        '## Oracle ceilings (val only)',
        '',
        '| Variant | Macro-F1 | Accuracy |',
        '|---|---:|---:|',
    ])
    for name in ('current', 'oracle_answerability', 'oracle_verdict', 'oracle_both'):
        result = metrics[name]
        lines.append(f"| {name} | {result['macro_f1']:.5f} | {result['accuracy']:.5f} |")
    lines.extend([
        '',
        '## Per-class (current policy)',
        '',
        '| Class | Precision | Recall | F1 | Support |',
        '|---|---:|---:|---:|---:|',
    ])
    for label in LABELS:
        result = metrics['current']['per_class'][label]
        lines.append(
            f"| {label} | {result['precision']:.4f} | {result['recall']:.4f} | "
            f"{result['f1']:.4f} | {result['support']} |"
        )
    lines.extend([
        '',
        '## Reading',
        '',
        report['reading'],
        '',
        '## Artifacts',
        '',
        '- `experiments/v5/error_analysis.json`',
        '- `experiments/v5/error_analysis.md`',
        '',
        'No test data was read. No model was fitted or tuned.',
        '',
    ])
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()

    split_rows = _load_jsonl(VAL_SPLIT)
    rerank_rows = {row['id']: row for row in _load_jsonl(VAL_RERANK)}
    policy_rows = {row['id']: row for row in _load_jsonl(VAL_POLICY)}
    tuning = _load_json(V4_TUNING)

    missing = sorted(set(row['id'] for row in split_rows) - set(rerank_rows))
    missing_policy = sorted(set(row['id'] for row in split_rows) - set(policy_rows))
    if missing or missing_policy:
        raise SystemExit(f'Missing val artifacts: rerank={missing[:5]}, policy={missing_policy[:5]}')

    selected_policy = tuning.get('selected', {}).get('policy', {}).get('best') or {}
    tau = selected_policy.get('tau')

    rows = []
    for split_row in split_rows:
        item = policy_rows[split_row['id']]
        hits = rerank_rows[split_row['id']]['hits']
        probabilities = np.array([item['calibrated_probabilities'][label] for label in LABELS],
                                 dtype=float)
        argmax = LABELS[int(probabilities.argmax())]
        final_prediction = item['prediction']
        answered = bool(item['answered'])
        p_answerable = float(item['p_answerable'])
        retrieval = _retrieval_stats(split_row, hits)
        row = {
            'id': split_row['id'],
            'claim': split_row.get('claim') or (split_row.get('claims') or [''])[0],
            'gold': split_row['label'],
            'prediction': final_prediction,
            'argmax': argmax,
            'answered': answered,
            'p_answerable': p_answerable,
            'calibrated_probabilities': item['calibrated_probabilities'],
            'retrieval': {key: value for key, value in retrieval.items()
                          if key != 'retrieved_pairs'},
            'bucket': _bucket(split_row['label'], final_prediction, argmax, answered,
                              p_answerable, tau, retrieval),
        }
        rows.append(row)

    gold = [row['gold'] for row in rows]
    predicted = [row['prediction'] for row in rows]
    metrics = _oracle_metrics(rows)

    verifiable = [row for row in rows if row['gold'] != 'INSUFFICIENT']
    retrieval_stats = {
        'n_verifiable': len(verifiable),
        'strict_recall': float(np.mean([row['retrieval']['strict_hit'] for row in verifiable])),
        'hit_recall': float(np.mean([row['retrieval']['hit_any'] for row in verifiable])),
        'page_recall': float(np.mean([row['retrieval']['page_hit'] for row in verifiable])),
        'complete_rank_mean': float(np.mean([
            row['retrieval']['complete_rank'] for row in verifiable
            if row['retrieval']['complete_rank'] is not None
        ])),
        'strict_misses': int(sum(not row['retrieval']['strict_hit'] for row in verifiable)),
    }

    bucket_counts = Counter(row['bucket'] for row in rows)
    bucket_by_gold = defaultdict(Counter)
    for row in rows:
        bucket_by_gold[row['gold']][row['bucket']] += 1

    examples = defaultdict(list)
    for row in rows:
        if row['bucket'] == 'correct':
            continue
        if len(examples[row['bucket']]) >= 8:
            continue
        examples[row['bucket']].append({
            key: row[key]
            for key in ('id', 'claim', 'gold', 'prediction', 'argmax', 'answered',
                        'p_answerable', 'bucket')
        } | {'retrieval': row['retrieval']})

    total_errors = sum(count for name, count in bucket_counts.items() if name != 'correct')
    retrieval_attributable = (
        bucket_counts.get('retrieval_miss_abstention', 0)
        + bucket_counts.get('retrieval_miss_verifier', 0)
    )
    answerability_attributable = (
        bucket_counts.get('answerability_abstention', 0)
        + bucket_counts.get('class_threshold_abstention', 0)
        + bucket_counts.get('verifier_abstention', 0)
    )
    nei_false_answers = bucket_counts.get('false_answer_nei', 0)

    reading = (
        f"Of {total_errors} val errors, {retrieval_attributable} are attributable to "
        f"a strict retrieval miss; {answerability_attributable} are abstentions with "
        f"retrievable evidence; and {nei_false_answers} are false answers on true "
        "INSUFFICIENT claims. The oracle-answerability ceiling shows how much of the "
        "gap is selective policy versus verdict quality. Use these buckets to choose "
        "whether P1 should prioritize the cross-encoder/grounding signal or P3 "
        "retrieval."
    )

    report = {
        'version': 'v5-p0.5-error-analysis',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val',
        'n': len(rows),
        'test_evaluated': False,
        'policy_tau': tau,
        'metrics': metrics,
        'retrieval': retrieval_stats,
        'buckets': dict(bucket_counts),
        'bucket_by_gold': {label: dict(counter) for label, counter in bucket_by_gold.items()},
        'examples': dict(examples),
        'inputs': {
            'val_split': str(VAL_SPLIT.relative_to(PROJECT_ROOT)),
            'val_rerank': str(VAL_RERANK.relative_to(PROJECT_ROOT)),
            'val_policy': str(VAL_POLICY.relative_to(PROJECT_ROOT)),
            'v4_tuning': str(V4_TUNING.relative_to(PROJECT_ROOT)),
            'sha256': {
                'val_split': _sha256(VAL_SPLIT),
                'val_rerank': _sha256(VAL_RERANK),
                'val_policy': _sha256(VAL_POLICY),
                'v4_tuning': _sha256(V4_TUNING),
            },
        },
        'reading': reading,
        'notes': [
            'Val only; no test data was read.',
            'No model was fitted, no threshold was selected, and no retrieval was rerun.',
            'Strict recall requires a complete gold evidence set in the top-5.',
            'Oracle-answerability keeps the current verdict but uses gold abstention; '
            'oracle-verdict keeps the current abstention but uses gold verdicts.',
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / 'error_analysis.json'
    markdown_path = args.out_dir / 'error_analysis.md'
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    markdown_path.write_text(_summary_markdown(report), encoding='utf-8')

    print(json.dumps({
        'n': len(rows),
        'macro_f1': metrics['current']['macro_f1'],
        'accuracy': metrics['current']['accuracy'],
        'strict_recall': retrieval_stats['strict_recall'],
        'hit_recall': retrieval_stats['hit_recall'],
        'buckets': dict(bucket_counts),
        'oracle_answerability_macro_f1': metrics['oracle_answerability']['macro_f1'],
        'oracle_verdict_macro_f1': metrics['oracle_verdict']['macro_f1'],
    }, indent=2))
    print(f'Saved -> {json_path.relative_to(PROJECT_ROOT)}')
    print(f'Saved -> {markdown_path.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
