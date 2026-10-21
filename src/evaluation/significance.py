"""Post-hoc paired significance tests: learned combiner vs the frozen rule.

This script does not fit anything, change a threshold, or re-evaluate test with a
new system. It only quantifies the difference that was already reported, using the
committed per-claim predictions and a deterministic replay of the fixed rule.

Two tests per retrieval method:
  * McNemar on correctness (chi-square with continuity correction + exact binomial)
  * paired bootstrap of the macro-F1 difference (identical resampled claims)

    python -m src.evaluation.significance
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import BIN_HALLUCINATED, BIN_SUPPORTED, LABELS, to_binary
from src.evaluation.metrics import multiclass_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide

METHODS = ('bm25', 'dense', 'hybrid')
OUT = PROJECT_ROOT / 'experiments/stage5_learned'
RETRIEVAL_CACHE = PROJECT_ROOT / 'data/processed/retrieval/fever_test.jsonl'
STAGE3 = PROJECT_ROOT / 'experiments/stage3_retrieval'
N_BOOT = 10_000
SEED = 42


def _encode(labels: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABELS)}
    return np.array([index[label] for label in labels], dtype=np.int64)


def _encode_binary(labels: list[str]) -> np.ndarray:
    index = {BIN_SUPPORTED: 0, BIN_HALLUCINATED: 1}
    return np.array([index[label] for label in labels], dtype=np.int64)


def _confusion(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y * n_classes + p, minlength=n_classes * n_classes
                       ).reshape(n_classes, n_classes)


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = _confusion(y, p, len(LABELS)).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)
    return float(f1.mean())


def _binary_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = _confusion(y, p, 2).astype(np.float64)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """a = candidate, b = reference. Positive b-count favours the candidate."""
    from scipy.stats import binomtest, chi2

    favouring = int((correct_a & ~correct_b).sum())
    against = int((~correct_a & correct_b).sum())
    discordant = favouring + against
    if discordant == 0:
        statistic, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        statistic = (abs(favouring - against) - 1) ** 2 / discordant
        p_chi2 = float(chi2.sf(statistic, 1))
        p_exact = float(binomtest(min(favouring, against), discordant, 0.5).pvalue)
    return {'favouring_candidate': favouring, 'favouring_reference': against,
            'discordant': discordant, 'chi2_continuity': float(statistic),
            'p_chi2': p_chi2, 'p_exact_binomial': p_exact}


def paired_delta(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
                 metric, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    observed = metric(y, candidate) - metric(y, reference)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.integers(0, n, n)
        deltas[i] = metric(y[sample], candidate[sample]) - metric(y[sample], reference[sample])
    below = float((deltas <= 0).mean())
    above = float((deltas >= 0).mean())
    p_value = min(1.0, max(2 * min(below, above), 1.0 / (n_boot + 1)))
    return {'observed': float(observed), 'mean': float(deltas.mean()),
            'lower_95': float(np.percentile(deltas, 2.5)),
            'upper_95': float(np.percentile(deltas, 97.5)),
            'p_two_sided': p_value, 'n_resamples': n_boot, 'seed': seed}


def load_fixed_and_learned(method: str, entail: float, contra: float, cache: dict):
    learned = [json.loads(line) for line in
               (OUT / f'predictions_{method}.jsonl').read_text(encoding='utf-8').splitlines()]
    fixed = []
    for row in learned:
        hits = cache[row['id']][method][:5]
        fixed.append(decide(aggregate_probabilities([h['nli'] for h in hits]), entail, contra))
    return learned, fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'significance.json')
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    args = parser.parse_args()

    cfg = load_config()
    entail = cfg['detector']['entail_threshold']
    contra = cfg['detector']['contra_threshold']
    cache = {row['id']: row for row in
             (json.loads(line) for line in RETRIEVAL_CACHE.read_text(encoding='utf-8').splitlines())}
    committed_path = Path(sorted(glob.glob(str(STAGE3 / 'metrics_*.json')))[-1])
    committed = json.loads(committed_path.read_text(encoding='utf-8'))
    protocol_path = OUT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else {}

    results = {}
    for method in METHODS:
        learned_rows, fixed_labels = load_fixed_and_learned(method, entail, contra, cache)
        gold = [row['gold'] for row in learned_rows]
        learned_labels = [row['prediction'] for row in learned_rows]
        y = _encode(gold)
        candidate = _encode(learned_labels)
        reference = _encode(fixed_labels)

        # Guard: the replayed fixed rule must reproduce the committed Stage 3 replay.
        expected = committed['pipeline'][f'{method}/thresh']['3way']
        replay = multiclass_metrics(gold, fixed_labels)
        if abs(replay['accuracy'] - expected['accuracy']) > 1e-9 or \
                abs(replay['macro_f1'] - expected['macro_f1']) > 1e-9:
            raise SystemExit(f'{method}: fixed-rule replay does not match the committed metrics')

        candidate_binary = _encode_binary([to_binary(label) for label in learned_labels])
        reference_binary = _encode_binary([to_binary(label) for label in fixed_labels])
        gold_binary = _encode_binary([to_binary(label) for label in gold])

        learned_metrics = multiclass_metrics(gold, learned_labels)
        delta_macro = paired_delta(y, candidate, reference, _macro_f1, args.n_boot)
        delta_binary = paired_delta(gold_binary, candidate_binary, reference_binary,
                                    _binary_f1, args.n_boot)
        per_class = {label: round(learned_metrics['per_class'][label]['f1'] -
                                  replay['per_class'][label]['f1'], 6)
                     for label in LABELS}
        results[method] = {
            'n': len(y),
            'fixed_rule': {'accuracy': replay['accuracy'], 'macro_f1': replay['macro_f1'],
                           'binary_f1': float(_binary_f1(gold_binary, reference_binary))},
            'learned': {'accuracy': learned_metrics['accuracy'],
                        'macro_f1': learned_metrics['macro_f1'],
                        'binary_f1': float(_binary_f1(gold_binary, candidate_binary))},
            'changed_predictions': int((candidate != reference).sum()),
            'per_class_f1_delta': per_class,
            'mcnemar_3way': mcnemar(candidate == y, reference == y),
            'paired_macro_f1_delta': delta_macro,
            'mcnemar_binary': mcnemar(candidate_binary == gold_binary,
                                      reference_binary == gold_binary),
            'paired_binary_f1_delta': delta_binary,
        }
        print(f'[{method}] delta macro-F1 {delta_macro["observed"]:+.4f} '
              f'CI [{delta_macro["lower_95"]:+.4f}, {delta_macro["upper_95"]:+.4f}] '
              f'p={delta_macro["p_two_sided"]:.2g}; McNemar p={results[method]["mcnemar_3way"]["p_chi2"]:.2g}',
              flush=True)

    report = {
        'analysis': 'post_hoc_paired_significance',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'refit': False, 'thresholds_changed': False, 'new_system_evaluated': False,
        'source_predictions': 'experiments/stage5_learned/predictions_*.jsonl',
        'fixed_rule_replay': {'entail': entail, 'contra': contra,
                              'reproduced': str(committed_path.relative_to(PROJECT_ROOT))},
        'stage5_protocol_test_evaluated_utc': protocol.get('test_evaluated_utc'),
        'mcnemar': 'chi-square with continuity correction; exact binomial reported too',
        'bootstrap': 'paired resampling of claims, candidate and reference share indices',
        'methods': results,
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Saved -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
