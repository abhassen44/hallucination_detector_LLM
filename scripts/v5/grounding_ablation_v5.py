"""v5 P1a: validation-only ablation of pretrained grounding features.

Joins the v4 30-feature claim matrix with cached MiniCheck/HHEM support scores,
fits both the three-class verdict model and the answerability head, and reports
the frozen v4 gates. No test data is read.

    python -m scripts.v5.grounding_ablation_v5 --train-size 3000
    python -m scripts.v5.grounding_ablation_v5 --train-size 9000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v3.answerability_v3 import (CLASS_THRESHOLDS, answerability_policy,
                                      cross_fit_base_probabilities, meets_constraints,
                                      meets_verifiable)
from src.datasets.schema import LABELS
from src.evaluation.calibration import (brier_score, cross_fit_vector_scaling_brier)
from src.evaluation.fit_detector import C_GRID, fit_model
from src.evaluation.metrics import multiclass_metrics
from src.evaluation.selective import metrics_for_answered
from src.utils.config import PROJECT_ROOT
from src.verification.features import (FEATURE_NAMES, build_matrix, load_split,
                                       similarity_matrix)
from src.verification.grounding_features import (build_grounding_features,
                                                 grounding_path, load_cache,
                                                 variant_columns)

OUT = PROJECT_ROOT / 'experiments' / 'v5'
CACHE = PROJECT_ROOT / 'data' / 'processed' / 'rerank'
FEATURE_CACHE = PROJECT_ROOT / 'data' / 'processed' / 'features'
TOP_K = 5


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _enriched(rows: list[dict], cached: list[dict]) -> list[dict]:
    by_id = {row['id']: row for row in rows}
    return [{**by_id[item['id']], 'rerank': item['hits'][:TOP_K]} for item in cached]


def _rows(train_size: int) -> tuple[list[dict], list[dict]]:
    base_cached = _jsonl(CACHE / 'train_hybrid_top8.jsonl')
    base_rows = load_split('train')
    train_rows = _enriched(base_rows, base_cached)
    if train_size == 9000:
        extra_cached = _jsonl(CACHE / 'train_extra_hybrid_top5_6000.jsonl')
        extra_rows = [{'id': row['id'], 'claim': row['claim'], 'label': row['label']}
                      for row in extra_cached]
        train_rows += _enriched(extra_rows, extra_cached)
    elif train_size != 3000:
        raise ValueError('train_size must be 3000 or 9000')
    val_rows = _enriched(load_split('val'), _jsonl(CACHE / 'val_hybrid_top8.jsonl'))
    return train_rows, val_rows


def _base_matrix(split: str, train_size: int, rows: list[dict], corpus,
                 sim_model) -> np.ndarray:
    path = FEATURE_CACHE / f'v5_base_{split}_{train_size}.npz'
    ids = [row['id'] for row in rows]
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if list(cached['ids']) == ids:
            return cached['matrix']
    if sim_model is None:
        raise SystemExit(f'Missing feature cache {path}; rerun without a warm cache.')
    sims = similarity_matrix(rows, 'rerank', corpus, sim_model, top=TOP_K)
    matrix = build_matrix(rows, 'rerank', sims, top=TOP_K)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), matrix=matrix)
    return matrix


def _policy_search(p_answer: np.ndarray, calibrated: np.ndarray,
                   gold: np.ndarray) -> dict:
    candidates = []
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        answered = answerability_policy(calibrated, p_answer, float(tau))
        candidates.append({'tau': float(tau),
                           **metrics_for_answered(gold, calibrated, answered)})
    feasible = [candidate for candidate in candidates if meets_constraints(candidate)]
    verifiable = [candidate for candidate in candidates if meets_verifiable(candidate)]
    best_coverage = (max(feasible, key=lambda candidate: candidate['coverage'])
                     if feasible else None)
    best_nei = (min(feasible, key=lambda candidate: (
        candidate['nei_coverage'] if candidate['nei_coverage'] is not None else 1.0,
        -candidate['coverage'],
        -(candidate['answered_precision'] or 0.0),
    )) if feasible else None)
    # Pre-test robustness: leave at least one percentage point of headroom on
    # the verifiable-coverage gate and do not lose more than 0.5 points of the
    # v4 overall coverage. Within that pool, minimise NEI false answers.
    robust = [candidate for candidate in feasible
              if (candidate['verifiable_coverage'] or 0.0) >= 0.81
              and (candidate['coverage'] or 0.0) >= 0.65]
    best_robust = (min(robust, key=lambda candidate: (
        candidate['nei_coverage'] if candidate['nei_coverage'] is not None else 1.0,
        -candidate['coverage'],
        -(candidate['answered_precision'] or 0.0),
    )) if robust else None)
    fallback = (min(verifiable, key=lambda candidate: candidate['nei_coverage'] or 1.0)
                if verifiable else None)
    floor = min((candidate['nei_coverage'] for candidate in verifiable
                 if candidate['nei_coverage'] is not None), default=None)
    return {'feasible': bool(feasible), 'best': best_robust or best_nei or fallback,
            'best_nei': best_nei, 'best_coverage': best_coverage,
            'best_robust': best_robust, 'robust_pool_size': len(robust),
            'fallback': fallback, 'n_feasible': len(feasible),
            'verifiable_n': len(verifiable), 'nei_floor_verifiable': floor,
            'feasible_candidates': sorted(
                feasible, key=lambda candidate: (
                    candidate['nei_coverage'] if candidate['nei_coverage'] is not None else 1.0,
                    -candidate['coverage']))}


GATE_AUROC = 0.79436


def _select_answerability(configs: list[dict]) -> tuple[dict, str]:
    """Pick the highest-AUROC configuration whose selective policy passes.

    The selective constraints are the frozen v2/v5 gates. If no configuration
    passes them, fall back to the highest-AUROC configuration so the negative
    result is still comparable; the fallback is marked explicitly.
    """
    feasible = [config for config in configs if config['policy']['feasible']]
    pool, mode = (feasible, 'feasible') if feasible else (configs, 'fallback')
    selected = max(pool, key=lambda config: (
        config['auroc'],
        -((config['policy']['best'] or {}).get('nei_coverage') or 1.0),
    ))
    return selected, mode


def _answerability_configs(train_features: np.ndarray, train_answerable: np.ndarray,
                           val_features: np.ndarray, val_answerable: np.ndarray) -> list[dict]:
    configs: list[dict] = []
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=5000, random_state=42,
                               class_weight='balanced'))
        model.fit(train_features, train_answerable)
        probability = model.predict_proba(val_features)[:, 1]
        configs.append({'name': f'logistic_C{c_value}', 'model': model,
                        'probability': probability,
                        'auroc': float(roc_auc_score(val_answerable, probability)),
                        'average_precision': float(average_precision_score(
                            val_answerable, probability))})

    for max_iter, learning_rate, leaves in ((300, 0.03, 15), (300, 0.05, 31),
                                            (500, 0.03, 31)):
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=leaves,
            l2_regularization=1.0, class_weight='balanced', random_state=42)
        model.fit(train_features, train_answerable)
        probability = model.predict_proba(val_features)[:, 1]
        configs.append({'name': f'hgb_{max_iter}_{learning_rate}_{leaves}',
                        'model': model, 'probability': probability,
                        'auroc': float(roc_auc_score(val_answerable, probability)),
                        'average_precision': float(average_precision_score(
                            val_answerable, probability))})

    for nei_weight in (2, 3, 4, 6):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight={0: 1.0, 1: float(nei_weight)},
            random_state=42)
        model.fit(train_features, train_answerable)
        probability = model.predict_proba(val_features)[:, 1]
        configs.append({'name': f'hgb_nei_weight_{nei_weight}',
                        'model': model, 'probability': probability,
                        'auroc': float(roc_auc_score(val_answerable, probability)),
                        'average_precision': float(average_precision_score(
                            val_answerable, probability))})
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-size', type=int, choices=(3000, 9000), default=3000)
    parser.add_argument('--variants', nargs='+', default=('base', 'mc', 'hhem', 'both', 'compact'),
                        choices=('base', 'mc', 'hhem', 'both', 'compact'))
    args = parser.parse_args()

    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    train_rows, val_rows = _rows(args.train_size)
    corpus = SentenceCorpus.from_fever()
    train_path = FEATURE_CACHE / f'v5_base_train_{args.train_size}.npz'
    val_path = FEATURE_CACHE / f'v5_base_val_{args.train_size}.npz'
    sim_model = (None if train_path.exists() and val_path.exists()
                 else SimilarityModel())
    train_x = _base_matrix('train', args.train_size, train_rows, corpus, sim_model)
    val_x = _base_matrix('val', args.train_size, val_rows, corpus, sim_model)
    train_y = np.array([LABELS.index(row['label']) for row in train_rows], dtype=np.int64)
    val_y = np.array([LABELS.index(row['label']) for row in val_rows], dtype=np.int64)
    train_answerable = (train_y != 2).astype(np.int64)
    val_answerable = (val_y != 2).astype(np.int64)

    grounding = {}
    for model in ('mc', 'hhem'):
        path = grounding_path(model, 'train', args.train_size)
        if not path.exists():
            raise SystemExit(f'Missing grounding cache: {path}')
        grounding[model] = load_cache(path)
        val_path = grounding_path(model, 'val', args.train_size)
        if not val_path.exists():
            raise SystemExit(f'Missing val grounding cache: {val_path}')
        grounding[f'{model}_val'] = load_cache(val_path)

    train_g, g_names = build_grounding_features(
        train_rows, {'mc': grounding['mc'], 'hhem': grounding['hhem']}, train_x)
    val_g, _ = build_grounding_features(
        val_rows, {'mc': grounding['mc_val'], 'hhem': grounding['hhem_val']}, val_x)
    if train_g.shape[1] != val_g.shape[1]:
        raise ValueError('train/val grounding feature widths differ')

    results = []
    for variant in args.variants:
        if variant == 'base':
            g_cols: list[int] = []
        else:
            g_cols = variant_columns(g_names, variant)
        cols = list(range(len(FEATURE_NAMES))) + [len(FEATURE_NAMES) + i for i in g_cols]
        train_matrix = np.hstack([train_x, train_g])
        val_matrix = np.hstack([val_x, val_g])
        model, selected_c, selection_score = fit_model(
            train_matrix, train_y, val_matrix, val_y, cols)
        train_columns = train_matrix[:, cols]
        val_columns = val_matrix[:, cols]
        predictions = model.predict(val_columns)
        val_metrics = multiclass_metrics(
            [LABELS[i] for i in val_y], [LABELS[i] for i in predictions])
        logits = model.decision_function(val_columns)
        calibrated, _, calibrator = cross_fit_vector_scaling_brier(logits, val_y,
                                                                    n_splits=2)
        train_oof = cross_fit_base_probabilities(train_columns, train_y,
                                                 c_value=selected_c, folds=5)
        val_probabilities = model.predict_proba(val_columns)

        # The answerability head sees the full v4 feature block, the base-model
        # probabilities, and the grounding block for this variant.
        answer_train = np.hstack([train_x, train_oof, train_g[:, g_cols]])
        answer_val = np.hstack([val_x, val_probabilities, val_g[:, g_cols]])
        configs = _answerability_configs(answer_train, train_answerable,
                                         answer_val, val_answerable)
        for config in configs:
            config['policy'] = _policy_search(config['probability'], calibrated, val_y)
            config['brier'] = float(np.mean(
                (config['probability'] - val_answerable) ** 2))
        selected, selection_mode = _select_answerability(configs)
        best_nei = (selected['policy']['best'] or {}).get('nei_coverage')
        results.append({
            'variant': variant,
            'n_features': len(cols),
            'selected_C': float(selected_c),
            'selection_macro_f1': float(selection_score),
            'val_macro_f1': float(val_metrics['macro_f1']),
            'val_per_class': val_metrics['per_class'],
            'calibration': {'brier': float(brier_score(val_y, calibrated)),
                            'parameters': calibrator},
            'answerability': {
                'selected': selected['name'],
                'selection_mode': selection_mode,
                'auroc': selected['auroc'],
                'auroc_gate_met': bool(selected['auroc'] > GATE_AUROC),
                'average_precision': selected['average_precision'],
                'brier': selected['brier'],
                'policy': selected['policy'],
                'best_feasible_auroc': max(
                    (config['auroc'] for config in configs if config['policy']['feasible']),
                    default=None),
                'configs': [{key: value for key, value in config.items()
                             if key not in ('probability', 'model')}
                            for config in configs],
            },
        })
        print(f'{variant:8s} macro={val_metrics["macro_f1"]:.4f} '
              f'C={selected_c:g} ans_auroc={selected["auroc"]:.4f} '
              f'feasible={selected["policy"]["feasible"]} '
              f'nei={best_nei}',
              flush=True)

    report = {
        'version': 'v5-grounding-ablation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'val',
        'test_evaluated': False,
        'train_size': args.train_size,
        'feature_names': FEATURE_NAMES,
        'grounding_feature_names': g_names,
        'grounding_cache': {
            'train': {model: str(grounding_path(model, 'train', args.train_size))
                      for model in ('mc', 'hhem')},
            'val': {model: str(grounding_path(model, 'val', args.train_size))
                    for model in ('mc', 'hhem')},
        },
        'gates': {
            'overall_precision_min': 0.75,
            'verifiable_precision_min': 0.94,
            'verifiable_coverage_min': 0.80,
            'nei_coverage_max': 0.40,
            'val_macro_f1_min': 0.69296,
            'answerability_auroc_min': 0.79436,
            'brier_max': 0.40870,
        },
        'results': results,
        'notes': [
            'Validation only; no test data was read.',
            'Grounding models are pretrained and were not fitted on FEVER.',
            'Labels are silver. No human-validation claim is made.',
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'grounding_ablation_{args.train_size}.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')

    lines = [
        f'# v5 grounding ablation ({args.train_size} train claims)',
        '',
        f'Created: {report["created_utc"]}',
        '',
        '| Variant | Features | C | Macro-F1 | SUPP F1 | CONTRA F1 | NEI F1 | Ans AUROC | AUROC gate | NEI floor | Feasible |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in results:
        per = row['val_per_class']
        policy = row['answerability']['policy']
        lines.append(
            f'| {row["variant"]} | {row["n_features"]} | {row["selected_C"]:g} '
            f'| {row["val_macro_f1"]:.4f} '
            f'| {per["SUPPORTED"]["f1"]:.4f} | {per["CONTRADICTED"]["f1"]:.4f} '
            f'| {per["INSUFFICIENT"]["f1"]:.4f} '
            f'| {row["answerability"]["auroc"]:.4f} '
            f'| {row["answerability"]["auroc_gate_met"]} '
            f'| {policy["nei_floor_verifiable"] if policy["nei_floor_verifiable"] is not None else "n/a"} '
            f'| {policy["feasible"]} |')
    lines += [
        '',
        'No test data was read at ablation time; the later v5 test pass is recorded in VALIDATION.md.',
        'Labels are silver; this is not human adjudication.',
    ]
    (OUT / f'grounding_ablation_{args.train_size}.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved -> {OUT / f"grounding_ablation_{args.train_size}.json"}')


if __name__ == '__main__':
    main()
