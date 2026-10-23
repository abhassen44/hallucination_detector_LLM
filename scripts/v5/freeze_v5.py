"""Freeze the selected v5 validation pipeline before the one test pass.

This script fits only on the v4 9k train split and selects only on FEVER val. It
writes model artifacts and a freeze manifest, refuses to touch test data, and
refuses to overwrite a manifest after the test pass has started.

    python -m scripts.v5.freeze_v5 --variant mc --train-size 9000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from scripts.v3.answerability_v3 import (CLASS_THRESHOLDS, answerability_policy,
                                      cross_fit_base_probabilities, meets_constraints)
from scripts.v5.grounding_ablation_v5 import (_answerability_configs, _base_matrix,
                                            _policy_search, _rows, _select_answerability)
from src.datasets.schema import LABELS
from src.evaluation.calibration import (apply_vector_scaling, brier_score,
                                        cross_fit_vector_scaling_brier,
                                        fit_vector_scaling_brier)
from src.evaluation.metrics import multiclass_metrics
from src.utils.config import PROJECT_ROOT
from src.verification.features import FEATURE_NAMES
from src.verification.grounding_features import (build_grounding_features,
                                                 grounding_path, load_cache,
                                                 variant_columns)

OUT = PROJECT_ROOT / 'experiments' / 'v5'
FROZEN = OUT / 'frozen'
GATES = {
    'val_macro_f1_min': 0.69296,
    'insufficient_f1_min': 0.53559,
    'answerability_auroc_min': 0.79436,
    'brier_max': 0.40870,
    'nei_coverage_max': 0.40,
    'overall_precision_min': 0.75,
    'verifiable_precision_min': 0.94,
    'verifiable_coverage_min': 0.80,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                       text=True).strip()
    except Exception:
        return 'unknown'


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('mc', 'hhem', 'both', 'compact'),
                        default='mc')
    parser.add_argument('--train-size', type=int, choices=(3000, 9000), default=9000)
    parser.add_argument('--ablation', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    freeze_path = OUT / 'freeze.json'
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding='utf-8'))
        if existing.get('test_started') or existing.get('test_evaluated'):
            raise SystemExit('Refusing to overwrite the freeze after the test pass started.')
        if not args.force:
            raise SystemExit(f'{_relative(freeze_path)} exists; use --force only before test_started.')
    ablation_path = args.ablation or OUT / f'grounding_ablation_{args.train_size}.json'
    if not ablation_path.exists():
        raise SystemExit(f'Missing ablation report: {ablation_path}')
    ablation = json.loads(ablation_path.read_text(encoding='utf-8'))
    selected_variant = next((row for row in ablation['results']
                             if row['variant'] == args.variant), None)
    if selected_variant is None:
        raise SystemExit(f'Variant {args.variant} not present in {ablation_path}')
    if not selected_variant['answerability']['policy']['feasible']:
        raise SystemExit('Selected variant does not pass the validation selective constraints.')
    selected_config = selected_variant['answerability']['selected']
    selected_c = float(selected_variant['selected_C'])

    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    train_rows, val_rows = _rows(args.train_size)
    corpus = SentenceCorpus.from_fever()
    # The ablation has already populated these caches; compute only if absent.
    train_cache = PROJECT_ROOT / 'data/processed/features' / f'v5_base_train_{args.train_size}.npz'
    val_cache = PROJECT_ROOT / 'data/processed/features' / f'v5_base_val_{args.train_size}.npz'
    sim_model = (None if train_cache.exists() and val_cache.exists() else SimilarityModel())
    train_x = _base_matrix('train', args.train_size, train_rows, corpus, sim_model)
    val_x = _base_matrix('val', args.train_size, val_rows, corpus, sim_model)
    train_y = np.array([LABELS.index(row['label']) for row in train_rows], dtype=np.int64)
    val_y = np.array([LABELS.index(row['label']) for row in val_rows], dtype=np.int64)

    grounding = {}
    for model in ('mc', 'hhem'):
        train_path = grounding_path(model, 'train', args.train_size)
        val_path = grounding_path(model, 'val', args.train_size)
        if not train_path.exists() or not val_path.exists():
            raise SystemExit(f'Missing grounding cache for {model}: {train_path}, {val_path}')
        grounding[model] = load_cache(train_path)
        grounding[f'{model}_val'] = load_cache(val_path)
    train_g, g_names = build_grounding_features(
        train_rows, {'mc': grounding['mc'], 'hhem': grounding['hhem']}, train_x)
    val_g, _ = build_grounding_features(
        val_rows, {'mc': grounding['mc_val'], 'hhem': grounding['hhem_val']}, val_x)
    g_cols = variant_columns(g_names, args.variant)
    cols = list(range(len(FEATURE_NAMES))) + [len(FEATURE_NAMES) + i for i in g_cols]
    train_matrix = np.hstack([train_x, train_g])[:, cols]
    val_matrix = np.hstack([val_x, val_g])[:, cols]

    from scripts.v2.calibrate_v2 import fit_base

    model = fit_base(train_matrix, train_y, c_value=selected_c)
    val_probabilities = model.predict_proba(val_matrix)
    val_predictions = val_probabilities.argmax(axis=1)
    val_metrics = multiclass_metrics([LABELS[i] for i in val_y],
                                     [LABELS[i] for i in val_predictions])
    if abs(val_metrics['macro_f1'] - selected_variant['val_macro_f1']) > 1e-9:
        raise SystemExit('Refit validation macro-F1 does not match the ablation report.')

    val_logits = model.decision_function(val_matrix)
    calibrated_oof, _, _ = cross_fit_vector_scaling_brier(val_logits, val_y, n_splits=2)
    calibrator = fit_vector_scaling_brier(val_logits, val_y)
    calibrated_full = apply_vector_scaling(val_logits, calibrator)

    train_oof = cross_fit_base_probabilities(train_matrix, train_y,
                                             c_value=selected_c, folds=5)
    answer_train = np.hstack([train_x, train_oof, train_g[:, g_cols]])
    answer_val = np.hstack([val_x, val_probabilities, val_g[:, g_cols]])
    train_answerable = (train_y != 2).astype(np.int64)
    val_answerable = (val_y != 2).astype(np.int64)
    configs = _answerability_configs(answer_train, train_answerable,
                                     answer_val, val_answerable)
    for config in configs:
        config['policy_crossfit'] = _policy_search(
            config['probability'], calibrated_oof, val_y)
        config['policy'] = _policy_search(
            config['probability'], calibrated_full, val_y)
        config['brier'] = float(np.mean(
            (config['probability'] - val_answerable) ** 2))
    chosen = next((config for config in configs if config['name'] == selected_config), None)
    if chosen is None:
        raise SystemExit(f'Selected answerability config {selected_config} was not reproduced.')
    selected, selection_mode = _select_answerability([
        {**config, 'policy': config['policy_crossfit']} for config in configs])
    if selected['name'] != selected_config:
        raise SystemExit('Answerability selection is not reproducible from the ablation report.')
    policy = chosen['policy']
    if not policy['feasible'] or policy['best'] is None:
        raise SystemExit('The selected answerability head does not pass the selective constraints.')
    metrics = policy['best']
    if metrics['nei_coverage'] is None or metrics['nei_coverage'] > GATES['nei_coverage_max']:
        raise SystemExit('NEI coverage gate failed.')
    if (metrics['answered_precision'] or 0) < GATES['overall_precision_min']:
        raise SystemExit('Overall precision gate failed.')
    if (metrics['verifiable_precision'] or 0) < GATES['verifiable_precision_min']:
        raise SystemExit('Verifiable precision gate failed.')
    if (metrics['verifiable_coverage'] or 0) < GATES['verifiable_coverage_min']:
        raise SystemExit('Verifiable coverage gate failed.')
    if val_metrics['macro_f1'] < GATES['val_macro_f1_min']:
        raise SystemExit('Macro-F1 gate failed.')
    if val_metrics['per_class']['INSUFFICIENT']['f1'] < GATES['insufficient_f1_min']:
        raise SystemExit('INSUFFICIENT F1 gate failed.')
    if chosen['auroc'] <= GATES['answerability_auroc_min']:
        raise SystemExit('Answerability AUROC gate failed.')
    brier_crossfit = brier_score(val_y, calibrated_oof)
    if brier_crossfit > GATES['brier_max']:
        raise SystemExit('Calibration Brier gate failed.')

    tau = float(policy['best']['tau'])
    answered_full = answerability_policy(calibrated_full, chosen['probability'], tau,
                                         thresholds=CLASS_THRESHOLDS)
    final_full = np.where(answered_full, calibrated_full.argmax(axis=1), 2)
    val_selective_metrics = multiclass_metrics(
        [LABELS[i] for i in val_y], [LABELS[i] for i in final_full])

    FROZEN.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': model,
        'columns': cols,
        'feature_names': FEATURE_NAMES,
        'grounding_feature_names': g_names,
        'grounding_columns': g_cols,
        'variant': args.variant,
        'train_size': args.train_size,
    }, FROZEN / 'verdict.joblib')
    joblib.dump({
        'model': chosen['model'],
        'feature_order': ['base_30', 'base_probabilities_3', 'grounding'],
        'variant': args.variant,
    }, FROZEN / 'answerability.joblib')
    (FROZEN / 'calibration.json').write_text(
        json.dumps({'method': 'vector_scaling_brier', 'parameters': calibrator},
                   indent=2) + '\n', encoding='utf-8')

    cache_paths = {
        'train_split': PROJECT_ROOT / 'data/splits/fever_train_v4_9000.jsonl',
        'val_split': PROJECT_ROOT / 'data/splits/fever_val.jsonl',
        'train_rerank': PROJECT_ROOT / 'data/processed/rerank/train_hybrid_top8.jsonl',
        'extra_rerank': PROJECT_ROOT / 'data/processed/rerank/train_extra_hybrid_top5_6000.jsonl',
        'val_rerank': PROJECT_ROOT / 'data/processed/rerank/val_hybrid_top8.jsonl',
        'train_grounding_mc': grounding_path('mc', 'train', args.train_size),
        'train_grounding_hhem': grounding_path('hhem', 'train', args.train_size),
        'val_grounding_mc': grounding_path('mc', 'val', args.train_size),
        'val_grounding_hhem': grounding_path('hhem', 'val', args.train_size),
    }
    freeze = {
        'version': 'v5-freeze',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'test_evaluated': False,
        'test_started': False,
        'git_head': _git_head(),
        'train_size': args.train_size,
        'variant': args.variant,
        'selection_mode': selection_mode,
        'verdict': {
            'model': 'StandardScaler + multinomial LogisticRegression',
            'selected_C': selected_c,
            'n_features': len(cols),
            'columns': cols,
            'val_macro_f1': val_metrics['macro_f1'],
            'val_per_class': val_metrics['per_class'],
        },
        'answerability': {
            'selected': chosen['name'],
            'auroc': chosen['auroc'],
            'average_precision': chosen['average_precision'],
            'brier': chosen['brier'],
            'tau': tau,
            'class_thresholds': dict(zip(LABELS, CLASS_THRESHOLDS.tolist())),
            'val_policy': policy['best'],
            'val_policy_crossfit': chosen['policy_crossfit']['best'],
            'val_feasible': policy['feasible'],
            'val_feasible_candidates': policy.get('feasible_candidates', []),
            'val_selective_metrics': val_selective_metrics,
        },
        'calibration': {'brier_crossfit': brier_crossfit,
                        'brier_full_val': brier_score(val_y, calibrated_full),
                        'parameters': calibrator},
        'gates': GATES,
        'hashes': {name: _sha256(path) for name, path in cache_paths.items()},
        'artifacts': {
            'verdict': _relative(FROZEN / 'verdict.joblib'),
            'answerability': _relative(FROZEN / 'answerability.joblib'),
            'calibration': _relative(FROZEN / 'calibration.json'),
        },
        'notes': [
            'Fit on 9k train; selected on val only. Test not read.',
            'Labels are silver. No human-validation claim is made.',
            'Learned path remains offline unless the serving gates are separately met.',
        ],
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'variant': args.variant,
        'train_size': args.train_size,
        'macro_f1': val_metrics['macro_f1'],
        'insufficient_f1': val_metrics['per_class']['INSUFFICIENT']['f1'],
        'answerability_auroc': chosen['auroc'],
        'tau': tau,
        'val_selective_macro_f1': val_selective_metrics['macro_f1'],
        'policy': {key: policy['best'][key] for key in
                   ('coverage', 'answered_precision', 'verifiable_coverage',
                    'verifiable_precision', 'nei_coverage')},
        'freeze': _relative(freeze_path),
    }, indent=2))


if __name__ == '__main__':
    main()
