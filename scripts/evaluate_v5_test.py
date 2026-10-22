"""Apply the frozen v5 pipeline to the FEVER test split exactly once.

Before the test pass, run ``--check-val`` to reproduce the frozen validation
metrics from the serialized artifacts. The test branch refuses to run if the
freeze manifest is absent, if test features are missing, or if the test pass was
already started. ``--recover`` is only for completing a started attempt that
wrote no metrics or predictions (for example, a pre-report serialization error).

    python -m scripts.evaluate_v5_test --check-val
    python -m scripts.evaluate_v5_test --prepare
    python -m scripts.evaluate_v5_test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.answerability_v3 import CLASS_THRESHOLDS, answerability_policy
from src.datasets.schema import LABELS
from src.evaluation.calibration import (apply_vector_scaling, brier_score,
                                        expected_calibration_error)
from src.evaluation.metrics import multiclass_metrics
from src.evaluation.selective import metrics_for_answered, policy_metrics
from src.utils.config import PROJECT_ROOT
from src.verification.features import build_matrix, load_split, similarity_matrix
from src.verification.grounding_features import (build_grounding_features,
                                                 grounding_path, load_cache)

OUT = PROJECT_ROOT / 'experiments' / 'v5'
FREEZE = OUT / 'freeze.json'
FROZEN = OUT / 'frozen'
CACHE = PROJECT_ROOT / 'data' / 'processed' / 'rerank'
FEATURE_CACHE = PROJECT_ROOT / 'data' / 'processed' / 'features'
TOP_K = 5


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _jsonable(value):
    """Convert numpy scalars/arrays so reports are strict JSON."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _enrich(split: str, rerank_path: Path) -> list[dict]:
    rows = load_split(split)
    cached = _jsonl(rerank_path)
    if [row['id'] for row in cached] != [row['id'] for row in rows]:
        raise SystemExit(f'{rerank_path} row ids do not match the {split} split.')
    by_id = {row['id']: row for row in rows}
    return [{**by_id[item['id']], 'rerank': item['hits'][:TOP_K]} for item in cached]


def _base_matrix(split: str, train_size: int, rows: list[dict], corpus,
                 sim_model) -> np.ndarray:
    path = FEATURE_CACHE / f'v5_base_{split}_{train_size}.npz'
    ids = [row['id'] for row in rows]
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if list(cached['ids']) == ids:
            return cached['matrix']
    sims = similarity_matrix(rows, 'rerank', corpus, sim_model, top=TOP_K)
    matrix = build_matrix(rows, 'rerank', sims, top=TOP_K)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), matrix=matrix)
    return matrix


def _load_grounding(split: str, train_size: int) -> tuple[dict[str, list[float]],
                                                          dict[str, list[float]]]:
    return ({model: load_cache(grounding_path(model, split, train_size))
             for model in ('mc', 'hhem')})


def _features(split: str, rerank_path: Path, train_size: int, corpus,
              sim_model) -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    rows = _enrich(split, rerank_path)
    base = _base_matrix(split, train_size, rows, corpus, sim_model)
    grounding = _load_grounding(split, train_size)
    matrix, names = build_grounding_features(rows, grounding, base)
    return rows, base, matrix, names


def _apply(rows: list[dict], base: np.ndarray, grounding: np.ndarray,
           verdict_artifact: dict, answer_artifact: dict, calibration: dict,
           tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = verdict_artifact['model']
    columns = verdict_artifact['columns']
    g_cols = verdict_artifact['grounding_columns']
    x = np.hstack([base, grounding])[:, columns]
    probabilities = model.predict_proba(x)
    logits = model.decision_function(x)
    calibrated = apply_vector_scaling(logits, calibration['parameters'])
    answer_features = np.hstack([base, probabilities, grounding[:, g_cols]])
    p_answerable = answer_artifact['model'].predict_proba(answer_features)[:, 1]
    answered = answerability_policy(calibrated, p_answerable, tau,
                                    thresholds=CLASS_THRESHOLDS)
    final = np.where(answered, calibrated.argmax(axis=1), 2).astype(np.int64)
    return final, calibrated, p_answerable, answered


def _metrics(gold: np.ndarray, final: np.ndarray, calibrated: np.ndarray,
             answered: np.ndarray) -> dict:
    predicted_labels = [LABELS[int(i)] for i in final]
    result = multiclass_metrics([LABELS[int(i)] for i in gold], predicted_labels)
    result['selective'] = metrics_for_answered(gold, calibrated, answered)
    result['policy_with_frozen_thresholds'] = policy_metrics(
        gold, calibrated, CLASS_THRESHOLDS)
    result['brier'] = brier_score(gold, calibrated)
    result['ece'] = expected_calibration_error(gold, calibrated)
    return result


def _check_val(freeze: dict) -> None:
    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    train_size = int(freeze['train_size'])
    corpus = SentenceCorpus.from_fever()
    train_cache = FEATURE_CACHE / f'v5_base_train_{train_size}.npz'
    val_cache = FEATURE_CACHE / f'v5_base_val_{train_size}.npz'
    sim_model = None if train_cache.exists() and val_cache.exists() else SimilarityModel()
    rows, base, grounding, _ = _features(
        'val', CACHE / 'val_hybrid_top8.jsonl', train_size, corpus, sim_model)
    gold = np.array([LABELS.index(row['label']) for row in rows], dtype=np.int64)
    verdict = joblib.load(FROZEN / 'verdict.joblib')
    answer = joblib.load(FROZEN / 'answerability.joblib')
    calibration = json.loads((FROZEN / 'calibration.json').read_text(encoding='utf-8'))
    tau = float(freeze['answerability']['tau'])
    final, calibrated, p_answerable, answered = _apply(rows, base, grounding, verdict, answer,
                                                        calibration, tau)
    metrics = _metrics(gold, final, calibrated, answered)
    answerability_auroc = float(roc_auc_score((gold != 2).astype(int), p_answerable))
    if abs(answerability_auroc - float(freeze['answerability']['auroc'])) > 1e-6:
        raise SystemExit('Val replay answerability AUROC mismatch: '
                         f'{answerability_auroc} vs {freeze["answerability"]["auroc"]}')
    raw_x = np.hstack([base, grounding])[:, verdict['columns']]
    raw_probabilities = verdict['model'].predict_proba(raw_x)
    raw_final = raw_probabilities.argmax(axis=1)
    raw_metrics = multiclass_metrics([LABELS[int(i)] for i in gold],
                                     [LABELS[int(i)] for i in raw_final])
    expected_raw_macro = float(freeze['verdict']['val_macro_f1'])
    if abs(raw_metrics['macro_f1'] - expected_raw_macro) > 1e-6:
        raise SystemExit(f'Val replay raw macro-F1 mismatch: {raw_metrics["macro_f1"]} '
                         f'vs {expected_raw_macro}')
    expected_selective = freeze['answerability'].get('val_selective_metrics')
    if expected_selective is None:
        raise SystemExit('Freeze manifest lacks val_selective_metrics; rerun freeze.')
    if abs(metrics['macro_f1'] - expected_selective['macro_f1']) > 1e-6:
        raise SystemExit(f'Val replay selective macro-F1 mismatch: {metrics["macro_f1"]} '
                         f'vs {expected_selective["macro_f1"]}')
    expected_policy = freeze['answerability']['val_policy']
    for key in ('coverage', 'answered_precision', 'verifiable_coverage',
                'verifiable_precision', 'nei_coverage'):
        actual = metrics['selective'][key]
        expected = expected_policy[key]
        if (actual is None) != (expected is None) or (
                actual is not None and abs(actual - expected) > 1e-6):
            raise SystemExit(f'Val replay selective mismatch for {key}: {actual} vs {expected}')
    print(json.dumps({
        'val_replay_raw_macro_f1': raw_metrics['macro_f1'],
        'val_replay_selective_macro_f1': metrics['macro_f1'],
        'val_replay_nei_coverage': metrics['selective']['nei_coverage'],
        'val_replay_answered_precision': metrics['selective']['answered_precision'],
        'val_replay_verifiable_coverage': metrics['selective']['verifiable_coverage'],
        'val_replay_verifiable_precision': metrics['selective']['verifiable_precision'],
        'val_replay_answerability_auroc': answerability_auroc,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-val', action='store_true')
    parser.add_argument('--prepare', action='store_true',
                        help='Build/cache test features without starting the test pass.')
    parser.add_argument('--recover', action='store_true',
                        help='Complete a started test pass that produced no report '
                             '(for example, a serialization failure before any metric '
                             'was written). Refuses if metrics or predictions exist.')
    args = parser.parse_args()

    if not FREEZE.exists():
        raise SystemExit(f'{_relative(FREEZE)} does not exist; freeze first.')
    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if args.check_val:
        _check_val(freeze)
        return

    recovered = False
    if freeze.get('test_started'):
        artifacts_exist = ((OUT / 'test_metrics.json').exists()
                           or (OUT / 'test_predictions.jsonl').exists())
        if not args.recover or freeze.get('test_evaluated') or artifacts_exist:
            raise SystemExit('The FEVER test pass was already started; refusing a second pass.')
        recovered = True
    test_rerank = CACHE / 'test_hybrid_top8.jsonl'
    test_mc = grounding_path('mc', 'test', int(freeze['train_size']))
    test_hhem = grounding_path('hhem', 'test', int(freeze['train_size']))
    missing = [path for path in (test_rerank, test_mc, test_hhem) if not path.exists()]
    if missing:
        raise SystemExit('Missing test features: ' + ', '.join(_relative(path) for path in missing))

    if args.prepare:
        from src.retrieval.corpus import SentenceCorpus
        from src.verification.similarity import SimilarityModel

        train_size = int(freeze['train_size'])
        corpus = SentenceCorpus.from_fever()
        train_cache = FEATURE_CACHE / f'v5_base_train_{train_size}.npz'
        test_cache = FEATURE_CACHE / f'v5_base_test_{train_size}.npz'
        sim_model = None if train_cache.exists() and test_cache.exists() else SimilarityModel()
        rows, base, grounding, names = _features(
            'test', test_rerank, train_size, corpus, sim_model)
        verdict = joblib.load(FROZEN / 'verdict.joblib')
        if verdict['grounding_feature_names'] != names:
            raise SystemExit('Test grounding feature names do not match the freeze.')
        print(json.dumps({'prepared_test_rows': len(rows),
                          'base_matrix_shape': list(base.shape),
                          'grounding_shape': list(grounding.shape),
                          'test_cache': _relative(test_cache)}, indent=2))
        return

    if not recovered:
        freeze['test_started'] = True
        freeze['test_started_utc'] = datetime.now(timezone.utc).isoformat()
        freeze['test_attempts'] = 1
        FREEZE.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    else:
        freeze['test_attempts'] = 2
        freeze['test_recovered_after_error'] = True
        freeze['test_recovered_utc'] = datetime.now(timezone.utc).isoformat()
        FREEZE.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')

    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    train_size = int(freeze['train_size'])
    corpus = SentenceCorpus.from_fever()
    train_cache = FEATURE_CACHE / f'v5_base_train_{train_size}.npz'
    test_cache = FEATURE_CACHE / f'v5_base_test_{train_size}.npz'
    sim_model = None if train_cache.exists() and test_cache.exists() else SimilarityModel()
    rows, base, grounding, names = _features(
        'test', test_rerank, train_size, corpus, sim_model)
    gold = np.array([LABELS.index(row['label']) for row in rows], dtype=np.int64)
    verdict = joblib.load(FROZEN / 'verdict.joblib')
    answer = joblib.load(FROZEN / 'answerability.joblib')
    calibration = json.loads((FROZEN / 'calibration.json').read_text(encoding='utf-8'))
    if verdict['grounding_feature_names'] != names:
        raise SystemExit('Test grounding feature names do not match the freeze.')
    tau = float(freeze['answerability']['tau'])
    final, calibrated, p_answerable, answered = _apply(rows, base, grounding, verdict,
                                                       answer, calibration, tau)
    metrics = _metrics(gold, final, calibrated, answered)
    metrics['answerability_auroc'] = float(
        roc_auc_score((gold != 2).astype(int), p_answerable))
    gates = freeze['gates']
    test_gates = {
        'answerability_auroc': bool(metrics['answerability_auroc']
                                    > gates['answerability_auroc_min']),
        'verifiable_precision': bool(
            (metrics['selective']['verifiable_precision'] or 0)
            >= gates['verifiable_precision_min']),
        'verifiable_coverage': bool(
            (metrics['selective']['verifiable_coverage'] or 0)
            >= gates['verifiable_coverage_min']),
        'overall_precision': bool(
            (metrics['selective']['answered_precision'] or 0)
            >= gates['overall_precision_min']),
        'nei_coverage': bool(
            (metrics['selective']['nei_coverage'] or 1)
            <= gates['nei_coverage_max']),
        'macro_f1': bool(metrics['macro_f1'] >= gates['val_macro_f1_min']),
        'insufficient_f1': bool(
            metrics['per_class']['INSUFFICIENT']['f1']
            >= gates['insufficient_f1_min']),
        'brier': bool(metrics['brier'] <= gates['brier_max']),
    }
    report = {
        'version': 'v5-test',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'test',
        'test_evaluated': True,
        'test_pass': 1,
        'attempts': 2 if recovered else 1,
        'recovered_after_error': bool(recovered),
        'freeze': _relative(FREEZE),
        'git_head': freeze.get('git_head'),
        'variant': freeze.get('variant'),
        'train_size': freeze.get('train_size'),
        'metrics': metrics,
        'test_gates': test_gates,
        'hashes': {
            'freeze': _sha256(FREEZE),
            'test_rerank': _sha256(test_rerank),
            'test_grounding_mc': _sha256(test_mc),
            'test_grounding_hhem': _sha256(test_hhem),
        },
        'notes': [
            'One final FEVER test pass; no component had a separate test pass.',
            'No test-driven tuning: the pipeline, features, thresholds, and seeds were frozen before this run.',
            'Labels are silver; this is not human adjudication.',
            'These are claim-level FEVER metrics, separate from any RAGTruth/out-of-domain response metrics.',
        ],
    }
    if recovered:
        report['notes'].append(
            'The first attempt stopped during JSON serialization before any metric '
            'or prediction was written. No test metric was observed or used to '
            'change the frozen pipeline; this recovery reran the identical pipeline.')
    report = _jsonable(report)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / 'test_metrics.json'
    predictions_path = OUT / 'test_predictions.jsonl'
    metrics_tmp = OUT / 'test_metrics.json.tmp'
    predictions_tmp = OUT / 'test_predictions.jsonl.tmp'
    prediction_lines = [
        json.dumps({
            'id': row['id'],
            'claim': row['claim'],
            'gold': row['label'],
            'prediction': LABELS[int(final[index])],
            'calibrated_probabilities': dict(
                zip(LABELS, calibrated[index].tolist())),
            'p_answerable': float(p_answerable[index]),
            'answered': bool(answered[index]),
        }, ensure_ascii=False)
        for index, row in enumerate(rows)
    ]
    predictions_tmp.write_text('\n'.join(prediction_lines) + '\n', encoding='utf-8')
    metrics_tmp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    predictions_tmp.replace(predictions_path)
    metrics_tmp.replace(metrics_path)
    freeze['test_evaluated'] = True
    freeze['test_evaluated_utc'] = report['created_utc']
    freeze['test_metrics'] = _relative(metrics_path)
    freeze['test_attempts'] = report['attempts']
    freeze['test_recovered_after_error'] = report['recovered_after_error']
    FREEZE.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(_jsonable({
        'macro_f1': metrics['macro_f1'],
        'insufficient_f1': metrics['per_class']['INSUFFICIENT']['f1'],
        'selective': {key: metrics['selective'][key] for key in
                      ('coverage', 'answered_precision', 'verifiable_coverage',
                       'verifiable_precision', 'nei_coverage')},
        'gates': test_gates,
    }), indent=2))


if __name__ == '__main__':
    main()
