"""Stage 5: fit the learned claim-level combiner on FEVER.

Protocol
--------
* Fit on the FEVER *train* split only.
* Select regularisation strength ``C`` and feature groups on *val* only.
* Evaluate the frozen model on *test* exactly once; the protocol file records
  whether that happened and later runs refuse to repeat it without an explicit
  override flag.
* The fixed 0.7/0.95 rule is replayed on the same rows as a reference.

Nothing here uses HaluEval: it has no claim-level gold labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.datasets.schema import LABELS
from src.evaluation.metrics import multiclass_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.verification.detector import aggregate_probabilities, decide
from src.verification.features import (FEATURE_GROUPS, FEATURE_NAMES, build_matrix, labels,
                                       load_split, similarity_matrix)

METHODS = ('bm25', 'dense', 'hybrid')
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SEED = 42
PROTOCOL_VERSION = 'stage5-learned-v1'

ABLATIONS = {
    'full': list(FEATURE_GROUPS),
    'no_nli': ['retrieval', 'similarity', 'claim'],
    'no_retrieval': ['nli', 'similarity', 'claim'],
    'no_similarity': ['nli', 'retrieval', 'claim'],
    'no_claim_features': ['nli', 'retrieval', 'similarity'],
    'nli_only': ['nli'],
    'retrieval_only': ['retrieval'],
    'similarity_only': ['similarity'],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_path(split: str, method: str) -> Path:
    return PROJECT_ROOT / 'data' / 'processed' / 'features' / f'sim_{split}_{method}.npz'


def similarities(split: str, method: str, rows: list[dict], corpus, model) -> np.ndarray:
    """Pretrained-similarity features, cached by split+method+row ids."""
    path = _cache_path(split, method)
    ids = [r['id'] for r in rows]
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if list(cached['ids']) == ids:
            return cached['sims']
    sims = similarity_matrix(rows, method, corpus, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), sims=sims)
    return sims


def fixed_rule_predictions(rows: list[dict], method: str, entail: float, contra: float) -> list[str]:
    return [decide(aggregate_probabilities([h['nli'] for h in r[method][:5]]), entail, contra)
            for r in rows]


def _encode(labels_: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABELS)}
    return np.array([index[label] for label in labels_], dtype=np.int64)


def fit_model(X_train, y_train, X_val, y_val, columns: list[int], seed: int = SEED):
    """Standardise, pick C on val macro-F1, return the fitted pipeline and its choice."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=c, max_iter=5000, random_state=seed))
        model.fit(X_train[:, columns], y_train)
        score = f1_score(y_val, model.predict(X_val[:, columns]), average='macro', zero_division=0)
        if best is None or score > best[0]:
            best = (score, c, model)
    return best[2], best[1], best[0]


def bootstrap_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int = 2000, seed: int = SEED) -> dict:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    scores = np.empty(n, dtype=np.float64)
    size = len(y_true)
    for i in range(n):
        idx = rng.integers(0, size, size)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0)
    return {'mean': float(scores.mean()), 'lower_95': float(np.percentile(scores, 2.5)),
            'upper_95': float(np.percentile(scores, 97.5)), 'n_resamples': n}


def evaluate(model, X, y, rows, columns) -> tuple[dict, list[dict]]:
    predictions = model.predict(X[:, columns])
    predicted_labels = [LABELS[i] for i in predictions]
    metrics = multiclass_metrics([LABELS[i] for i in y], predicted_labels)
    probabilities = model.predict_proba(X[:, columns])
    output = [{'id': r['id'], 'claim': r['claim'], 'gold': gold, 'prediction': pred,
               'probabilities': dict(zip(LABELS, probs.tolist()))}
              for r, gold, pred, probs in zip(rows, (LABELS[i] for i in y), predicted_labels, probabilities)]
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS + ('all',), default='all')
    parser.add_argument('--out', type=Path, default=PROJECT_ROOT / 'experiments/stage5_learned')
    parser.add_argument('--allow-test-rerun', action='store_true',
                        help='Permit a second test evaluation (exploratory only).')
    parser.add_argument('--skip-mlflow', action='store_true')
    args = parser.parse_args()
    methods = METHODS if args.method == 'all' else (args.method,)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / 'protocol.json'
    cfg = load_config()
    entail = cfg['detector']['entail_threshold']
    contra = cfg['detector']['contra_threshold']

    splits = {name: load_split(name) for name in ('train', 'val', 'test')}
    cache_paths = {name: PROJECT_ROOT / 'data/processed/retrieval' / f'fever_{name}.jsonl'
                   for name in splits}
    protocol = {
        'version': PROTOCOL_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'fit_split': 'train', 'selection_split': 'val', 'report_split': 'test',
        'labels': list(LABELS),
        'methods': list(methods),
        'feature_groups': FEATURE_GROUPS,
        'feature_names': FEATURE_NAMES,
        'c_grid': list(C_GRID),
        'seed': SEED,
        'model': 'StandardScaler + multinomial LogisticRegression',
        'similarity_model': cfg['embeddings']['model'],
        'nli_model': cfg['nli']['model'],
        'fixed_rule': {'entail_threshold': entail, 'contra_threshold': contra},
        'retrieval_top_k': 5,
        'cache_sha256': {name: _sha256(p) for name, p in cache_paths.items()},
        'row_counts': {name: len(rows) for name, rows in splits.items()},
        'training_performed': True,
        'test_evaluated': False,
    }
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding='utf-8'))
        if previous.get('test_evaluated') and not args.allow_test_rerun:
            raise SystemExit('The test split was already evaluated under this protocol. '
                             'Pass --allow-test-rerun only for an explicitly exploratory rerun.')
        if any(previous.get(k) != protocol.get(k) for k in
               ('cache_sha256', 'feature_names', 'methods', 'c_grid', 'seed')):
            raise SystemExit('Protocol inputs differ from the stored run; use a separate output directory.')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    from src.retrieval.corpus import SentenceCorpus
    from src.verification.similarity import SimilarityModel

    corpus = SentenceCorpus.from_fever()
    similarity_model = SimilarityModel()

    matrices, golds, sims = {}, {}, {}
    for name, rows in splits.items():
        for method in methods:
            sims[(name, method)] = similarities(name, method, rows, corpus, similarity_model)
            matrices[(name, method)] = build_matrix(rows, method, sims[(name, method)])
        golds[name] = _encode(labels(rows))
    print('Features ready:', {f'{n}/{m}': matrices[(n, m)].shape for n in splits for m in methods}, flush=True)

    results, predictions, ablation_results = {}, {}, {}
    for method in methods:
        X = {name: matrices[(name, method)] for name in splits}
        y = golds

        # --- Reference: the fixed 0.7/0.95 rule on identical rows.
        baselines = {}
        for name in ('val', 'test'):
            pred = fixed_rule_predictions(splits[name], method, entail, contra)
            baselines[name] = multiclass_metrics([LABELS[i] for i in y[name]], pred)

        # --- Full learned model.
        columns = list(range(len(FEATURE_NAMES)))
        model, chosen_c, val_score = fit_model(X['train'], y['train'], X['val'], y['val'], columns)
        val_metrics, _ = evaluate(model, X['val'], y['val'], splits['val'], columns)
        test_metrics, test_predictions = evaluate(model, X['test'], y['test'], splits['test'],
                                                  columns)
        pred_array = np.array([LABELS.index(p['prediction']) for p in test_predictions])
        predictions[method] = test_predictions

        # --- Interpretability.
        coefficients = {}
        scaler = model.named_steps['standardscaler']
        classifier = model.named_steps['logisticregression']
        for class_index, label in enumerate(LABELS):
            order = np.argsort(-np.abs(classifier.coef_[class_index]))[:12]
            coefficients[label] = [{'feature': FEATURE_NAMES[i],
                                    'weight': float(classifier.coef_[class_index][i]),
                                    'standardised_weight': float(classifier.coef_[class_index][i])}
                                   for i in order]
        from sklearn.inspection import permutation_importance
        importance = permutation_importance(model, X['val'][:, columns], y['val'],
                                            scoring='f1_macro', n_repeats=10, random_state=SEED)
        permutation = sorted(
            ({'feature': FEATURE_NAMES[i], 'importance_mean': float(mean), 'importance_std': float(std)}
             for i, mean, std in zip(columns, importance.importances_mean, importance.importances_std)),
            key=lambda row: -row['importance_mean'])

        results[method] = {
            'selected_C': chosen_c,
            'val_macro_f1_at_selection': float(val_score),
            'val': val_metrics,
            'test': test_metrics,
            'test_bootstrap_macro_f1': bootstrap_macro_f1(y['test'], pred_array),
            'fixed_rule': baselines,
            'coefficients': coefficients,
            'permutation_importance': permutation,
            'features_used': len(columns),
            'scaler_mean': dict(zip(FEATURE_NAMES, scaler.mean_.tolist())),
        }
        print(f'[{method}] C={chosen_c} val macro-F1={val_score:.4f} '
              f'test macro-F1={test_metrics["macro_f1"]:.4f}', flush=True)

        # --- Ablations: re-select C and refit for every feature subset.
        method_ablations = {}
        for name, groups in ABLATIONS.items():
            subset = [FEATURE_NAMES.index(feature) for group in groups
                      for feature in FEATURE_GROUPS[group]]
            ablated, ablated_c, _ = fit_model(X['train'], y['train'], X['val'], y['val'], subset)
            ablated_metrics, _ = evaluate(ablated, X['test'], y['test'], splits['test'], subset)
            method_ablations[name] = {'groups': groups, 'selected_C': ablated_c,
                                      'n_features': len(subset), 'test': ablated_metrics}
            print(f'  ablation {name:20s} test macro-F1={ablated_metrics["macro_f1"]:.4f}',
                  flush=True)
        ablation_results[method] = method_ablations

    timestamp = datetime.now(timezone.utc).isoformat()
    protocol['test_evaluated'] = True
    protocol['test_evaluated_utc'] = timestamp
    report = {'protocol': protocol, 'timestamp': timestamp, 'methods': results}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out / 'metrics_ablations.json').write_text(
        json.dumps({'protocol': protocol['version'], 'timestamp': timestamp,
                    'ablations': ablation_results}, indent=2) + '\n', encoding='utf-8')
    for method, rows in predictions.items():
        with (out / f'predictions_{method}.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    if not args.skip_mlflow:
        try:
            log_mlflow(out, results, ablation_results)
        except Exception as exc:  # tracking must never hide the results
            print(f'MLflow logging skipped: {exc}', flush=True)

    print(json.dumps({m: {'val_macro_f1': r['val']['macro_f1'],
                          'test_macro_f1': r['test']['macro_f1'],
                          'fixed_rule_test_macro_f1': r['fixed_rule']['test']['macro_f1']}
                      for m, r in results.items()}, indent=2), flush=True)


def log_mlflow(out: Path, results: dict, ablations: dict) -> None:
    import os
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    import mlflow
    from mlflow.tracking import MlflowClient

    db = PROJECT_ROOT / 'data/cache/mlflow.db'
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{db.as_posix()}')
    client = MlflowClient()
    name = 'hallucination-stage5-learned'
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(PROJECT_ROOT / 'mlartifacts').as_uri())
    digest = _sha256(out / 'metrics.json')
    runs = []
    for method, result in results.items():
        for variant, metrics in [('learned', result['test'])] + [
                (f'ablation_{k}', v['test']) for k, v in ablations[method].items() if k != 'full']:
            signature = f'{method}/{variant}'
            previous = client.search_runs([experiment_id], filter_string=(
                f"tags.source_sha256 = '{digest}' AND tags.variant = '{signature}'"))
            if previous:
                runs.append({'variant': signature, 'run_id': previous[0].info.run_id})
                continue
            with mlflow.start_run(experiment_id=experiment_id, run_name=signature) as run:
                mlflow.set_tags({'source_sha256': digest, 'variant': signature,
                                 'training_performed': 'true'})
                mlflow.log_params({'method': method, 'variant': variant,
                                   'selected_C': result['selected_C'],
                                   'fit_split': 'train', 'selection_split': 'val',
                                   'report_split': 'test'})
                mlflow.log_metrics({'accuracy': float(metrics['accuracy']),
                                    'macro_f1': float(metrics['macro_f1'])})
                mlflow.log_artifact(str(out / 'metrics.json'))
                runs.append({'variant': signature, 'run_id': run.info.run_id})
    (out / 'mlflow_runs.json').write_text(
        json.dumps({'experiment_id': experiment_id, 'source_sha256': digest, 'runs': runs},
                   indent=2) + '\n', encoding='utf-8')
    print(f'MLflow: {len(runs)} runs in {name}', flush=True)


if __name__ == '__main__':
    main()
