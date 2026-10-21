"""Evidence-coverage diagnostic for v2.

The flag predicts *strict retrieval failure*: at least one claim does not have a
complete gold evidence set inside the top-5 retrieved sentences. Features are the
serving-available retrieval/NLI/similarity/claim signals; gold evidence is used
only to build the training label, never as a feature.

The model is fitted on train and selected on val. The threshold is chosen on val
under a flag-rate budget, then frozen for the single test pass.
"""
from __future__ import annotations

import numpy as np

from src.datasets.schema import INSUFFICIENT


def gold_evidence(examples) -> dict[str, list[set[tuple[str, int]]]]:
    return {example.id: [{(page, int(sent)) for page, sent in group}
                         for group in example.meta.get('evidence_sets', [])]
            for example in examples}


def strict_failure_labels(rows: list[dict], gold: dict[str, list[set[tuple[str, int]]]],
                          method: str = 'hybrid', top_k: int = 5) -> tuple[np.ndarray, list[dict]]:
    """Binary labels for verifiable rows with a defined gold evidence set.

    Returns the boolean failure vector and the rows it applies to. INSUFFICIENT
    claims and claims with missing evidence are excluded: "retrieval failure" is
    undefined without a gold set.
    """
    labels, kept = [], []
    for row in rows:
        if row['label'] == INSUFFICIENT:
            continue
        groups = gold.get(row['id']) or []
        if not groups:
            continue
        top = {(hit['page_id'], int(hit['sent_id'])) for hit in row[method][:top_k]}
        labels.append(not any(group <= top for group in groups))
        kept.append(row)
    return np.array(labels, dtype=np.int64), kept


def fit_flag(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray,
             val_y: np.ndarray, c_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
             seed: int = 42):
    """Select C on val AUROC and return the fitted model plus its parameters."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    for c_value in c_grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=5000, random_state=seed,
                               class_weight='balanced'))
        model.fit(train_x, train_y)
        score = roc_auc_score(val_y, model.predict_proba(val_x)[:, 1])
        if best is None or score > best[0]:
            best = (score, c_value, model)
    auc, c_value, model = best
    scaler = model.named_steps['standardscaler']
    classifier = model.named_steps['logisticregression']
    parameters = {
        'C': float(c_value),
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'coef': classifier.coef_[0].tolist(),
        'intercept': float(classifier.intercept_[0]),
        'val_auroc_at_selection': float(auc),
    }
    return model, parameters


def apply_flag(parameters: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(parameters['mean'])
    scale = np.asarray(parameters['scale'])
    coef = np.asarray(parameters['coef'])
    z = ((matrix - mean) / scale) @ coef + parameters['intercept']
    return 1.0 / (1.0 + np.exp(-z))


def select_threshold(scores: np.ndarray, failures: np.ndarray, errors: np.ndarray,
                     min_flag_rate: float = 0.05, max_flag_rate: float = 0.25,
                     min_enrichment: float = 2.0, n_bootstrap: int = 2000,
                     seed: int = 42) -> dict:
    """Pick the highest-coverage threshold whose *lower-bound* enrichment clears 2x.

    The point estimate alone is fragile at val's size, so the threshold is chosen
    on the 5th percentile of a row bootstrap. The highest flag rate that keeps that
    lower bound at or above ``min_enrichment`` wins; this spends the flag budget on
    warning coverage instead of picking the tiny, most-enriched slice.
    """
    rng = np.random.default_rng(seed)
    base_error = float(errors.mean()) if len(errors) else 0.0
    indices = rng.integers(0, len(scores), size=(n_bootstrap, len(scores)))
    candidates = []
    for rate in np.arange(min_flag_rate, max_flag_rate + 1e-9, 0.01):
        threshold = float(np.quantile(scores, 1.0 - rate))
        flag = scores >= threshold
        actual_rate = float(flag.mean())
        if actual_rate == 0 or actual_rate > max_flag_rate:
            continue
        flagged_error = float(errors[flag].mean())
        enrichment = flagged_error / base_error if base_error > 0 else float('inf')
        capture = float((flag & (errors > 0)).sum() / max((errors > 0).sum(), 1))
        bootstrap = []
        for index in indices:
            sampled_flag = flag[index]
            sampled_error = errors[index]
            sampled_base = sampled_error.mean()
            bootstrap.append(sampled_error[sampled_flag].mean() / sampled_base
                             if sampled_base > 0 and sampled_flag.any() else 0.0)
        candidates.append({
            'threshold': threshold,
            'target_flag_rate': float(rate),
            'flag_rate': actual_rate,
            'base_error_rate': base_error,
            'flagged_error_rate': flagged_error,
            'error_enrichment': enrichment,
            'error_enrichment_lower_95': float(np.percentile(bootstrap, 5)),
            'error_capture': capture,
        })
    if not candidates:
        raise ValueError('no threshold inside the flag-rate budget')
    eligible = [candidate for candidate in candidates
                if candidate['error_enrichment_lower_95'] >= min_enrichment]
    if eligible:
        best = max(eligible, key=lambda candidate: candidate['flag_rate'])
    else:
        best = max(candidates, key=lambda candidate: candidate['error_enrichment'])
    flag = scores >= best['threshold']
    best['n_flagged'] = int(flag.sum())
    best['n'] = int(len(scores))
    best['n_failures'] = int(failures.sum())
    best['failure_rate'] = float(failures.mean())
    best['flagged_failure_rate'] = float(failures[flag].mean())
    best['failure_enrichment'] = (best['flagged_failure_rate'] / best['failure_rate']
                                  if best['failure_rate'] > 0 else None)
    best['selection_rule'] = ('highest flag rate with bootstrap 5th-percentile '
                              f'enrichment >= {min_enrichment}')
    return best
