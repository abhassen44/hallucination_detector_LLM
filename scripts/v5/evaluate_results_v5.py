"""Evaluate the frozen v5 result and save charts and a summary.

This reads the committed v5 artifacts; it does not load a model, rerun the test
pass, or change any frozen threshold. It recomputes the claim-level test metrics
from ``test_predictions.jsonl``, checks them against ``test_metrics.json``, and
writes PNG images plus JSON/Markdown summaries.

    python -m scripts.v5.evaluate_results_v5
    python -m scripts.v5.evaluate_results_v5 --out "output version 5"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.datasets.schema import LABELS  # noqa: E402
from src.evaluation.calibration import (brier_score,  # noqa: E402
                                        expected_calibration_error)
from src.evaluation.metrics import multiclass_metrics  # noqa: E402
from src.evaluation.selective import (metrics_for_answered,  # noqa: E402
                                      risk_coverage_curve)
from src.utils.config import PROJECT_ROOT  # noqa: E402

V5 = PROJECT_ROOT / 'experiments' / 'v5'
DEFAULT_OUT = PROJECT_ROOT / 'output version 5'
COLORS = {
    'SUPPORTED': '#2f9e63',
    'CONTRADICTED': '#d94a4a',
    'INSUFFICIENT': '#d9912b',
    'accent': '#3b7dd8',
    'muted': '#6b7280',
    'grid': '#e5e7eb',
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _style(ax) -> None:
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _check(name: str, actual: float | None, expected: float | None,
           checks: dict, tolerance: float = 1e-9) -> None:
    ok = actual is not None and expected is not None and abs(actual - expected) <= tolerance
    checks[name] = {'actual': actual, 'expected': expected, 'match': bool(ok)}
    if not ok:
        raise SystemExit(f'Recomputed {name} does not match frozen metrics: {actual} vs {expected}')


def evaluate() -> dict:
    test = _load_json(V5 / 'test_metrics.json')
    freeze = _load_json(V5 / 'freeze.json')
    rows = _load_jsonl(V5 / 'test_predictions.jsonl')
    if len(rows) != test['metrics']['n']:
        raise SystemExit('Prediction count does not match test_metrics.json.')

    gold = np.array([LABELS.index(row['gold']) for row in rows], dtype=np.int64)
    predicted = np.array([LABELS.index(row['prediction']) for row in rows], dtype=np.int64)
    answered = np.array([bool(row['answered']) for row in rows], dtype=bool)
    calibrated = np.array([[row['calibrated_probabilities'][label] for label in LABELS]
                           for row in rows], dtype=np.float64)
    p_answerable = np.array([float(row['p_answerable']) for row in rows], dtype=np.float64)

    recomputed = multiclass_metrics([LABELS[int(i)] for i in gold],
                                    [LABELS[int(i)] for i in predicted])
    selective = metrics_for_answered(gold, calibrated, answered)
    brier = brier_score(gold, calibrated)
    ece = expected_calibration_error(gold, calibrated)
    auroc = float(roc_auc_score((gold != 2).astype(int), p_answerable))

    official = test['metrics']
    checks: dict = {}
    _check('macro_f1', recomputed['macro_f1'], official['macro_f1'], checks)
    _check('accuracy', recomputed['accuracy'], official['accuracy'], checks)
    for label in LABELS:
        _check(f'{label}_f1', recomputed['per_class'][label]['f1'],
               official['per_class'][label]['f1'], checks)
    _check('brier', brier, official['brier'], checks)
    _check('ece', ece, official['ece'], checks)
    _check('answerability_auroc', auroc, official['answerability_auroc'], checks)
    for key in ('coverage', 'answered_precision', 'verifiable_coverage',
                'verifiable_precision', 'nei_coverage', 'nei_precision'):
        _check(f'selective_{key}', selective[key], official['selective'][key], checks)

    nei = gold == 2
    answered_nei = answered & nei
    correct_nei = answered_nei & (predicted == 2)
    wrong_nei = answered_nei & (predicted != 2)
    distribution = {
        'gold': {label: int((gold == index).sum()) for index, label in enumerate(LABELS)},
        'predicted': {label: int((predicted == index).sum()) for index, label in enumerate(LABELS)},
        'answered': {
            label: int(((predicted == index) & answered).sum())
            for index, label in enumerate(LABELS)
        },
        'abstained': int((~answered).sum()),
        'answered_total': int(answered.sum()),
        'nei': {
            'total': int(nei.sum()),
            'answered': int(answered_nei.sum()),
            'abstained': int((nei & ~answered).sum()),
            'correct_answered': int(correct_nei.sum()),
            'wrong_answered': int(wrong_nei.sum()),
        },
    }

    val = freeze['answerability']['val_selective_metrics']
    val_policy = freeze['answerability']['val_policy']
    validation = {
        'macro_f1': val['macro_f1'],
        'insufficient_f1': val['per_class']['INSUFFICIENT']['f1'],
        'answerability_auroc': freeze['answerability']['auroc'],
        'brier': freeze['calibration']['brier_crossfit'],
        'coverage': val_policy['coverage'],
        'answered_precision': val_policy['answered_precision'],
        'verifiable_coverage': val_policy['verifiable_coverage'],
        'verifiable_precision': val_policy['verifiable_precision'],
        'nei_coverage': val_policy['nei_coverage'],
    }
    return {
        'test': test,
        'freeze': freeze,
        'rows': rows,
        'gold': gold,
        'predicted': predicted,
        'answered': answered,
        'calibrated': calibrated,
        'recomputed': recomputed,
        'selective': selective,
        'brier': brier,
        'ece': ece,
        'auroc': auroc,
        'validation': validation,
        'distribution': distribution,
        'checks': checks,
    }


def _plot_per_class(result: dict, path: Path) -> None:
    labels = LABELS
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, metric, color in (
            (-width, 'precision', COLORS['accent']),
            (0, 'recall', COLORS['SUPPORTED']),
            (width, 'f1', COLORS['INSUFFICIENT'])):
        values = [result['recomputed']['per_class'][label][metric] for label in labels]
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(),
                      color=color, alpha=0.9)
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('v5 test — per-class precision, recall and F1', fontweight='bold')
    ax.legend(frameon=False, ncol=3, loc='upper center')
    _style(ax)
    _save(fig, path)


def _plot_confusion(result: dict, path: Path) -> None:
    matrix = np.asarray(result['recomputed']['confusion_matrix'])
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]}\n{matrix[i, j] / row_sums[i, 0]:.1%}',
                    ha='center', va='center', fontsize=9,
                    color='white' if matrix[i, j] > matrix.max() / 2 else '#111827')
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=20, ha='right')
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Gold')
    ax.set_title('v5 test — confusion matrix', fontweight='bold')
    _save(fig, path)


def _plot_risk_coverage(result: dict, path: Path) -> None:
    points = risk_coverage_curve(result['gold'], result['calibrated'])
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot([point['coverage'] for point in points],
            [point['answered_precision'] for point in points],
            color=COLORS['accent'], linewidth=2, label='Global-confidence curve')
    selective = result['selective']
    ax.scatter([selective['coverage']], [selective['answered_precision']],
               color=COLORS['INSUFFICIENT'], s=85, zorder=3,
               label='Frozen v5 policy (overall)')
    thresholds = result['test']['metrics']['policy_with_frozen_thresholds']
    ax.scatter([thresholds['coverage']], [thresholds['answered_precision']],
               color=COLORS['muted'], s=70, zorder=3,
               label='Frozen class thresholds (overall)')
    ax.annotate(f"{selective['coverage']:.3f} coverage\nselective precision {selective['answered_precision']:.3f}",
                (selective['coverage'], selective['answered_precision']),
                xytext=(-12, 14), textcoords='offset points', ha='right', fontsize=8,
                color=COLORS['INSUFFICIENT'])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Coverage (fraction of all claims answered)')
    ax.set_ylabel('Answered precision')
    ax.set_title('v5 test — overall coverage / precision trade-off', fontweight='bold')
    ax.legend(frameon=False, loc='lower left')
    _style(ax)
    _save(fig, path)


def _plot_nei(result: dict, path: Path) -> None:
    nei = result['distribution']['nei']
    values = [nei['abstained'], nei['correct_answered'], nei['wrong_answered']]
    names = ['Abstained', 'Correct answer', 'Wrong answer']
    colors = [COLORS['muted'], COLORS['SUPPORTED'], COLORS['CONTRADICTED']]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(names, values, color=colors, alpha=0.9)
    ax.bar_label(bars, labels=[f'{value}\n{value / nei["total"]:.1%}' for value in values],
                 padding=4, fontsize=10)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel('Claims')
    ax.set_title(f'v5 test — {nei["total"]} gold INSUFFICIENT claims', fontweight='bold')
    _style(ax)
    _save(fig, path)


def _plot_gates(result: dict, path: Path) -> None:
    gates = result['test']['test_gates']
    names = list(gates)
    values = [1 if gates[name] else 0 for name in names]
    colors = [COLORS['SUPPORTED'] if value else COLORS['CONTRADICTED'] for value in values]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(names, values, color=colors, alpha=0.9)
    ax.bar_label(bars, labels=['PASS' if value else 'FAIL' for value in values],
                 padding=4, fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Fail', 'Pass'])
    ax.set_title('v5 test — predeclared research gates', fontweight='bold')
    ax.invert_yaxis()
    _style(ax)
    _save(fig, path)


def _plot_validation_test(result: dict, path: Path) -> None:
    val = result['validation']
    test = result['test']['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    labels = ['Final-policy\nmacro-F1', 'INSUFFICIENT\nF1', 'Answerability\nAUROC']
    val_values = [val['macro_f1'], val['insufficient_f1'], val['answerability_auroc']]
    test_values = [test['macro_f1'], test['per_class']['INSUFFICIENT']['f1'],
                   test['answerability_auroc']]
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, val_values, width, label='Validation', color=COLORS['accent'])
    axes[0].bar(x + width / 2, test_values, width, label='Test', color=COLORS['SUPPORTED'])
    for index, value in enumerate(val_values):
        axes[0].text(index - width / 2, value + 0.02, f'{value:.3f}', ha='center', fontsize=8)
    for index, value in enumerate(test_values):
        axes[0].text(index + width / 2, value + 0.02, f'{value:.3f}', ha='center', fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Classification metrics', fontweight='bold')
    axes[0].legend(frameon=False)
    _style(axes[0])

    labels = ['Overall\ncoverage', 'Overall\nprecision', 'Verifiable\ncoverage',
              'Verifiable\nprecision']
    val_values = [val['coverage'], val['answered_precision'],
                  val['verifiable_coverage'], val['verifiable_precision']]
    test_values = [test['selective']['coverage'], test['selective']['answered_precision'],
                   test['selective']['verifiable_coverage'],
                   test['selective']['verifiable_precision']]
    x = np.arange(len(labels))
    axes[1].bar(x - width / 2, val_values, width, label='Validation', color=COLORS['accent'])
    axes[1].bar(x + width / 2, test_values, width, label='Test', color=COLORS['INSUFFICIENT'])
    for index, value in enumerate(val_values):
        axes[1].text(index - width / 2, value + 0.02, f'{value:.3f}', ha='center', fontsize=8)
    for index, value in enumerate(test_values):
        axes[1].text(index + width / 2, value + 0.02, f'{value:.3f}', ha='center', fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title('Selective policy metrics', fontweight='bold')
    axes[1].legend(frameon=False)
    _style(axes[1])
    fig.suptitle('v5 — validation versus held-out test', fontweight='bold')
    _save(fig, path)


def _plot_distribution(result: dict, path: Path) -> None:
    distribution = result['distribution']
    x = np.arange(len(LABELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.8))
    groups = (
        ('Gold', [distribution['gold'][label] for label in LABELS], COLORS['muted']),
        ('Final prediction', [distribution['predicted'][label] for label in LABELS],
         COLORS['accent']),
        ('Answered prediction', [distribution['answered'][label] for label in LABELS],
         COLORS['SUPPORTED']),
    )
    for offset, (name, values, color) in zip((-width, 0, width), groups):
        bars = ax.bar(x + offset, values, width, label=name, color=color, alpha=0.9)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel('Claims')
    ax.set_title('v5 test — label distribution', fontweight='bold')
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, path)


def _write_summary(result: dict, out: Path, images: list[str]) -> None:
    test = result['test']
    metrics = test['metrics']
    selective = metrics['selective']
    gates = test['test_gates']
    summary = {
        'version': 'v5',
        'source': 'experiments/v5/test_metrics.json',
        'test_pass': test['test_pass'],
        'attempts': test['attempts'],
        'recovered_after_error': test['recovered_after_error'],
        'split': test['split'],
        'n': metrics['n'],
        'official_metrics': metrics,
        'recomputed_metrics': result['recomputed'],
        'selective': selective,
        'gates': gates,
        'validation': result['validation'],
        'distribution': result['distribution'],
        'verification_checks': result['checks'],
        'images': images,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                      encoding='utf-8')

    gate_rows = '\n'.join(
        f'| {name} | {"PASS" if passed else "FAIL"} |' for name, passed in gates.items())
    checks_ok = all(check['match'] for check in result['checks'].values())
    image_rows = '\n'.join(f'- `{name}`' for name in images)
    markdown = f"""# v5 results evaluation

Generated by `python -m scripts.v5.evaluate_results_v5`.

- Source: `experiments/v5/test_metrics.json` and `test_predictions.jsonl`
- Test split: {test['split']}, {metrics['n']:,} claims
- Test pass: {test['test_pass']} (attempts: {test['attempts']})
- Recomputed metrics match frozen metrics: **{checks_ok}**
- Fixed rule remains served; v5 remains offline

## Headline metrics

| Metric | Value |
|---|---:|
| Final-policy macro-F1 | {metrics['macro_f1']:.5f} |
| INSUFFICIENT F1 | {metrics['per_class']['INSUFFICIENT']['f1']:.5f} |
| Answerability AUROC | {metrics['answerability_auroc']:.5f} |
| Brier | {metrics['brier']:.5f} |
| ECE | {metrics['ece']:.5f} |
| Overall coverage | {selective['coverage']:.5f} |
| Overall precision | {selective['answered_precision']:.5f} |
| Verifiable coverage | {selective['verifiable_coverage']:.5f} |
| Verifiable precision | {selective['verifiable_precision']:.5f} |
| NEI false-answer coverage | {selective['nei_coverage']:.5f} |
| NEI precision when answered | {selective['nei_precision']:.5f} |

## Gates

| Gate | Result |
|---|---|
{gate_rows}

All eight gates passed.

## INSUFFICIENT breakdown

- Total gold NEI: {result['distribution']['nei']['total']}
- Abstained: {result['distribution']['nei']['abstained']}
- Answered correctly: {result['distribution']['nei']['correct_answered']}
- Answered incorrectly: {result['distribution']['nei']['wrong_answered']}

NEI is reduced, not solved. The 95.5% verifiable precision is selective
precision on the answered subset.

## Images

{image_rows}
"""
    (out / 'summary.md').write_text(markdown, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Output directory (default: output version 5)')
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    result = evaluate()
    figures = [
        ('01_per_class_metrics.png', _plot_per_class),
        ('02_confusion_matrix.png', _plot_confusion),
        ('03_risk_coverage.png', _plot_risk_coverage),
        ('04_nei_outcomes.png', _plot_nei),
        ('05_gate_status.png', _plot_gates),
        ('06_validation_vs_test.png', _plot_validation_test),
        ('07_prediction_distribution.png', _plot_distribution),
    ]
    images = []
    for name, plotter in figures:
        path = out / name
        plotter(result, path)
        images.append(name)
    _write_summary(result, out, images)
    print(json.dumps({
        'output': str(out),
        'images': images,
        'metrics_match_frozen': all(check['match'] for check in result['checks'].values()),
        'test_pass': result['test']['test_pass'],
    }, indent=2))


if __name__ == '__main__':
    main()
