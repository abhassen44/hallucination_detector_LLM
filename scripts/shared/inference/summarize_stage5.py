"""Render experiments/stage5_learned/REPORT.md from the committed JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from src.datasets.schema import LABELS
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / 'experiments/stage5_learned'


def _load(name):
    path = OUT / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def _table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(
        '---:' if i else '---' for i in range(len(header))) + '|']
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return lines


def main() -> None:
    metrics = _load('metrics.json')
    if metrics is None:
        raise SystemExit('Run src.evaluation.fit_detector first')
    ablations = _load('metrics_ablations.json')
    rerank = _load('val_study_rerank.json')
    context = _load('val_study_context.json')
    protocol = metrics['protocol']
    model = metrics['methods']

    lines = ['# Stage 5 — learned claim-level detector', '',
             'Fit on the FEVER train split, selected on val, evaluated on test **once**.',
             f"Model: `{protocol['model']}`; NLI `{protocol['nli_model']}`; "
             f"similarity `{protocol['similarity_model']}`; seed {protocol['seed']}.", '',
             'Rows: ' + ', '.join(f'{split} {count}' for split, count in
                                  protocol['row_counts'].items()) + '.',
             f"Test evaluated: {protocol['test_evaluated']} "
             f"({protocol.get('test_evaluated_utc', 'n/a')}).", '',
             '## Headline', '']
    lines += _table(
        ['Retrieval', 'Val acc', 'Val macro-F1', 'Test acc', 'Test macro-F1',
         '95% CI (test)', 'Fixed rule test F1', 'Delta'],
        [[method, f"{result['val']['accuracy']:.3f}", f"{result['val']['macro_f1']:.3f}",
          f"{result['test']['accuracy']:.3f}", f"{result['test']['macro_f1']:.3f}",
          f"[{result['test_bootstrap_macro_f1']['lower_95']:.3f}, "
          f"{result['test_bootstrap_macro_f1']['upper_95']:.3f}]",
          f"{result['fixed_rule']['test']['macro_f1']:.3f}",
          f"{result['test']['macro_f1'] - result['fixed_rule']['test']['macro_f1']:+.3f}"]
         for method, result in model.items()])
    lines += ['', 'The confidence interval is a 2,000-resample bootstrap over test claims.',
              'Paired McNemar and paired-bootstrap tests against the fixed rule are reported',
              'below.', '',
              '## Per-class F1 on test', '']
    rows = []
    for method, result in model.items():
        for label in LABELS:
            learned = result['test']['per_class'][label]
            fixed = result['fixed_rule']['test']['per_class'][label]
            rows.append([method, label, f"{learned['precision']:.3f}",
                         f"{learned['recall']:.3f}", f"{learned['f1']:.3f}",
                         f"{fixed['f1']:.3f}"])
    lines += _table(['Retrieval', 'Label', 'P', 'R', 'F1', 'Fixed-rule F1'], rows)

    significance = _load('significance.json')
    if significance:
        lines += ['', '## Paired significance (post-hoc)', '',
                  'Committed test predictions only: no refit, no threshold change, no second',
                  'test pass. The fixed rule is replayed deterministically and checked against',
                  'the committed Stage 3 metrics before testing.', '']
        rows = []
        for method, result in significance['methods'].items():
            delta = result['paired_macro_f1_delta']
            mcnemar = result['mcnemar_3way']
            rows.append([method, f"{delta['observed']:+.4f}",
                         f"[{delta['lower_95']:+.4f}, {delta['upper_95']:+.4f}]",
                         f"{mcnemar['favouring_candidate']} / {mcnemar['favouring_reference']}",
                         f"{mcnemar['p_chi2']:.2g}", f"{mcnemar['p_exact_binomial']:.2g}"])
        lines += _table(['Retrieval', 'Delta macro-F1', '95% CI', 'b / c', 'McNemar p',
                         'exact p'], rows)
        lines += ['', 'Every interval excludes zero. Bootstrap p-values bottom out at',
                  '1/10,001; the exact McNemar p is the sharper statement.', '']

    decomposition = _load('decomposition.json')
    if decomposition and decomposition['methods'].get('hybrid'):
        lines += ['', '## Where the error lives (1,324 verifiable claims)', '',
                  'Every system restricted to the Stage 2 oracle subset. "Introduced" means a',
                  'confident error where the oracle is right; selective accuracy is measured on',
                  'the self-selected answered subset, with the oracle computed on the same rows.', '']
        rows = []
        for system, result in decomposition['methods']['hybrid'].items():
            buckets = result['error_buckets']
            rows.append([system, f"{result['coverage']:.3f}",
                         f"{result['selective_accuracy']:.3f}",
                         f"{result['oracle_accuracy_answered']:.3f}",
                         result['pipeline_error'],
                         buckets['abstained_oracle_correct'],
                         buckets['abstained_oracle_wrong'],
                         buckets['confident_oracle_wrong'],
                         buckets['confident_oracle_correct']])
        lines += _table(['System', 'Coverage', 'Acc answered', 'Oracle answered', 'Errors',
                         'Abstain / oracle right', 'Abstain / oracle wrong',
                         'Confident / oracle wrong', 'Confident / oracle right'], rows)
        lines += ['', 'The learned combiner trades coverage for precision: it abstains on more',
                  'claims but introduces far fewer confident errors. Most residual error is',
                  'abstention on claims the oracle answers correctly, i.e. evidence coverage.',
                  'All methods are in `decomposition.json`; interpretation is in',
                  '`docs/architecture/v1.md`.', '']

    if ablations:
        lines += ['', '## Feature ablation (test macro-F1)', '',
                  'Every ablation re-selects `C` on val before the single test pass.', '']
        variants = list(next(iter(ablations['ablations'].values())).keys())
        rows = [[variant] + [f"{ablations['ablations'][method][variant]['test']['macro_f1']:.3f}"
                             for method in model] for variant in variants]
        lines += _table(['Variant'] + [m for m in model], rows)
        drops = {variant: sum(ablations['ablations'][m][variant]['test']['macro_f1'] -
                              ablations['ablations'][m]['full']['test']['macro_f1']
                              for m in model) / len(model) for variant in variants}
        worst = min(drops, key=drops.get)
        lines += ['', f'The weakest variant is `{worst}` '
                      f'({drops[worst]:+.3f} macro-F1 on average versus the full model).']

    hybrid = model.get('hybrid')
    if hybrid:
        lines += ['', '## What the model relies on (hybrid)', '',
                  'Permutation importance on val, macro-F1 decrease when shuffled:', '']
        rows = [[item['feature'], f"{item['importance_mean']:.4f}",
                 f"{item['importance_std']:.4f}"]
                for item in hybrid['permutation_importance'][:12]]
        lines += _table(['Feature', 'Mean decrease', 'Std'], rows)

    for name, study, title in (('rerank', rerank, 'Val-only: cross-encoder reranking'),
                               ('context', context, 'Val-only: neighbour-sentence context')):
        if not study:
            continue
        lines += ['', f'## {title}', '',
                  'Val split only. No test evaluation, no fitting, no threshold search.', '']
        if name == 'rerank':
            rows = []
            for method, result in study['methods'].items():
                rows.append([method,
                             f"{result['retrieval']['baseline']['@5']['strict_recall']:.3f}",
                             f"{result['retrieval']['reranked']['@5']['strict_recall']:.3f}",
                             f"{result['pipeline']['baseline']['macro_f1']:.3f}",
                             f"{result['pipeline']['candidate']['macro_f1']:.3f}",
                             f"{result['pipeline']['delta']['macro_f1']:+.3f}"])
            lines += _table(['Retrieval', 'Strict R@5', 'Reranked R@5',
                             'Fixed-rule F1', 'Reranked F1', 'Delta'], rows)
        else:
            rows = [[method, f"{result['pipeline']['baseline']['macro_f1']:.3f}",
                     f"{result['pipeline']['candidate']['macro_f1']:.3f}",
                     f"{result['pipeline']['delta']['macro_f1']:+.3f}",
                     result['changed_predictions']]
                    for method, result in study['methods'].items()]
            lines += _table(['Retrieval', 'Fixed-rule F1', 'Context F1', 'Delta',
                             'Changed predictions'], rows)

    lines += ['', '## Limitations', '',
              '- FEVER claims only: HaluEval has no claim-level gold labels, so no claim-level',
              '  HaluEval score is reported anywhere.',
              '- The retrieval corpus is 247,848 sentences from 50k FEVER pages. Open-domain',
              '  coverage is limited and the HaluEval Wikipedia results reflect that.',
              '- Logistic regression is linear; interactions between NLI and retrieval signals',
              '  are only captured through the explicit product-free feature design.',
              '- One training seed is used. The bootstrap interval covers test sampling noise,',
              '  not training variance.',
              '- Probabilities are not calibrated. The reported probabilities are model outputs,',
              '  not factual confidence.',
              '- The fixed 0.7/0.95 rule was inherited from Stage 3 and is not re-tuned here.', '']

    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved -> {OUT / "REPORT.md"}')


if __name__ == '__main__':
    main()
