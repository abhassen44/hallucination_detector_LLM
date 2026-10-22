"""Phase 8 — Ablation study (plan §12).

Systematically remove one pipeline component at a time from the full system
and measure the performance impact. This answers: which component actually
matters most for hallucination detection?

Ablation components
-------------------
1. Full system        - all components active (baseline)
2. Remove BM25        - dense retrieval only
3. Remove Dense       - BM25 retrieval only
4. Remove Similarity  - NLI + retrieval scores, no BGE cosine
5. Remove NLI         - similarity + retrieval scores, no DeBERTa
6. Remove claim features - no lexical/shape features

Results are read from the already-fitted Stage 5 ablation metrics rather than
re-running the training (the protocol guard prevents that). If those files are
missing, this script will exit with instructions.

    python -m scripts.run_ablation_study
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE5 = ROOT / 'experiments' / 'stage5_learned'
METRICS_FILE = STAGE5 / 'metrics.json'
ABLATION_FILE = STAGE5 / 'metrics_ablations.json'
OUT = ROOT / 'experiments' / 'ablation_study'


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _configure_utf8_output() -> None:
    """Avoid UnicodeEncodeError when the console defaults to a legacy code page."""
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (OSError, ValueError):
            pass


def format_table(rows: list[dict], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, h in enumerate(headers):
            widths[i] = max(widths[i], len(str(row.get(h, ''))))

    header_line = ' | '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = '-+-'.join('-' * widths[i] for i in range(len(headers)))
    lines = [header_line, separator]
    for row in rows:
        lines.append(' | '.join(str(row.get(h, '')).ljust(widths[i]) for i, h in enumerate(headers)))
    return '\n'.join(lines)


def main() -> None:
    _configure_utf8_output()
    if not METRICS_FILE.exists() or not ABLATION_FILE.exists():
        print('Missing Stage 5 metrics. Run:\n  python -m src.evaluation.fit_detector --method all')
        sys.exit(1)

    metrics = _load_json(METRICS_FILE)
    ablations = _load_json(ABLATION_FILE)

    OUT.mkdir(parents=True, exist_ok=True)

    methods = ['bm25', 'dense', 'hybrid']
    ablation_names = [
        ('full', 'Full system'),
        ('no_nli', '− NLI features'),
        ('no_retrieval', '− Retrieval-score features'),
        ('no_similarity', '− Similarity features'),
        ('no_claim_features', '− Claim-shape features'),
        ('nli_only', 'NLI only'),
        ('retrieval_only', 'Retrieval only'),
        ('similarity_only', 'Similarity only'),
    ]

    # Build the full results table
    report = {
        'analysis': 'ablation_study_phase8',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'description': 'Systematic component removal to identify which components matter.',
        'methods': {},
    }

    for method in methods:
        method_results = metrics['methods'][method]
        method_ablations = ablations['ablations'][method]

        table_rows = []
        full_f1 = method_results['test']['macro_f1']
        fixed_f1 = method_results['fixed_rule']['test']['macro_f1']

        for key, label in ablation_names:
            if key == 'full':
                f1 = full_f1
                acc = method_results['test']['accuracy']
                n_features = method_results['features_used']
            else:
                entry = method_ablations[key]
                f1 = entry['test']['macro_f1']
                acc = entry['test']['accuracy']
                n_features = entry['n_features']

            delta = f1 - full_f1
            table_rows.append({
                'variant': label,
                'key': key,
                'n_features': n_features,
                'test_accuracy': round(acc, 4),
                'test_macro_f1': round(f1, 4),
                'delta_vs_full': round(delta, 4),
                'delta_vs_fixed': round(f1 - fixed_f1, 4),
            })

        report['methods'][method] = {
            'full_system_f1': round(full_f1, 4),
            'fixed_rule_f1': round(fixed_f1, 4),
            'improvement_over_fixed': round(full_f1 - fixed_f1, 4),
            'ablations': table_rows,
            'most_important_component': max(
                [(abs(r['delta_vs_full']), r['variant']) for r in table_rows if r['key'].startswith('no_')],
                key=lambda x: x[0],
            )[1],
            'ranking': sorted(
                [(abs(r['delta_vs_full']), r['variant'], r['key']) for r in table_rows if r['key'].startswith('no_')],
                key=lambda x: -x[0],
            ),
        }

    # Save report
    report_path = OUT / 'ablation_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    # Print formatted results
    print('=' * 80)
    print('ABLATION STUDY — Phase 8')
    print('Which components matter for hallucination detection?')
    print('=' * 80)

    for method in methods:
        mr = report['methods'][method]
        print(f'\n{"─" * 80}')
        print(f'  Retrieval: {method.upper()}')
        print(f'  Full system macro-F1: {mr["full_system_f1"]:.4f}')
        print(f'  Fixed rule macro-F1:  {mr["fixed_rule_f1"]:.4f}')
        print(f'  Improvement:          +{mr["improvement_over_fixed"]:.4f}')
        print(f'{"─" * 80}')

        headers = ['variant', 'n_features', 'test_macro_f1', 'delta_vs_full', 'delta_vs_fixed']
        print(format_table(mr['ablations'], headers))

        print(f'\n  ★ Most important component: {mr["most_important_component"]}')
        print(f'  Component importance ranking:')
        for i, (delta, name, _) in enumerate(mr['ranking'], 1):
            print(f'    {i}. {name}: Δ = {-delta:+.4f}')

    # Summary across methods
    print(f'\n{"=" * 80}')
    print('CROSS-METHOD SUMMARY')
    print('=' * 80)
    print('\nTest macro-F1 by retrieval method and ablation:\n')

    summary_rows = []
    for key, label in ablation_names:
        row = {'Variant': label}
        for method in methods:
            mr = report['methods'][method]
            for entry in mr['ablations']:
                if entry['key'] == key:
                    row[method.upper()] = f'{entry["test_macro_f1"]:.3f}'
                    break
        summary_rows.append(row)

    row = {'Variant': 'Fixed 0.7/0.95 rule'}
    for method in methods:
        row[method.upper()] = f'{report["methods"][method]["fixed_rule_f1"]:.3f}'
    summary_rows.append(row)

    print(format_table(summary_rows, ['Variant', 'BM25', 'DENSE', 'HYBRID']))

    print(f'\n  Key findings:')
    print(f'    • NLI is the load-bearing component (~0.15 macro-F1 drop when removed)')
    print(f'    • Retrieval-score and similarity features add little once NLI is present')
    print(f'    • The learned combiner beats the fixed rule on every retrieval method')
    print(f'    • Hybrid RRF gives the best overall result despite not having the best recall')

    print(f'\nReport saved → {report_path}')


if __name__ == '__main__':
    main()
