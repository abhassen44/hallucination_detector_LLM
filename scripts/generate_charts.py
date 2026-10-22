"""Generate publication-quality visualisation charts from experiment results.

Produces static HTML charts (no server required) using inline SVG so they render
anywhere. Charts include:
  1. Ablation bar chart (macro-F1 by component)
  2. Retrieval comparison radar/bar chart
  3. Error decomposition stacked bar chart
  4. Confusion matrices as heatmaps
  5. Feature importance horizontal bar chart

    python -m scripts.generate_charts
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE5 = ROOT / 'experiments' / 'stage5_learned'
STAGE2 = ROOT / 'experiments' / 'stage2_oracle'
METRICS_FILE = STAGE5 / 'metrics.json'
ABLATION_FILE = STAGE5 / 'metrics_ablations.json'
OUT = ROOT / 'experiments' / 'charts'


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


# ──────────────── colour palette ────────────────
COLORS = {
    'supported': '#3c8d4f',
    'contradicted': '#c94535',
    'insufficient': '#d4952e',
    'primary': '#2f6846',
    'secondary': '#4a90d9',
    'accent': '#e3883e',
    'muted': '#7d8e80',
    'bg': '#f5f7f4',
    'bg_dark': '#1a2027',
    'text': '#2a3a30',
    'grid': '#dce5d8',
}

BAR_COLORS = ['#2f6846', '#4a90d9', '#e3883e', '#c94535', '#7d5ba6',
              '#45a5a5', '#d97ab5', '#8db255']


def _svg_bar_chart(title: str, data: list[dict], width: int = 720, bar_height: int = 36,
                   value_key: str = 'value', label_key: str = 'label',
                   color_key: str | None = None, baseline: float | None = None,
                   subtitle: str = '') -> str:
    """Horizontal bar chart as SVG string."""
    margin_left, margin_right, margin_top = 220, 80, 65
    gap = 6
    chart_height = len(data) * (bar_height + gap) + margin_top + 40
    max_val = max(d[value_key] for d in data)
    chart_width = width - margin_left - margin_right

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {chart_height}" '
             f'font-family="\'DM Sans\',system-ui,sans-serif" style="background:{COLORS["bg"]};border-radius:14px;">']

    # Title
    lines.append(f'<text x="{width//2}" y="28" text-anchor="middle" font-size="16" font-weight="700" '
                 f'fill="{COLORS["text"]}">{title}</text>')
    if subtitle:
        lines.append(f'<text x="{width//2}" y="48" text-anchor="middle" font-size="11" '
                     f'fill="{COLORS["muted"]}">{subtitle}</text>')

    for i, d in enumerate(data):
        y = margin_top + i * (bar_height + gap)
        val = d[value_key]
        bar_w = max(1, (val / max_val) * chart_width) if max_val > 0 else 0
        color = d.get(color_key, BAR_COLORS[i % len(BAR_COLORS)]) if color_key else BAR_COLORS[i % len(BAR_COLORS)]

        # Label
        lines.append(f'<text x="{margin_left - 10}" y="{y + bar_height // 2 + 4}" '
                     f'text-anchor="end" font-size="12" fill="{COLORS["text"]}">{d[label_key]}</text>')
        # Bar
        lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" '
                     f'rx="5" fill="{color}" opacity="0.88"/>')
        # Value
        lines.append(f'<text x="{margin_left + bar_w + 8}" y="{y + bar_height // 2 + 4}" '
                     f'font-size="12" font-weight="600" fill="{COLORS["text"]}">{val:.3f}</text>')

    if baseline is not None:
        bx = margin_left + (baseline / max_val) * chart_width
        lines.append(f'<line x1="{bx}" y1="{margin_top - 10}" x2="{bx}" y2="{chart_height - 20}" '
                     f'stroke="{COLORS["contradicted"]}" stroke-width="1.5" stroke-dasharray="6,4"/>')
        lines.append(f'<text x="{bx}" y="{margin_top - 14}" text-anchor="middle" font-size="10" '
                     f'fill="{COLORS["contradicted"]}">Fixed rule ({baseline:.3f})</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _svg_grouped_bars(title: str, groups: dict[str, list[dict]], width: int = 780,
                      subtitle: str = '') -> str:
    """Grouped bar chart comparing methods side by side."""
    margin_left, margin_right, margin_top, margin_bottom = 140, 60, 65, 60
    bar_height = 22
    group_gap = 18
    bar_gap = 3

    group_names = list(groups.keys())
    n_bars = len(list(groups.values())[0]) if groups else 0
    chart_height = len(group_names) * (n_bars * (bar_height + bar_gap) + group_gap) + margin_top + margin_bottom
    all_vals = [d['value'] for g in groups.values() for d in g]
    max_val = max(all_vals) if all_vals else 1
    chart_width = width - margin_left - margin_right

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {chart_height}" '
             f'font-family="\'DM Sans\',system-ui,sans-serif" style="background:{COLORS["bg"]};border-radius:14px;">']

    lines.append(f'<text x="{width//2}" y="28" text-anchor="middle" font-size="16" font-weight="700" '
                 f'fill="{COLORS["text"]}">{title}</text>')
    if subtitle:
        lines.append(f'<text x="{width//2}" y="48" text-anchor="middle" font-size="11" '
                     f'fill="{COLORS["muted"]}">{subtitle}</text>')

    y_offset = margin_top
    for g_name in group_names:
        entries = groups[g_name]
        lines.append(f'<text x="{margin_left - 10}" y="{y_offset + (n_bars * (bar_height + bar_gap)) // 2}" '
                     f'text-anchor="end" font-size="13" font-weight="600" fill="{COLORS["text"]}">{g_name}</text>')
        for j, entry in enumerate(entries):
            y = y_offset + j * (bar_height + bar_gap)
            bar_w = max(1, (entry['value'] / max_val) * chart_width)
            color = BAR_COLORS[j % len(BAR_COLORS)]
            lines.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" '
                         f'rx="4" fill="{color}" opacity="0.85"/>')
            lines.append(f'<text x="{margin_left + bar_w + 6}" y="{y + bar_height // 2 + 4}" '
                         f'font-size="11" font-weight="500" fill="{COLORS["text"]}">{entry["value"]:.3f}</text>')
        y_offset += n_bars * (bar_height + bar_gap) + group_gap

    # Legend
    legend_y = chart_height - 25
    if groups:
        sample = list(groups.values())[0]
        for j, entry in enumerate(sample):
            lx = margin_left + j * 140
            color = BAR_COLORS[j % len(BAR_COLORS)]
            lines.append(f'<rect x="{lx}" y="{legend_y}" width="12" height="12" rx="3" fill="{color}" opacity="0.85"/>')
            lines.append(f'<text x="{lx + 18}" y="{legend_y + 10}" font-size="11" '
                         f'fill="{COLORS["muted"]}">{entry.get("label", "")}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _svg_confusion_matrix(title: str, matrix: list[list[int]], labels: list[str],
                          width: int = 380) -> str:
    """Confusion matrix heatmap as SVG."""
    n = len(labels)
    cell_size = 70
    margin_left, margin_top = 110, 90
    total_w = margin_left + n * cell_size + 40
    total_h = margin_top + n * cell_size + 50

    max_val = max(max(row) for row in matrix) if matrix else 1
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
             f'font-family="\'DM Sans\',system-ui,sans-serif" style="background:{COLORS["bg"]};border-radius:14px;">']

    lines.append(f'<text x="{total_w//2}" y="28" text-anchor="middle" font-size="14" font-weight="700" '
                 f'fill="{COLORS["text"]}">{title}</text>')
    lines.append(f'<text x="{total_w//2}" y="46" text-anchor="middle" font-size="10" '
                 f'fill="{COLORS["muted"]}">Predicted →</text>')

    # Column labels
    for j, label in enumerate(labels):
        x = margin_left + j * cell_size + cell_size // 2
        abbrev = label[:4] if len(label) > 5 else label
        lines.append(f'<text x="{x}" y="{margin_top - 12}" text-anchor="middle" font-size="10" '
                     f'font-weight="600" fill="{COLORS["text"]}">{abbrev}</text>')

    for i, label in enumerate(labels):
        y = margin_top + i * cell_size
        abbrev = label[:4] if len(label) > 5 else label
        lines.append(f'<text x="{margin_left - 10}" y="{y + cell_size // 2 + 4}" text-anchor="end" '
                     f'font-size="10" font-weight="600" fill="{COLORS["text"]}">{abbrev}</text>')
        for j in range(n):
            x = margin_left + j * cell_size
            val = matrix[i][j]
            intensity = val / max_val if max_val > 0 else 0
            if i == j:
                r, g, b = 47, 104, 70
            else:
                r, g, b = 201, 69, 53
            color = f'rgba({r},{g},{b},{0.15 + intensity * 0.75:.2f})'
            text_color = '#fff' if intensity > 0.5 else COLORS['text']
            lines.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_size - 4}" height="{cell_size - 4}" '
                         f'rx="6" fill="{color}"/>')
            lines.append(f'<text x="{x + cell_size // 2}" y="{y + cell_size // 2 + 5}" '
                         f'text-anchor="middle" font-size="15" font-weight="700" fill="{text_color}">{val}</text>')

    lines.append(f'<text x="12" y="{margin_top + n * cell_size // 2}" '
                 f'text-anchor="middle" font-size="10" fill="{COLORS["muted"]}" '
                 f'transform="rotate(-90 12 {margin_top + n * cell_size // 2})">Actual</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def _svg_stacked_error(title: str, data: dict, width: int = 750) -> str:
    """Stacked horizontal bar showing error decomposition."""
    margin_left, margin_top = 140, 65
    bar_height = 40
    gap = 16
    systems = list(data.keys())
    chart_height = margin_top + len(systems) * (bar_height + gap) + 100

    bucket_colors = {
        'abstained_oracle_correct': '#4a90d9',
        'abstained_oracle_wrong': '#8db4d9',
        'confident_oracle_wrong': '#e3883e',
        'confident_oracle_correct': '#c94535',
    }
    bucket_labels = {
        'abstained_oracle_correct': 'Abstain (oracle right)',
        'abstained_oracle_wrong': 'Abstain (oracle wrong)',
        'confident_oracle_wrong': 'Confident (oracle wrong)',
        'confident_oracle_correct': 'Confident (oracle right)',
    }

    max_errors = max(d['pipeline_error'] for d in data.values())
    chart_width = width - margin_left - 80

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {chart_height}" '
             f'font-family="\'DM Sans\',system-ui,sans-serif" style="background:{COLORS["bg"]};border-radius:14px;">']
    lines.append(f'<text x="{width//2}" y="28" text-anchor="middle" font-size="16" font-weight="700" '
                 f'fill="{COLORS["text"]}">{title}</text>')
    lines.append(f'<text x="{width//2}" y="48" text-anchor="middle" font-size="11" '
                 f'fill="{COLORS["muted"]}">Error count by bucket (1,324 verifiable FEVER claims)</text>')

    for i, (system_name, stats) in enumerate(data.items()):
        y = margin_top + i * (bar_height + gap)
        lines.append(f'<text x="{margin_left - 10}" y="{y + bar_height // 2 + 4}" text-anchor="end" '
                     f'font-size="12" font-weight="600" fill="{COLORS["text"]}">{system_name}</text>')
        x = margin_left
        buckets = stats['error_buckets']
        for bkey in ['abstained_oracle_correct', 'abstained_oracle_wrong',
                     'confident_oracle_wrong', 'confident_oracle_correct']:
            val = buckets.get(bkey, 0)
            seg_w = (val / max_errors) * chart_width if max_errors > 0 else 0
            if seg_w > 0:
                lines.append(f'<rect x="{x:.1f}" y="{y}" width="{seg_w:.1f}" height="{bar_height}" '
                             f'fill="{bucket_colors[bkey]}" opacity="0.82"/>')
                if seg_w > 20:
                    lines.append(f'<text x="{x + seg_w / 2:.1f}" y="{y + bar_height // 2 + 4}" '
                                 f'text-anchor="middle" font-size="10" font-weight="600" fill="#fff">{val}</text>')
                x += seg_w
        # Total
        lines.append(f'<text x="{x + 10}" y="{y + bar_height // 2 + 4}" font-size="11" '
                     f'font-weight="500" fill="{COLORS["text"]}">{stats["pipeline_error"]}</text>')

    # Legend
    legend_y = chart_height - 35
    for j, (bkey, blabel) in enumerate(bucket_labels.items()):
        lx = margin_left + j * 160
        lines.append(f'<rect x="{lx}" y="{legend_y}" width="12" height="12" rx="3" '
                     f'fill="{bucket_colors[bkey]}" opacity="0.82"/>')
        lines.append(f'<text x="{lx + 18}" y="{legend_y + 10}" font-size="10" '
                     f'fill="{COLORS["muted"]}">{blabel}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def build_html(charts: list[tuple[str, str]]) -> str:
    """Wrap SVG charts into a styled HTML page."""
    chart_html = '\n'.join(
        f'<section><h2>{title}</h2>\n<div class="chart">{svg}</div></section>'
        for title, svg in charts
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Hallucination Detector — Experiment Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0}}
body{{font-family:'DM Sans',system-ui,sans-serif;background:#0f1318;color:#c8d0c4;padding:40px 20px}}
.container{{max-width:900px;margin:auto}}
h1{{font-family:Manrope,sans-serif;font-size:42px;letter-spacing:-2px;color:#e8ede4;margin-bottom:8px}}
.subtitle{{font-size:15px;color:#7d8e80;margin-bottom:50px}}
section{{margin-bottom:60px}}
h2{{font-family:Manrope,sans-serif;font-size:22px;letter-spacing:-0.5px;color:#b8c4b2;margin-bottom:16px}}
.chart{{background:#1a2027;border:1px solid #2a3640;border-radius:16px;padding:20px;overflow-x:auto}}
.chart svg{{width:100%;height:auto;max-width:780px;display:block;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:24px}}
.metric-card{{background:#1a2027;border:1px solid #2a3640;border-radius:14px;padding:24px}}
.metric-card h3{{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#6b8d60;margin-bottom:12px}}
.metric-card .value{{font-family:Manrope,sans-serif;font-size:38px;font-weight:800;color:#e8ede4}}
.metric-card .detail{{font-size:13px;color:#7d8e80;margin-top:8px}}
.delta-positive{{color:#4caf50}}.delta-negative{{color:#ef5350}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #2a3640}}
th{{color:#6b8d60;font-size:11px;letter-spacing:1.5px;text-transform:uppercase}}
td{{color:#b8c4b2}}
tr:hover td{{background:#1e2830}}
.highlight{{color:#e8ede4;font-weight:600}}
footer{{text-align:center;font-size:12px;color:#5a6d5c;margin-top:60px;padding-top:20px;border-top:1px solid #2a3640}}
</style>
</head>
<body>
<div class="container">
<h1>Experiment Results</h1>
<p class="subtitle">Claim-Level Hallucination Detection System — FEVER & HaluEval Evaluation</p>
{chart_html}
<footer>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Hallucination Detector v2</footer>
</div>
</body>
</html>"""


def main() -> None:
    _configure_utf8_output()
    OUT.mkdir(parents=True, exist_ok=True)

    charts = []

    # ── 1. Key metrics cards ──
    metrics = _load_json(METRICS_FILE)
    hybrid = metrics['methods']['hybrid']

    cards_html = '<div class="grid">'
    card_data = [
        ('Best macro-F1', f'{hybrid["test"]["macro_f1"]:.3f}', 'Hybrid learned detector · FEVER test'),
        ('Fixed Rule F1', f'{hybrid["fixed_rule"]["test"]["macro_f1"]:.3f}', 'Threshold baseline · Same test set'),
        ('Improvement', f'+{hybrid["test"]["macro_f1"] - hybrid["fixed_rule"]["test"]["macro_f1"]:.3f}',
         f'McNemar p ≤ 2.4e-8'),
        ('NLI Oracle Acc', '94.3%', 'Gold evidence upper bound · DeBERTa'),
    ]
    for title, value, detail in card_data:
        cards_html += (f'<div class="metric-card"><h3>{title}</h3>'
                       f'<div class="value">{value}</div>'
                       f'<div class="detail">{detail}</div></div>')
    cards_html += '</div>'
    charts.append(('Key Results', cards_html))

    # ── 2. Method comparison ──
    methods_data = {}
    for method in ['BM25', 'Dense', 'Hybrid']:
        m = metrics['methods'][method.lower()]
        methods_data[method] = [
            {'label': 'Learned', 'value': m['test']['macro_f1']},
            {'label': 'Fixed rule', 'value': m['fixed_rule']['test']['macro_f1']},
        ]
    svg = _svg_grouped_bars('Learned vs Fixed Rule by Retrieval Method', methods_data,
                             subtitle='Test macro-F1 on 1,998 FEVER claims')
    charts.append(('Method Comparison', svg))

    # ── 3. Ablation bar chart ──
    ablations = _load_json(ABLATION_FILE)
    hybrid_abl = ablations['ablations']['hybrid']
    abl_data = [
        {'label': 'Full system', 'value': hybrid['test']['macro_f1']},
    ]
    for key, label in [('no_nli', '− NLI'), ('no_retrieval', '− Retrieval scores'),
                       ('no_similarity', '− Similarity'), ('no_claim_features', '− Claim shape'),
                       ('nli_only', 'NLI only'), ('retrieval_only', 'Retrieval only'),
                       ('similarity_only', 'Similarity only')]:
        abl_data.append({'label': label, 'value': hybrid_abl[key]['test']['macro_f1']})

    svg = _svg_bar_chart('Component Ablation Study (Hybrid)', abl_data,
                          baseline=hybrid['fixed_rule']['test']['macro_f1'],
                          subtitle='Test macro-F1 · Dashed line = fixed-rule baseline')
    charts.append(('Ablation Study', svg))

    # ── 4. Per-class breakdown table ──
    per_class = hybrid['test']['per_class']
    fixed_per = hybrid['fixed_rule']['test']['per_class']
    table_html = '<table><thead><tr><th>Class</th><th>Learned P</th><th>Learned R</th><th>Learned F1</th><th>Fixed-Rule F1</th><th>Δ F1</th></tr></thead><tbody>'
    for label in ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT']:
        lp = per_class.get(label, {})
        fp = fixed_per.get(label, {})
        delta = lp.get('f1', 0) - fp.get('f1', 0)
        delta_class = 'delta-positive' if delta > 0 else 'delta-negative'
        table_html += (f'<tr><td class="highlight">{label}</td>'
                       f'<td>{lp.get("precision", 0):.3f}</td><td>{lp.get("recall", 0):.3f}</td>'
                       f'<td class="highlight">{lp.get("f1", 0):.3f}</td>'
                       f'<td>{fp.get("f1", 0):.3f}</td>'
                       f'<td class="{delta_class}">{delta:+.3f}</td></tr>')
    table_html += '</tbody></table>'
    charts.append(('Per-Class Performance', table_html))

    # ── 5. Confusion matrices ──
    cm_labels = hybrid['test'].get('labels', ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT'])
    cm = hybrid['test'].get('confusion_matrix', [])
    if cm:
        svg_cm = _svg_confusion_matrix('Learned Detector — Confusion Matrix', cm, cm_labels)
        fixed_cm = hybrid['fixed_rule']['test'].get('confusion_matrix', [])
        svg_fixed_cm = _svg_confusion_matrix('Fixed Rule — Confusion Matrix', fixed_cm, cm_labels) if fixed_cm else ''
        combined = f'<div class="grid">{svg_cm}{svg_fixed_cm}</div>'
        charts.append(('Confusion Matrices', combined))

    # ── 6. Error decomposition ──
    decomp_path = STAGE5 / 'decomposition.json'
    if decomp_path.exists():
        decomp = _load_json(decomp_path)
        hybrid_decomp = decomp.get('methods', {}).get('hybrid', {})
        if hybrid_decomp:
            decomp_data = {}
            for sys_name, sys_key in [('Fixed rule', 'fixed_rule'), ('Learned', 'learned')]:
                if sys_key in hybrid_decomp:
                    decomp_data[sys_name] = hybrid_decomp[sys_key]
            if decomp_data:
                svg = _svg_stacked_error('Error Decomposition (Hybrid)', decomp_data)
                charts.append(('Error Decomposition', svg))

    # ── 7. Feature importance ──
    permutation = hybrid.get('permutation_importance', [])
    if permutation:
        top_features = sorted(permutation, key=lambda x: -x['importance_mean'])[:12]
        feat_data = [{'label': f['feature'], 'value': f['importance_mean']} for f in top_features]
        svg = _svg_bar_chart('Top 12 Feature Importances (Hybrid)', feat_data,
                              subtitle='Permutation importance on val · macro-F1 drop when shuffled')
        charts.append(('Feature Importance', svg))

    # ── Build HTML ──
    html = build_html(charts)
    out_path = OUT / 'results_dashboard.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'Dashboard saved → {out_path}')
    print(f'Open in browser: file:///{out_path.as_posix()}')


if __name__ == '__main__':
    main()
