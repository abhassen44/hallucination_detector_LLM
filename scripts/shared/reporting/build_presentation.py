"""Build the v5 project presentation deck (.pptx).

Generates an 11-slide deck covering the problem, architecture, research
evolution, v5 method, evaluation protocol, test results, error analysis,
application, and limitations. Charts are embedded from the committed v5
result report (``output version 5/``).

    python -m scripts.shared.reporting.build_presentation [--out PATH]

Requires ``python-pptx`` (see ``requirements-dev.txt``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from src.utils.config import PROJECT_ROOT

ROOT = PROJECT_ROOT
CHARTS = ROOT / 'output version 5'
DEFAULT_OUT = ROOT / 'presentation' / 'claim_level_hallucination_detection.pptx'

# ──────────────── palette (matches project chart colours) ────────────────
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
PRIMARY = RGBColor(0x2F, 0x68, 0x46)
ACCENT = RGBColor(0x4A, 0x90, 0xD9)
WARN = RGBColor(0xD4, 0x95, 0x2E)
FAIL = RGBColor(0xC9, 0x45, 0x35)
TEXT = RGBColor(0x2A, 0x3A, 0x30)
MUTED = RGBColor(0x6B, 0x75, 0x6E)
LIGHT_BG = RGBColor(0xF5, 0xF7, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BOX_BORDER = RGBColor(0xC9, 0xD6, 0xC6)

DECK_TITLE = 'Claim-Level Hallucination Detection System'
REPO_URL = 'github.com/abhassen44/hallucination_detector_LLM'


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


# ──────────────── slide / shape helpers ────────────────

def _blank_slide(prs: Presentation) -> object:
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rect(slide, x: float, y: float, w: float, h: float, fill: RGBColor,
          line: RGBColor | None = None, shape: int = MSO_SHAPE.RECTANGLE) -> object:
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def _text(slide, x: float, y: float, w: float, h: float, text: str, size: int,
          color: RGBColor = TEXT, bold: bool = False, align: int = PP_ALIGN.LEFT,
          anchor: int = MSO_ANCHOR.TOP, italic: bool = False) -> object:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = 'Calibri'
    return box


def _bullets(slide, x: float, y: float, w: float, h: float, items: list[str],
             size: int = 16, color: RGBColor = TEXT, space_after: int = 10,
             bullet: str = '•  ') -> object:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.LEFT
        para.space_after = Pt(space_after)
        run = para.add_run()
        run.text = f'{bullet}{item}'
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = 'Calibri'
    return box


def _image_fit(slide, path: Path, x: float, y: float, w: float, h: float,
               frame: bool = True) -> None:
    """Insert a picture centred in (x, y, w, h), preserving aspect ratio."""
    with PILImage.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    box_ar = w / h
    if ar >= box_ar:
        dw, dh = w, w / ar
    else:
        dh, dw = h, h * ar
    left = x + (w - dw) / 2
    top = y + (h - dh) / 2
    if frame:
        _rect(slide, left - 0.06, top - 0.06, dw + 0.12, dh + 0.12, WHITE,
              line=BOX_BORDER)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top),
                             Inches(dw), Inches(dh))


def _content_header(slide, title: str, kicker: str | None = None) -> None:
    _rect(slide, 0.0, 0.0, 13.333, 0.16, PRIMARY)
    if kicker:
        _text(slide, 0.55, 0.28, 12.2, 0.3, kicker, 11, MUTED, bold=True)
        _text(slide, 0.55, 0.56, 12.2, 0.75, title, 28, NAVY, bold=True)
    else:
        _text(slide, 0.55, 0.34, 12.2, 0.9, title, 30, NAVY, bold=True)


def _footer(slide, number: int) -> None:
    _text(slide, 0.55, 7.08, 8.0, 0.3, f'{DECK_TITLE} — v5', 10, MUTED)
    _text(slide, 12.35, 7.08, 0.55, 0.3, str(number), 10, MUTED,
          align=PP_ALIGN.RIGHT)


def _flow_row(slide, boxes: list[tuple[str, str]], left: float, top: float,
              width: float, height: float, arrow_w: float = 0.34) -> None:
    n = len(boxes)
    total_arrows = arrow_w * (n - 1)
    bw = (width - total_arrows) / n
    for i, (title, sub) in enumerate(boxes):
        x = left + i * (bw + arrow_w)
        shp = _rect(slide, x, top, bw, height, LIGHT_BG, line=BOX_BORDER,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p1 = tf.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        r1 = p1.add_run()
        r1.text = title
        r1.font.size = Pt(12)
        r1.font.bold = True
        r1.font.color.rgb = NAVY
        r1.font.name = 'Calibri'
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(10)
        r2.font.color.rgb = MUTED
        r2.font.name = 'Calibri'
        if i < n - 1:
            ax = x + bw + 0.02
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, Inches(ax), Inches(top + height / 2 - 0.11),
                Inches(arrow_w - 0.04), Inches(0.22))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = ACCENT
            arrow.line.fill.background()


def _table(slide, x: float, y: float, w: float, h: float, headers: list[str],
           rows: list[list[str]], col_widths: list[float], font_size: int = 12,
           header_size: int = 12) -> object:
    table = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x),
                                   Inches(y), Inches(w), Inches(h)).table
    for i, cw in enumerate(col_widths):
        table.columns[i].width = Inches(cw)
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = header
        for para in cell.text_frame.paragraphs:
            para.alignment = PP_ALIGN.LEFT
            for run in para.runs:
                run.font.size = Pt(header_size)
                run.font.bold = True
                run.font.color.rgb = WHITE
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 0 else LIGHT_BG
            cell.text = value
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = TEXT
    return table


# ──────────────── slides ────────────────

def slide_title(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _rect(slide, 0.0, 0.0, 13.333, 7.5, NAVY)
    _rect(slide, 0.0, 7.28, 13.333, 0.22, PRIMARY)
    _text(slide, 0.9, 1.9, 11.5, 1.4, DECK_TITLE, 42, WHITE, bold=True)
    _rect(slide, 0.95, 3.35, 2.1, 0.06, PRIMARY)
    _text(slide, 0.9, 3.6, 11.5, 0.6,
          'LLM answer → atomic claims → evidence retrieval → selective NLI verification',
          19, RGBColor(0xC7, 0xD3, 0xE4))
    _text(slide, 0.9, 4.6, 11.5, 0.45,
          'Python · PyTorch · Transformers · BM25 + FAISS RAG · FastAPI · React',
          15, ACCENT, bold=True)
    _text(slide, 0.9, 5.5, 11.5, 0.9,
          'v5 · one held-out FEVER test pass · all eight preregistered gates passed\n'
          f'{REPO_URL}',
          13, MUTED)
    _text(slide, 0.9, 6.7, 11.5, 0.4,
          'Served path uses the fixed calibrated rule; the learned detector stays offline.',
          12, RGBColor(0x9F, 0xB3, 0xC9), italic=True)


def slide_problem(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Why claim-level verification?', 'PROBLEM')
    _bullets(slide, 0.7, 1.5, 7.3, 4.6, [
        'Whole-response scoring hides compound answers: one LLM reply can be partly '
        'supported and partly fabricated.',
        'We assess each atomic factual claim individually, not the response as a whole.',
        'Three-way verdict with evidence: SUPPORTED, CONTRADICTED, or INSUFFICIENT.',
        'Not-enough-info (NEI) is the hardest class — a wrong answer is worse than '
        'a calibrated abstention.',
        'Serving must therefore be selective: answer only when evidence supports a verdict.',
    ], size=17, space_after=14)
    _flow_row(slide, [
        ('LLM answer', 'Gemma/Ollama extraction'),
        ('Atomic claims', 'subject-predicate-object'),
        ('Per-claim verdicts', '3-way + evidence'),
    ], left=0.7, top=5.15, width=7.3, height=1.05)
    _rect(slide, 8.4, 1.5, 4.4, 5.2, LIGHT_BG, line=BOX_BORDER,
          shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    _text(slide, 8.7, 1.75, 3.8, 0.4, 'Datasets & corpus', 16, NAVY, bold=True)
    _text(slide, 8.7, 2.4, 3.8, 4.1,
          '• FEVER claims (train / val / test)\n'
          '• 247,848-sentence Wikipedia corpus\n'
          '• HaluEval QA for transfer analysis\n'
          '• RAGTruth span-level annotations\n'
          '• Silver labels — research only',
          14, TEXT)


def slide_architecture(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Pipeline architecture', 'SYSTEM')
    _flow_row(slide, [
        ('1 · Claim extraction', 'Ollama LLM JSON pipeline'),
        ('2 · Hybrid retrieval', 'BM25 + BGE dense (RRF)'),
        ('3 · Evidence fusion', 'top-k sentence evidence'),
        ('4 · Verification', 'DeBERTa NLI + similarity'),
        ('5 · Selective policy', 'frozen 0.7 / 0.95 rule'),
        ('6 · Verdict & UI', 'FastAPI + React'),
    ], left=0.35, top=1.6, width=12.6, height=1.5)
    _bullets(slide, 0.7, 3.6, 11.9, 3.2, [
        'Claims are deduplicated and mapped to concise factual propositions before '
        'retrieval.',
        'Retrieval fuses lexical BM25 with dense BGE embeddings over a '
        '247,848-sentence Wikipedia corpus; strict Recall@5 reaches 90.2% (dense).',
        'Verification combines DeBERTa-v3 NLI with BGE semantic similarity; '
        'oracle-evidence accuracy is 94.3%.',
        'The selective policy abstains on low-confidence claims; the app serves the '
        'fixed calibrated rule, never the learned model.',
        'The FastAPI app serves the built React UI on one origin (/verify, /retrieve, '
        '/detect, /health).',
    ], size=15, space_after=9)


def slide_evolution(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Research evolution v1 → v5', 'VERSIONS')
    _table(slide, 0.45, 1.55, 12.45, 4.9,
           ['Version', 'Focus', 'Outcome'],
           [
               ['v1', 'Learned verifier (logistic regression on NLI, similarity, retrieval features)',
                'Test: macro-F1 0.702 vs 0.633 fixed rule'],
               ['v2', 'Temperature scaling + selective policy + abstention flag',
                'Test: Brier / abstention gates failed → stays offline'],
               ['v3', 'Vector scaling + answerability booster, richer diagnostics',
                'Val only — no test pass spent'],
               ['v4', 'Cross-encoder rerank + 9k train + NEI tuning',
                'Val only — no test pass spent (negative result)'],
               ['v5', 'MiniCheck grounding features + answerability head + robust policy',
                'Test: all eight preregistered gates passed'],
           ],
           col_widths=[0.9, 6.4, 5.15], font_size=13, header_size=13)
    _text(slide, 0.55, 6.6, 12.2, 0.45,
          'Discipline: exactly one FEVER test pass in total; every earlier version '
          'stopped on validation.',
          12, MUTED, italic=True)


def slide_method(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'v5 method: grounding-aware NEI', 'RESEARCH')
    _flow_row(slide, [
        ('Grounding features', 'MiniCheck MC-LLM / HHEM'),
        ('NLI + similarity', 'DeBERTa-v3 + BGE'),
        ('Answerability head', 'HGB, distilled from policy'),
        ('Robust policy', 'frozen thresholds (τ = 0.62)'),
        ('Verdict / abstain', 'abstain → INSUFFICIENT'),
    ], left=0.35, top=1.6, width=12.6, height=1.5)
    _bullets(slide, 0.7, 3.7, 7.5, 3.1, [
        'MiniCheck grounding scores judge each claim against its retrieved evidence '
        'and become calibrated features.',
        'A distilled answerability head models when NEI claims can reliably be '
        'answered.',
        'The robust policy maps abstentions to INSUFFICIENT and keeps answered '
        'precision high.',
        'Frames: 9,000 train / 999 val / 1,998 test; frozen before the single test '
        'pass.',
    ], size=15, space_after=9)
    _rect(slide, 8.5, 3.7, 4.3, 2.9, LIGHT_BG, line=BOX_BORDER,
          shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    _text(slide, 8.75, 3.9, 3.8, 0.35, 'Serving boundary', 15, NAVY, bold=True)
    _text(slide, 8.75, 4.4, 3.8, 2.1,
          'Served app → fixed 0.7 / 0.95 rule\n'
          'Learned v5 → offline research only\n'
          'No serving promotion of v5.',
          13, TEXT)


def slide_protocol(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Evaluation protocol', 'RIGOUR')
    _bullets(slide, 0.7, 1.55, 7.6, 5.0, [
        'FEVER splits: 3,000 train / 999 val / 1,998 test claims; 666 per class '
        'at test time.',
        'One test pass total: pipeline, features, thresholds, and seeds frozen in '
        'experiments/v5/freeze.json before the run.',
        'Eight preregistered gates evaluated on test: macro-F1, INSUFFICIENT F1, '
        'answerability AUROC, Brier, overall precision / coverage, verifiable '
        'precision / coverage, NEI coverage.',
        'Metric hygiene: FEVER claim-level metrics are kept separate from '
        'HaluEval answer-level and RAGTruth span-level metrics.',
        'Labels are silver (LLM-proposed), so no human-adjudication claim is made.',
    ], size=15, space_after=10)
    _rect(slide, 8.65, 1.55, 4.1, 4.95, LIGHT_BG, line=BOX_BORDER,
          shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    _text(slide, 8.95, 1.8, 3.5, 0.35, 'Freeze checklist', 15, NAVY, bold=True)
    _text(slide, 8.95, 2.3, 3.5, 4.0,
          '✓ Feature set locked\n'
          '✓ Thresholds frozen\n'
          '✓ Seeds fixed\n'
          '✓ Gates predeclared\n'
          '✓ Single test execution\n'
          '✓ Predictions archived',
          14, TEXT)


def slide_results(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Held-out test results (1,998 claims)', 'RESULTS')
    _table(slide, 0.45, 1.6, 6.3, 4.7,
           ['Metric', 'Value'],
           [
               ['Macro-F1', '0.7369'],
               ['Accuracy', '0.7372'],
               ['SUPPORTED F1', '0.8601'],
               ['CONTRADICTED F1', '0.7059'],
               ['INSUFFICIENT F1', '0.6448'],
               ['Answerability AUROC', '0.8381'],
               ['Brier / ECE', '0.3752 / 0.0218'],
               ['Verifiable precision @ coverage', '0.9550 @ 0.8176'],
           ],
           col_widths=[3.6, 2.7], font_size=13, header_size=13)
    _image_fit(slide, CHARTS / '06_validation_vs_test.png', 7.05, 1.6, 5.75, 4.6)
    _text(slide, 7.05, 6.35, 5.8, 0.4,
          'Validation vs test — v5 generalises without val-tuned inflation.',
          12, MUTED, italic=True, align=PP_ALIGN.CENTER)


def slide_per_class(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Per-class behaviour and confusion', 'RESULTS · DEEP DIVE')
    _image_fit(slide, CHARTS / '01_per_class_metrics.png', 0.55, 1.6, 6.1, 4.7)
    _image_fit(slide, CHARTS / '02_confusion_matrix.png', 6.85, 1.6, 6.0, 4.7)
    _text(slide, 0.55, 6.45, 6.1, 0.4,
          'SUPPORTED is strong; CONTRADICTED/INSUFFICIENT trade off.',
          12, MUTED, italic=True, align=PP_ALIGN.CENTER)
    _text(slide, 6.85, 6.45, 6.0, 0.4,
          'Main confusions: INSUFFICIENT ↔ CONTRADICTED.',
          12, MUTED, italic=True, align=PP_ALIGN.CENTER)


def slide_nei(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'NEI outcomes and the risk–coverage trade-off',
                    'ERROR ANALYSIS')
    _image_fit(slide, CHARTS / '04_nei_outcomes.png', 0.55, 1.55, 6.1, 4.4)
    _image_fit(slide, CHARTS / '03_risk_coverage.png', 6.85, 1.55, 6.0, 4.4)
    _bullets(slide, 0.7, 6.15, 11.9, 0.8, [
        'Of 666 NEI claims: 431 abstained (64.7%), 2 answered correctly, 233 '
        'answered wrong — answered NEI precision is only 0.85%.',
        'Effective verifiable accuracy ≈ 0.8176 × 0.9550 ≈ 0.781 — the policy buys '
        'precision with coverage.',
    ], size=13, space_after=4)


def slide_application(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Application: FastAPI + React demo', 'SERVING')
    _flow_row(slide, [
        ('React client', 'bounded 240 s calls'),
        ('FastAPI app', 'v0.3.1 hardened'),
        ('Endpoints', '/verify · /retrieve · /detect'),
        ('Pipelines', 'extraction → RAG → policy'),
        ('Response', 'verdict + evidence citations'),
    ], left=0.35, top=1.6, width=12.6, height=1.4)
    _bullets(slide, 0.7, 3.55, 6.6, 3.3, [
        'Two modes: Wikipedia retrieval evidence and user-supplied source '
        'verification.',
        'Hardened: request IDs, CORS local allowlist, nosniff / cache headers, '
        'safe metrics parsing, 503 handling with build command.',
        'Serialized inference keeps GPU work stable; the built React bundle is '
        'served on one origin.',
    ], size=14, space_after=8)
    _rect(slide, 7.6, 3.55, 5.2, 3.2, LIGHT_BG, line=BOX_BORDER,
          shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    _text(slide, 7.9, 3.8, 4.6, 0.35, 'Correctness of the demo', 15, NAVY, bold=True)
    _text(slide, 7.9, 4.25, 4.6, 2.4,
          '• 163 Python + 7 frontend tests\n'
          '• Live smoke tests: /health, /, /charts,\n   /verify, /retrieve, /detect\n'
          '• Serves the fixed 0.7 / 0.95 rule —\n   learned detector stays offline',
          13, TEXT)


def slide_limitations(prs: Presentation) -> None:
    slide = _blank_slide(prs)
    _content_header(slide, 'Limitations and conclusion', 'CLOSE')
    _bullets(slide, 0.7, 1.55, 11.9, 3.1, [
        'v5 is the best research version but is NOT promoted to serving; the fixed '
        'calibrated rule remains the served path.',
        'NEI answer precision is still poor (0.85%); abstention covers most NEI '
        'claims instead of answering them.',
        'Labels are silver; there is no human-adjudication claim, and NER/NN '
        'latency for the learned path is unmeasured.',
        'Overall answered precision is 0.787 at 0.663 coverage — useful, but not '
        'production vetting.',
    ], size=15, space_after=10)
    _rect(slide, 0.7, 4.8, 11.9, 1.9, LIGHT_BG, line=BOX_BORDER,
          shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    _text(slide, 1.0, 5.0, 11.3, 0.35, 'Conclusion & next steps', 16, NAVY, bold=True)
    _text(slide, 1.0, 5.45, 11.3, 1.2,
          'Evidence-grounded, selective claim verification works and is honest about '
          'its boundary. Next: improve NEI answer quality, human validation of a '
          'sample, latency gates, and an optional opt-in grounded verifier port '
          '(defaults off).',
          14, TEXT)


def build_deck(out_path: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_problem(prs)
    slide_architecture(prs)
    slide_evolution(prs)
    slide_method(prs)
    slide_protocol(prs)
    slide_results(prs)
    slide_per_class(prs)
    slide_nei(prs)
    slide_application(prs)
    slide_limitations(prs)
    for i, slide in enumerate(prs.slides, start=1):
        if i != 1:
            _footer(slide, i)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Output .pptx path (default: presentation/)')
    args = parser.parse_args()
    out = build_deck(args.out)
    print(f'Deck written: {out}')


if __name__ == '__main__':
    main()
