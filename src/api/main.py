"""Run with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000."""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
APP_VERSION = '0.3.1'
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
Method = Literal['bm25', 'dense', 'hybrid']
_UI = PROJECT_ROOT / 'frontend' / 'dist'
_UI_INDEX = _UI / 'index.html'
_DEFAULT_ALLOWED_ORIGINS = (
    'http://127.0.0.1:5173',
    'http://localhost:5173',
    'http://127.0.0.1:4173',
    'http://localhost:4173',
)


def _allowed_origins() -> list[str]:
    raw = os.getenv('API_ALLOWED_ORIGINS', ','.join(_DEFAULT_ALLOWED_ORIGINS))
    if raw.strip() == '*':
        return ['*']
    return [origin.strip() for origin in raw.split(',') if origin.strip()]


app = FastAPI(title='Claim-Level Hallucination Detector', version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Accept', 'X-Request-ID'],
    expose_headers=['X-Request-ID'],
)
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


@app.middleware('http')
async def operational_headers(request, call_next):
    """Attach tracing and safe cache headers without touching inference state."""
    incoming = request.headers.get('X-Request-ID', '').strip()
    request_id = incoming if 0 < len(incoming) <= 128 else uuid.uuid4().hex
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    response.headers['X-Content-Type-Options'] = 'nosniff'
    path = request.url.path
    if path in ('/', '/charts'):
        response.headers['Cache-Control'] = 'no-cache'
    elif path.startswith('/assets/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    elif path in ('/health', '/claims', '/detect', '/retrieve', '/verify') \
            or path.startswith('/metrics'):
        response.headers['Cache-Control'] = 'no-store'
    return response


class RequestModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ClaimsRequest(RequestModel):
    answer: Text
    question: Text | None = None


class DetectRequest(ClaimsRequest):
    evidence: list[Text] | None = Field(default=None, max_length=20)
    method: Method = 'hybrid'
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveRequest(RequestModel):
    claim: Text
    method: Method = 'hybrid'
    top_k: int = Field(default=5, ge=1, le=20)


class VerifyRequest(RequestModel):
    claim: Text
    evidence: list[Text] = Field(max_length=20)


@lru_cache(maxsize=1)
def _extractor():
    from src.generation.claim_extraction import ClaimExtractor
    return ClaimExtractor()


@lru_cache(maxsize=1)
def _detector():
    from src.verification.detector import Detector
    return Detector(extractor=_extractor())


def get_detector():
    try:
        with _model_lock:
            return _detector()
    except Exception as exc:
        logger.exception('Unable to load inference dependencies')
        raise HTTPException(503, 'Models or retrieval indexes are unavailable. See README.md and docs/architecture/v5.md.') from exc


def get_extractor():
    try:
        with _model_lock:
            return _extractor()
    except Exception as exc:
        logger.exception('Unable to load extractor')
        raise HTTPException(503, 'Claim extractor is unavailable. Check the Ollama configuration.') from exc


def run_inference(call):
    try:
        with _inference_lock:
            return call()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception('Inference failed')
        raise HTTPException(503, 'Inference failed. Check server logs and model availability.') from exc


@app.get('/health')
def health():
    return {'status': 'ok', 'models_loaded': bool(_detector.cache_info().currsize),
            'mode': 'inference_only', 'detector': 'threshold_baseline',
            'version': APP_VERSION,
            'ui_built': _UI_INDEX.is_file() and (_UI / 'assets').is_dir()}


@app.post('/claims')
def claims(body: ClaimsRequest, extractor=Depends(get_extractor)):
    r = run_inference(lambda: extractor.extract_with_metadata(body.answer, body.question))
    return {'claims': r.claims, 'extraction_fallback': r.is_fallback,
            'extraction_model': extractor.llm.model}


@app.post('/detect')
def detect(body: DetectRequest, detector=Depends(get_detector)):
    return run_inference(lambda: detector.detect(body.question, body.answer, body.method, body.top_k, evidence=body.evidence))


@app.post('/retrieve')
def retrieve(body: RetrieveRequest, detector=Depends(get_detector)):
    return {'evidence': run_inference(lambda: detector.retrieve(body.claim, body.method, body.top_k))}


@app.post('/verify')
def verify(body: VerifyRequest, detector=Depends(get_detector)):
    return run_inference(lambda: detector.verify(body.claim, [{'text': e} for e in body.evidence]))


@app.get('/metrics')
def metrics():
    root = PROJECT_ROOT / 'experiments'
    paths = set()
    for pattern in ('stage*/metrics*.json', 'baseline_review/*_metrics.json',
                    'ablation_study/*.json', 'v2/metrics_v2.json'):
        paths.update(root.glob(pattern))
    saved = {}
    for p in sorted(paths):
        try:
            saved[p.relative_to(root).as_posix()] = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.warning('Skipping unreadable metrics file: %s', p)
    return {'source': 'saved_experiment_results', 'results': saved}


@app.get('/metrics/summary')
def metrics_summary():
    """Lightweight summary of key metrics for the dashboard."""
    stage5 = PROJECT_ROOT / 'experiments' / 'stage5_learned' / 'metrics.json'
    ablations = PROJECT_ROOT / 'experiments' / 'stage5_learned' / 'metrics_ablations.json'
    decomp = PROJECT_ROOT / 'experiments' / 'stage5_learned' / 'decomposition.json'
    significance = PROJECT_ROOT / 'experiments' / 'stage5_learned' / 'significance.json'

    summary = {'stage5': {}, 'ablations': {}, 'decomposition': {}, 'significance': {}}
    for key, path in [('stage5', stage5), ('ablations', ablations),
                       ('decomposition', decomp), ('significance', significance)]:
        if path.exists():
            try:
                summary[key] = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
    return summary


@app.get('/charts')
def charts():
    """Serve the results dashboard if generated."""
    chart_path = PROJECT_ROOT / 'experiments' / 'charts' / 'results_dashboard.html'
    if chart_path.exists():
        return FileResponse(chart_path, media_type='text/html')
    raise HTTPException(404, 'Charts not generated. Run: python -m scripts.shared.reporting.generate_charts')


# Serve the built React UI from the same origin. API routes remain available without it.
if (_UI / 'assets').is_dir():
    app.mount('/assets', StaticFiles(directory=_UI / 'assets'), name='assets')


@app.get('/', include_in_schema=False)
def home():
    if _UI_INDEX.is_file() and (_UI / 'assets').is_dir():
        return FileResponse(_UI_INDEX, media_type='text/html')
    raise HTTPException(503, 'Frontend build not found. Run: npm run build --prefix frontend')
