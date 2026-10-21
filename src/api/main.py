"""Run with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000."""
from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
Method = Literal['bm25', 'dense', 'hybrid']
app = FastAPI(title='Claim-Level Hallucination Detector', version='0.2.0')
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


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
        raise HTTPException(503, 'Models or retrieval indexes are unavailable. See docs/RUNNING.md.') from exc


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
            'mode': 'inference_only', 'detector': 'threshold_baseline'}


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
    for pattern in ('stage*/metrics*.json', 'baseline_review/*_metrics.json'):
        paths.update(root.glob(pattern))
    saved = {}
    for p in sorted(paths):
        try:
            saved[p.relative_to(root).as_posix()] = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            logger.warning('Skipping unreadable metrics file: %s', p)
    return {'source': 'saved_experiment_results', 'results': saved}


# Serve the built React UI from the same origin. API routes remain available without it.
_ui = PROJECT_ROOT / 'frontend' / 'dist'
if _ui.is_dir():
    app.mount('/assets', StaticFiles(directory=_ui / 'assets'), name='assets')

    @app.get('/', include_in_schema=False)
    def home():
        return FileResponse(_ui / 'index.html')
