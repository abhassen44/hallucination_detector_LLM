"""Inference-only baseline. No training, fitting, or threshold search occurs here."""
from __future__ import annotations

import numpy as np

from src.datasets.schema import SUPPORTED, CONTRADICTED, INSUFFICIENT
from src.generation.claim_extraction import ClaimExtractor
from src.utils.config import load_config


def aggregate_probabilities(probabilities):
    """Same max-evidence rule as the Stage 3 baseline, in E/N/C order."""
    if not len(probabilities):
        return np.array([0., 1., 0.])
    p = np.asarray(probabilities)
    e, c = float(p[:, 0].max()), float(p[:, 2].max())
    if e >= c:
        c = min(c, 1 - e)
    else:
        e = min(e, 1 - c)
    return np.array([e, max(0., 1 - e - c), c])


def decide(p, entail=0.7, contra=0.95):
    e, _, c = p
    if e >= entail and e >= c:
        return SUPPORTED
    if c >= contra and c > e:
        return CONTRADICTED
    return INSUFFICIENT


class Detector:
    def __init__(self, extractor=None, retriever=None, nli=None, cfg=None):
        self.cfg = cfg or load_config()
        self.extractor = extractor or ClaimExtractor()
        self._retriever = retriever
        if nli is None:
            from src.verification.nli import NLIModel
            nli = NLIModel(cfg=self.cfg['nli'])
        self.nli = nli

    @property
    def retriever(self):
        if self._retriever is None:
            from src.retrieval.hybrid import Retriever
            self._retriever = Retriever(self.cfg)
        return self._retriever

    def retrieve(self, claim, method='hybrid', top_k=5):
        return self.retriever.search(claim, method=method, k=top_k)

    def verify(self, claim, evidence):
        probs = self.nli.predict([e['text'] for e in evidence], [claim] * len(evidence),
                                 show_progress=False) if evidence else np.empty((0, 3))
        aggregated = aggregate_probabilities(probs)
        thresholds = self.cfg.get('detector', {})
        label = decide(aggregated, thresholds.get('entail_threshold', 0.7),
                       thresholds.get('contra_threshold', 0.95))
        conflict = bool(len(probs) and
                        np.max(probs[:, 0]) >= thresholds.get('entail_threshold', 0.7) and
                        np.max(probs[:, 2]) >= thresholds.get('contra_threshold', 0.95))
        # These are NLI signals, not calibrated confidence in a factual decision.
        return {'claim': claim, 'label': label, 'evidence_conflict': conflict,
                'signals': dict(zip(('entailment', 'neutral', 'contradiction'), aggregated.tolist())),
                'evidence': [dict(e, nli=dict(zip(('entailment', 'neutral', 'contradiction'), p.tolist())))
                             for e, p in zip(evidence, probs)]}

    def detect(self, question, answer, method='hybrid', top_k=5, evidence=None):
        extracted = self.extractor.extract_with_metadata(answer, question)
        if len(extracted.claims) > 50:
            raise ValueError('Answer produced more than 50 claims; submit a shorter answer.')
        results = [self.verify(c, [{'text': text, 'source': 'provided'} for text in evidence]
                               if evidence is not None else self.retrieve(c, method, top_k))
                   for c in extracted.claims]
        labels = [r['label'] for r in results]
        overall = (CONTRADICTED if CONTRADICTED in labels else
                   SUPPORTED if labels and all(l == SUPPORTED for l in labels) else INSUFFICIENT)
        return {'label': overall, 'claims': results, 'extraction_fallback': extracted.is_fallback,
                'method': 'provided' if evidence is not None else method,
                'evidence_source': 'provided' if evidence is not None else 'wikipedia',
                'detector': 'threshold_baseline',
                'extraction_model': self.extractor.llm.model,
                'warnings': ['Sentence splitting was used because claim extraction failed.']
                if extracted.is_fallback else []}
