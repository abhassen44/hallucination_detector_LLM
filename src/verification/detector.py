"""Inference-only baseline. No training, fitting, or threshold search occurs here."""
from __future__ import annotations

import numpy as np

from src.datasets.schema import SUPPORTED, CONTRADICTED, INSUFFICIENT
from src.generation.claim_extraction import ClaimExtractor
from src.utils.config import load_config
from src.verification.evidence import acceptance, select_evidence


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
    def __init__(self, extractor=None, retriever=None, nli=None, cfg=None, reranker=None,
                 reviewer=None):
        self.cfg = cfg or load_config()
        self.extractor = extractor or ClaimExtractor()
        self._retriever = retriever
        self._reranker = reranker
        self._reviewer = reviewer
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

    def retrieve(self, claim, method='hybrid', top_k=5, rerank=False):
        if not rerank:
            return self.retriever.search(claim, method=method, k=top_k)
        depth = max(top_k, self.cfg.get('retrieval', {}).get('rerank_depth', 50))
        candidates = self.retriever.search(claim, method=method, k=depth)
        if not candidates:
            return []
        if self._reranker is None:
            from src.retrieval.rerank import Reranker
            self._reranker = Reranker(local_files_only=True)
        scores = np.asarray(self._reranker.score(claim, [e['text'] for e in candidates]))
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError('Reranker returned invalid candidate scores')
        order = np.argsort(-scores, kind='stable')[:top_k]
        return [dict(candidates[i], rerank_score=float(scores[i])) for i in order]

    def verify(self, claim, evidence, evidence_policy='all', grounded_review=False):
        selected = select_evidence(evidence, evidence_policy)
        selected_ids = {id(hit) for hit in selected}
        probs = self.nli.predict([e['text'] for e in evidence], [claim] * len(evidence),
                                 show_progress=False) if evidence else np.empty((0, 3))
        used = [i for i, hit in enumerate(evidence) if id(hit) in selected_ids]
        aggregated = aggregate_probabilities(probs[used])
        thresholds = self.cfg.get('detector', {})
        label = decide(aggregated, thresholds.get('entail_threshold', 0.7),
                       thresholds.get('contra_threshold', 0.95))
        conflict = bool(len(probs) and
                        np.max(probs[:, 0]) >= thresholds.get('entail_threshold', 0.7) and
                        np.max(probs[:, 2]) >= thresholds.get('contra_threshold', 0.95))
        # These are NLI signals, not calibrated confidence in a factual decision.
        result = {'claim': claim, 'label': label, 'evidence_conflict': conflict,
                'evidence_policy': evidence_policy,
                'decision': acceptance(label, has_evidence=bool(evidence), conflict=conflict),
                'signals': dict(zip(('entailment', 'neutral', 'contradiction'), aggregated.tolist())),
                'evidence': [dict(e, used_for_verdict=i in used,
                                  nli=dict(zip(('entailment', 'neutral', 'contradiction'), p.tolist())))
                             for i, (e, p) in enumerate(zip(evidence, probs))]}
        if grounded_review:
            from src.verification.grounded import GroundedVerifier, apply_review, needs_review
            if needs_review(result):
                if self._reviewer is None:
                    self._reviewer = GroundedVerifier()
                return apply_review(result, self._reviewer.review(claim, evidence))
            result['grounded_review'] = {'status': 'not_required'}
        return result

    def detect(self, question, answer, method='hybrid', top_k=5, evidence=None, rerank=False,
               evidence_policy='all', grounded_review=False):
        extracted = self.extractor.extract_with_metadata(answer, question)
        if len(extracted.claims) > 50:
            raise ValueError('Answer produced more than 50 claims; submit a shorter answer.')
        results = [self.verify(c, [{'text': text, 'source': 'provided'} for text in evidence]
                               if evidence is not None else self.retrieve(c, method, top_k, rerank=rerank),
                               evidence_policy=evidence_policy, grounded_review=grounded_review)
                   for c in extracted.claims]
        labels = [r['label'] for r in results]
        overall = (CONTRADICTED if CONTRADICTED in labels else
                   SUPPORTED if labels and all(l == SUPPORTED for l in labels) else INSUFFICIENT)
        reasons = sorted({reason for r in results for reason in r['decision']['reasons']})
        if not results:
            reasons.append('no_factual_claims')
        if extracted.is_fallback:
            reasons.append('extraction_fallback')
        return {'label': overall, 'claims': results, 'extraction_fallback': extracted.is_fallback,
                'decision': {'status': 'abstain' if reasons else 'answer',
                             'reasons': reasons, 'policy': 'evidence_review_v1'},
                'reranked': bool(rerank and evidence is None),
                'evidence_policy': evidence_policy,
                'grounded_review_enabled': grounded_review,
                'method': 'provided' if evidence is not None else method,
                'evidence_source': 'provided' if evidence is not None else 'wikipedia',
                'detector': 'threshold_with_grounded_review' if grounded_review else 'threshold_baseline',
                'extraction_model': self.extractor.llm.model,
                'warnings': ['Sentence splitting was used because claim extraction failed.']
                if extracted.is_fallback else []}
