"""Pretrained grounding scorers used only as v5 features.

The scorers are inference-only. They are not fitted on FEVER and are not used by
the serving path. A score is the probability that a claim is supported by one
evidence sentence, in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer


def _device(device: str | None) -> str:
    if device:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _dtype(device: str):
    return torch.float16 if device.startswith('cuda') else torch.float32


def _batches(values: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


class MiniCheckScorer:
    """MiniCheck-Flan-T5-Large sentence-level support scorer."""

    model_id = 'lytang/MiniCheck-Flan-T5-Large'
    support_token_id = 209
    no_support_token_id = 3

    def __init__(self, device: str | None = None, batch_size: int = 4,
                 max_length: int = 512) -> None:
        self.device = _device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype=_dtype(self.device)).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.eval()

    def score_pairs(self, documents: Sequence[str], claims: Sequence[str]) -> np.ndarray:
        if len(documents) != len(claims):
            raise ValueError('documents and claims must have the same length')
        scores: list[float] = []
        for batch in _batches(list(zip(documents, claims)), self.batch_size):
            texts = [
                f'predict: {document}{self.tokenizer.eos_token}{claim}'
                for document, claim in batch
            ]
            inputs = self.tokenizer(
                texts, max_length=self.max_length, truncation=True,
                padding=True, return_tensors='pt').to(self.device)
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].size(0), 1), dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=decoder_input_ids,
                ).logits.squeeze(1)
                token_logits = logits[:, [self.no_support_token_id, self.support_token_id]]
                probabilities = torch.softmax(token_logits.float(), dim=-1)[:, 1]
            scores.extend(probabilities.cpu().numpy().tolist())
        return np.asarray(scores, dtype=np.float32)

    def score_evidence(self, claim: str, evidence: Sequence[str]) -> np.ndarray:
        return self.score_pairs(list(evidence), [claim] * len(evidence))


class HHEMScorer:
    """Vectara HHEM-2.1-Open premise/hypothesis consistency scorer."""

    model_id = 'vectara/hallucination_evaluation_model'

    def __init__(self, device: str | None = None, batch_size: int = 8) -> None:
        self.device = _device(device)
        self.batch_size = batch_size
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=True,
            torch_dtype=_dtype(self.device)).to(self.device)
        self.model.eval()

    def score_pairs(self, documents: Sequence[str], claims: Sequence[str]) -> np.ndarray:
        if len(documents) != len(claims):
            raise ValueError('documents and claims must have the same length')
        scores: list[float] = []
        for batch in _batches(list(zip(documents, claims)), self.batch_size):
            pairs = [(str(document), str(claim)) for document, claim in batch]
            with torch.inference_mode():
                probabilities = self.model.predict(pairs)
            scores.extend(probabilities.float().cpu().numpy().tolist())
        return np.asarray(scores, dtype=np.float32)

    def score_evidence(self, claim: str, evidence: Sequence[str]) -> np.ndarray:
        return self.score_pairs(list(evidence), [claim] * len(evidence))


@dataclass(frozen=True)
class GroundingSummary:
    maximum: float
    mean: float
    top2_mean: float
    margin: float
    count_above_half: int
    count_below_twenty: int


def summarize_scores(scores: Sequence[float]) -> GroundingSummary:
    values = np.asarray(list(scores), dtype=np.float32)
    if values.size == 0:
        return GroundingSummary(0.0, 0.0, 0.0, 0.0, 0, 0)
    ordered = np.sort(values)[::-1]
    top2 = float(ordered[:2].mean()) if ordered.size > 1 else float(ordered[0])
    return GroundingSummary(
        maximum=float(ordered[0]),
        mean=float(values.mean()),
        top2_mean=top2,
        margin=float(ordered[0] - (ordered[1] if ordered.size > 1 else 0.0)),
        count_above_half=int((values >= 0.5).sum()),
        count_below_twenty=int((values < 0.2).sum()),
    )
