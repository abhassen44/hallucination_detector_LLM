"""Cross-encoder reranking over cached retrieval candidates.

Pretrained only: no fine-tuning, no threshold search. Used by the val-only
improvement study, not by the default inference path.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = 'BAAI/bge-reranker-base'


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64,
                 max_length: int = 512, device: str = 'auto'):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = CrossEncoder(model_name, max_length=max_length, device=self.device)

    def score(self, claim: str, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [(claim, text) for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                    show_progress_bar=show_progress)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(self, claim: str, candidates: list[tuple[int, float]],
               corpus, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k candidates after cross-encoder scoring."""
        scores = self.score(claim, [corpus.texts[doc] for doc, _ in candidates])
        order = np.argsort(-scores)[:k]
        return [(candidates[i][0], float(scores[i])) for i in order]
