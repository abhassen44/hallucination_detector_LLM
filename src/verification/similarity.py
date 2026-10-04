"""Semantic-similarity verifier (Method A). A weak signal, not proof (plan §8)."""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config


class SimilarityModel:
    def __init__(self, model_name: str | None = None, cfg: dict | None = None):
        cfg = cfg or load_config()["embeddings"]
        self.model_name = model_name or cfg["model"]
        self.batch_size = cfg["batch_size"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )

    def max_similarity(self, claims: list[str], evidence_lists: list[list[str]]) -> np.ndarray:
        """For each claim, max cosine similarity to any of its evidence passages (0 if none)."""
        flat = [e for ev in evidence_lists for e in ev]
        c_emb = self.encode(claims, show_progress=True)
        e_emb = self.encode(flat, show_progress=True) if flat else np.zeros((0, c_emb.shape[1]))
        out, pos = np.zeros(len(claims), dtype=np.float32), 0
        for i, ev in enumerate(evidence_lists):
            if ev:
                out[i] = float((e_emb[pos : pos + len(ev)] @ c_emb[i]).max())
            pos += len(ev)
        return out
