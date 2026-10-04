"""NLI verifier (Premise = evidence, Hypothesis = claim).

Returns probabilities in a fixed order: [entailment, neutral, contradiction],
regardless of how the underlying checkpoint orders its labels.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.datasets.schema import CONTRADICTED, INSUFFICIENT, SUPPORTED
from src.utils.config import load_config

NLI_ORDER = ("entailment", "neutral", "contradiction")
NLI_TO_LABEL = {0: SUPPORTED, 1: INSUFFICIENT, 2: CONTRADICTED}


class NLIModel:
    def __init__(self, model_name: str | None = None, cfg: dict | None = None):
        cfg = cfg or load_config()["nli"]
        self.model_name = model_name or cfg["model"]
        self.batch_size = cfg["batch_size"]
        self.max_length = cfg["max_length"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = cfg.get("fp16", True) and self.device == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        self.col_idx = self._resolve_label_columns()

    def _resolve_label_columns(self) -> list[int]:
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        idx = []
        for name in NLI_ORDER:
            match = [i for i, lab in id2label.items() if lab.startswith(name[:5])]
            if not match:
                raise ValueError(f"{self.model_name}: cannot find '{name}' in {id2label}")
            idx.append(match[0])
        return idx

    @torch.inference_mode()
    def predict(self, premises: list[str], hypotheses: list[str], show_progress: bool = True) -> np.ndarray:
        """Returns array (N, 3) of [P(entail), P(neutral), P(contradict)]."""
        assert len(premises) == len(hypotheses)
        n = len(premises)
        out = np.zeros((n, 3), dtype=np.float32)
        # Sort by length -> far less padding per batch.
        order = sorted(range(n), key=lambda i: len(premises[i]) + len(hypotheses[i]))
        it = range(0, n, self.batch_size)
        for s in tqdm(it, desc="NLI", disable=not show_progress):
            ids = order[s : s + self.batch_size]
            enc = self.tokenizer(
                [premises[i] for i in ids], [hypotheses[i] for i in ids],
                truncation="only_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self.col_idx].cpu().numpy()
            out[ids] = probs
        return out


def probs_to_labels(probs: np.ndarray) -> list[str]:
    return [NLI_TO_LABEL[int(i)] for i in probs.argmax(axis=1)]


def aggregate_max(per_evidence: list[np.ndarray]) -> np.ndarray:
    """Claim-level aggregation over several evidence passages.

    Strongest entailment vs strongest contradiction; neutral is what remains.
    """
    if not per_evidence:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    p = np.stack(per_evidence)
    e, c = p[:, 0].max(), p[:, 2].max()
    if e >= c:
        return np.array([e, 1 - e - min(c, 1 - e), min(c, 1 - e)], dtype=np.float32)
    return np.array([min(e, 1 - c), 1 - c - min(e, 1 - c), c], dtype=np.float32)
