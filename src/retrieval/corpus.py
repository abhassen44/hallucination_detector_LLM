"""Sentence-level retrieval corpus.

Retrieval unit = one Wikipedia sentence, rendered as ``"<Title>: <sentence>"`` (same
format as the gold context in the FEVER splits). Keeping the title is important: many
FEVER claims are only resolvable through the page entity name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.datasets.schema import read_jsonl
from src.utils.config import PROJECT_ROOT


@dataclass
class SentenceCorpus:
    texts: list[str]                 # "Title: sentence"
    keys: list[tuple[str, int]]      # (page_id, sent_id)

    def __len__(self) -> int:
        return len(self.texts)

    @classmethod
    def from_fever(cls, path: Path | None = None) -> "SentenceCorpus":
        path = path or PROJECT_ROOT / "data" / "processed" / "fever_corpus.jsonl"
        texts, keys = [], []
        for page in read_jsonl(path):
            for sid, sent in enumerate(page["sentences"]):
                if sent:
                    texts.append(f"{page['title']}: {sent}")
                    keys.append((page["page_id"], sid))
        return cls(texts, keys)
