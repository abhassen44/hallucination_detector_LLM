"""Unified example schema shared by every dataset.

Label design (resolves the 2-vs-3 class mismatch in the plan):
  * ``label``        - 3-way claim-level label: SUPPORTED / CONTRADICTED / INSUFFICIENT.
                       ``None`` when the dataset has no claim-level gold (HaluEval).
  * ``binary_label`` - answer-level: "supported" / "hallucinated". Available for all data.
                       For FEVER, INSUFFICIENT is mapped to "hallucinated" (unsupported claim).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SUPPORTED, CONTRADICTED, INSUFFICIENT = "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"
LABELS = (SUPPORTED, CONTRADICTED, INSUFFICIENT)

BIN_SUPPORTED, BIN_HALLUCINATED = "supported", "hallucinated"


def to_binary(label: str) -> str:
    return BIN_SUPPORTED if label == SUPPORTED else BIN_HALLUCINATED


@dataclass
class Example:
    id: str
    source: str                      # "fever" | "halueval"
    question: str | None             # None for FEVER (claims only)
    answer: str                      # text to verify (FEVER: the claim itself)
    claims: list[str]                # atomic claims (FEVER: [claim]; HaluEval: filled in Phase 3)
    context: list[str]               # gold evidence text (FEVER sentences / HaluEval knowledge)
    label: str | None                # 3-way claim label or None
    binary_label: str                # "supported" | "hallucinated"
    group: str                       # examples sharing a group never cross splits
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(path: str | Path, rows: Iterable[Example | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict() if isinstance(r, Example) else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_examples(path: str | Path) -> list[Example]:
    return [Example(**d) for d in read_jsonl(path)]
