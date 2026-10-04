"""HaluEval (QA subset) loader.

Each source row has a right and a hallucinated answer for the same question.
Both become examples sharing a ``group`` so they always land in the same split
(otherwise the model could "see" the question during training).
HaluEval has no claim-level gold, so ``label`` is None and ``claims`` is filled
later by the claim extractor (Phase 3).
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.datasets.download import download
from src.datasets.schema import BIN_HALLUCINATED, BIN_SUPPORTED, Example, write_jsonl
from src.datasets.splits import split_dataset


def _read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def load_halueval(cfg: dict, raw_dir: Path, splits_dir: Path, seed: int) -> dict[str, list[Example]]:
    hcfg = cfg["datasets"]["halueval"]
    path = download(hcfg["url"], raw_dir / "halueval" / "qa_data.json")
    rows = _read_rows(path)

    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[: hcfg["max_groups"]])

    examples: list[Example] = []
    for i in idx:
        r = rows[i]
        g = f"halueval-qa-{i}"
        ctx = [r["knowledge"].strip()] if r.get("knowledge") else []
        for kind, ans, blab in (
            ("right", r["right_answer"], BIN_SUPPORTED),
            ("hallucinated", r["hallucinated_answer"], BIN_HALLUCINATED),
        ):
            examples.append(Example(
                id=f"{g}-{kind}", source="halueval", question=r["question"].strip(),
                answer=ans.strip(), claims=[], context=ctx, label=None,
                binary_label=blab, group=g, meta={"answer_type": kind},
            ))

    splits = split_dataset(examples, cfg["datasets"]["splits"], seed)
    for split, exs in splits.items():
        for e in exs:
            e.meta["split"] = split
        write_jsonl(splits_dir / f"halueval_{split}.jsonl", exs)
        dist = Counter(e.binary_label for e in exs)
        print(f"[halueval] {split}: {len(exs)} | " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    return splits
