"""Group-aware train/val/test splitting."""
from __future__ import annotations

import random
from collections import defaultdict

from src.datasets.schema import Example


def split_dataset(examples: list[Example], ratios: dict[str, float], seed: int) -> dict[str, list[Example]]:
    """Split by ``group`` so related examples never leak across splits."""
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "split ratios must sum to 1"
    groups: dict[str, list[Example]] = defaultdict(list)
    for e in examples:
        groups[e.group].append(e)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    out: dict[str, list[Example]] = {}
    start, names = 0, list(ratios)
    for i, name in enumerate(names):
        end = len(keys) if i == len(names) - 1 else start + round(len(keys) * ratios[name])
        out[name] = [e for k in keys[start:end] for e in groups[k]]
        start = end
    return out
