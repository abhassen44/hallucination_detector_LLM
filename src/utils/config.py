"""Config loading and global seeding."""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    """Resolve a config path relative to project root and ensure parent exists."""
    p = PROJECT_ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int | None = None) -> None:
    seed = seed if seed is not None else load_config()["seed"]
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
