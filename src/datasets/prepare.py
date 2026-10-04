"""Stage 1 entry point.

    python -m src.datasets.prepare --dataset all
    python -m src.datasets.prepare --dataset halueval
"""
from __future__ import annotations

import argparse

from src.utils.config import PROJECT_ROOT, load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["all", "fever", "halueval"], default="all")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    splits = PROJECT_ROOT / cfg["paths"]["splits"]
    for p in (raw, processed, splits):
        p.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "halueval"):
        from src.datasets.halueval import load_halueval
        load_halueval(cfg, raw, splits, seed)
    if args.dataset in ("all", "fever"):
        from src.datasets.fever import load_fever
        load_fever(cfg, raw, processed, splits, seed)


if __name__ == "__main__":
    main()
