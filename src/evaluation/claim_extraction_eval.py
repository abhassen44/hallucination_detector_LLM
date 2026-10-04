"""Stage 4 — Claim Extraction Evaluation & Quality Report.

Runs LLM-based claim extraction on HaluEval splits, computes quality statistics,
saves metrics to experiments/stage4_claims/, and prints a summary.

FEVER claims are single-sentence by construction, so only HaluEval needs
LLM decomposition.  This script:
  1. Extracts claims for the requested HaluEval splits (cached via SQLite)
  2. Validates claim quality (short/long/duplicate/fallback detection)
  3. Computes per-split and per-label statistics
  4. Saves JSON metrics and writes updated splits with .claims populated

Usage:
    python -m src.evaluation.claim_extraction_eval
    python -m src.evaluation.claim_extraction_eval --split test --limit 50
    python -m src.evaluation.claim_extraction_eval --split all --save-report
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime

from src.claims.extractor import (
    ClaimExtractionStats,
    ClaimExtractor,
    assess_claims,
)
from src.datasets.schema import load_examples
from src.generation.claim_extraction import process_dataset
from src.generation.llm import OllamaLLM
from src.utils.config import PROJECT_ROOT, load_config, set_seed

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "experiments" / "stage4_claims"


def _evaluate_split(
    split: str,
    extractor: ClaimExtractor,
    limit: int | None = None,
    inspect_n: int = 10,
) -> dict:
    """Extract claims, validate quality, return metrics dict."""
    exs = process_dataset(split, extractor, dataset="halueval", limit=limit)
    if not exs:
        return {}

    stats_all = ClaimExtractionStats()
    stats_by_label: dict[str, ClaimExtractionStats] = {}
    issues: list[dict] = []

    for e in exs:
        report = assess_claims(e.id, e.answer, e.claims)
        stats_all.add(report)

        label = e.binary_label
        if label not in stats_by_label:
            stats_by_label[label] = ClaimExtractionStats()
        stats_by_label[label].add(report)

        if report.has_issues:
            issues.append({
                "id": e.id,
                "binary_label": e.binary_label,
                "answer": e.answer[:200],
                "num_claims": report.num_claims,
                "short_claims": report.short_claims,
                "long_claims": report.long_claims,
                "duplicate_pairs": report.duplicate_pairs,
                "is_fallback": report.is_fallback,
            })

    stats_all.print_summary(f"HaluEval {split}")

    # Per-label breakdown
    for label, st in sorted(stats_by_label.items()):
        st.print_summary(f"HaluEval {split} — {label}")

    # Print sample extractions
    if inspect_n > 0 and exs:
        print(f"\n--- Sample Extractions (HaluEval {split}, first {min(inspect_n, len(exs))}) ---")
        for e in exs[:inspect_n]:
            print(f"[{e.binary_label:>13s}] {e.id}")
            print(f"  Q: {e.question}")
            print(f"  A: {e.answer}")
            print(f"  Claims ({len(e.claims)}):")
            for i, c in enumerate(e.claims):
                print(f"    {i+1}. {c}")
            print()

    result = {
        "split": split,
        "n_examples": len(exs),
        "overall": stats_all.summary(),
        "by_label": {k: v.summary() for k, v in sorted(stats_by_label.items())},
        "n_issues": len(issues),
    }
    if issues:
        result["sample_issues"] = issues[:20]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 evaluation: claim extraction quality on HaluEval."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test",
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent LLM workers")
    parser.add_argument("--limit", type=int, default=None, help="Process first N examples only")
    parser.add_argument("--inspect", type=int, default=10, help="Print first N examples")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report to experiments/")
    args = parser.parse_args()

    set_seed()
    cfg = load_config()
    llm = OllamaLLM()
    extractor = ClaimExtractor(llm, max_workers=args.workers)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    results: dict = {"splits": {}, "timestamp": datetime.now().isoformat(timespec="seconds")}
    for s in splits:
        result = _evaluate_split(s, extractor, limit=args.limit, inspect_n=args.inspect)
        if result:
            results["splits"][s] = result

    results["llm_model"] = llm.model

    # Save metrics
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved → {out_path}")

    # Summary table across splits
    if len(results["splits"]) > 1:
        print("\n=== Cross-split Summary ===")
        print(f"{'split':8s} {'n':>6s} {'claims':>7s} {'avg/ex':>7s} {'fallback%':>10s} {'issues%':>9s}")
        for s, r in results["splits"].items():
            o = r["overall"]
            print(
                f"{s:8s} {o['total_examples']:6d} {o['total_claims']:7d} "
                f"{o['avg_claims_per_example']:7.2f} {o['fallback_rate']:10.1%} "
                f"{o['issue_rate']:9.1%}"
            )


if __name__ == "__main__":
    main()
