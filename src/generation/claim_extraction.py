"""Atomic Claim Extraction (Phase 3 / Stage 4 of plan.md).

Decomposes an answer (or text) into atomic, standalone, verifiable factual claims.
Each claim:
  1. Contains exactly ONE factual assertion (atomic).
  2. Is self-contained: resolves pronouns (e.g. 'it', 'he') to the explicit entity.
  3. Omits conversational fluff ('Sure!', 'In my opinion', 'Based on the context').
  4. Contextualises short entity answers (e.g. Question: 'Who won in 2010?' Answer: 'Spain'
     -> Claim: 'Spain won in 2010.').

Usage:
    python -m src.generation.claim_extraction --split all
    python -m src.generation.claim_extraction --split test --limit 50 --inspect 20
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.datasets.schema import Example, load_examples, write_jsonl
from src.generation.llm import OllamaLLM
from src.utils.config import PROJECT_ROOT, load_config, set_seed

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a precise factual claim extraction system.
Your job is to deconstruct an answer into a list of atomic, self-contained factual claims.
Each claim must:
- Be an independent, complete grammatical sentence.
- State exactly ONE factual proposition that can be verified as true or false.
- Fully resolve all pronouns and entities (do NOT use 'he', 'she', 'it', 'they' without stating the subject).
- Never include filler phrases, opinions, or hedges.
- If given a Question and a short Answer, turn the answer into a complete factual statement addressing the question.

Output format: Return ONLY a valid JSON array of strings, e.g. ["Claim 1.", "Claim 2."]."""


def _fallback_split(text: str) -> list[str]:
    """Fallback if LLM is unavailable or fails JSON: simple sentence boundary split."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    clean = [s.strip() for s in sents if len(s.strip()) > 3]
    return clean or [text.strip()]


class ClaimExtractor:
    def __init__(self, llm: OllamaLLM | None = None, max_workers: int = 4):
        self.llm = llm or OllamaLLM()
        self.max_workers = max_workers

    def _build_prompt(self, text: str, question: str | None = None) -> str:
        if question and question.strip():
            return (
                f"Question: {question.strip()}\n"
                f"Answer: {text.strip()}\n\n"
                f"Extract all atomic, standalone factual claims made by the answer in the context of the question.\n"
                f"JSON array of claims:"
            )
        return (
            f"Statement: {text.strip()}\n\n"
            f"Extract all atomic, standalone factual claims.\n"
            f"JSON array of claims:"
        )

    def extract_claims(self, text: str, question: str | None = None, use_cache: bool = True) -> list[str]:
        """Extract atomic claims for a single text.

        Returns a list of claim strings.  Falls back to sentence-split on
        LLM failure or empty results.
        """
        text = text.strip()
        if not text:
            return []
        prompt = self._build_prompt(text, question)
        try:
            res = self.llm.generate_json(prompt, system=EXTRACTION_SYSTEM_PROMPT, use_cache=use_cache)
            claims = self._parse_response(res)
            if claims:
                return claims
        except Exception as e:
            logger.debug("LLM claim extraction failed: %s", e)
        logger.debug("Falling back to sentence split for: %s", text[:80])
        return _fallback_split(text)

    @staticmethod
    def _parse_response(res) -> list[str]:
        """Extract claim strings from various LLM response formats."""
        # Direct list: ["claim1", "claim2"]
        if isinstance(res, list):
            return [str(c).strip() for c in res if str(c).strip()]
        # Dict with "claims" key: {"claims": [...]}
        if isinstance(res, dict):
            for key in ("claims", "facts", "statements", "propositions"):
                if key in res and isinstance(res[key], list):
                    return [str(c).strip() for c in res[key] if str(c).strip()]
        return []

    def extract_batch(
        self,
        items: list[tuple[str, str | None]],
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> list[list[str]]:
        """Batch extraction with concurrent worker threads."""
        n = len(items)
        results: list[list[str] | None] = [None] * n

        def _task(idx: int, t: str, q: str | None):
            return idx, self.extract_claims(t, q, use_cache=use_cache)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_task, i, t, q) for i, (t, q) in enumerate(items)]
            it = as_completed(futures)
            if show_progress:
                it = tqdm(it, total=n, desc="Extracting claims")
            for f in it:
                idx, claims = f.result()
                results[idx] = claims

        return [r if r is not None else [] for r in results]


def process_dataset(
    split: str,
    extractor: ClaimExtractor,
    dataset: str = "halueval",
    limit: int | None = None,
    save_in_place: bool = True,
) -> list[Example]:
    """Run claim extraction on a dataset split and optionally save results.

    For FEVER, claims are already single-sentence; this adds them to the
    claims list if missing.  For HaluEval the LLM decomposes multi-sentence
    answers into atomic claims.
    """
    splits_dir = PROJECT_ROOT / "data" / "splits"
    path = splits_dir / f"{dataset}_{split}.jsonl"
    if not path.exists():
        print(f"[claim_extraction] Skipping {path} (not found)")
        return []

    exs = load_examples(path)
    if limit:
        exs = exs[:limit]

    if dataset == "fever":
        # FEVER examples are single claims already; ensure .claims is populated
        for e in exs:
            if not e.claims:
                e.claims = [e.answer]
        print(f"[claim_extraction] FEVER {split}: {len(exs)} examples, claims = answer (no decomposition needed)")
    else:
        # HaluEval: decompose with LLM
        items = [(e.answer, e.question) for e in exs]
        print(f"[claim_extraction] Processing {len(items)} examples from {dataset} {split}...")
        claims_list = extractor.extract_batch(items, show_progress=True)
        for e, claims in zip(exs, claims_list):
            e.claims = claims

    if save_in_place:
        # For HaluEval: update the split file with extracted claims
        # For FEVER: only update if claims were missing
        write_jsonl(path, exs)
        print(f"[claim_extraction] Updated {path} with extracted claims.")
    return exs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Extract atomic claims from HaluEval answers using LLM."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test",
        help="Which split to process (default: test)",
    )
    parser.add_argument(
        "--dataset", choices=["halueval", "fever", "all"], default="halueval",
        help="Which dataset to process (default: halueval)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent LLM workers")
    parser.add_argument("--limit", type=int, default=None, help="Process first N examples only")
    parser.add_argument("--inspect", type=int, default=10, help="Print first N examples")
    args = parser.parse_args()

    set_seed()
    llm = OllamaLLM()
    extractor = ClaimExtractor(llm, max_workers=args.workers)

    # Fix: --split all now correctly includes all three splits
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    datasets = ["halueval", "fever"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        for s in splits:
            exs = process_dataset(s, extractor, dataset=ds, limit=args.limit)
            if args.inspect > 0 and exs:
                print(f"\n--- Sample Inspections ({ds}/{s}) ---")
                for e in exs[: args.inspect]:
                    print(f"ID: {e.id} ({e.binary_label})")
                    print(f"  Q: {e.question}")
                    print(f"  A: {e.answer}")
                    print(f"  Claims ({len(e.claims)}):")
                    for c in e.claims:
                        print(f"    - {c}")
                    print()


if __name__ == "__main__":
    main()
