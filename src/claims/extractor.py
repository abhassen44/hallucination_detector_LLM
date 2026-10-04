"""Claim quality analysis and validation (Stage 4 — plan §6).

Wraps the LLM-based ClaimExtractor (src.generation.claim_extraction) with
quality checks that identify potentially degenerate extractions:
  - Too few or too many claims for a given answer
  - Very short claims (< 5 words) that may be fragments
  - Very long claims that likely contain multiple assertions
  - Near-duplicate claims within the same answer

Also provides aggregate statistics for manual inspection and reporting.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# Re-export the core extractor so callers can import from src.claims
from src.generation.claim_extraction import ClaimExtractor  # noqa: F401


# ---------------------------------------------------------------------------
# Quality heuristics
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _has_near_duplicate(claims: list[str], threshold: float = 0.85) -> list[tuple[int, int]]:
    """Find pairs of claims that are near-duplicates via token overlap (Jaccard)."""
    normed = [set(_normalise(c).split()) for c in claims]
    dups = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            if not normed[i] or not normed[j]:
                continue
            jaccard = len(normed[i] & normed[j]) / len(normed[i] | normed[j])
            if jaccard >= threshold:
                dups.append((i, j))
    return dups


# ---------------------------------------------------------------------------
# Per-example quality report
# ---------------------------------------------------------------------------

@dataclass
class ClaimQualityReport:
    """Quality assessment for claims extracted from a single answer."""
    example_id: str
    answer_word_count: int
    num_claims: int
    avg_claim_words: float
    short_claims: list[int]       # indices of claims with < 5 words
    long_claims: list[int]        # indices of claims with > 40 words
    duplicate_pairs: list[tuple[int, int]]
    is_fallback: bool             # True if the LLM failed and we used sentence split

    @property
    def has_issues(self) -> bool:
        return bool(self.short_claims or self.long_claims or self.duplicate_pairs or self.is_fallback)


def assess_claims(
    example_id: str,
    answer: str,
    claims: list[str],
    *,
    short_threshold: int = 5,
    long_threshold: int = 40,
    dup_threshold: float = 0.85,
) -> ClaimQualityReport:
    """Assess the quality of extracted claims for a single example."""
    wcs = [_word_count(c) for c in claims]
    return ClaimQualityReport(
        example_id=example_id,
        answer_word_count=_word_count(answer),
        num_claims=len(claims),
        avg_claim_words=float(np.mean(wcs)) if wcs else 0.0,
        short_claims=[i for i, w in enumerate(wcs) if w < short_threshold],
        long_claims=[i for i, w in enumerate(wcs) if w > long_threshold],
        duplicate_pairs=_has_near_duplicate(claims, dup_threshold),
        is_fallback=False,  # caller should set True if extraction fell back
    )


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

@dataclass
class ClaimExtractionStats:
    """Aggregate statistics over a batch of examples."""
    total_examples: int = 0
    total_claims: int = 0
    examples_with_issues: int = 0
    fallback_count: int = 0
    claims_per_example: list[int] = field(default_factory=list)
    words_per_claim: list[float] = field(default_factory=list)
    short_claim_count: int = 0
    long_claim_count: int = 0
    duplicate_pair_count: int = 0

    def add(self, report: ClaimQualityReport) -> None:
        self.total_examples += 1
        self.total_claims += report.num_claims
        self.claims_per_example.append(report.num_claims)
        if report.num_claims:
            self.words_per_claim.append(report.avg_claim_words)
        if report.has_issues:
            self.examples_with_issues += 1
        if report.is_fallback:
            self.fallback_count += 1
        self.short_claim_count += len(report.short_claims)
        self.long_claim_count += len(report.long_claims)
        self.duplicate_pair_count += len(report.duplicate_pairs)

    def summary(self) -> dict:
        cpe = np.array(self.claims_per_example) if self.claims_per_example else np.zeros(1)
        wpc = np.array(self.words_per_claim) if self.words_per_claim else np.zeros(1)
        return {
            "total_examples": self.total_examples,
            "total_claims": self.total_claims,
            "avg_claims_per_example": float(cpe.mean()),
            "median_claims_per_example": float(np.median(cpe)),
            "min_claims": int(cpe.min()),
            "max_claims": int(cpe.max()),
            "avg_words_per_claim": float(wpc.mean()),
            "examples_with_issues": self.examples_with_issues,
            "issue_rate": self.examples_with_issues / max(self.total_examples, 1),
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / max(self.total_examples, 1),
            "short_claims": self.short_claim_count,
            "long_claims": self.long_claim_count,
            "duplicate_pairs": self.duplicate_pair_count,
            "claims_distribution": dict(Counter(self.claims_per_example)),
        }

    def print_summary(self, label: str = "") -> None:
        s = self.summary()
        tag = f" ({label})" if label else ""
        print(f"\n{'='*60}")
        print(f"Claim Extraction Quality Stats{tag}")
        print(f"{'='*60}")
        print(f"  Examples:          {s['total_examples']}")
        print(f"  Total claims:      {s['total_claims']}")
        print(f"  Avg claims/ex:     {s['avg_claims_per_example']:.2f}")
        print(f"  Median claims/ex:  {s['median_claims_per_example']:.1f}")
        print(f"  Range:             [{s['min_claims']}, {s['max_claims']}]")
        print(f"  Avg words/claim:   {s['avg_words_per_claim']:.1f}")
        print(f"  Fallback rate:     {s['fallback_rate']:.1%} ({s['fallback_count']}/{s['total_examples']})")
        print(f"  Issue rate:        {s['issue_rate']:.1%} ({s['examples_with_issues']}/{s['total_examples']})")
        print(f"    Short claims:    {s['short_claims']}")
        print(f"    Long claims:     {s['long_claims']}")
        print(f"    Duplicate pairs: {s['duplicate_pairs']}")
        print(f"  Claims distribution: {dict(sorted(s['claims_distribution'].items()))}")
        print(f"{'='*60}")
