# v2 proposals

P1 regenerates the provisional LLM categories under taxonomy v2 here:

- `extraction.jsonl` — keyed by `id`; Pass/Fail plus an optional cause.
- `verification.jsonl` — keyed by `(id, condition)`.

The v1 proposals in `experiments/stage6_errors/` are frozen and must not be
overwritten: taxonomy v2 renames `nli_failure` to `verifier_error`, so a v1
proposal cannot be mapped onto v2 without losing information.

Proposal files are inputs to `scripts.shared.review.review --suggestions`, not labels. The
reviewer sees them, accepts or overrides them, and the choice is recorded in
`review_meta.suggestion_used`.
