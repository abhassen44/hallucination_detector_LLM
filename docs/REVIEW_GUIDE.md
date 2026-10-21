# Review the extraction outputs

## v2 full pass (current)

The v2 queues are seeded copies of the v1 full queues, so the 37 completed pilot
verdicts are already present:

- `experiments/v2/queues/extraction.jsonl` — 171 rows, 17 seeded, **154 pending**
- `experiments/v2/queues/verification.jsonl` — 103 rows, 20 seeded, **83 pending**

The commands below describe the **optional full human pass**. The current label
artifacts use the LLM proposals for the 237 non-pilot rows; rerun
`scripts.materialize_v2_labels --force` after any human pass to promote the new
verdicts. For the lighter-weight option, fill `experiments/v2/audit_sheet.csv`
(40 blind rows, 10–15 minutes) and score it with `scripts.score_v2_audit`; that
supports a “human-audited sample” claim, not a full human-label claim.

Taxonomy v2 is in `experiments/v2/taxonomy.json`. Verification drops the v1
`nli_failure` catch-all: use `verifier_error` when the needed evidence span is
present and clear but the verdict was wrong, and `ambiguous_evidence` when the
shown evidence cannot settle the claim. The full definitions and precedence rules
are in the taxonomy file. Extraction is **Pass/Fail only** for this pass: the
category and the four diagnostic Booleans are not collected.

```sh
.venv/bin/python -m scripts.review \
  --queue experiments/v2/queues/extraction.jsonl \
  --suggestions experiments/v2/proposals/extraction.jsonl \
  --taxonomy experiments/v2/taxonomy.json --reviewer Abhas

.venv/bin/python -m scripts.review \
  --queue experiments/v2/queues/verification.jsonl \
  --suggestions experiments/v2/proposals/verification.jsonl \
  --taxonomy experiments/v2/taxonomy.json --reviewer Abhas
```

Suggestions are visible for the full pass; pressing Enter accepts the displayed
value and records `suggestion_used`, while typing overrides it. Verification
proposals suggest only a category, not a verdict: type `p` or `f` on every
verification row. After the pass, 20% of rows are re-annotated blind (no
suggestions) to measure intra-annotator agreement.

## v1 extraction queue

The queue is `experiments/stage4_claims/human_review.jsonl`. It contains all 121
heuristically flagged examples plus 50 seeded comparison examples without flags,
covering every split and both answer labels. The three malformed-response
fallbacks appear first and include the original raw response. No human quality
judgments have been assigned automatically.

The blind pilot in `experiments/pilot_v1/` is complete and passed (κ 0.88 Pass/Fail,
0.80 category agreement, one row unannotated; see
`experiments/pilot_v1/PILOT.md`). The full queue below is the next task, and
proposals may be shown now.

For the full pass, work through the queue in order. Read the question, answer and
extracted claims. Write a short observable description in `human_review.open_coding`,
then set `pass_fail_assessment` to `Pass` or `Fail`. Leave categories empty during
the first pass; derive them from the observations rather than guessing root causes.

## Use the review tool

```sh
.venv/bin/python -m scripts.review \
  --queue experiments/stage4_claims/human_review.jsonl \
  --suggestions experiments/stage6_errors/provisional_extraction.jsonl
```

The tool shows one pending row at a time. It backs the queue up to
`human_review.jsonl.bak` before the first write, never offers a row that already
has a judgment, writes atomically, and records progress in `review_progress.json`.
Press Enter to accept a displayed suggestion, or type your own value; typing
nothing leaves the field unset.

`experiments/stage6_errors/provisional_extraction.jsonl` contains Gemma's proposal
for every extraction row (verdict, category, rationale, confidence). It is a
**proposal**, not a label. The reviewer remains responsible for every judgment.

Assess extraction, not the truth of the original answer. A deliberately false
HaluEval answer should be extracted faithfully, without silently correcting it.
Use the question to interpret short answers and pronouns. Check whether the
claims preserve the answer's assertions, cover the factual content, state one
assertion each, and make sense without the original question. An answer such as
"It's hard to tell" may legitimately yield no factual claims.

`fidelity`, `atomicity`, `coverage` and `self_contained` are optional human Boolean
judgments. Leave them null until reviewed. Heuristic flags are screening signals,
not measured errors. Counts overlap: one example may be both short and duplicate.
The flagged queue is deliberately enriched for issues and does not estimate the
population error rate. The 50 comparison examples are also stratified, not a
simple random sample; report sample design alongside any reviewed statistics.

Preserve original IDs and extraction outputs. Save corrections separately rather
than editing test answers, labels, or evidence. Do not feed human test annotations
into fitting or tuning; the Stage 5 fitter reads only the FEVER train split.

Regenerate this queue before annotation with:

```sh
.venv/bin/python -m scripts.prepare_extraction_review
```

The generator refuses to overwrite a queue containing completed human judgments.
For the existing FEVER failure samples, follow the same open-coding-first process.

Workflow reference: [Langfuse error analysis](https://langfuse.com/guides/cookbook/error-analysis-llm-applications).

## Syntax-only proposals and verification errors

`experiments/stage4_claims/json_recovery_proposals.json` contains deterministic
JSON-delimiter repair proposals for all three malformed responses. Every
model-produced word is retained. A fidelity review on 2026-10-21 accepted all three
(`human_accepted: true`, `reviewed_by: assistant (AI review)`) and recorded the
reason in `fidelity_check`. They are still **not applied** to the splits: applying
them would change three HaluEval examples and require regenerating the committed
frozen-rule predictions, which belongs in a separate version.

`experiments/stage5_inference/error_review.jsonl` contains a separate 103-row
verification failure sample, balanced where possible across false positives and
false negatives within each evidence condition. The rows cover 93 unique claims:
10 claims repeat under a second evidence condition, so counts are per row and
suggestions are keyed by `(id, condition)`. Its empty review fields follow the same
workflow. This is a different review task from extraction quality.
`experiments/stage6_errors/provisional_verification.jsonl` holds Gemma's proposed
category (not a verdict) for each of these rows; review them with the same
`--suggestions` flag only after the blind pilot is complete.
