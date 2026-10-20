# Review the extraction outputs

The queue is `experiments/stage4_claims/human_review.jsonl`. It contains all 121
heuristically flagged examples plus 50 seeded comparison examples without flags,
covering every split and both answer labels. The three malformed-response
fallbacks appear first and include the original raw response. No human quality
judgments have been assigned automatically.

Start with the first 30–50 examples. Read the question, answer and extracted
claims. Write a short observable description in `human_review.open_coding`, then
set `pass_fail_assessment` to `Pass` or `Fail`. Leave categories empty during the
first pass; derive them from the observations rather than guessing root causes.

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
into fitting or tuning; training remains deferred.

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
model-produced word is retained. The proposals have not been applied to the
splits or benchmarks; `human_accepted` stays null. Review factual fidelity against
the question and answer before using the proposed claims.

`experiments/stage5_inference/error_review.jsonl` contains a separate 103-example
verification failure sample, balanced where possible across false positives and
false negatives within each evidence condition. Its empty review fields follow
the same workflow. This is a different review task from extraction quality.
