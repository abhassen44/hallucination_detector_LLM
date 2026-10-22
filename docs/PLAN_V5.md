# Version 5 plan

**Status:** application integration complete; v5 research is planned. No v5 model
result or serving promotion is claimed.
**Research baseline:** `v4.0` (`ver-4`).
**Integration target:** bring the newer API, UI, and reporting work from `main` into
this v5 branch while preserving the v4 experiment history.

## Goal

Improve claim-level verification where v4 stopped: recognition of insufficient
evidence (NEI), without reducing performance on verifiable claims. Preserve a
separate open-domain evaluation track because the current FEVER-derived corpus
does not provide broad HaluEval coverage.

## Baseline and evidence

- v1/v2 learned hybrid detector: FEVER test macro-F1 0.702; this is the existing
  reported test result, not a v5 comparison target on validation.
- v4 reranker plus refit: validation macro-F1 0.69296 and INSUFFICIENT F1 0.53559.
- v4's 9k training arm reached NEI 0.408 at 0.800 verifiable coverage; joint
  threshold search found no feasible point under the 0.40 NEI gate. v4 stopped
  before its variant test pass.
- Existing error decomposition points to missing/insufficient retrieved evidence
  as a major remaining error source. NLI is load-bearing in the ablation results.

## Work packages

### P0 — integrate application and research tracks

- Integrate the latest `main` API, UI, and ablation/reporting changes into `ver-5`.
- Keep the v4 report, protocol artifacts, and frozen serving decision intact.
- Update the canonical README/results/status pages to identify v4 as the latest
  completed research version and v5 as active work.
- Do not load or serve the learned detector as part of this integration.

### P1 — evidence-sufficiency / NEI model

- Start from the v4 top-5 reranker and 9k training candidate.
- Compare a separate answerability head and a two-stage answerability-then-verdict
  model against the existing combiner.
- Add training-only hard negatives by deleting relevant evidence or swapping in
  evidence from an unrelated page; keep claim IDs grouped across derived examples.
- Use train for fitting and validation for model/threshold selection. Keep the
  existing FEVER test split out of iteration.

### P2 — retrieval coverage and out-of-domain evaluation

- Record evidence recall, complete-evidence recall, NEI false-answer rate, and
  retrieval latency alongside verdict metrics.
- Evaluate a broader or user-supplied source corpus separately from FEVER.
- Add an external RAG evaluation (for example, RAGTruth) with response/span metrics
  reported separately from FEVER claim-level metrics.

### P3 — validation gate and final evaluation

- Compare macro-F1 and per-class F1, with particular attention to INSUFFICIENT.
- Require a validation improvement over the v4 top-5 candidate and the predeclared
  NEI gate at the required verifiable coverage before spending a test pass.
- Freeze the selected pipeline, features, thresholds, seeds, and artifact hashes
  before one final FEVER test evaluation. Report latency and memory for reranking.
- Keep the learned path offline unless every serving gate passes and a separate
  serving decision is recorded.

## Out of scope for the integration step

- No new FEVER test evaluation, model fitting, threshold tuning, or promotion.
- No replacement of the MNLI model without a validation comparison.
- No direct comparison between FEVER claim scores and RAGTruth response/span
  scores.

## Completion checklist

- [x] v4.0 history remains in the branch.
- [x] Latest `main` API/UI/reporting work is integrated.
- [x] README and canonical results describe the actual branch/version status.
- [ ] Tests/build are run only when explicitly requested.
