# Version 5 plan

**Status:** complete. One FEVER test pass has been spent and all eight predeclared
research gates passed. The learned path remains offline; no serving promotion is
claimed.
**Research baseline:** `v4.0` (`ver-4`).
**Integration target:** bring the newer API, UI, and reporting work from `main` into
this v5 branch while preserving the v4 experiment history.

## Goal

Improve claim-level verification where v4 stopped: recognition of insufficient
evidence (NEI), without reducing performance on verifiable claims. Preserve a
separate open-domain evaluation track because the current FEVER-derived corpus
does not provide broad HaluEval coverage.

## Outcome (2026-10-22)

MiniCheck-only grounding features at 9k train were selected on validation. The
robust policy (`tau=0.62`) passed all eight research gates on validation, the
pipeline was frozen, and one test pass was spent.

- Validation: raw macro-F1 **0.70681**, INSUFFICIENT F1 **0.54941**, answerability
  AUROC **0.81957**, verifiable coverage **0.81381** at **0.97048** precision,
  NEI false-answer coverage **0.36336**.
- Test: final-policy macro-F1 **0.73694**, INSUFFICIENT F1 **0.64483**, AUROC
  **0.83809**, verifiable coverage **0.81757** at **0.95500** precision, overall
  coverage **0.66266** at **0.78701** precision, NEI false-answer coverage
  **0.35285**.
- All eight gates passed. NEI answer precision remains 0.85% on the NEI claims the
  policy still answers, so NEI is reduced rather than solved.
- The one test pass was completed after a serialization-only recovery; no test
  metric was observed before the fix, and the frozen pipeline did not change.
- Full report: [`experiments/v5/VALIDATION.md`](../experiments/v5/VALIDATION.md).
- The learned path stays offline; serving promotion is a separate decision.

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
- Update the canonical README/results/status pages to identify v4 as the prior
  completed baseline and v5 as the current completed research version.
- Do not load or serve the learned detector as part of this integration.

### P1 — evidence-sufficiency / NEI model (complete)

- Start from the v4 top-5 reranker and 9k training candidate.
- Compare a separate answerability head and a two-stage answerability-then-verdict
  model against the existing combiner.
- Added pretrained MiniCheck/HHEM grounding features as the new signal; the
  answerability head consumed the grounding block.
- Kept train for fitting and validation for model/threshold selection. The FEVER
  test split stayed out of iteration until the freeze.

### P2 — retrieval coverage and out-of-domain evaluation (not part of this pass)

- Record evidence recall, complete-evidence recall, NEI false-answer rate, and
  retrieval latency alongside verdict metrics.
- Evaluate a broader or user-supplied source corpus separately from FEVER.
- Add an external RAG evaluation (for example, RAGTruth) with response/span metrics
  reported separately from FEVER claim-level metrics.

### P3 — validation gate and final evaluation (complete)

- Compared macro-F1 and per-class F1, with particular attention to INSUFFICIENT.
- Required a validation improvement over the v4 top-5 candidate and the predeclared
  NEI gate at the required verifiable coverage before spending a test pass.
- Froze the selected pipeline, features, thresholds, seeds, and artifact hashes
  before the single FEVER test evaluation.
- Kept the learned path offline; no serving decision was made.

## Out of scope for the integration step

- No further FEVER test evaluation, post-test fitting, threshold tuning, or
  promotion.
- No replacement of the MNLI model without a validation comparison.
- No direct comparison between FEVER claim scores and RAGTruth response/span
  scores.

## Completion checklist

- [x] v4.0 history remains in the branch.
- [x] Latest `main` API/UI/reporting work is integrated.
- [x] README and canonical results describe the actual branch/version status.
- [x] Validation-only grounding ablation and robust policy selection.
- [x] Frozen artifacts and manifest hashed before test.
- [x] One FEVER test pass recorded; no component test passes.
- [x] Tests/build run.
- [ ] Serving promotion (not claimed; learned path remains offline).
