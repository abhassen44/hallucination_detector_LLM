# Current results and improvement priorities

Reviewed 2026-10-22 on `feat/ver-2-work`, based on `origin/feature/ver-2`
at `e7f0af5`. This review reads existing predictions and caches only. No model
training, fitting, threshold search, new inference, or new test pass was performed.
Historical results and serving behavior are unchanged.

Implementation follow-up: see `experiments/improvements/REPORT.md` for the
subsequent non-training changes and validation-only evidence-selection results.
The original baseline remains available as the default; new selection and
reranking options are experimental.

## Current performance

| System / evaluation | Current result | Interpretation |
|---|---|---|
| FEVER learned hybrid, 1,998 test claims | Accuracy 70.82%; macro-F1 0.7020 | Better than fixed-rule macro-F1 0.6328; remains offline |
| FEVER v2 calibration | ECE 0.0280 → 0.0228; Brier 0.39656 → 0.39725 | Confidence changes; class predictions and F1 do not |
| FEVER v2 selective policy | 73.55% accuracy among accepted predictions at 72.47% acceptance | Falls short of the original 90% overall precision goal |
| HaluEval provided knowledge, claim argmax | Answer accuracy 73.33% | Useful evidence of the value of source-grounded verification |
| HaluEval Wikipedia retrieval, fixed rule | Answer accuracy 50.50%; 595/600 flagged | Essentially the always-hallucinated baseline on this balanced dataset |

The API still serves the fixed 0.7/0.95 rule. The learned results describe an
offline experiment, not the running application's accuracy. The learned combiner,
calibrator, and coverage flag were fitted in the branch's historical experiments;
none was fitted during this review.

## Findings that change the priorities

### 1. NEI errors and the acceptance policy are the main FEVER problem

Saved test predictions reproduce the published accuracy, macro-F1, and selected
accuracy. Before the policy, 322 of 666 gold INSUFFICIENT (NEI) claims receive
SUPPORTED or CONTRADICTED: 322 of the model's 583 errors (55.2%).

The policy accepts 321 NEI claims, of which only four are correct. It rejects
340 of the 344 correct NEI predictions while retaining 317 of the 322 incorrect
ones. Those 317 cases account for 82.8% of all 383 accepted errors. The same
pattern exists on validation: 154 correct NEI predictions rejected, 172 wrong
ones retained. This is not solved by merely increasing the NEI threshold.

The current `verifiable_precision` calculation conditions on **gold** being
SUPPORTED/CONTRADICTED. That is a legitimate diagnostic but is unavailable as a
filter at inference time. It excludes the NEI false assertions that dominate
accepted errors. Likewise, `per_class[*].precision` and `nei_precision` in
`src/evaluation/selective.py` are accuracy within selected **gold-class** subsets,
not standard precision for a predicted class.

| Accepted predicted verdict | Correct / accepted, test | Actual precision |
|---|---:|---:|
| SUPPORTED | 584 / 711 | 82.14% |
| CONTRADICTED | 477 / 731 | 65.25% |
| INSUFFICIENT | 4 / 6 | 66.67% (very small sample) |

Thus “94.14% verifiable precision” must not be presented as the reliability of
supported/contradicted verdicts for arbitrary incoming claims. On validation,
accepted predicted-class precision is similarly 82.10% supported and 64.29%
contradicted.

**Next:** distinguish a factual verdict (`INSUFFICIENT` included) from a separate
operational `abstain` decision. Evaluate precision by *predicted* class, false
assertions on gold NEI, acceptance, and overall accepted accuracy together.
Design a new acceptance policy on development data under these objectives;
do not adjust frozen v2 thresholds using these test results.

### 2. FEVER evidence utilization needs attention before retrieval-only tuning

The earlier decomposition calls recoverable abstentions an evidence/retrieval
loss, but an oracle-correct claim does not prove retrieval failed. Joining saved
verdicts to actual gold-evidence coverage gives a more specific diagnosis:

| Verifiable claims predicted INSUFFICIENT | Validation | Test |
|---|---:|---:|
| Complete gold set already in top five | 88 | 154 |
| Complete gold set missing from top five | 25 | 41 |
| Total | 113 | 195 |

About 78–79% of these false-INSUFFICIENT cases have complete retrieved gold
evidence. On test, 203 of all 261 verifiable classification errors have complete
gold evidence in the top five (154 insufficient + 49 wrong factual verdicts).
This does not isolate NLI versus the learned combiner, and a missing annotated
set does not prove no alternative evidence exists. It does show that assigning
most recoverable abstentions to missing retrieval overstates that cause.

**Next:** on development cases with complete gold sets, compare sentence NLI,
relevance/reranker scores, conflicting distractors, and the final combiner output.
Separate single-sentence from multi-sentence evidence requirements. Test small,
relevant evidence sets and relevance-aware aggregation instead of independent
max-entailment/max-contradiction over every hit. These are hypotheses, not measured
gains. Keep `INSUFFICIENT` when evidence does not address the claim: a different
attribute or an omitted fact is not automatically a contradiction.

### 3. Two rank features carry no useful rank information

`row_features` computes `e_idx = argmax(e)` and then `_rank(e, e_idx)`;
the maximum's rank in its own sorted values is always one for a unique maximum.
The same occurs for contradiction. All 1,998 test rows have both ranks equal to
one. Validation has 998 unique maxima per feature and one tied maximum; ties
do not rescue the intended retrieval-rank signal.

**Next:** version the feature definition so the intended retrieval positions are
`e_idx + 1` and `c_idx + 1`, with a test where the strongest NLI evidence occurs
away from the first hit. Do not silently change features underneath saved model
coefficients. Measuring any gain requires a later authorized refit and development
evaluation; this review does not change them or promise a gain.

### 4. HaluEval needs broader evidence coverage and a clearer answer-level task

The 247,848-sentence FEVER corpus is narrow for HaluEval. Provided-knowledge
verification reaches 73.33% accuracy versus 50.50% with Wikipedia retrieval,
although this comparison changes both evidence source and, in the best row,
decision rule. Provided claims with the same fixed rule still reach 72.0%.

**Next:** prioritize the provided-source workflow for grounded checks. For open
domain use, add a broader, versioned source corpus and measure evidence coverage
separately from verdict accuracy. Keep source provenance and reproducible
snapshots. Expansion can add distracting evidence, so measure NEI false
assertions as well as retrieval recall.

Do not equate missing retrieval evidence with proven hallucination in the user
interface. HaluEval's binary benchmark can retain its documented mapping, while
the application distinguishes unsupported from contradicted. Also track answer
completeness/relevance separately: an answer can contain true claims yet omit
what the question asks (the existing user review notes case `6814`). HaluEval
does not provide gold labels for individual extracted claims.

## Recommended order of work

| Priority | Work | Evidence / acceptance criterion | Training needed? |
|---|---|---|---|
| 1 | Correct metric names and add predicted-class reliability plus explicit abstention reporting | Reproduce existing outputs and expose NEI false assertions without hiding them behind the gold-verifiable subset | No |
| 2 | Prepare development diagnostics for complete-evidence failures; evaluate relevance-aware evidence selection | Reduce false assertions and false INSUFFICIENT on development data without sacrificing per-class reliability; report latency | No new training for pretrained inference experiments |
| 3 | Test pretrained cross-encoder reranking as a candidate | Existing hybrid fixed-rule validation F1 0.5849 → 0.6076; strict R@5 0.8964 → 0.9174. Compare using the same decision path; measure cost and latency | No for fixed-rule study; a fair learned-path refit needs authorization |
| 4 | Improve HaluEval evidence coverage and source-grounded workflow | Beat the 50.5% retrieval baseline with genuine supported/hallucinated discrimination and measured source coverage | No model training required; indexing is needed |
| 5 | Repair rank features under a new feature schema, then revisit learned policy/calibration | New development results against frozen baseline; never load old coefficients against altered feature meanings | Later refitting required; deferred |
| 6 | Validate the failure taxonomy with humans | Existing 37 human / 237 silver labels and assistant-estimated audit cannot establish population failure rates | No |

Reranking is promising for the fixed-rule path, but its reported gains must not
be added to the learned model's F1: that combination has not been demonstrated.

## Low-priority directions and evaluation safeguards

- Further weighted-RRF tuning has little evidence of value: best validation
  macro-F1 improvement is only 0.0016, smaller than the reported bootstrap-refit
  validation standard deviation of about 0.0057. That standard deviation is a
  scale reference, not a paired significance test of the RRF change.
- Blindly adding neighboring sentences already worsened validation fixed-rule
  F1 for every retrieval method. Prefer selective evidence assembly.
- Temperature scaling preserves argmax, so it cannot improve classification
  F1. The small Brier increase is statistically unresolved; retain the original
  failed gate rather than relaxing it after seeing test. Defer new calibrator
  fitting and use separate development/calibration data in a future protocol.
- The coverage flag is diagnostic, not a demonstrated accuracy improvement.
  Its AUROC drops from 0.774 validation to 0.734 test; keep it offline.
- Validation has already supported many choices. Use fresh development data or
  nested development splits for the next iteration, freeze one candidate, then
  evaluate on a fresh held-out set. This review is post-hoc inspection of existing
  test outputs, not evidence that any proposed change generalizes.
- Human open-coding judgments remain human work. Existing silver category counts
  describe an enriched review queue, not population error prevalence. No human
  labels were supplied or changed in this review.

## Reproduce this review

Run `python -m scripts.review_current_results`. It reads saved predictions,
replays the already-frozen acceptance policy, checks row keys and counts,
reproduces headline accuracy/F1/accepted accuracy, and hashes the inputs. It
checks retrieval caches against the historical protocol, permitting only the
explicit LF/CRLF byte conversion required by the Windows-origin hashes. It
does not import or run any model.

Machine-readable counts: `experiments/results_review/saved_prediction_audit.json`.
Historical references: `experiments/v2/REPORT.md`,
`experiments/stage5_learned/REPORT.md`,
`experiments/stage5_inference/REPORT.md`, and `docs/SERVING_DECISION_V2.md`.
