# Results

Canonical numbers for the project. Every table states its evaluation unit, the
decision rule, and whether any fitting was involved. Historical reports are kept
unchanged: [Stage 2/3](PROGRESS.md), [frozen-rule HaluEval](../experiments/stage5_inference/REPORT.md),
[Stage 5 learned](../experiments/stage5_learned/REPORT.md), [provisional errors](../experiments/stage6_errors/SUMMARY.md).

Last updated: 2026-10-22.

## v5 — grounding features, one test pass (2026-10-22)

v5 adds pretrained **MiniCheck-Flan-T5-Large** grounding scores to the 30 v4
features and refits the verdict and answerability heads on 9,000 FEVER train
claims. Model/threshold selection used validation only. The pipeline was frozen
and completed one final test pass on the 1,998-claim test split. All eight
predeclared research gates pass. The learned path remains offline; no serving
promotion is claimed.

**Validation (999 claims).** The robust policy at `tau=0.62` reaches raw macro-F1
**0.70681**, INSUFFICIENT F1 **0.54941**, answerability AUROC **0.81957**, Brier
**0.38753**, overall coverage **0.66366** at **0.79487** precision, verifiable
coverage **0.81381** at **0.97048** precision, and NEI false-answer coverage
**0.36336**. Against v4's 9k candidate, macro-F1 improves 1.39 points and NEI
false answers fall 4.50 points, at a cost of 0.60 points of overall coverage.

**Test (1,998 claims, one final pass).**

| Metric | Test value | Reference |
|---|---:|---:|
| Final-policy macro-F1 (abstain -> INSUFFICIENT) | **0.73694** | >= 0.69296 |
| INSUFFICIENT F1 | **0.64483** | >= 0.53559 |
| Answerability AUROC | **0.83809** | > 0.79436 |
| Brier | **0.37519** | <= 0.40870 |
| Overall coverage / precision | 0.66266 / **0.78701** | >= 0.65 / >= 0.75 |
| Verifiable coverage / precision | **0.81757 / 0.95500** | >= 0.80 / >= 0.94 |
| NEI false-answer coverage | **0.35285** | <= 0.40 |

The policy answered 1,324 of 1,998 claims. It abstained on 64.71% of NEI claims;
the 35.29% it answered were wrong 99.15% of the time. v5 therefore reduces NEI
false answers and improves the other gates, but **NEI answer precision remains
unsolved** and overall precision is 78.7%, not 95.5%. The 95.5% number is
verifiable precision on the answered subset; if abstentions are scored wrong, the
effective binary accuracy on verifiable claims is `0.81757 * 0.95500 = 0.7808`.
Full validation/test report:
[`experiments/v5/VALIDATION.md`](../experiments/v5/VALIDATION.md). One test
invocation failed during JSON serialization before writing any metric; the
identical frozen pipeline was rerun after a serializer-only fix. The report records
both attempts. No test-driven tuning was performed.

## Protocol discipline

| Stage | Fitted? | Selection data | Report data |
|---|---|---|---|
| Stage 2 oracle NLI | No | threshold on val | test |
| Stage 3 retrieval + fixed rule | No | thresholds on val | test |
| Frozen-rule HaluEval | No | thresholds inherited | test |
| Stage 5 learned combiner | **Yes** (logistic regression) | `C` and feature groups on val | test, once |
| v5 grounding + answerability | **Yes** (logistic + HGB) | `C`, answerability config, `tau` on val | test, once |
| Val-only improvement studies | No | none | val only |

Test scores below come from a single evaluation under a protocol written before the
pass. `experiments/stage5_learned/protocol.json` stores the split roles, feature list,
`C` grid, seed and SHA-256 of every retrieval cache; a test asserts the caches still
match.

## Datasets

| Dataset | Splits | Unit of gold label |
|---|---|---|
| FEVER | 3,000 / 999 / 1,998 claims | claim: SUPPORTED / CONTRADICTED / INSUFFICIENT |
| HaluEval QA | 2,800 / 600 / 600 answers | answer: supported / hallucinated |
| Corpus | 247,848 sentences from 50k FEVER pages | none (retrieval only) |

HaluEval has no claim-level gold labels. Any claim-level HaluEval number would be
invented, so none is reported.

## Stage 2 — oracle evidence (upper bound)

Gold evidence supplied to DeBERTa-v3-base-mnli; no retrieval error.

| Setting | Test n | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| FEVER, premise = concatenated gold evidence | 1,324 | 0.943 | 0.943 |
| FEVER, premise = strongest single gold sentence | 1,324 | 0.940 | 0.940 |
| FEVER, embedding similarity only | 1,324 | 0.700 | 0.697 |
| HaluEval, whole answer, argmax | 600 | 0.587 | 0.625 (binary F1) |
| HaluEval, whole answer, val-tuned threshold | 600 | 0.708 | 0.660 (binary F1) |

The 3-way FEVER figure (0.906 accuracy) is not comparable to the 2-way figure: the
oracle test split contains no NOT-ENOUGH-INFO claims, so the INSUFFICIENT column is
empty by construction.

## Stage 3 — retrieval and the fixed rule

Strict recall requires a complete gold evidence set inside the top-k, on the 1,332
verifiable test claims.

| Retrieval | Strict R@1 | Strict R@5 | Strict R@10 | Fixed-rule acc | Fixed-rule macro-F1 | FEVER score |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.612 | 0.808 | 0.871 | 0.621 | 0.606 | 0.546 |
| Dense (BGE) | **0.718** | **0.902** | **0.932** | 0.638 | 0.612 | 0.595 |
| Hybrid (RRF) | 0.679 | 0.886 | 0.917 | **0.652** | **0.633** | **0.600** |

Hybrid RRF does not beat dense on retrieval, but it does produce the best fixed-rule
classification. That inconsistency is one reason for fitting a combiner.

## Stage 5 — learned claim-level detector

Features: 30 deterministic signals in four groups — NLI maxima/means/counts/ranks,
retrieval scores and page agreement, BGE claim–evidence cosine, claim shape. Model:
standardised multinomial logistic regression. Fit on train, `C` and feature subset
selected on val, test evaluated once.

| Retrieval | Val acc | Val macro-F1 | Test acc | Test macro-F1 | 95% bootstrap CI | Fixed rule test F1 | Delta |
|---|---:|---:|---:|---:|---|---:|---:|
| BM25 | 0.662 | 0.655 | 0.686 | 0.682 | [0.663, 0.702] | 0.606 | +0.076 |
| Dense | 0.701 | 0.694 | 0.708 | 0.701 | [0.681, 0.721] | 0.612 | +0.090 |
| Hybrid | 0.684 | 0.676 | **0.708** | **0.702** | [0.682, 0.721] | 0.633 | +0.069 |

Per-class F1 on test, hybrid:

| Label | Learned P | Learned R | Learned F1 | Fixed-rule F1 |
|---|---:|---:|---:|---:|
| SUPPORTED | 0.816 | 0.892 | 0.852 | 0.786 |
| CONTRADICTED | 0.653 | 0.716 | 0.683 | 0.674 |
| INSUFFICIENT | 0.638 | 0.517 | 0.571 | 0.438 |

The largest gain is INSUFFICIENT (+0.133), the class the fixed rule almost never
predicted. Contradiction improves only marginally, which is consistent with the
retrieval-bound nature of that error.

### Paired significance (post-hoc)

The learned-vs-fixed comparison is paired on the same 1,998 test claims. Δ is the
difference in 3-way macro-F1, resampled with 10,000 paired bootstrap draws. McNemar
uses per-claim correctness; `b`/`c` are (learned right, fixed wrong) and (learned
wrong, fixed right).

| Retrieval | Δ macro-F1 | 95% CI | b / c | McNemar χ² p | exact p |
|---|---:|---|---:|---:|---:|
| BM25 | +0.0762 | [0.0571, 0.0957] | 279 / 149 | 4.5e-10 | 3.3e-10 |
| Dense | +0.0899 | [0.0698, 0.1098] | 283 / 143 | 1.6e-11 | 1.1e-11 |
| Hybrid | +0.0692 | [0.0501, 0.0892] | 254 / 142 | 2.4e-08 | 2.0e-08 |

Every interval excludes zero, and the discordant pairs favour the learned combiner
by 1.7–2.0 to 1. This is a post-hoc test of committed predictions: no refit, no
threshold change, and no new system was evaluated. Fixed-rule labels are a
deterministic replay that reproduces the committed Stage 3 accuracy and macro-F1.
Bootstrap p-values bottom out at 1/10,001; the exact McNemar p is the sharper
statement. Artifact: `experiments/stage5_learned/significance.json`.

### Where the error lives — 1,324 verifiable claims

The 0.702 headline is measured on all 1,998 claims, including 674 NEI claims with
no gold evidence. Restricting every system to the Stage 2 oracle's 1,324 verifiable
claims separates *coverage* from *verdict quality*. Buckets are exhaustive and
mutually exclusive; “introduced” means a confident error where the oracle is right.

| System | Coverage | Acc when answered | Oracle acc, same claims | Errors | Abstain, oracle right | Abstain, oracle wrong | Confident, oracle wrong | Confident, oracle right |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid fixed | 0.931 | 0.882 | 0.952 | 236 | 75 | 16 | 42 | 103 |
| Hybrid learned | 0.856 | 0.943 | 0.953 | 256 | 169 | 22 | 44 | 21 |

The same pattern holds for BM25 and dense (`decomposition.json`): when the learned
combiner commits, it is within one point of the oracle on the same answered claims
(0.941–0.943 vs 0.950–0.953). The decision rule is not the main bottleneck.

- The learned combiner trades coverage for precision: hybrid abstentions rise
  91 → 191, while confident errors introduced by retrieved evidence fall 103 → 21.
- 169 of its 256 errors (66%) are abstentions on claims the oracle answers
  correctly — evidence the retriever did not surface, not a wrong verdict.
- Of 65 confident errors, 44 are also oracle errors; only 21 are introduced.
- Net incremental error against the oracle is 169 + 21 − 9 = 181.

Confident errors are jointly retrieval and verifier errors, so the buckets bound
rather than isolate blame. Selective accuracy is measured on a self-selected
subset; the oracle column is computed on exactly the same answered claims. Oracle
verdicts are a cached reproduction of the frozen Stage 2 NLI-concat, checked
cell-for-cell against the committed 2×2 confusion matrix; no model was refitted or
re-selected.

Serving consequence: the learned combiner stays offline in v1 because its
probabilities are uncalibrated and it abstains on 27.0% of test claims. The demo
continues to serve the fixed rule; see
[SERVING_DECISION.md](SERVING_DECISION.md).

### Ablations (test macro-F1)

Each ablation re-selects `C` on val before its single test pass.

| Variant | BM25 | Dense | Hybrid |
|---|---:|---:|---:|
| full | 0.682 | 0.701 | 0.702 |
| no NLI features | 0.527 | 0.565 | 0.547 |
| no retrieval-score features | 0.688 | 0.681 | 0.701 |
| no similarity features | 0.682 | 0.699 | 0.698 |
| no claim-shape features | 0.671 | 0.685 | 0.693 |
| NLI only | 0.664 | 0.656 | 0.673 |
| retrieval only | 0.439 | 0.506 | 0.430 |
| similarity only | 0.451 | 0.473 | 0.457 |

NLI is the load-bearing component (~0.15 macro-F1). Retrieval-score and similarity
features add little once NLI is present; claim-shape features contribute ~0.01–0.02,
mostly by helping separate short entity claims from unverifiable ones.

## Val-only improvement studies

Both studies use the 999 val claims and inherit the frozen 0.7/0.95 thresholds.
They perform no fitting and no threshold search, and the test split is never
touched. They ask whether a change is worth promoting.

### Cross-encoder reranking

Cached candidates are rescored with the pretrained `BAAI/bge-reranker-base`
(depth 50, depth 20 for hybrid), the top 5 are kept, and NLI is re-run.

| Retrieval | Strict R@1 | Strict R@5 | Fixed-rule macro-F1 | Reranked | Delta |
|---|---|---|---:|---:|---:|
| BM25 | 0.625 → 0.707 | 0.838 → 0.904 | 0.566 | 0.601 | +0.035 |
| Dense | 0.722 → 0.694 | 0.913 → 0.913 | 0.586 | 0.616 | +0.030 |
| Hybrid | 0.688 → 0.703 | 0.896 → 0.917 | 0.585 | 0.608 | +0.023 |

For reference, the learned combiner on the same val split reaches 0.655 / 0.694 /
0.676 with the **unreranked** candidates — above every reranked fixed-rule number.
Dense reranking even loses strict R@1 while gaining macro-F1: it promotes supporting
sentences over the single best gold sentence, which helps the rule but is not a
retrieval improvement. Reranking is not promoted in v1; it adds a 1.1 GB
cross-encoder to serving and needs a full refit to be compared fairly.

### Neighbour-sentence premises

Premise = previous + current + next sentence from the same page.

| Retrieval | Fixed-rule F1 | Context F1 | Delta | Changed predictions |
|---|---:|---:|---:|---:|
| BM25 | 0.566 | 0.558 | −0.008 | 156 |
| Dense | 0.586 | 0.567 | −0.019 | 128 |
| Hybrid | 0.585 | 0.571 | −0.014 | 139 |

Worse for all three. SUPPORTED F1 drops 0.035–0.049 while contradiction is
essentially unchanged: extra sentences dilute the premise and push the NLI model
towards neutral. Not promoted.

## HaluEval — frozen rule (no fitting)

600 test answers, answer-level binary score, positive class = hallucinated.

| Evidence | Unit | Accuracy | F1 |
|---|---|---:|---:|
| Provided knowledge | whole answer (argmax) | 0.587 | 0.625 |
| Provided knowledge | claims (argmax) | **0.733** | **0.743** |
| Provided knowledge | claims (fixed rule) | 0.720 | 0.739 |
| BM25 Wikipedia | claims (fixed rule) | 0.505 | 0.668 |
| Dense Wikipedia | claims (fixed rule) | 0.505 | 0.668 |
| Hybrid Wikipedia | claims (fixed rule) | 0.505 | 0.668 |

Always flagging every answer scores 0.500 accuracy and 0.667 F1 on this balanced
set. The Wikipedia rows are therefore not evidence of discrimination: the hybrid
rule flags 595/600 answers, and the corpus is FEVER-derived while HaluEval spans
general topics. The provided-knowledge row is an evidence upper bound, not a
retrieval result.

## Error analysis — provisional

`scripts/shared/review/analyze_errors.py` asked Gemma to classify every prepared review case.
These are **proposals**, not human labels; all 274 are flagged
`provisional_llm_assisted: true`.

| Queue | Records | Usable | Pass | Fail |
|---|---:|---:|---:|---:|
| Extraction | 171 | 171 | 149 | 22 |

| Verification cause (proposed) | Count |
|---|---:|
| retrieval_failure | 56 |
| nli_failure | 38 |
| entity_confusion | 4 |
| multi_hop_reasoning | 4 |
| numerical_reasoning | 1 |

The review samples are enriched for failures, so these counts describe the sample,
not the population. Verification rows cover 103 cases but only 93 unique claims: 10
claims appear under two evidence conditions, so counts are per row and proposals are
keyed by `(id, condition)`, never by id alone. The 40 pilot rows are annotated; the
full 274-row pass remains unassigned. `scripts/shared/review/review.py` walks the queues,
optionally showing the proposal, and refuses to re-offer completed rows.

### Blind pilot — passed with a completeness deviation

Two 20-case blind queues validated the taxonomy before the full pass:
`experiments/pilot_v1/extraction_queue.jsonl` (balanced 10 Pass / 10 Fail) and
`verification_queue.jsonl` (stratified across proposed causes, one condition per
claim). The gate was fixed before annotation: Cohen's κ ≥ 0.60 on extraction
Pass/Fail and exact category agreement ≥ 0.50 on verification errors.

| Check | Observed | Gate |
|---|---:|---:|
| Extraction Pass/Fail κ (17 labeled verdicts) | **0.881** | ≥ 0.60 |
| Verification exact category agreement (20/20) | **0.80** (16/20) | ≥ 0.50 |
| Verification category κ | 0.709 | — |

Completeness deviation: 39/40 rows were annotated. Three extraction rows have a
proposal but no binary verdict (`halueval-qa-9023-hallucinated`,
`halueval-qa-1653-hallucinated`, `halueval-qa-3451-hallucinated`); the queue is
19/20 complete. Enumerating all eight possible verdict assignments gives κ between
**0.600 and 0.900**, so every completion clears the threshold and the decision is
invariant to the missing judgments. The bootstrap 95% interval for the observed κ
is wide ([0.605, 1.000]) because n is small; the point estimate and the adversarial
worst case both meet the gate.

Verification disagreements (4): `nli_failure → ambiguous_evidence` (2),
`nli_failure → claim_extraction_failure` (1), `nli_failure → multi_hop_reasoning`
(1). Extraction cause labels diverged on 3 of 7 Fail rows (entity confusion,
ambiguous evidence, temporal information where the proposal said claim extraction
failure). The gate passes, but these divergences argue for clearer category
definitions — or restricting extraction to Pass/Fail — in the full pass.

The pilot is balanced by design, not a prevalence sample. Protocol and commands are
in `experiments/pilot_v1/PILOT.md`; the scored artifact is
`experiments/pilot_v1/pilot_agreement.json`.

## v2 - calibration, selective policy, coverage flag

v2 froze the v1 verdict path and added a validity/serving layer. The full report is
in `experiments/v2/REPORT.md`; the serving decision is
[SERVING_DECISION_V2.md](SERVING_DECISION_V2.md).

**Labels.** The 237-row human pass was replaced by materialized LLM proposals at
the user's direction: 37 rows keep their human pilot verdict, 237 are silver labels
(`human_reviewed: false`). Pilot noise: extraction κ 0.881 (n=17), verification
category agreement 0.75 (n=20). Taxonomy v2 splits `nli_failure` into
`verifier_error` and keeps `ambiguous_evidence` distinct. The 40-row audit was filled
by an assistant estimate, not a human; agreement with the proposals is low
(extraction 60.0%, κ 0.20; verification categories 57.9%), so the labels remain
silver and not human-validated. A separate qualitative review of 15 HaluEval errors
maps to 11 retrieval failures, 2 ambiguous-evidence cases, 1 verifier error, and 1
answer-level omission outside the claim taxonomy; it is not part of the 274 labels
and changes no metric (`experiments/v2/HALUEVAL_ERROR_REVIEW.md`).

**Calibration (val, out-of-fold).** Temperature 1.122 with 2-fold cross-fitting:
ECE 0.03317 → 0.02301 (30.6% relative reduction), Brier 0.42303 → 0.42210, NLL
0.70858 → 0.70572. Two hundred bootstrap refits on train: val macro-F1
0.6721 ± 0.0057, 95% [0.6609, 0.6837].

**Selective policy (val).** Class-conditional thresholds, selected under the
fallback because 90% overall precision was infeasible at ≥50% coverage:
SUPPORTED 0.50, CONTRADICTED 0.38, INSUFFICIENT 0.84. Val coverage 0.7197,
answered precision 0.7316, verifiable coverage 0.8168 and precision 0.9614
(CP lower 0.9416). NEI coverage 0.5255 at 0.0171 precision.

**Coverage flag (val).** Predicts strict retrieval failure; AUROC 0.774, flag rate
0.191, error enrichment 2.40× (bootstrap lower 2.09), false-abstention enrichment
2.78×.

**Single test pass.** Macro-F1 0.70202 (unchanged), ECE 0.02279, Brier 0.39725 vs
0.39656 uncalibrated. The policy answered 84.61% of verifiable claims at 94.14%
precision (CP lower 92.61%), 72.47% of all claims at 73.55% precision, and 48.20%
of NEI claims at 1.25% precision. The coverage flag reached AUROC 0.734 (gate 0.75)
but enriched errors 2.66× and false abstentions 3.40×; it is omitted from serving.

**Retrieval stretch (val only).** Weighted RRF was tuned on val by re-fusing the
cached BM25/dense lists. Best BM25 weight 0.25 gave val macro-F1 0.67798 vs 0.67636
for the v1 hybrid (+0.0016, not material); dense-only reached 0.6761 with higher
strict R@5 (0.9129 vs 0.8964). The retrieval-variant test pass was not spent and no
variant is promoted.

**Decision.** No promotion: the Brier “no worse” gate failed by 0.00069 (paired 95%
CI [-0.00079, 0.00219], so not distinguishable from zero, but the gate was
pre-registered as a point comparison), and the flag missed its AUROC gate. The
learned detector remains offline; the fixed rule stays the default.

## v3 - stopped before the test pass

v3 targeted the v2 promotion blockers: Brier, NEI, and coverage-flag stability.
P1 selected vector scaling fitted on Brier (val Brier 0.42161 vs 0.42303; ECE
0.0255). P2's answerability model reached val AUROC 0.765, but the best policy
under the verifiable constraints left NEI coverage at 0.483 (target 0.40) and
overall precision at 0.741 (target 0.75). A joint threshold search found no
feasible point. A v3.1 diagnostic added uncertainty and coverage-flag features; the
best AUROC rose only to 0.769 and the NEI floor stayed at 0.477, so v3 stopped
before P4. **No v3 test pass was spent** and the learned detector remains offline.
Report: `experiments/v3/REPORT.md`.

## v4 - reranker and data expansion, stopped before test

Reranking the top-50 hybrid candidates and refitting the logistic combiner gives
val macro-F1 **0.69296** vs 0.67636 and INSUFFICIENT F1 **0.53559** vs 0.52073;
top-8 is flat, so top-5 is kept. Expanding train from 3k to 6k/9k balanced claims
improves answerability AUROC from 0.765 to 0.781/0.787 and lowers the NEI floor from
0.483 to 0.453/0.423. Cost-sensitive tuning reaches **0.408** NEI at verifiable
coverage 0.800 and overall precision 0.768; a joint threshold search still finds no
feasible point (best NEI 0.417). v4 stops before the test pass. **No v4 test pass
was spent** and the learned detector remains offline. Report:
`experiments/v4/REPORT.md`.

## Reproduce

```sh
python -m scripts.reproduce --stage checks
python -m src.evaluation.fit_detector --method all
python -m scripts.shared.inference.summarize_stage5
python -m src.evaluation.significance
python -m src.evaluation.decomposition
python -m scripts.reproduce --stage all --include-heavy
```

## Limitations

- **Corpus coverage dominates open-domain behaviour.** The 247,848-sentence
  FEVER-derived corpus is the main reason the HaluEval Wikipedia rows sit at chance.
- **Single-seed point estimate.** v2 adds 200 bootstrap refits on train (val
  macro-F1 0.6721 ± 0.0057), but the reported model is still one seed; the test
  interval covers sampling noise, not the full training distribution.
- **Linear combiner.** Interactions are only captured through explicit features.
- **Calibration is not served.** Temperature scaling lowers ECE but slightly worsens
  Brier on test; the learned path remains offline, so raw probabilities are not
  presented as calibrated confidence.
- **NLI is MNLI-only.** No FEVER-tuned checkpoint is used, which avoids leakage but
  leaves in-domain performance lower than a FEVER-trained model would give.
- **NEI remains unsolved.** The v2 policy abstains on 51.8% of NEI test claims, but
  the 48.2% it still answers are wrong 98.8% of the time; overall answered precision
  is 73.6%, not 90%. v5 improves NEI false-answer coverage to 35.3%, but the NEI
  claims it still answers have only 0.85% precision.
- **v5 precision is selective.** Verifiable precision 95.5% is measured on the
  81.8% of verifiable claims the policy answers; the effective forced-choice
  accuracy is 78.1% when abstentions are counted wrong.
- **Labels are mostly silver.** Only 37 of 274 taxonomy labels are human; the other
  237 are LLM proposals, and the 40-row blind audit is pending. Population-level
  failure-mode counts remain provisional.
