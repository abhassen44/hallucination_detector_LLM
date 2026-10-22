# v5 validation and test report

Status: **complete.** One FEVER test pass has been spent. All eight predeclared
research gates passed. The learned path remains offline; no serving promotion is
claimed.

This report contains validation-only selection numbers and the single held-out
test result. Test metrics live in `experiments/v5/test_metrics.json`; per-claim
predictions live in `experiments/v5/test_predictions.jsonl`.

## Question

v4 stopped at validation NEI false-answer coverage **0.408** under the 0.40 gate.
v5 asks whether a new, pretrained grounding signal can distinguish insufficient
evidence from supported/contradicted claims without sacrificing verifiable
precision or coverage.

The new signal is **MiniCheck-Flan-T5-Large** support scores for the five reranked
evidence sentences. HHEM-2.1-Open was evaluated as a second grounding model. Both
are inference-only and were not fitted on FEVER.

## Protocol

- Fit: FEVER train, 9,000 claims (3,000 original + 6,000 v4 expansion).
- Select: FEVER validation, 999 claims. No test data was read before the freeze.
- Report: FEVER test, 1,998 claims, one final pass (see execution note), after
  `experiments/v5/freeze.json` was written.
- Verdict model: standardised multinomial logistic regression; `C` selected on
  validation macro-F1.
- Answerability model: HistGradientBoosting with class weight `{0: 1, 1: 3}`;
  selected among a fixed configuration grid on validation.
- Selective policy: frozen v2 class thresholds, answerability gate `tau`, and the
  v5 robust rule: among feasible policies, require verifiable coverage **>= 0.81**
  and overall coverage **>= 0.65**, then minimise NEI false answers. The v4 rule
  maximised coverage among feasible policies instead.
- Labels are silver FEVER labels. No human adjudication is claimed.

### Test-pass execution note

The first test invocation stopped during JSON serialization after inference but
before any metric or prediction was written. No metric was emitted or inspected.
The serializer was fixed to convert NumPy scalars, and the identical frozen
pipeline was rerun through `--recover`. The report records `attempts: 2` and
`recovered_after_error: true`. No feature, model, threshold, or policy changed
between attempts.

## Grounding feature ablation

The 9k ablation joins the 30 v4 features with cached grounding scores and
interactions. Selected-policy metrics are validation-only.

| Variant | Features | Val macro-F1 | INSUFF F1 | Ans. AUROC | Brier | NEI floor | Feasible |
|---|---:|---:|---:|---:|---:|---:|---|
| base (no grounding) | 30 | 0.6852 | 0.5222 | 0.7944 | 0.4087 | 0.4084 | no |
| MiniCheck (mc) | 45 | **0.7068** | **0.5494** | **0.8196** | **0.3875** | **0.3574** | yes |
| HHEM only | 45 | 0.6966 | 0.5281 | 0.8035 | 0.3939 | 0.3724 | yes |
| MiniCheck + HHEM | 62 | 0.7060 | 0.5467 | 0.8182 | 0.3901 | 0.3604 | yes |
| compact two-model | 48 | 0.6954 | 0.5309 | 0.8163 | 0.3911 | 0.3544 | yes |

MiniCheck-only is the selected variant: it is the simplest passing candidate, has
the best macro-F1 and INSUFFICIENT F1, and its 9k answerability AUROC is the best
of the feasible variants. The two-model and compact variants do not improve the
NEI floor on this validation split.

## Frozen candidate

- Verdict features: 30 v4 features + 15 MiniCheck score/interaction features.
- Verdict `C`: 0.3; validation raw macro-F1: **0.70681**.
- Answerability configuration: `hgb_nei_weight_3`; validation AUROC: **0.81957**.
- Full-calibrator answerability gate: **`tau=0.62`**.
- Freeze artifact: `experiments/v5/freeze.json`.
- Frozen models: `experiments/v5/frozen/verdict.joblib` and
  `experiments/v5/frozen/answerability.joblib`.

### Selected-policy validation metrics

| Metric | v4 9k tuned | v5 MiniCheck | Gate | Pass |
|---|---:|---:|---:|---|
| Raw macro-F1 | 0.69296 | **0.70681** | > 0.69296 | yes |
| INSUFFICIENT F1 | 0.53559 | **0.54941** | > 0.53559 | yes |
| Answerability AUROC | 0.79436 | **0.81957** | > 0.79436 | yes |
| Calibration Brier | 0.40870 | **0.38753** | <= 0.40870 | yes |
| Overall coverage | 0.66967 | 0.66366 | >= 0.65 (policy) | yes |
| Overall precision | 0.76831 | **0.79487** | >= 0.75 | yes |
| Verifiable coverage | 0.80030 | **0.81381** | >= 0.80 | yes |
| Verifiable precision | 0.96435 | **0.97048** | >= 0.94 | yes |
| NEI false-answer coverage | 0.40841 | **0.36336** | <= 0.40 | yes |

The selective final-prediction macro-F1 is **0.73965** (v4 current policy:
0.69090). This is a secondary number because the final policy uses abstention; the
raw macro-F1 is the comparable model-quality number.

The v5 candidate trades **0.60 points of overall answered coverage** relative to
v4 for a **4.50-point reduction in NEI false answers**, while verifiable coverage
improves by **1.35 points** and verifiable precision by **0.61 points**. Overall
precision improves by **2.66 points**. That is the intended v5 trade-off.

## Test result

The frozen pipeline completed one final test pass on 1,998 test claims.

| Metric | Test value | Frozen gate | Pass |
|---|---:|---:|---|
| Final-policy macro-F1 (abstain -> INSUFFICIENT) | **0.73694** | >= 0.69296 | yes |
| Accuracy | 0.73724 | - | - |
| SUPPORTED F1 | 0.86012 | - | - |
| CONTRADICTED F1 | 0.70588 | - | - |
| INSUFFICIENT F1 | **0.64483** | >= 0.53559 | yes |
| Answerability AUROC | **0.83809** | > 0.79436 | yes |
| Calibration Brier | **0.37519** | <= 0.40870 | yes |
| ECE | 0.02179 | - | - |
| Overall coverage | 0.66266 | >= 0.65 (policy) | yes |
| Overall answered precision | **0.78701** | >= 0.75 | yes |
| Verifiable coverage | **0.81757** | >= 0.80 | yes |
| Verifiable precision | **0.95500** | >= 0.94 | yes |
| Verifiable precision CP lower 95% | 0.94095 | - | - |
| NEI false-answer coverage | **0.35285** | <= 0.40 | yes |
| NEI precision when answered | 0.00851 | - | - |

All eight frozen test gates passed: macro-F1, INSUFFICIENT F1, answerability
AUROC, Brier, overall precision, verifiable coverage, verifiable precision, and
NEI false-answer coverage. Overall coverage is the robust-policy selection
constraint (`>= 0.65`), not a ninth gate. The policy answered **1,324 / 1,998**
claims at 78.70% overall precision, and **1,089 / 1,332** verifiable claims at
95.50% precision. It abstained on **64.71%** of NEI claims; the 35.29% it answered
were wrong **99.15%** of the time. NEI recognition improved, but NEI answer
precision is still the dominant failure mode.

If abstentions are scored as wrong, the effective binary accuracy on verifiable
claims is `0.81757 * 0.95500 = 0.7808`. That is the honest forced-choice
comparison to binary grounding models; the 95.5% figure is selective precision at
81.8% coverage, not accuracy on all verifiable claims.

## Reproducibility checks

`python -m scripts.evaluate_v5_test --check-val` replayed the frozen artifacts and
matched:

- raw macro-F1 0.7068115
- selective macro-F1 0.7396540
- NEI coverage 0.3633634
- overall precision 0.7948718
- verifiable coverage 0.8138138
- verifiable precision 0.9704797
- answerability AUROC 0.8195718

The test reranker/NLI cache and both test grounding caches were built only after
the freeze manifest existed. Test features were prepared without starting the test
pass; the single final evaluation was then run. The test metrics are recorded in
`experiments/v5/test_metrics.json` and include SHA-256 hashes of the freeze and all
test caches. The `hashes.freeze` entry identifies the manifest state written when
evaluation began; the final manifest adds only the post-test bookkeeping fields
(`test_evaluated`, timestamps, and metric path).

## Scope

These are claim-level FEVER metrics. They are not response-, span-, or
retrieval-augmented-generation metrics and must not be compared directly with
RAGTruth or other out-of-domain response evaluations. The learned path remains
offline unless a separate serving decision passes every serving gate. No v5
serving promotion is claimed here.
