# v2 report — calibrated, coverage-aware claim verification

Status: complete through P6 (P6 is a val-only feasibility study). **No serving promotion.**
Baseline: `v1.0` = `f79f0ff`. Branch: `feature/ver-2`.

## TL;DR

| Question | Result |
|---|---|
| Did the frozen verdict path change? | No. Test macro-F1 **0.7020**, identical to v1; 95% bootstrap [0.6821, 0.7213]. |
| Did calibration improve ECE? | Yes: test ECE **0.0280 → 0.0228**. |
| Did it preserve Brier? | **No**: 0.39656 → 0.39725 (+0.00069; 95% CI [-0.00079, 0.00219]). The pre-registered strict gate fails. |
| Does the selective policy hold? | Yes: verifiable precision **94.14%** at **84.61%** coverage; CP 95% lower bound 92.61%. Overall precision 73.55% at 72.47% coverage. |
| Does the coverage flag hold? | On val, AUROC 0.774 (pass); on test, AUROC **0.734** (fail). **Omitted from serving.** |
| Serving decision | **Learned model stays offline**; the fixed 0.7/0.95 rule remains the default. |

## Labels and taxonomy (P1)

The planned 237-row human pass was replaced by materialized LLM proposals at the
user's direction. The label artifacts are therefore **silver labels**, not human
annotations:

- **37 rows** keep the human pilot verdict (`label_source: human_queue`).
- **237 rows** take the LLM proposal (`label_source: llm_proposal`,
  `human_reviewed: false`).
- Pilot label noise: extraction κ **0.881** (17 rows); verification category
  agreement **0.75** (20 rows). The v1 figure of 0.80 used the old categories;
  splitting `nli_failure` accounts for the change.
- A blind **40-row audit sheet** is generated but not yet completed. Without it,
  the claim is “LLM-proposed labels with pilot-estimated noise,” not
  “human-validated.”

Taxonomy v2 keeps eight verification categories, replacing `nli_failure` with
`verifier_error`. Extraction is Pass/Fail only for this pass.

Label distribution: extraction 150 Pass / 21 Fail. Verification (all Fail by
queue construction): retrieval 65, verifier_error 18, multi-hop 7, extraction 5,
entity 4, ambiguous 3, numerical 1.

## Calibration (P2, val only)

Base model: v1 hybrid, frozen C = 1.0, fitted on train. Val macro-F1 reproduces v1
exactly (0.67636). Temperature scaling is fitted with 2-fold cross-fitting inside
val; reported metrics are out-of-fold.

| Metric | Uncalibrated | Calibrated (OOF) |
|---|---:|---:|
| ECE (10-bin) | 0.03317 | **0.02301** |
| Brier | 0.42303 | 0.42210 |
| NLL | 0.70858 | 0.70572 |
| Accuracy | 0.68368 | 0.68368 |

Relative ECE reduction: **30.6%** (gate ≥25%); Brier not worse on val. Per-fold
temperatures: 1.1353 / 1.1080; full-val temperature **1.1220**.

Training variance from 200 bootstrap refits on train, evaluated on val:
macro-F1 **0.6721 ± 0.0057**, 95% [0.6609, 0.6837]; ECE 0.0382 ± 0.0075.

## Selective policy (P3, val only)

Policy family: answer the argmax class only when its calibrated probability clears
a class-specific threshold; otherwise abstain.

The pre-registered 90% overall precision target was **infeasible at ≥50%
coverage** on val (it needed roughly 24% coverage). The fallback therefore selected
maximum coverage subject to verifiable precision, with a Clopper–Pearson 95% lower
bound of 0.94.

Frozen thresholds: **SUPPORTED 0.50, CONTRADICTED 0.38, INSUFFICIENT 0.84.**

| Val metric | Value |
|---|---:|
| Coverage (all claims) | 0.7197 |
| Answered precision | 0.7316 |
| Verifiable coverage | 0.8168 |
| Verifiable precision | 0.9614 (CP lower 0.9416) |
| NEI coverage | 0.5255 |
| NEI precision | 0.0171 |
| NEI abstention | 0.4745 |

Per class: SUPPORTED 90.1% coverage / 96.3% precision; CONTRADICTED 73.3% / 95.9%;
INSUFFICIENT 52.6% / 1.7%. The last row is the unresolved NEI problem: the policy
abstains on 47.4% of NEI claims, but the NEI claims it still answers are almost
always wrong.

## Coverage flag (P3, val; evaluated on test in P4)

Target: strict retrieval failure (no complete gold evidence set in top-5).
Features are serving-available retrieval/NLI/similarity/claim signals; gold
evidence is used only for the label.

Train: 3,000 claims; val: 666 labelled verifiable claims, 69 failures (10.4%).
Val AUROC **0.774**, AP 0.296 (base failure rate 0.104).

The threshold was selected by a bootstrap lower bound: highest flag rate whose
5th-percentile error enrichment stays ≥2.0. Val operating point: flag rate
**19.1%**, flagged error rate 50.4% vs base 21.0% (**2.40×**; bootstrap lower
2.09), failure enrichment 2.81×, false-abstention enrichment 2.78×.

## Test pass (P4, one pass)

Test n = 1,998. Thresholds, temperature, and flag were frozen before the pass.

| Gate | Threshold | Test value | Result |
|---|---|---:|---|
| Macro-F1 | ≥ 0.697 | 0.70202 | **Pass** |
| ECE | ≤ 0.05 | 0.02279 | **Pass** |
| Brier | not worse than raw | 0.39725 vs 0.39656 | **Fail** |
| Verifiable precision | ≥ 0.94 | 0.94144 | **Pass** |
| Verifiable precision, CP lower | ≥ 0.90 | 0.92609 | **Pass** |
| Coverage | ≥ 0.50 | 0.72472 | **Pass** |
| Verifiable coverage | ≥ 0.75 | 0.84610 | **Pass** |
| Answered precision | ≥ 0.70 | 0.73550 | **Pass** |
| Coverage-flag AUROC | ≥ 0.75 | 0.73374 | **Fail** |
| Coverage-flag enrichment | ≥ 2.0 | 2.66× | Pass |
| Coverage-flag rate | ≤ 0.25 | 0.1524 | Pass |
| **Promotion** | all hard gates | — | **Fail** |

The Brier difference is tiny and not distinguishable from zero in a paired
bootstrap (+0.00069, 95% CI [-0.00079, 0.00219]), but the gate was written as a
point comparison. It is reported as a failure rather than reinterpreted after
seeing test.

The coverage flag still enriches test errors (2.66× at 15.2% flag rate) and
false abstentions (3.40×), but its AUROC misses the frozen 0.75 bar, so it is
**not served**; it remains a diagnostic.

Test selective policy: coverage 72.47% [70.47, 74.37], answered precision 73.55%
[71.33, 75.78], verifiable coverage 84.61% [82.73, 86.47], verifiable precision
94.14% [92.70, 95.48], NEI coverage 48.20%. Per class: SUPPORTED 92.3% / 95.0%;
CONTRADICTED 76.9% / 93.2%; INSUFFICIENT 48.2% / 1.2%.

One test pass was used. A bootstrap denominator bug in coverage CIs was corrected
after the pass from the saved per-row predictions; point estimates and gates were
unaffected. No model or threshold was changed.

## Retrieval feasibility (P6, val only)

Weighted RRF was tuned on val by re-fusing the cached BM25/dense top-50 lists. The
weight grid was `w_bm25 ∈ {0, 0.25, 0.5, 0.75, 1.0}`; NLI was computed only for
fused top-5 hits missing cached scores. At `w=0.5` the study reproduces the v1
hybrid exactly (val macro-F1 0.67636), which validates the implementation.

| BM25 weight | Val macro-F1 | Fixed-rule val macro-F1 | Strict R@5 |
|---:|---:|---:|---:|
| 0.00 (dense only) | 0.6761 | 0.5805 | 0.9129 |
| 0.25 | **0.6780** | 0.5843 | 0.9009 |
| 0.50 (v1 hybrid) | 0.6764 | 0.5849 | 0.8964 |
| 0.75 | 0.6756 | 0.5686 | 0.8754 |
| 1.00 (BM25 only) | 0.6554 | 0.5656 | 0.8378 |

The best point is **+0.0016** macro-F1 over v1, which is not a material gain and was
not tested. The retrieval-variant test pass remains unused. A retrieval change would
not fix the Brier or NEI blockers anyway.

## Decision (P5)

The learned detector **remains offline**. The fixed rule stays the serving default.

Reasons:

1. The calibration gate fails on Brier, even though ECE improves.
2. The coverage flag fails its AUROC gate on test and is omitted.
3. The selective policy is strong on verifiable claims but cannot fix the NEI
   problem; overall answered precision is 73.6%, not 90%.

v2 still delivers the validity layer: calibrated confidence, a frozen and tested
selective policy, a measured coverage diagnostic, training-variance estimates, and
an explicit label-integrity caveat.

## Limitations

- **Silver labels.** Only 37 of 274 taxonomy labels are human. The 40-row audit is
  pending, so failure-mode counts are LLM-proposed with pilot-estimated noise.
- **Val was reused.** C was selected on val in v1; calibration, policy, and flag
  were also selected on val. Each step is cross-fitted or frozen, but selection
  optimism remains.
- **One test pass.** No test-driven iteration. That is a strength for validity and
  a limit on how much can be claimed from a single split.
- **Flag instability.** Val AUROC 0.774 vs test 0.734; the diagnostic is not stable
  enough to serve.
- **NEI remains unsolved.** 48.2% of NEI test claims receive a confident answer,
  and only 1.2% of those answers are correct.
- **Latency not measured.** Serving was not attempted, so the ≤50 ms added-latency
  gate was not evaluated.

## Reproduce

```powershell
$env:PYTHONPATH="$PWD"
.\.venv\Scripts\python.exe -m scripts.seed_v2_queues --force
.\.venv\Scripts\python.exe -m scripts.propose_v2 --queue all --force
.\.venv\Scripts\python.exe -m scripts.materialize_v2_labels --force
.\.venv\Scripts\python.exe -m scripts.calibrate_v2 --force
.\.venv\Scripts\python.exe -m scripts.selective_policy_v2 --force
.\.venv\Scripts\python.exe -m scripts.coverage_flag_v2 --force
# The test pass is intentionally guarded: this was the one allowed run.
.\.venv\Scripts\python.exe -m scripts.test_pass_v2
```

Artifacts: `experiments/v2/{protocol,taxonomy,label_manifest,calibration,policy,coverage_flag,metrics_v2}.json`,
`experiments/v2/{queues,proposals}`, `experiments/v2/labels_llm_proposed_*.jsonl`,
`experiments/v2/{val_predictions_v2,test_predictions_v2,val_coverage_flags}.jsonl`,
and `experiments/v2/audit_sheet.csv`.
