# Serving decision — v2

**Decision:** the learned detector **stays offline**. The FastAPI service and React
UI continue to serve the frozen 0.7/0.95 rule. v2 does not promote the learned path,
and the coverage warning is not served.

Date: 2026-10-21. Scope: v2, FEVER claim-level scoring. Baseline: `v1.0`
(`f79f0ff`). Reversible: all artifacts are reproducible.

## Gate results

| Gate | Threshold | Test | Result |
|---|---|---:|---|
| Macro-F1 | ≥ 0.697 | 0.70202 | Pass |
| ECE | ≤ 0.05 | 0.02279 | Pass |
| Brier | not worse than uncalibrated | 0.39725 vs 0.39656 | **Fail** |
| Frozen selective policy | val-derived minimums | see below | Pass |
| Coverage flag | AUROC ≥ 0.75 | 0.73374 | **Fail** |
| Added latency | ≤ 50 ms P95 | not measured | Not evaluated |

Promotion requires all hard gates. Brier fails; the flag also fails and is omitted.
No promotion.

## What the policy achieved

On the single test pass, the frozen class-conditional policy answered:

- **84.6%** of verifiable claims at **94.1%** precision (CP lower 92.6%);
- **72.5%** of all claims at **73.6%** precision;
- **48.2%** of NEI claims — and only 1.2% of those answers were correct.

The verifiable subset is serving-grade in isolation. The overall number is not,
because a confident SUPPORTED/CONTRADICTED answer on an NEI claim is still an error,
and the model cannot reliably recognise that case.

## Why not serve the uncalibrated model?

The uncalibrated model has ECE 0.0280 (also ≤0.05) and Brier 0.39656, so a naive
reading might promote it. But its selective thresholds were fitted on
*calibrated* probabilities and frozen for the calibrated system. Re-deriving them
for raw scores would be a different system and would require another protocol and
another test pass. That is not done here.

## Why not serve the coverage flag?

The val AUROC was 0.774, just above the 0.75 gate. Test AUROC was 0.734. The flag
still enriched errors (2.66× at a 15.2% flag rate) and false abstentions (3.40×),
but the pre-registered stability bar was missed. It remains an offline diagnostic
and a candidate for a v3 study with more training data or a better target.

## Revisit when

1. A calibration method preserves or improves **both** ECE and Brier on test
   (vector scaling, isotonic with a proper multiclass treatment, or a
   Brier-constrained temperature fit). The current +0.0007 Brier change is within
   noise but still fails the written gate.
2. The coverage flag reaches ≥0.75 AUROC on a held-out pass with a stable flag rate.
3. There is a serving policy that improves NEI recognition, not just verifiable
   precision. Today 48.2% of NEI claims still receive confident wrong answers.
4. A **human** 40-row audit is completed (the current sheet is assistant-estimated),
   or the full human pass is run, so taxonomy claims are not based on silver labels.

All four are required for a future promotion; a higher offline macro-F1 alone is
not sufficient.

## What is still served

| Path | Detector | Learned model loaded? |
|---|---|---|
| FastAPI `/detect`, `/verify` | Frozen 0.7/0.95 rule + claim extraction | No |
| React UI | Same API | No |
| Offline evaluation | Fixed rule and learned combiner | Yes (analysis only) |

No serving code changed in v2. The learned artifacts remain reproducible under
`experiments/stage5_learned/` and `experiments/v2/`.
