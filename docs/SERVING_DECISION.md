# Serving decision — v1

**Decision (owner review pending):** the learned Stage 5 combiner stays an **offline
research artifact**. The FastAPI service and React UI continue to serve the frozen
0.7/0.95 rule, and any demo that shows claim verdicts must present `INSUFFICIENT`
as a first-class outcome with a coverage note.

Date: 2026-10-21. Scope: v1, FEVER claim-level scoring. This is reversible; the
learned model remains fitted, logged, and reproducible.

## What is served today

| Path | Detector | Learned model loaded? |
|---|---|---|
| FastAPI `/detect`, `/verify` | Frozen 0.7/0.95 rule + claim extraction | No |
| React UI | Same API | No |
| Offline evaluation | Fixed rule **and** learned combiner | Yes (analysis only) |

The serving path is inference-only: no fitting, no threshold search, no dependence
on Stage 5 artifacts.

## Evidence for the decision

1. **The learned model is better on the measured task.** Test macro-F1 0.702
   (hybrid) vs 0.633 for the fixed rule; paired Δ +0.069 [0.050, 0.089], McNemar
   p = 2.4e-8. The gain is not sampling noise.
2. **Most remaining error is coverage, not wrong confidence.** On the 1,324
   verifiable claims the learned hybrid answers 85.6% at 0.943 accuracy; the oracle
   on the same answered rows is 0.953. Two thirds of its errors are abstentions on
   claims the oracle answers correctly.
3. **It trades coverage for precision.** Across all 1,998 test claims it returns
   `INSUFFICIENT` for 27.0%, versus 15.4% for the fixed hybrid rule. A demo that
   hides that abstention rate would overstate coverage.
4. **Probabilities are uncalibrated.** They are model outputs, not confidence in
   factual truth; serving them as calibrated scores would be misleading.
5. **Scope is FEVER-only.** HaluEval has no claim-level gold, so the learned
   detector has no claim-level HaluEval score. The demo's open-domain Wikipedia
   mode is outside its validated distribution.
6. **The error taxonomy passed its pilot, not the full pass.** The blind 40-case
   pilot met the gate (κ 0.88 Pass/Fail, 0.80 category agreement) with one row
   unannotated; the full 274-row annotation is still pending, so failure-mode
   counts remain LLM proposals.

## Alternatives considered

| Option | Why not now |
|---|---|
| Serve the learned model by default | Uncalibrated probabilities, FEVER-only validation, no claim-level HaluEval score, and a 27% abstention rate the UI does not explain |
| Serve behind a feature flag for demos | Acceptable later, but requires the coverage warning and a calibration study first; no v1 deadline pressure to expose it |
| Retrain or recalibrate before deciding | Out of v1 scope; would need a new protocol and a fresh test pass |

## Consequences

- The UI reports the fixed rule's verdicts. `INSUFFICIENT` already exists in the
  schema, so the serving contract does not change.
- Documentation must describe 0.702 as an **offline upper-bound-style research
  result**, not as the deployed detector's accuracy.
- The 0.633 fixed-rule number is the deployed system's measured quality; the gap
  between 0.633 and 0.702 is evidence that a learned combiner is worth pursuing,
  not a claim about the running demo.

## Revisit when

1. The blind pilot passes and the full 274-row annotation is complete (taxonomy
   stable enough to report failure modes). The pilot gate is met; the full pass is
   still pending.
2. A calibration study (reliability diagram, ECE, threshold-free scores) is done on
   validation data.
3. A coverage diagnostic explains when the retriever is likely to miss evidence, so
   the UI can warn instead of silently abstaining.
4. Corpus expansion or a reranker is promoted, which changes the feature
   distribution and requires a refit plus a new single test pass.

Promotion to serving requires all four, not just a higher offline macro-F1.
