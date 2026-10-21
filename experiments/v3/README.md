# v3 — serving-ready calibration and answerability

Baseline: v2 at `e7f0af5` (tag `v2.0`). Branch: `ver-3`.
Protocol: [protocol.json](protocol.json). Plan: `C:\Users\Abhas\.opencode\plan\v3-plan.md`.

v3 is not a macro-F1 chase. It fixes the v2 promotion blockers: Brier regression,
confident NEI answers, and coverage-flag instability.

## Status

| Phase | Scope | Status |
|---|---|---|
| P0 | Freeze v2, branch, protocol | **Done** |
| P1 | Calibration bake-off (val only) | Next |
| P2 | Answerability / NEI model (val only) | Pending |
| P3 | Coverage-flag stability (val only) | Pending |
| P4 | Single test pass | Pending |
| P5 | Serving decision / feature flag | Pending |

## Hard gates (one test pass)

- Macro-F1 ≥ 0.697
- ECE ≤ 0.05; Brier ≤ uncalibrated on the same test rows
- Verifiable precision ≥ 0.94 (CP lower ≥ 0.90), verifiable coverage ≥ 0.80
- Overall coverage ≥ 0.55, overall precision ≥ 0.75
- NEI coverage ≤ 0.40 (v2: 0.482)
- Coverage flag passes or is omitted
- P95 added latency ≤ 50 ms

## Discipline

- Fit on train, select/calibrate on val, **one** test pass.
- If P1 cannot beat Brier on val, v3 stops before the test pass.
- If the NEI target is infeasible on val, the pre-registered fallback applies before
  test; no post-test reinterpretation.
- The fixed 0.7/0.95 rule stays the default unless every hard gate passes.
- Labels remain silver; no human-validation claim.

## Layout

```text
experiments/v3/
  protocol.json
  calibration.json        # P1 bake-off
  answerability.json      # P2
  coverage_flag.json      # P3
  policy.json             # frozen before P4
  metrics_v3.json         # P4
  test_predictions_v3.jsonl
  REPORT.md
```

v2 artifacts under `experiments/v2/` are read-only.
