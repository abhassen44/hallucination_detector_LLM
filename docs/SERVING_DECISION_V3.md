# Serving decision — v3

**Decision:** no v3 promotion. The v3 study stopped before the test pass because the
pre-registered NEI gate was unreachable on val. The learned detector remains offline
and the fixed 0.7/0.95 rule remains the default.

Date: 2026-10-21. Baseline: v2 at `e7f0af5` (tag `v2.0`). Branch: `ver-3`.
Full report: [experiments/v3/REPORT.md](../experiments/v3/REPORT.md).

## Why it stopped

| Phase | Result | Consequence |
|---|---|---|
| P1 calibration | Vector scaling fitted on Brier: val **0.42161** vs raw **0.42303**; ECE 0.0255 | Real val improvement, not enough alone |
| P2 answerability | Val AUROC **0.765**; best policy leaves NEI coverage **0.483** and overall precision **0.741** | Fails NEI ≤0.40 and overall ≥0.75 |
| v3.1 diagnostic | Added uncertainty + coverage-flag score; best AUROC **0.769**, NEI floor **0.477** | Still infeasible |
| Joint threshold search | **0** feasible points; NEI floor 0.483 under verifiable constraints | No policy can meet the gates |
| P4 test | **Not run** | v3 test pass remains unused |

The v2 failure was calibration; v3 fixed that on val. The remaining blocker is the
NEI problem: the answerability model cannot separate NEI claims from verifiable
claims well enough without losing more verifiable coverage than the protocol allows.

## What did not change

- No serving code changed.
- The fixed rule is still the demo’s detector.
- v2 artifacts and the v2 decision stand.
- The 274 taxonomy labels remain silver; no human-validation claim.

## Revisit when

1. A stronger NEI signal exists: per-claim evidence coverage, a dedicated
   retrieval-failure model, or more NEI training data.
2. A v3.1 protocol deliberately relaxes the NEI/overall targets with a written
   rationale before looking at test (for example NEI ≤0.48, overall ≥0.74).
3. There is a reason to spend the unused test pass on a calibration-only release;
   the current policy would still fail the NEI/overall gates, so that alone is not
   enough.

Until then, the serving decision is unchanged: learned offline, fixed rule default.
