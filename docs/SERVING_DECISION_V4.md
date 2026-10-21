# Serving decision — v4

**Decision:** no v4 promotion. The v4 study stopped before the test pass because
the pre-registered NEI gate was still unreachable on val. The learned detector
remains offline; the fixed 0.7/0.95 rule remains the default.

Date: 2026-10-21. Baseline: v3 at `e9974c6`. Branch: `ver-4`.
Report: [experiments/v4/REPORT.md](../experiments/v4/REPORT.md).

## Why it stopped

| Lever | Result |
|---|---|
| Reranker + top-5 refit | Val macro-F1 **0.69296** (+0.0166 vs v1), INSUFFICIENT F1 0.53559 |
| Train 6k | Answerability AUROC 0.781, NEI floor 0.453 |
| Train 9k | AUROC 0.787, NEI floor 0.423 |
| Cost-sensitive 9k model | AUROC 0.794, NEI **0.408** at vcov 0.800, precision 0.768 |
| Joint threshold search | No feasible point; best NEI **0.417** at vcov 0.809 |
| Test pass | **Not spent** |

The missing margin is small (0.008 NEI at the tuned point), but it is real: crossing
it requires either a better NEI signal or giving up verifiable coverage. The
macro-F1 gain and the NEI gain also pull in opposite directions at 9k.

## What did improve

- Reranker + top-5 is a genuine offline macro-F1 improvement and raises
  INSUFFICIENT F1.
- More training data improves NEI separation (AUROC 0.765 → 0.794 tuned).
- Calibration work from v3 remains valid.

None of that is enough to justify serving.

## Revisit when

1. A stronger NEI signal exists (claim-level evidence labels, a dedicated
   retrieval-failure classifier, or more NEI examples).
2. A two-stage verdict model with hard negatives is tested on val.
3. A deliberate v4.1 protocol accepts a weaker NEI target (e.g. ≤0.42); that is a
   weaker claim, not a free improvement.

Until then: learned offline, fixed rule default.
