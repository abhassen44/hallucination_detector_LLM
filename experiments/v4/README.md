# v4 — retrieval stack, more data, NEI stage

Baseline: v3 stopped at `e9974c6` (branch `ver-3`). Branch: `ver-4`.
Protocol: [protocol.json](protocol.json). Plan: `C:\Users\Abhas\.opencode\plan\v4-plan.md`.

v4 targets the v3 wall: NEI separability. It also tests the reranker + refit for
macro-F1.

## Status

| Phase | Scope | Status |
|---|---|---|
| P0 | Freeze v3, branch, protocol | **Done** |
| P1 | Reranker top-8 + refit (val) | **Done** — top-5 wins; val macro-F1 0.6930 |
| P2 | Train 6k → 9k, refit (val) | Pending |
| P3 | Two-stage NEI + hard negatives (val) | Pending |
| P4 | Verifier ceiling / concat top-3 | Pending |
| P5 | One test pass + serving decision | Pending |

## P1 result

Reranked hybrid, refit logistic combiner:

| Arm | Val macro-F1 | INSUFFICIENT F1 | Keep? |
|---|---:|---:|---|
| v1 hybrid | 0.67636 | 0.52073 | baseline |
| rerank top-5 | **0.69296** | **0.53559** | **Yes** |
| rerank top-8 | 0.67732 | 0.51581 | No |

Top-5 clears the keep rule: +0.0166 macro-F1 and +0.0149 INSUFFICIENT F1. The
top-8 arm dilutes the gain, so v4 uses **reranker + top-5**. Details:
[retrieval_stack.json](retrieval_stack.json).

## Key discipline

- **One** v4 test pass; no component test passes.
- P1 first because retrieval changes invalidate feature caches.
- Keep the reranker only if val macro-F1 ≥0.68136 (v1 +0.005) and INSUFFICIENT F1
  drops no more than 0.01.
- Expand train 3k → 6k, then 9k only if the answerability curve is still rising.
- If NEI ≤0.40 is still unreachable on val, stop before test.

## Layout

```text
experiments/v4/
  protocol.json
  retrieval_stack.json     # P1
  data_expansion.json      # P2
  nei_stage.json           # P3
  verifier_study.json      # P4
  metrics_v4.json          # P5
  test_predictions_v4.jsonl
  REPORT.md
```

v3 artifacts under `experiments/v3/` are read-only.
