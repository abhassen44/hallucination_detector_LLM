# v4 — retrieval stack, more data, NEI stage

**Closed.** v4 stopped before the test pass: the best val NEI coverage was 0.408
against the 0.40 gate, and the test pass remains unused. The learned detector stays
offline. See [REPORT.md](REPORT.md) and
[../docs/SERVING_DECISION_V4.md](../docs/SERVING_DECISION_V4.md).

Baseline: v3 stopped at `e9974c6` (branch `ver-3`). Branch: `ver-4`.
Protocol: [protocol.json](protocol.json). Plan: `C:\Users\Abhas\.opencode\plan\v4-plan.md`.

v4 targets the v3 wall: NEI separability. It also tests the reranker + refit for
macro-F1.

## Status

| Phase | Scope | Status |
|---|---|---|
| P0 | Freeze v3, branch, protocol | **Done** |
| P1 | Reranker top-8 + refit (val) | **Done** — top-5 wins; val macro-F1 0.6930 |
| P2 | Train 6k → 9k, refit (val) | **Done** — NEI 0.423 at 9k; still short |
| P3 | Answerability tuning (val) | **Done** — NEI 0.408; joint search still infeasible |
| P4 | Verifier ceiling / concat top-3 | Not run |
| P5 | One test pass + serving decision | **Not run; test pass unused** |

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

## Outcome

6k and 9k data expansion improved answerability (AUROC 0.765 → 0.781 → 0.787), and
cost-sensitive tuning pushed NEI coverage to **0.408**. That is still above the
0.40 gate; a joint threshold search found no feasible point (best NEI 0.417 at
verifiable coverage 0.809). The base val macro-F1 also fell at 9k (0.685), so the
macro-F1 and NEI goals conflict. v4 stopped before the test pass. Full report:
[REPORT.md](REPORT.md).

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
