# v5 P0.5 — validation error analysis

Created: 2026-10-22T12:02:38.221535+00:00
Split: val (999 claims); test_evaluated: false

Pre-P1 snapshot. The final v5 validation/test report is `VALIDATION.md`.

## Headline

- Current val macro-F1: **0.69090**
- Current val accuracy: **0.69870**
- Verifiable strict recall@5: **0.9174**
- Verifiable hit recall@5: **0.9715**
- Verifiable page recall@5: **0.9895**

## Error buckets

| Bucket | Count | Share of errors |
|---|---:|---:|
| false_answer_nei | 171 | 56.8% |
| answerability_abstention | 84 | 27.9% |
| verifier_error | 21 | 7.0% |
| retrieval_miss_abstention | 18 | 6.0% |
| retrieval_miss_verifier | 4 | 1.3% |
| verifier_abstention | 3 | 1.0% |

## Oracle ceilings (val only)

| Variant | Macro-F1 | Accuracy |
|---|---:|---:|
| current | 0.69090 | 0.69870 |
| oracle_answerability | 0.86857 | 0.86987 |
| oracle_verdict | 0.80979 | 0.82883 |
| oracle_both | 1.00000 | 1.00000 |

## Per-class (current policy)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUPPORTED | 0.8054 | 0.8949 | 0.8478 | 333 |
| CONTRADICTED | 0.6575 | 0.7147 | 0.6849 | 333 |
| INSUFFICIENT | 0.6067 | 0.4865 | 0.5400 | 333 |

## Reading

Of 301 val errors, 22 are attributable to a strict retrieval miss; 87 are abstentions with retrievable evidence; and 171 are false answers on true INSUFFICIENT claims. The oracle-answerability ceiling shows how much of the gap is selective policy versus verdict quality. Use these buckets to choose whether P1 should prioritize the cross-encoder/grounding signal or P3 retrieval.

## Artifacts

- `experiments/v5/error_analysis.json`
- `experiments/v5/error_analysis.md`

No test data was read. No model was fitted or tuned.
