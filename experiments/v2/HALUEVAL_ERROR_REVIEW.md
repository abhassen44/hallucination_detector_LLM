# HaluEval error review (qualitative)

User-supplied root-cause review of 15 HaluEval errors, checked against
`experiments/v2/queues/verification.jsonl` and the frozen predictions. This is a
**qualitative review**, not part of the 274-row taxonomy labels and not a new test
pass.

## Mapping summary

| v2 category | Cases |
|---|---:|
| `retrieval_failure` | 11 |
| `ambiguous_evidence` | 2 |
| `verifier_error` | 1 |
| Outside the claim-level taxonomy | 1 |
| **Total** | **15** |

All 15 rows are verification failures by construction (gold ≠ predicted). They are
all `Fail`; the category is the informative part.

## Cases

| ID | Condition | Gold → Pred | User root cause | v2 mapping |
|---|---|---|---|---|
| `halueval-qa-1287-right` | provided | supported → hallucinated | Co-reference miss: evidence names Kendall/Kylie as co-hosts and one as KUWTK star, but NLI returns neutral [0, 1, 0]. | `verifier_error` |
| `halueval-qa-1909-hallucinated` | provided | hallucinated → supported | Extrapolation: evidence says PBGV is a dog breed and Toy Trawler is a spaniel, but not "small dog"; NLI over-supports [0.96, 0.04, 0]. | `ambiguous_evidence` |
| `halueval-qa-1990-hallucinated` | dense | hallucinated → supported | Entity mismatch: Bob Dole was Ford's running mate, but retrieval returned Mike Padden / G.H.W. Bush snippets. | `retrieval_failure` |
| `halueval-qa-2023-right` | dense | supported → hallucinated | Irrelevant/distractor retrieval. | `retrieval_failure` |
| `halueval-qa-3211-right` | bm25 | supported → hallucinated | Hard-Fi confused with Hard Working Americans; Suede absent. | `retrieval_failure` |
| `halueval-qa-3486-right` | bm25 | supported → hallucinated | Ghana Club 100 not retrieved. | `retrieval_failure` |
| `halueval-qa-4605-right` | dense | supported → hallucinated | Other English actresses retrieved; Jodie Prenger absent. | `retrieval_failure` |
| `halueval-qa-5114-right` | hybrid | supported → hallucinated | My Santa / birth year not retrieved. | `retrieval_failure` |
| `halueval-qa-517-right` | hybrid | supported → hallucinated | Other Lorraines/Vivians retrieved; Hansberry/Gulland dates absent. | `retrieval_failure` |
| `halueval-qa-5557-right` | hybrid | supported → hallucinated | Other railway stations retrieved; Carnforth absent. | `retrieval_failure` |
| `halueval-qa-8136-right` | bm25 | supported → hallucinated | Kato Vermio retrieved instead of Kato Khandwala. | `retrieval_failure` |
| `halueval-qa-9735-right` | bm25 | supported → hallucinated | Other films/directors retrieved; The Suspect absent. | `retrieval_failure` |
| `halueval-qa-6814-hallucinated` | provided | hallucinated → supported | Grouped under "superficial entity overlap". | **Outside taxonomy:** the context supports the claims; the answer is wrong because it omits "Sweatin' to the Oldies", an answer-completeness failure. |
| `halueval-qa-7591-hallucinated` | bm25 | hallucinated → supported | Grouped under "superficial entity overlap". | `retrieval_failure` — Aspasius is absent from all retrieved snippets. |
| `halueval-qa-9021-hallucinated` | provided | hallucinated → supported | Yankees/trolley overlap; the reference sentence's "The team" is ambiguous across Yankees/Giants/Dodgers. | `ambiguous_evidence` (entity confusion plausible) |

## Two corrections to the pasted table

1. **`6814` is not a claim-level verification error.** The claims are supported by
   the supplied context. HaluEval marks the answer hallucinated because it fails to
   answer "what line of videos?" (`Sweatin' to the Oldies`). The claim verifier
   cannot detect an omission that was never extracted.
2. **`8136` has two queue rows.** The retrieval-failure description matches the
   `bm25_extracted_claims` row (`Kato Vermio`). The `provided_extracted_claims` row
   is a separate `claim_extraction_failure`: the extracted claim is only the
   fragment `"Kato Khandwala"` and NLI is neutral (E=0.497).

## What this does and does not change

- It supports the v2 finding that failure is dominated by **retrieval coverage**,
  not the decision rule: 11 of 15 reviewed errors are retrieval failures.
- It does **not** change the v2 test metrics, serving decision, or the 274-row
  label artifacts. These are qualitative review notes, not new taxonomy labels.
- It reinforces the HaluEval scope caveat: answer-level gold includes omissions
  that a claim-level verifier cannot see.

Artifact: [halueval_error_review.json](halueval_error_review.json).
