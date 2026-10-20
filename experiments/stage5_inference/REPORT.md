# Frozen-rule HaluEval inference evaluation

No training, classifier fitting, threshold search, or new LLM calls were performed.
All results use the same 600 test answers and stored Gemma extractions.
Positive class: hallucinated. Scores are answer-level; no claim-level gold labels exist.

| Evidence / text unit | Decision rule | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| provided_whole_answer | argmax | 0.587 | 0.572 | 0.690 | 0.625 |
| provided_whole_answer | fixed_threshold | 0.577 | 0.557 | 0.747 | 0.638 |
| provided_extracted_claims | argmax | 0.733 | 0.717 | 0.770 | 0.743 |
| provided_extracted_claims | fixed_threshold | 0.720 | 0.692 | 0.793 | 0.739 |
| bm25_extracted_claims | argmax | 0.507 | 0.503 | 0.993 | 0.668 |
| bm25_extracted_claims | fixed_threshold | 0.505 | 0.503 | 0.997 | 0.668 |
| dense_extracted_claims | argmax | 0.505 | 0.503 | 0.993 | 0.667 |
| dense_extracted_claims | fixed_threshold | 0.505 | 0.503 | 0.997 | 0.668 |
| hybrid_extracted_claims | argmax | 0.505 | 0.503 | 0.993 | 0.667 |
| hybrid_extracted_claims | fixed_threshold | 0.505 | 0.503 | 0.997 | 0.668 |

Provided-context results use supplied knowledge, not Wikipedia search.
Wikipedia results use the limited FEVER corpus and should be interpreted separately.
Rules and thresholds were frozen before prediction. The existing 0.7/0.95 thresholds
were inherited from FEVER; they were not optimized for HaluEval.
Any result differences are descriptive comparisons, not statistical significance claims.
The two test fallback extractions and two examples without claims are included,
so counts stay at 600 for every method. Predicted probabilities are uncalibrated.

## Human error review

The sample contains up to 20 false positives and 20 false negatives per evidence
condition using the fixed-threshold rule, with original inputs and evidence attached.
Fields for observations, pass/fail and category remain unassigned. This enriched
failure sample does not estimate population error rates.

## Interpretation

An always-hallucinated predictor scores 50% accuracy and F1 0.667 on this balanced set.
The hybrid fixed rule flags 595 of 600 answers. Its F1 near 0.668
is consequently close to this trivial reference, not evidence of useful discrimination.
The experiment demonstrates the importance of evidence quality. It does not by itself
separate corpus coverage, ranking relevance and NLI errors into causal categories.
