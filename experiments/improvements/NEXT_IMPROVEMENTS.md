# Remaining improvements after focused evidence selection

This is a diagnostic review of the existing 999 FEVER validation claims. No model
was loaded, trained, fitted, or queried. No test-set candidate evaluation was
performed, and serving behavior was not changed. These exploratory comparisons
reuse validation and require confirmation on new held-out data.

## Findings

The hybrid first-passage policy makes 309 classification errors: 145 gold NEI
claims receive factual verdicts (100 contradicted, 45 supported), 148 verifiable
claims become insufficient, and 16 verifiable claims receive the opposite factual
verdict. Its review rule accepts 537 claims with 139 errors; 131 of those errors
(94.2%) are false factual assertions on NEI. The principal accepted-error problem
is therefore determining whether a passage actually settles the claim.

First-passage selection fixes 167 errors made by five-passage aggregation, but
loses 87 previously correct predictions (net +80 correct). More evidence is not
automatically helpful. Among its 164 verifiable errors, 128 already have a complete
gold evidence set somewhere in the original five passages. In 92 verifiable
errors, a lower-ranked passage yields the correct single-sentence frozen-rule
verdict. This is oracle diagnostic headroom, not a deployable way to choose a
passage or a promise of 92 recoverable cases without new errors.

Multi-sentence gold evidence is harder: hybrid first-passage errors occur on
34/68 (50.0%) such claims, versus 130/598 (21.7%) claims with a single-sentence
gold alternative. The 68-case subgroup is small. Combining carefully selected
evidence is worth investigating, but indiscriminate neighbor concatenation was
already negative in the historical study.

The conflict guard rejects 104 correct factual verdicts (96 supported, eight
contradicted) because an opposing NLI signal exists elsewhere in the original
five passages. Relevance-aware conflict adjudication could improve acceptance,
but simply removing the guard can also reintroduce false assertions. Gold
correctness cannot be used to select which warnings to suppress in production.

## Checks that discourage easy shortcuts

Acceptance-only consensus retains the same hybrid first-passage factual verdict;
it merely rejects disagreements and requires each participating method's review
rule to accept. All retrievers share the same NLI verifier and corpus.

| Acceptance rule | Accepted accuracy | Fraction accepted | NEI false assertions |
|---|---:|---:|---:|
| Hybrid first-passage | 74.12% | 53.75% | 131 |
| Dense + hybrid agree and both accept | 75.83% | 48.05% | 109 |
| BM25 + dense + hybrid agree and all accept | 77.11% | 41.54% | 90 |

Consensus provides modest reliability gains by refusing more cases; it does not
reach 90% accepted accuracy or improve classification macro-F1, since verdicts
are unchanged. Shared verifier mistakes survive agreement.

Two fixed fallback checks expand evidence only when the first-passage verdict is
INSUFFICIENT. Both are worse than keeping first-passage selection:

| Hybrid evidence policy | Macro-F1 | Accepted accuracy | NEI false assertions |
|---|---:|---:|---:|
| First passage | 0.6919 | 74.12% | 131 |
| If insufficient, expand to first three | 0.6632 | 68.00% | 197 |
| If insufficient, expand to first page's hits | 0.6705 | 68.52% | 192 |

Do not promote these fallback rules. They recover some verifiable claims while
turning too many genuine NEI cases into wrong factual verdicts.

Dense first-passage macro-F1 is 0.6977 versus hybrid's 0.6919, but a 0.0059 gap on
the same reused validation set is not proof of superiority. A default switch
would need a paired comparison and confirmation on a new held-out set.

## Recommended next experiments, in order

1. **Check claim–evidence alignment before accepting contradictions.** The
   current NLI can be confident about contradiction without a reliable factual
   decision. Test an independent pretrained verifier or the already configured
   Gemma model on development cases, including both correct and incorrect
   verdicts. Ask whether the evidence addresses the same entity, relationship,
   time, and qualifiers before it can refute the claim. This is a hypothesis:
   measure NEI false assertions, supported/contradicted precision, accepted
   fraction, latency, and cost. Do not manufacture human labels or treat another
   model's judgment as gold.
2. **Rerank, then adjudicate evidence conflicts by relevance.** Finish the
   optional pretrained reranker download and evaluate first-passage selection
   after reranking. Use relevance to decide which opposing passages deserve
   review; do not erase opposing evidence or assume ranker scores are factual
   confidence. Compare to the same fixed-rule path. Earlier reranking gains
   cannot be added to the new focused-selection gain.
3. **Assemble small evidence sets for claims requiring multiple facts.** Test
   focused complementary sentences and preserve their page/time/entity
   provenance. Trigger a second look based on serving-available signals, not
   gold evidence sets. The failed neutral-only fallbacks show that merely adding
   more hits is insufficient. Keep the first-passage baseline as the control.
4. **Improve source coverage for open-domain/HaluEval checks.** This remains a
   separate corpus limitation: supplied knowledge achieved 72.0% answer accuracy
   with the fixed rule versus 50.5% with the narrow Wikipedia corpus. Choose a
   versioned source snapshot and measure relevant evidence availability before
   judging verdict changes. Expansion can introduce more distractors.
5. **Use a new evaluation set before promotion.** Freeze the candidate and its
   acceptance criteria on development data, then run one held-out comparison.
   Track accepted accuracy and acceptance together; reporting a higher accuracy
   on an ever-smaller selected subset alone is misleading.

All five can start without training. Using corrected rank features in the
learned combiner and recalibrating probabilities would require later fitting;
those remain deferred. Extraction and answer-completeness changes need separate
HaluEval evaluation; FEVER consists of prewritten claims, so improving claim
extraction cannot improve the FEVER score reported here.

Reproduce the counts with `.venv/bin/python -m scripts.analyze_remaining_errors`.
The script verifies input hashes against the previous evidence study and writes
`remaining_error_audit.json`. It does not overwrite previous experiment results.
