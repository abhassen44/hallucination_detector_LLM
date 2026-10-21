# Results

Canonical numbers for the project. Every table states its evaluation unit, the
decision rule, and whether any fitting was involved. Historical reports are kept
unchanged: [Stage 2/3](PROGRESS.md), [frozen-rule HaluEval](../experiments/stage5_inference/REPORT.md),
[Stage 5 learned](../experiments/stage5_learned/REPORT.md), [provisional errors](../experiments/stage6_errors/SUMMARY.md).

Last updated: 2026-10-21.

## Protocol discipline

| Stage | Fitted? | Selection data | Report data |
|---|---|---|---|
| Stage 2 oracle NLI | No | threshold on val | test |
| Stage 3 retrieval + fixed rule | No | thresholds on val | test |
| Frozen-rule HaluEval | No | thresholds inherited | test |
| Stage 5 learned combiner | **Yes** (logistic regression) | `C` and feature groups on val | test, once |
| Val-only improvement studies | No | none | val only |

Test scores below come from a single evaluation under a protocol written before the
pass. `experiments/stage5_learned/protocol.json` stores the split roles, feature list,
`C` grid, seed and SHA-256 of every retrieval cache; a test asserts the caches still
match.

## Datasets

| Dataset | Splits | Unit of gold label |
|---|---|---|
| FEVER | 3,000 / 999 / 1,998 claims | claim: SUPPORTED / CONTRADICTED / INSUFFICIENT |
| HaluEval QA | 2,800 / 600 / 600 answers | answer: supported / hallucinated |
| Corpus | 247,848 sentences from 50k FEVER pages | none (retrieval only) |

HaluEval has no claim-level gold labels. Any claim-level HaluEval number would be
invented, so none is reported.

## Stage 2 — oracle evidence (upper bound)

Gold evidence supplied to DeBERTa-v3-base-mnli; no retrieval error.

| Setting | Test n | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| FEVER, premise = concatenated gold evidence | 1,324 | 0.943 | 0.943 |
| FEVER, premise = strongest single gold sentence | 1,324 | 0.940 | 0.940 |
| FEVER, embedding similarity only | 1,324 | 0.700 | 0.697 |
| HaluEval, whole answer, argmax | 600 | 0.587 | 0.625 (binary F1) |
| HaluEval, whole answer, val-tuned threshold | 600 | 0.708 | 0.660 (binary F1) |

The 3-way FEVER figure (0.906 accuracy) is not comparable to the 2-way figure: the
oracle test split contains no NOT-ENOUGH-INFO claims, so the INSUFFICIENT column is
empty by construction.

## Stage 3 — retrieval and the fixed rule

Strict recall requires a complete gold evidence set inside the top-k, on the 1,332
verifiable test claims.

| Retrieval | Strict R@1 | Strict R@5 | Strict R@10 | Fixed-rule acc | Fixed-rule macro-F1 | FEVER score |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.612 | 0.808 | 0.871 | 0.621 | 0.606 | 0.546 |
| Dense (BGE) | **0.718** | **0.902** | **0.932** | 0.638 | 0.612 | 0.595 |
| Hybrid (RRF) | 0.679 | 0.886 | 0.917 | **0.652** | **0.633** | **0.600** |

Hybrid RRF does not beat dense on retrieval, but it does produce the best fixed-rule
classification. That inconsistency is one reason for fitting a combiner.

## Stage 5 — learned claim-level detector

Features: 30 deterministic signals in four groups — NLI maxima/means/counts/ranks,
retrieval scores and page agreement, BGE claim–evidence cosine, claim shape. Model:
standardised multinomial logistic regression. Fit on train, `C` and feature subset
selected on val, test evaluated once.

| Retrieval | Val acc | Val macro-F1 | Test acc | Test macro-F1 | 95% bootstrap CI | Fixed rule test F1 | Delta |
|---|---:|---:|---:|---:|---|---:|---:|
| BM25 | 0.662 | 0.655 | 0.686 | 0.682 | [0.663, 0.702] | 0.606 | +0.076 |
| Dense | 0.701 | 0.694 | 0.708 | 0.701 | [0.681, 0.721] | 0.612 | +0.090 |
| Hybrid | 0.684 | 0.676 | **0.708** | **0.702** | [0.682, 0.721] | 0.633 | +0.069 |

Per-class F1 on test, hybrid:

| Label | Learned P | Learned R | Learned F1 | Fixed-rule F1 |
|---|---:|---:|---:|---:|
| SUPPORTED | 0.816 | 0.892 | 0.852 | 0.786 |
| CONTRADICTED | 0.653 | 0.716 | 0.683 | 0.674 |
| INSUFFICIENT | 0.638 | 0.517 | 0.571 | 0.438 |

The largest gain is INSUFFICIENT (+0.133), the class the fixed rule almost never
predicted. Contradiction improves only marginally, which is consistent with the
retrieval-bound nature of that error.

### Ablations (test macro-F1)

Each ablation re-selects `C` on val before its single test pass.

| Variant | BM25 | Dense | Hybrid |
|---|---:|---:|---:|
| full | 0.682 | 0.701 | 0.702 |
| no NLI features | 0.527 | 0.565 | 0.547 |
| no retrieval-score features | 0.688 | 0.681 | 0.701 |
| no similarity features | 0.682 | 0.699 | 0.698 |
| no claim-shape features | 0.671 | 0.685 | 0.693 |
| NLI only | 0.664 | 0.656 | 0.673 |
| retrieval only | 0.439 | 0.506 | 0.430 |
| similarity only | 0.451 | 0.473 | 0.457 |

NLI is the load-bearing component (~0.15 macro-F1). Retrieval-score and similarity
features add little once NLI is present; claim-shape features contribute ~0.01–0.02,
mostly by helping separate short entity claims from unverifiable ones.

## Val-only improvement studies

Both studies use the 999 val claims and inherit the frozen 0.7/0.95 thresholds.
They perform no fitting and no threshold search, and the test split is never
touched. They ask whether a change is worth promoting.

### Cross-encoder reranking

Cached candidates are rescored with the pretrained `BAAI/bge-reranker-base`
(depth 50, depth 20 for hybrid), the top 5 are kept, and NLI is re-run.

| Retrieval | Strict R@1 | Strict R@5 | Fixed-rule macro-F1 | Reranked | Delta |
|---|---|---|---:|---:|---:|
| BM25 | 0.625 → 0.707 | 0.838 → 0.904 | 0.566 | 0.601 | +0.035 |
| Dense | 0.722 → 0.694 | 0.913 → 0.913 | 0.586 | 0.616 | +0.030 |
| Hybrid | 0.688 → 0.703 | 0.896 → 0.917 | 0.585 | 0.608 | +0.023 |

For reference, the learned combiner on the same val split reaches 0.655 / 0.694 /
0.676 with the **unreranked** candidates — above every reranked fixed-rule number.
Dense reranking even loses strict R@1 while gaining macro-F1: it promotes supporting
sentences over the single best gold sentence, which helps the rule but is not a
retrieval improvement. Reranking is not promoted in v1; it adds a 1.1 GB
cross-encoder to serving and needs a full refit to be compared fairly.

### Neighbour-sentence premises

Premise = previous + current + next sentence from the same page.

| Retrieval | Fixed-rule F1 | Context F1 | Delta | Changed predictions |
|---|---:|---:|---:|---:|
| BM25 | 0.566 | 0.558 | −0.008 | 156 |
| Dense | 0.586 | 0.567 | −0.019 | 128 |
| Hybrid | 0.585 | 0.571 | −0.014 | 139 |

Worse for all three. SUPPORTED F1 drops 0.035–0.049 while contradiction is
essentially unchanged: extra sentences dilute the premise and push the NLI model
towards neutral. Not promoted.

## HaluEval — frozen rule (no fitting)

600 test answers, answer-level binary score, positive class = hallucinated.

| Evidence | Unit | Accuracy | F1 |
|---|---|---:|---:|
| Provided knowledge | whole answer (argmax) | 0.587 | 0.625 |
| Provided knowledge | claims (argmax) | **0.733** | **0.743** |
| Provided knowledge | claims (fixed rule) | 0.720 | 0.739 |
| BM25 Wikipedia | claims (fixed rule) | 0.505 | 0.668 |
| Dense Wikipedia | claims (fixed rule) | 0.505 | 0.668 |
| Hybrid Wikipedia | claims (fixed rule) | 0.505 | 0.668 |

Always flagging every answer scores 0.500 accuracy and 0.667 F1 on this balanced
set. The Wikipedia rows are therefore not evidence of discrimination: the hybrid
rule flags 595/600 answers, and the corpus is FEVER-derived while HaluEval spans
general topics. The provided-knowledge row is an evidence upper bound, not a
retrieval result.

## Error analysis — provisional

`scripts/analyze_errors.py` asked Gemma to classify every prepared review case.
These are **proposals**, not human labels; all 274 are flagged
`provisional_llm_assisted: true`.

| Queue | Records | Usable | Pass | Fail |
|---|---:|---:|---:|---:|
| Extraction | 171 | 171 | 149 | 22 |

| Verification cause (proposed) | Count |
|---|---:|
| retrieval_failure | 56 |
| nli_failure | 38 |
| entity_confusion | 4 |
| multi_hop_reasoning | 4 |
| numerical_reasoning | 1 |

The review samples are enriched for failures, so these counts describe the sample,
not the population. Human judgments remain unassigned; `scripts/review.py` walks the
queues, optionally showing the proposal, and refuses to re-offer completed rows.

## Reproduce

```sh
python -m scripts.reproduce --stage checks
python -m src.evaluation.fit_detector --method all
python -m scripts.summarize_stage5
python -m scripts.reproduce --stage all --include-heavy
```

## Limitations

- **Corpus coverage dominates open-domain behaviour.** The 247,848-sentence
  FEVER-derived corpus is the main reason the HaluEval Wikipedia rows sit at chance.
- **Single-seed model.** The bootstrap interval covers test sampling noise, not
  training variance.
- **Linear combiner.** Interactions are only captured through explicit features.
- **Uncalibrated probabilities.** Reported probabilities are model outputs, not
  confidence in factual truth.
- **NLI is MNLI-only.** No FEVER-tuned checkpoint is used, which avoids leakage but
  leaves in-domain performance lower than a FEVER-trained model would give.
- **Human annotation incomplete.** Every qualitative claim here is either computed
  or explicitly provisional.
