# Stage 5 — learned claim-level detector

Fit on the FEVER train split, selected on val, evaluated on test **once**.
Model: `StandardScaler + multinomial LogisticRegression`; NLI `MoritzLaurer/DeBERTa-v3-base-mnli`; similarity `BAAI/bge-small-en-v1.5`; seed 42.

Rows: train 3000, val 999, test 1998.
Test evaluated: True (2026-10-21T06:35:04.895155+00:00).

## Headline

| Retrieval | Val acc | Val macro-F1 | Test acc | Test macro-F1 | 95% CI (test) | Fixed rule test F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.662 | 0.655 | 0.686 | 0.682 | [0.663, 0.702] | 0.606 | +0.076 |
| dense | 0.701 | 0.694 | 0.708 | 0.701 | [0.681, 0.721] | 0.612 | +0.090 |
| hybrid | 0.684 | 0.676 | 0.708 | 0.702 | [0.682, 0.721] | 0.633 | +0.069 |

The confidence interval is a 2,000-resample bootstrap over test claims.
Paired McNemar and paired-bootstrap tests against the fixed rule are reported
below.

## Per-class F1 on test

| Retrieval | Label | P | R | F1 | Fixed-rule F1 |
|---|---:|---:|---:|---:|---:|
| bm25 | SUPPORTED | 0.814 | 0.863 | 0.838 | 0.748 |
| bm25 | CONTRADICTED | 0.639 | 0.682 | 0.660 | 0.647 |
| bm25 | INSUFFICIENT | 0.588 | 0.514 | 0.548 | 0.423 |
| dense | SUPPORTED | 0.798 | 0.890 | 0.842 | 0.786 |
| dense | CONTRADICTED | 0.667 | 0.718 | 0.691 | 0.664 |
| dense | INSUFFICIENT | 0.639 | 0.517 | 0.571 | 0.385 |
| hybrid | SUPPORTED | 0.816 | 0.892 | 0.852 | 0.786 |
| hybrid | CONTRADICTED | 0.653 | 0.716 | 0.683 | 0.674 |
| hybrid | INSUFFICIENT | 0.638 | 0.517 | 0.571 | 0.438 |

## Paired significance (post-hoc)

Committed test predictions only: no refit, no threshold change, no second
test pass. The fixed rule is replayed deterministically and checked against
the committed Stage 3 metrics before testing.

| Retrieval | Delta macro-F1 | 95% CI | b / c | McNemar p | exact p |
|---|---:|---:|---:|---:|---:|
| bm25 | +0.0762 | [+0.0571, +0.0957] | 279 / 149 | 4.5e-10 | 3.3e-10 |
| dense | +0.0899 | [+0.0698, +0.1098] | 283 / 143 | 1.6e-11 | 1.1e-11 |
| hybrid | +0.0692 | [+0.0501, +0.0892] | 254 / 142 | 2.4e-08 | 2e-08 |

Every interval excludes zero. Bootstrap p-values bottom out at
1/10,001; the exact McNemar p is the sharper statement.


## Where the error lives (1,324 verifiable claims)

Every system restricted to the Stage 2 oracle subset. "Introduced" means a
confident error where the oracle is right; selective accuracy is measured on
the self-selected answered subset, with the oracle computed on the same rows.

| System | Coverage | Acc answered | Oracle answered | Errors | Abstain / oracle right | Abstain / oracle wrong | Confident / oracle wrong | Confident / oracle right |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule | 0.931 | 0.882 | 0.952 | 236 | 75 | 16 | 42 | 103 |
| learned | 0.856 | 0.943 | 0.953 | 256 | 169 | 22 | 44 | 21 |

The learned combiner trades coverage for precision: it abstains on more
claims but introduces far fewer confident errors. Most residual error is
abstention on claims the oracle answers correctly, i.e. evidence coverage.
All methods are in `decomposition.json`; interpretation is in
`docs/architecture/v1.md`.


## Feature ablation (test macro-F1)

Every ablation re-selects `C` on val before the single test pass.

| Variant | bm25 | dense | hybrid |
|---|---:|---:|---:|
| full | 0.682 | 0.701 | 0.702 |
| no_nli | 0.527 | 0.564 | 0.546 |
| no_retrieval | 0.687 | 0.681 | 0.701 |
| no_similarity | 0.682 | 0.699 | 0.698 |
| no_claim_features | 0.671 | 0.685 | 0.693 |
| nli_only | 0.663 | 0.656 | 0.673 |
| retrieval_only | 0.438 | 0.506 | 0.430 |
| similarity_only | 0.451 | 0.473 | 0.457 |

The weakest variant is `retrieval_only` (-0.237 macro-F1 on average versus the full model).

## What the model relies on (hybrid)

Permutation importance on val, macro-F1 decrease when shuffled:

| Feature | Mean decrease | Std |
|---|---:|---:|
| nli_e_max | 0.1615 | 0.0097 |
| nli_e_mean | 0.0827 | 0.0077 |
| nli_n_count_50 | 0.0405 | 0.0073 |
| sim_max | 0.0372 | 0.0113 |
| nli_c_count_95 | 0.0319 | 0.0067 |
| nli_c_mean | 0.0284 | 0.0084 |
| nli_e_top2 | 0.0253 | 0.0065 |
| ret_mean_score | 0.0244 | 0.0075 |
| nli_margin_max | 0.0185 | 0.0064 |
| claim_has_negation | 0.0128 | 0.0033 |
| ret_score_std | 0.0124 | 0.0027 |
| sim_top1 | 0.0114 | 0.0078 |

## Val-only: cross-encoder reranking

Val split only. No test evaluation, no fitting, no threshold search.

| Retrieval | Strict R@5 | Reranked R@5 | Fixed-rule F1 | Reranked F1 | Delta |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.838 | 0.904 | 0.566 | 0.600 | +0.035 |
| dense | 0.913 | 0.913 | 0.586 | 0.616 | +0.030 |
| hybrid | 0.896 | 0.917 | 0.585 | 0.608 | +0.023 |

## Val-only: neighbour-sentence context

Val split only. No test evaluation, no fitting, no threshold search.

| Retrieval | Fixed-rule F1 | Context F1 | Delta | Changed predictions |
|---|---:|---:|---:|---:|
| bm25 | 0.566 | 0.558 | -0.008 | 156 |
| dense | 0.586 | 0.567 | -0.019 | 128 |
| hybrid | 0.585 | 0.571 | -0.014 | 139 |

## Limitations

- FEVER claims only: HaluEval has no claim-level gold labels, so no claim-level
  HaluEval score is reported anywhere.
- The retrieval corpus is 247,848 sentences from 50k FEVER pages. Open-domain
  coverage is limited and the HaluEval Wikipedia results reflect that.
- Logistic regression is linear; interactions between NLI and retrieval signals
  are only captured through the explicit product-free feature design.
- One training seed is used. The bootstrap interval covers test sampling noise,
  not training variance.
- Probabilities are not calibrated. The reported probabilities are model outputs,
  not factual confidence.
- The fixed 0.7/0.95 rule was inherited from Stage 3 and is not re-tuned here.

