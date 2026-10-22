# Grounded second-verifier validation

2026-10-22. Model: `gemma4:31b-cloud`, pretrained inference only. No training,
fitting, threshold search, or new test evaluation. The serving default is unchanged;
this is an opt-in experiment.

## Results on the same 999 FEVER validation claims

Both systems start with hybrid retrieval, the first retrieved passage, and the
frozen entailment 0.7 / contradiction 0.95 thresholds.

| Metric | Focused NLI baseline | With second verifier |
|---|---:|---:|
| Three-class macro-F1 | 0.6919 | **0.7192** |
| Classification accuracy | 69.07% | **71.37%** |
| Accepted factual-verdict accuracy | 74.12% | **82.92%** |
| Fraction accepted | 53.75% (537/999) | **56.86% (568/999)** |
| Accepted false assertions on gold NEI | 131 | **91** |
| Accepted SUPPORTED precision | 82.50% | **85.05%** |
| Accepted CONTRADICTED precision | 69.14% | **80.52%** |

54 predictions corrected, 31 regressed: net +23 correct. The paired macro-F1
improvement is +0.02739, with a 2,000-resample row-bootstrap 95% interval of
[+0.00986, +0.04481]. This interval is conditional on reused validation, not an
unbiased held-out generalization estimate.

Contradiction recall falls from 72.37% to 64.56%, while NEI recall rises from
56.46% to 72.67%. The change trades some genuine contradictions for more cautious
verdicts. Accepted accuracy still falls short of 90%.

## Fixed policy and output checks

The second verifier runs only for a baseline CONTRADICTED verdict or strong NLI
evidence conflict. It sees the claim and passages without gold labels, the NLI
verdict, or baseline confidence. It checks the asserted entity, relationship,
time, quantity, negation, and qualifiers. Missing information is not refutation;
genuine unresolved source conflicts remain insufficient. Complementary passages
can be considered together.

A factual verdict needs an aligned-evidence boolean, a valid passage ID, and an
exact contiguous quote. These structural checks do not prove logical entailment;
the model can still make semantic errors. The policy confirms the original
factual verdict or defers to INSUFFICIENT, never flipping it to its opposite.
Confirmed conflicts can be accepted, while the original label, original review
decision, raw conflict flag, citations, and explanation remain inspectable.
Invalid output causes abstention. Authentication/rate-limit failures remain
service failures; they do not silently become confident baseline predictions.

The fixed balanced pilot (150 claims, 50/class, seed 20261022) included successes
and errors. Pilot macro-F1 improved 0.6859 → 0.7190. The unchanged prompt and
policy were then evaluated on all 999 validation claims, reusing pilot responses.
These are overlapping development evaluations, not independent validation sets.

Of 999 rows, 526 did not require review. Of 473 reviews, 464 (98.10%) passed output
validation and nine failed exact-quote validation. All nine caused regressions,
not gains, and are included in every headline score. Excluding those nine from
both systems gives macro-F1 0.6894 → 0.7254 as a sensitivity analysis. Quotes were
not relaxed or repaired after seeing results.

Mean review time was about 3.99 seconds including API/cache time, excluding
unreviewed rows. Cached pilot responses make this neither an uncached latency
benchmark nor serving P95. Added cloud-request cost was not measured in currency.

## Usage and verification

Enable **Double-check contradictions** in the UI, or add `grounded_review: true`
to `/detect` or `/verify`. Default requests make no extra review calls. The table
above additionally uses `method: "hybrid"` and `evidence_policy: "top1"`; other
combinations are not covered by these scores. Reviews accept at most 20 passages
and 48,000 total claim/evidence characters, rejecting oversized inputs instead of
silently truncating evidence.

157 tests passed; two existing calibration-fitting tests were deselected to honor
the no-fitting constraint. The frontend build passed. The browser also completed
a real supplied-source check: NLI initially called a video-game count claim
CONTRADICTED using a passage about Taylor Lautner's television roles. The second
verifier returned INSUFFICIENT and explained the missing video-game/year evidence.
That is a smoke check, not another accuracy benchmark.

Model judgments are marked `human_reviewed: false`; FEVER gold labels remain the
scoring reference. No original datasets, human labels, historical experiment
results, or test budgets were changed. Generic LLM pretraining may have included
FEVER or related facts; source-only instructions and quote checks do not establish
absence of contamination. The cloud alias exposes no immutable weights revision.
Raw responses are cached separately in `data/cache/grounded_verifier_v1.sqlite`.

The learned combiner and corrected rank features were not refitted. Corpus
expansion, optional reranker installation, and answer completeness remain separate
work. Before promotion, freeze the candidate and confirm precision, coverage,
latency, and cost on fresh held-out data.

## Replay

```sh
.venv/bin/python -m scripts.evaluate_grounded_verifier --limit 150 --workers 3
.venv/bin/python -m scripts.evaluate_grounded_verifier --limit 999 --workers 4
.venv/bin/python -m scripts.summarize_grounded_verifier
```

Keep the original model/configuration and prompt version. The runner checks frozen
IDs, source hashes, prompt hash, and model before resuming from checkpoint.
Artifacts: `protocol.json`, `predictions.jsonl`, `metrics.json`,
`paired_analysis.json`, `invalid_diagnostics.json`, and `ui-grounded-review.png`.
