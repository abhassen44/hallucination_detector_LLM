# Inference improvements — 2026-10-22

No model training or fitting was performed. New candidate comparisons use only
the 999 cached FEVER validation claims. The test split was not evaluated for any
new candidate. The separate frozen-policy reliability report only recounts
existing v2 outputs, including the already-published test outputs.

## Evidence selection results

All methods retain the original entailment 0.7 / contradiction 0.95 thresholds.
The baseline aggregates up to five passages; the focused option uses the first
retrieved passage. Candidate definitions were fixed in code before running the
comparison. No thresholds were searched. Validation has been reused historically,
so these are exploratory results, not an unbiased estimate for deployment.

| Retrieval | Baseline macro-F1 | First-passage macro-F1 | Change | Baseline accuracy | First-passage accuracy |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5656 | 0.6682 | +0.1026 | 59.16% | 66.47% |
| Dense | 0.5858 | 0.6977 | +0.1119 | 61.86% | 69.77% |
| Hybrid | 0.5849 | 0.6919 | +0.1069 | 61.06% | 69.07% |

For hybrid, INSUFFICIENT F1 improves from 0.3672 to 0.5620. With the explicit
review rule below, accepted false factual assertions on gold NEI drop from 228 to
131 out of 333. Acceptance also falls from 73.37% to 53.75%; accuracy among
accepted verdicts improves from 65.21% to 74.12%. This does not meet a 90%
reliability goal.

The tradeoff is evidence coverage: hybrid complete gold-set selection falls
597 → 458 out of 666 verifiable claims. First-passage selection can discard
multi-sentence evidence, and false INSUFFICIENT with complete selected gold rises
30 → 43. It is therefore an **experimental option**, not a new default. These
numbers are for the fixed rule, not an improvement measured on the learned model.

Top-three and top-page policies are also recorded for comparison. Neither
outperforms first-passage selection on validation. No candidate has been promoted
or evaluated on test, and the Brier/coverage-flag gates from v2 remain unchanged.

## Implemented behavior

- `evidence_policy` accepts `all` (default), `top1`, `top3`, or `same_page` for
  retrieved evidence. `/verify` accepts all/top1/top3 for supplied passages.
- The UI exposes the first-passage option, marks it experimental, and keeps the
  supplied-source workflow on all supplied passages. It labels passages excluded
  from the verdict so users can inspect what was ignored.
- The verdict label remains separate from a new `decision.status` (`answer` or
  `abstain`). Missing evidence, an INSUFFICIENT label, evidence conflict, extraction
  fallback, or no factual claims trigger review. These are operational checks,
  not a calibrated confidence guarantee. An insufficient verdict is not proof
  that the claim is false.
- Conflict detection examines **all** supplied top-k passages, including passages
  excluded by the focused policy. A conflicting passage cannot disappear from the
  review decision simply because it was not used for the verdict.
- `rerank: true` optionally scores a deeper candidate pool with pretrained
  `BAAI/bge-reranker-base`. Retrieval scores and provenance are retained alongside
  separate `rerank_score` values. It is lazy-loaded and bypassed for supplied
  sources. It uses pretrained inference only; no fine-tuning occurs. This
  combination with focused selection has not been benchmarked and remains opt-in.
- Corrected rank features are available through `build_rank_corrected_matrix`.
  They use new names and schema `stage5-features-v2-retrieval-rank`; consuming them
  under a legacy schema raises an error. The existing builders and saved models
  retain their original definitions. Using corrected features in a learned model
  requires a later refit, which is deferred.
- New reliability reports expose true precision by **predicted** class, accuracy
  by **gold** subgroup, NEI false assertions, and acceptance. Historical v2 metrics
  are retained for reproducibility and explicitly documented as legacy subgroup
  metrics. `/metrics` includes the new reports.

## Commands

From the repository root:

```sh
# Cached validation comparison; refuses to overwrite an existing result.
.venv/bin/python -m scripts.evaluate_evidence_policies --out experiments/improvements-new
# Recount existing policy outputs, no new model inference or selection.
.venv/bin/python -m scripts.report_policy_reliability
# Start API and built UI.
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Example request for the focused evidence option:

```json
{"answer": "Paris is in France.", "method": "hybrid", "evidence_policy": "top1"}
```

Use `rerank: true` to enable the optional pretrained reranker after installing
its weights (see `docs/RUNNING.md`). `/health` reports local readiness; the UI
disables reranking until setup is complete, and an API request fails fast with
an actionable error instead of downloading a model inside the request.
Retrieval itself remains limited to the existing
FEVER corpus. Broader open-domain evidence coverage is **not implemented** here:
it needs a chosen source snapshot and a separately measured corpus expansion.
Human labeling and new learned-model/calibrator fitting also remain deferred.

## Validation

- 149 tests passed, including serving/study parity, evidence conflicts, source
  bypass, reranking order/provenance, feature-schema compatibility, reliability
  denominators, and historical-artifact checks. Two existing calibration-fitting
  tests were deselected to honor the no-fitting constraint.
- The frontend production build passed.
- Optional reranker setup was attempted using both the native transfer and
  standard HTTPS. The native transfer stalled; HTTPS then exhausted retries after
  read timeouts and DNS-resolution failures. Its real-model smoke check could not
  complete. The integration passes ordering/provenance and unavailable-model
  tests, but live reranking remains disabled until the pretrained weights can be
  downloaded. No download process is left running.
- A real local `/verify` request with pretrained NLI returned SUPPORTED for
  "Paris is in France" against the France passage, marked the unrelated Saturn
  passage as excluded, and returned an independent `answer` decision. This is a
  wiring smoke test, not an accuracy benchmark.
- The browser completed a provided-source `/detect` check with the configured
  `gemma4:31b-cloud` extractor: two supported claims, with the source and NLI
  signals visible. A preview is saved as `ui-preview.png`.
- A pre-existing cache-hash check was made portable across LF/CRLF checkouts.
  Frozen data and hashes were not rewritten; a content-change test confirms that
  altered records are still rejected.

Artifacts: `metrics_evidence_policies.json`, `validation_predictions.jsonl`, and
`metrics_frozen_policy_reliability.json`. Historical reports and v2 serving
promotion decisions remain intact.
