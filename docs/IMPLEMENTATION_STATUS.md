# Current implementation status

## 2026-10-21 — learned Stage 5

The earlier "do not train or fit any models" constraint was lifted for Stage 5.
Exactly one learned model now exists (a multinomial logistic combiner on FEVER);
every other component is pretrained and inference-only. No threshold search was
run and the HaluEval test split was not used for fitting.

Implemented:

- Stage 5 feature builder (`src/verification/features.py`): 30 deterministic
  features from cached retrieval scores and NLI probabilities, plus pretrained
  BGE similarity. Gold labels are never read while building features.
- Stage 5 fitter (`src/evaluation/fit_detector.py`): train → val `C`/feature
  selection → **one** test evaluation, guarded by `protocol.json`. Re-running
  refuses to touch test again without an explicit exploratory override.
- Eight feature ablations per retrieval method, each re-selecting `C` on val.
- Bootstrap 95% intervals, standardised coefficients, and permutation importance.
- MLflow logging in a separate experiment (`hallucination-stage5-learned`).
- Val-only improvement studies (`src/evaluation/val_studies.py`): pretrained
  cross-encoder reranking and neighbour-sentence premises. Test is untouched.
- Review tooling (`scripts/review.py`) with atomic write-back, backup, completed-row
  protection, and optional display of provisional LLM suggestions.
- Gemma-assisted provisional error analysis (`scripts/analyze_errors.py`) over all
  274 prepared review cases; human fields are not modified.
- Windows text-IO hardening: all `read_text`/`write_text`/text `open` calls now
  declare UTF-8, with a regression test.
- `/metrics` now serves every saved metrics file, including `baseline_review`.
- `scripts/resolve_recovery.py` recorded decisions on the three JSON syntax repairs;
  `scripts/reproduce.py` provides one entry point per stage.
- The suite now has **57 passing tests** (was 37), including portability, feature,
  review, Stage 5 artifact, and error-analysis schema checks.

Results (FEVER test, 1,998 claims):

- hybrid 0.702 macro-F1, dense 0.701, BM25 0.682, versus 0.633 / 0.612 / 0.606 for
  the frozen 0.7/0.95 rule. See [RESULTS.md](RESULTS.md).
- Ablation: dropping the NLI group costs ~0.15 macro-F1; dropping similarity or
  retrieval-score features barely changes the score.

Val-only studies (999 val claims, no test pass):

- Cross-encoder reranking lifts fixed-rule macro-F1 by +0.035 (BM25), +0.030 (dense)
  and +0.023 (hybrid), but stays below the learned combiner on the same split.
- Neighbour-sentence premises reduce macro-F1 by 0.008–0.019.
- Neither change is promoted in v1.

Validation and limitations:

- Protocol inputs are hashed; a test asserts the committed retrieval caches still
  match the hashes recorded at fit time.
- Human judgments remain unassigned. The Stage 6 categories are **LLM proposals**,
  not human labels, and the review sample is enriched for failures, so it cannot
  estimate population error rates.
- HaluEval has no claim-level gold, so the learned combiner is scored on FEVER only.
- The learned model is linear and single-seed; the reported interval is a bootstrap
  over test claims, not a significance test against the fixed rule.
- Probabilities are uncalibrated.
- The three accepted JSON repairs are **not** applied to the dataset; applying them
  would change three HaluEval examples and require regenerating the frozen-rule
  predictions.

## Previous milestone — inference-only (2026-10-20)

At that point the user constraint forbade any training or fitting, and no training
or threshold search was run during that update.

- Safe limited extraction and atomic dataset writes.
- Per-example extraction fallback provenance and quality-report integration.
- Strict rejection of non-string items in claim extraction responses.
- Inference-only threshold detector with evidence and answer aggregation.
- FastAPI detection, extraction, retrieval, verification, saved metrics, and health.
- React inspection UI with evidence details, loading, errors, and fallback warnings.
- Offline replay of committed NLI predictions and human error-review sample.
- Focused dataset regression, detector, and API tests using simulated dependencies.

Validation and limitations at that milestone:

- Offline replay reproduces BM25, dense, and hybrid baselines. Hybrid: 1,998 FEVER
  test claims; accuracy 0.652152, macro-F1 0.632820.
- 37 regression, cache reproduction, and API tests passed; React production build
  passed.
- Pretrained BGE embeddings and DeBERTa NLI weights downloaded and verified.
- Full 247,848-sentence BM25 and exact dense indexes built.
- Ollama is running; requested gemma4:31b-cloud is registered and accessible.
- All 4,000 HaluEval examples were processed with gemma4:31b-cloud, producing 6,462
  claims with three explicitly flagged sentence-split fallbacks (0.075%).
- Quality heuristics flagged 121 examples (3.025%) for human review.
- Frozen-rule inference evaluation is complete on all 600 HaluEval test answers:
  73.3% accuracy (argmax) and 72.0% (the app's fixed rule) with supplied context,
  versus 58.7% and 57.7% whole-answer baselines. Wikipedia retrieval is only 50.5%.
- Ten inference-only comparisons are logged in local MLflow; zero models registered.
- Review artifacts contain 171 extraction cases, 103 verification errors, and three
  syntax-only recovery proposals.
- Browser rendering, missing-model handling, and Gemma-backed detection verified.

See RUNNING.md for setup and commands.
