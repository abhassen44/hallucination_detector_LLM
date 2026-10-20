# Current implementation status

User constraint: do not train or fit any models. No training or threshold search
was run during this update.

Implemented:

- Safe limited extraction and atomic dataset writes.
- Per-example extraction fallback provenance and quality-report integration.
- Strict rejection of non-string items in claim extraction responses.
- Inference-only threshold detector with evidence and answer aggregation.
- FastAPI detection, extraction, retrieval, verification, saved metrics, and health.
- React inspection UI with evidence details, loading, errors, and fallback warnings.
- Offline replay of committed NLI predictions and human error-review sample.
- Focused dataset regression, detector, and API tests using simulated dependencies.

Validation and limitations:

- Offline replay reproduces BM25, dense, and hybrid baselines. Hybrid: 1,998 FEVER test claims; accuracy 0.652152, macro-F1 0.632820.
  Reproduces the committed hybrid threshold baseline; not a newly trained result.
- 37 regression, cache reproduction, and API tests pass; React production build passes.
- Pretrained BGE embeddings and DeBERTa NLI weights downloaded and verified.
- Full 247,848-sentence BM25 and exact dense indexes built. macOS uses NumPy
  inner products to avoid a native FAISS/PyTorch OpenMP conflict.
- Ollama is running; requested gemma4:31b-cloud is registered and accessible.
- Live API smoke checks pass, including Gemma extraction without fallback and
  optional supplied-source detection.
- A 20-example HaluEval sample produced 24 claims with no fallback or heuristic
  flags. All 4,000 examples have now been processed with gemma4:31b-cloud.
  Reprocessing the cache with the parser fixes produced 6,462 claims, with three
  explicitly flagged sentence-split fallbacks (0.075%). All original questions,
  answers, labels, contexts, group IDs, and row counts were preserved.
- Quality heuristics flagged 121 examples (3.025%) for human review; these flags
  are not measured extraction errors. The raw cache contains all 4,000 responses.
- Frozen-rule inference evaluation is complete on all 600 HaluEval test answers.
  Claim decomposition with provided context: 73.3% accuracy (argmax), 72.0%
  (the app's fixed rule), versus whole-answer baselines of 58.7% and 57.7%.
- BM25, dense and hybrid Wikipedia retrieval comparisons are complete. Their
  fixed-rule accuracy is only 50.5%; hybrid flags 595/600 answers. This remains
  a substantial limitation, not a successful open-domain benchmark.
- Ten inference-only comparisons are logged in local MLflow; zero models are
  registered. Corpus/index/model fingerprints and runtime versions are saved.
- The app accepts optional source text and skips Wikipedia retrieval in that mode.
- Review artifacts contain 171 extraction cases, 103 verification errors, and
  three syntax-only recovery proposals. All human judgments remain unassigned.
- Learned classifier fitting and full learned-model ablations remain deferred. No synthetic or answer-level labels were substituted for gold
  claim labels.
- Browser rendering, missing-model handling, and Gemma-backed detection verified.
  A paraphrased claim in the browser sample triggered a false contradiction: the
  existing threshold baseline remains sensitive to wording and conflicting evidence.
  A conflict warning now exposes simultaneous strong support/refutation without
  changing the baseline classification rule.
- Human error categories are not assigned automatically; annotation is pending.
- Unit/API tests simulate dependencies; the separate live smoke report verifies
  real model/index integration, not benchmark accuracy.

See RUNNING.md for setup and commands. Older Stage 4 'Done' labels refer to code
implementation, not completed dataset processing or validated extraction quality.
