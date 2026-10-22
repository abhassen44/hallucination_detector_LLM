# Run the inference application

Serving the API requires no training. The Stage 5 combiner is already fitted and
committed; the API serves the fixed-rule baseline by default. Refitting is an
offline experiment stage described below.

## Environment

Use Python 3.10+ and Node.js 20.19+ (or 22.12+).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

The lightweight requirements support API startup and offline tests. For actual
NLI inference and retrieval, also install the model dependencies:

```sh
pip install -r requirements-inference.txt
```

PyTorch uses CPU when CUDA is unavailable. CUDA is optional, including on macOS.
Exact local dependency snapshots are recorded in `requirements-app.lock.txt`
(lightweight checks) and `requirements-inference.lock.txt` (including inference
libraries). They reflect the current macOS/Python 3.13 environment.

The first inference downloads pretrained Hugging Face models and builds retrieval
indexes, which can take several minutes and requires disk space and internet.
On macOS, exact dense search uses NumPy inner products instead of importing
FAISS into the PyTorch process, avoiding incompatible OpenMP runtimes in native
wheels. Other platforms retain FAISS. macOS embeddings use Metal (MPS) when
available; set `embeddings.device: cpu` in config to override it. Both indexes use
the same normalized embedding vectors and cosine-scoring rule.

These are inference/setup operations, not training. Use one server worker to avoid
duplicating model memory.

## Ollama and datasets

Run Ollama separately and ensure `config.yaml` points to an available model.
The selected extraction model is `gemma4:31b-cloud`, as configured in
`config.yaml`. The model name is recorded in API responses and dataset metadata.
`OLLAMA_MODEL` is an optional explicit override for separate experiments; do not
set it when reproducing this cloud-model run.

Cloud models may need authentication and incur provider usage. Check availability
with `curl http://localhost:11434/api/tags` before bulk extraction.

The cloned repository contains FEVER corpus, retrieval caches, and dataset splits.
If missing, prepare datasets with `python -m src.datasets.prepare --dataset all`.
This can download a large Wikipedia archive.

Build the retrieval indexes and check pretrained NLI before serving:

```sh
python -m src.setup_inference
```

The current local setup uses `gemma4:31b-cloud`; cloud access was verified. Local
Qwen was used only for diagnostics and is not the selected extraction model.
Cloud requests use the JSON-requesting prompt and local parsing, because Ollama
cloud does not support schema-constrained output formats. Local models use a
claim-array schema to avoid JSON generation loops.

This writes a readiness record to `experiments/inference_setup/readiness.json`
without training or contacting Ollama.

## React and API

```sh
npm ci --prefix frontend
npm run build --prefix frontend
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the interface and `/docs` for the API explorer.
Build the frontend before starting the API so its assets are mounted. For frontend
development, run `npm run dev --prefix frontend`; Vite proxies requests to port 8000.

Endpoints:

- `GET /health`: process health and whether detector models have loaded; not proof
  that Ollama or all model dependencies are ready.
- `POST /claims`: `{ "answer": "...", "question": "..." }`.
- `POST /retrieve`: `{ "claim": "...", "method": "hybrid", "top_k": 5 }`.
- `POST /verify`: `{ "claim": "...", "evidence": ["..."] }`.
- `POST /detect`: `{ "question": "...", "answer": "...", "method": "hybrid" }`.
- `GET /metrics`: saved Stage 2–6 experiment results, not live performance.

Input is limited to 12,000 characters per text, 20 evidence passages, and 50
extracted claims per answer. The API serializes inference to bound model memory.
Model/index initialization errors and inference failures return HTTP 503. This
local research demo has no authentication; keep it bound to localhost.

## Supplied-source verification

The optional **Source text** field checks the answer against your passage instead
of searching Wikipedia. The corresponding API field is `evidence: ["passage"]`.
Omit it or use null for Wikipedia search. An explicit empty array checks against
no evidence and yields INSUFFICIENT; it never silently falls back to search.
Provided-source detection and `/verify` do not initialize the Wikipedia retriever.

Supported means supported by the supplied evidence, not independently proven true.
The verifier has a 512-token input window; long passages may be truncated. Use
concise source passages containing the relevant facts and inspect the evidence.

## Baseline interpretation

The detector uses hybrid retrieval plus NLI and the existing fixed Stage 3
thresholds (entailment 0.7, contradiction 0.95). No new thresholds are selected.
Answer labels are CONTRADICTED if any claim is contradicted, SUPPORTED if every
claim is supported, and INSUFFICIENT otherwise. No claims yields INSUFFICIENT.
This heuristic aggregation has now been evaluated on 600 HaluEval test answers;
see the frozen-rule report below. No new thresholds were selected.
The API exposes NLI signals rather than calibrated confidence in factual truth.

## Learned detector (Stage 5)

Stage 5 is fitted offline; the API does **not** load it.

```sh
.venv/bin/python -m src.evaluation.fit_detector --method all
.venv/bin/python -m scripts.summarize_stage5
```

This writes features from the committed retrieval caches, selects `C` and feature
groups on val, evaluates test once, and writes
`experiments/stage5_learned/{metrics.json,metrics_ablations.json,REPORT.md}` plus
per-claim predictions. `protocol.json` records the cache hashes, the fit/selection/
report split, and whether test was evaluated. A second run refuses to re-evaluate
test unless you pass `--allow-test-rerun`, which marks the result exploratory.

## Val-only improvement studies

```sh
.venv/bin/python -m src.evaluation.val_studies --study all
```

Reranks cached candidates with the pretrained `BAAI/bge-reranker-base` and rebuilds
NLI premises with neighbouring sentences, on the **val split only**. No test pass,
no fitting, no threshold search. Results go to `experiments/stage5_learned/`.

On some CUDA driver combinations the cross-encoder raises
`CUDA error: an illegal memory access`. Pass `--device cpu` (and one method at a
time, `--methods hybrid`) to finish deterministically; results merge into the same
JSON file. The study is a one-off analysis, so CPU speed is acceptable.

## Post-hoc evidence (no new test pass)

```sh
.venv/bin/python -m src.evaluation.significance
.venv/bin/python -m src.evaluation.decomposition
```

`significance` replays the fixed rule from the retrieval cache, checks it against the
committed Stage 3 accuracy/macro-F1, then runs McNemar and a 10,000-draw paired
bootstrap on the committed per-claim predictions. `decomposition` restricts every
system to the 1,324 verifiable test claims and caches frozen Stage 2 NLI-concat
verdicts (`experiments/stage5_learned/oracle_predictions_test.jsonl`) so errors can
be split into abstention and confident-error buckets. Neither script fits anything,
changes a threshold, or evaluates a new system. Both write JSON next to the Stage 5
artifacts.

## Blind taxonomy pilot

```sh
.venv/bin/python -m scripts.make_pilot          # only if the queues need rebuilding
.venv/bin/python -m scripts.review \
  --queue experiments/pilot_v1/extraction_queue.jsonl --reviewer <name>
.venv/bin/python -m scripts.review \
  --queue experiments/pilot_v1/verification_queue.jsonl --reviewer <name>
.venv/bin/python -m scripts.pilot_agreement \
  --queue experiments/pilot_v1/extraction_queue.jsonl \
  --queue experiments/pilot_v1/verification_queue.jsonl
```

Do not pass `--suggestions` during the pilot. `--force` on `make_pilot` destroys
existing judgments. Gate and protocol: `experiments/pilot_v1/PILOT.md`.

The v1 pilot is complete (pass with a completeness deviation: κ 0.88, category
agreement 0.80, 39/40 rows). `--allow-incomplete` scores annotated rows and bounds
κ over every completion of the missing verdicts; use it only with that deviation
recorded.

## Review tooling

```sh
# Gemma first pass (proposals only; human fields untouched)
.venv/bin/python -m scripts.analyze_errors --queue all

# Interactive human review, optionally showing the proposals
.venv/bin/python -m scripts.review \
  --queue experiments/stage4_claims/human_review.jsonl \
  --suggestions experiments/stage6_errors/provisional_extraction.jsonl

# Progress without prompting
.venv/bin/python -m scripts.review --summary
```

The queue is backed up before the first write, completed rows are never offered
again, and `review_progress.json` records counts. Suggestions are keyed by
`(id, condition)` because verification rows repeat claims across evidence conditions.
See `docs/REVIEW_GUIDE.md`.

Every stage is also available from one entry point:

```sh
.venv/bin/python -m scripts.reproduce --stage checks
.venv/bin/python -m scripts.reproduce --stage all --include-heavy
```

If a content-generation/parsing error occurs, extraction falls back to sentence
splitting. Authentication failures and persistent rate limits instead stop the
bulk run; queued requests are cancelled and completed outputs remain cached.
HTTP 429 responses trigger shared backoff and respect numeric Retry-After values. The API displays a
warning, and dataset outputs record fallback provenance in example metadata.

## Offline baseline reproduction

```sh
.venv/bin/python -m src.evaluation.cached_baseline --method hybrid
```

This replays fixed decisions from the saved test cache without model calls,
training, fitting, or tuning. It writes metrics and a deterministic sample of up
to 20 failures per gold class to `experiments/baseline_review/`. Category and notes
fields are intentionally blank for human review. This is a reproduction check,
not a new independent benchmark. The cache is tied to the original corpus and
model; regenerate it when those change.

## Inference-only evaluation and tracking

```sh
# After downloading models/indexes with src.setup_inference:
HF_HUB_OFFLINE=1 .venv/bin/python -m src.evaluation.halueval_claims --evidence all
.venv/bin/python -m scripts.finalize_inference_provenance
.venv/bin/python -m scripts.summarize_inference
# Optional local MLflow logging, no external tracking account:
pip install -r requirements-tracking.txt
.venv/bin/python -m scripts.log_inference_mlflow
```

Results are in `experiments/stage5_inference/REPORT.md`. Ten frozen-rule
comparisons use the same 600 test answers. No fitting, new cloud generation, or
threshold search occurs. Provided-knowledge claim verification scores 73.3%
accuracy with argmax and 72.0% with the app's fixed rule; the corresponding
whole-answer baselines score 58.7% and 57.7%. The provided-context condition is an
evidence upper bound, not a retrieval result. Earlier Stage 2's 70.8% whole-answer
result used a different, validation-tuned threshold and is not a same-rule ablation.

Wikipedia retrieval is weak on this dataset: the fixed rule scores 50.5% accuracy
and flags 595/600 answers. Its F1 near 0.668 is close to always flagging every answer
(F1 0.667 on the balanced set); it should not be read as strong performance.

NLI predictions are checkpointed in `data/cache/halueval_nli_pairs.jsonl` and keyed
by inputs and model configuration. A sidecar checks runtime versions and the
resolved cached model snapshot. Interrupted final cache records can be recomputed;
completed corrupt records fail clearly. Original dataset, corpus, index, model
files and runtime fingerprints are recorded with the report. Preserve the frozen
results when changing inputs or models: choose separate `--output` and `--cache`
paths for a new run.

MLflow stores ten inference runs in workspace-local `data/cache/mlflow.db` with
artifacts under `mlartifacts/`. No models are trained or registered. The run-ID
manifest is `experiments/stage5_inference/mlflow_runs.json`. Tracking telemetry is
disabled for this exporter. Optional tracking dependencies are separate from the
lightweight app/test and inference requirements.

## Remaining work

1. Human review: 171 extraction cases (121 flagged plus 50 comparisons) and 103
   verification error samples are prepared and pre-annotated by Gemma as
   *proposals*. Use `scripts/review.py`; the fields stay null until a human decides.
2. Three syntax-only recovery proposals were reviewed and accepted but **not applied**.
   Applying them changes three HaluEval examples, so it belongs in a separate version
   with regenerated frozen-rule predictions.
3. Cross-encoder reranking and neighbour-sentence premises were evaluated on val only.
   Promoting either requires recomputing train/test features and refitting Stage 5.
4. The retrieval corpus remains FEVER-derived; open-domain coverage is the main limit
   on the HaluEval Wikipedia numbers.

Stage 5 training is limited to the FEVER train split. HaluEval labels are answer-level,
not claim-level gold, so no claim-level HaluEval score is reported anywhere. The
original dataset labels, questions, answers, contexts, group IDs, and row counts were
verified unchanged during extraction.
# Experimental inference improvements

The original all-passages threshold baseline remains the default. The
**Evidence to use** dropdown offers **First retrieved passage · experimental**;
on cached FEVER validation it reduces distractor-driven errors, but it can also
discard evidence needed for multi-sentence claims. These are validation-only
findings, not a new test score. Supplied-source checks keep all supplied text.
See `experiments/improvements/REPORT.md` for the results and tradeoffs.

API clients can pass `evidence_policy: "top1"` to `/detect` or `/verify`.
`/detect` additionally supports `top3` and `same_page` for retrieved evidence;
`same_page` requires page provenance and cannot be used with plain supplied text.
Every claim retains its verdict and now includes a separate `decision.status`:
`answer` or `abstain`, plus review reasons. Abstention is not another factual
label. Missing evidence alone does not establish that an answer is false.

Optional pretrained reranking is off by default. To install its weights once,
from the repository root on macOS/Linux:

```sh
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "from src.retrieval.rerank import Reranker; Reranker()"
```

This downloads a pretrained model and does not train or fit anything. Refresh the
app after setup. `/health` reports `reranker_available`, and the checkbox remains
disabled until weights and tokenizer are present. The server loads only local
weights: an unavailable optional reranker returns 503 with a setup explanation.
Set `rerank: true` on `/detect` or `/retrieve` to use it; supplied-source requests
bypass search and reranking. The reranking/focused-selection combination is
experimental and has not been evaluated on a new held-out set.

## Optional second evidence verifier

Enable **Double-check contradictions**, or add `grounded_review: true` to `/detect`
or `/verify`. This uses the configured `gemma4:31b-cloud` model for extra inference
on contradictions and conflicting evidence; it does not train or fit a model.
It is off by default and adds cloud requests and latency.

The second model sees only the claim and passages. It must provide exact evidence
quotes to confirm a factual verdict. Disagreement or invalid output causes an
INSUFFICIENT verdict and operational abstention. The original verdict remains in
`baseline_label`, with the separate `grounded_review` explanation and citations.
Uncontested supported verdicts are not all double-checked by this policy.

On the 999 reused FEVER validation claims, hybrid first-passage macro-F1 improved
0.6919 → 0.7192 and accepted accuracy 74.12% → 82.92%. This is not a new test score
or a 90% reliability guarantee. See `experiments/grounded_pilot_999/REPORT.md` for
regressions, quote failures, latency limitations, and replay commands.
