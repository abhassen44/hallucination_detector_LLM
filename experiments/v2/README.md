# v2 — calibrated, coverage-aware claim verification

Track A (validity/serving) with a Track B retrieval stretch. The pre-registered
protocol is [protocol.json](protocol.json); the v1 baseline is tag `v1.0`
(commit `f79f0ff`) and its artifacts are read-only.

## Status

| Phase | Scope | Status |
|---|---|---|
| P0 | Freeze v1, scaffold, protocol, seed pilot judgments | **Done** |
| P1 | Taxonomy v2, proposals, 274/274 labels, 20% blind re-annotation | Proposals done; annotation pending |
| P2 | Calibration + training variance (val only) | Pending |
| P3 | Selective policy + coverage flag (val only) | Pending |
| P4 | One test pass for the calibrated system | Pending |
| P5 | Serving decision / feature flag | Pending |
| P6 | Retrieval variant (stretch, one test pass) | Pending |

## P0 result

- Tag `v1.0` at `f79f0ff`.
- Queues seeded from the prepared full queues: **171 extraction / 103 verification**,
  all `(id, condition)` keys unique.
- The 40-row pilot overlaps the full queues. **37 verdicts were copied** (17
  extraction, 20 verification); 3 extraction rows remain pending (two have partial
  notes, one is untouched). They are listed in [queue_seed.json](queue_seed.json).
- Seed rows are keyed by `(id, condition)`; the pilot copies are never modified.

## P1 state

- [taxonomy.json](taxonomy.json) locks the v2 categories. Verification renames the
  v1 `nli_failure` catch-all to `verifier_error` and keeps `ambiguous_evidence`
  distinct. Extraction collects **Pass/Fail only** (no causes, no diagnostic
  Booleans) for this pass.
- [proposals/verification.jsonl](proposals/verification.jsonl) was regenerated under
  taxonomy v2 (103/103 usable). Extraction proposals are copied from the frozen v1
  file because the Pass/Fail task is unchanged.
- Three seeded pilot rows used the v1 `nli_failure` label; all three had clear
  evidence and wrong NLI verdicts, so they were mapped to `verifier_error`. The
  mapping is listed in [queue_seed.json](queue_seed.json).
- Pending annotation: **154 extraction rows**, **83 verification rows**.

## Layout

```
experiments/v2/
  protocol.json              # frozen gates, split hashes, test budget
  queue_seed.json            # what was copied from the pilot, source hashes
  queues/
    extraction.jsonl         # 171 rows, 17 seeded verdicts
    verification.jsonl       # 103 rows, 20 seeded verdicts
  proposals/                 # taxonomy-v2 LLM proposals (P1)
  human_labels_*.jsonl       # filled in P1
  agreement.json             # P1 re-annotation agreement
  calibration.json           # P2
  risk_coverage.json         # P3
  coverage_flag.json         # P3
  policy.json                # frozen selective policy (before P4)
  metrics_v2.json            # P4
  REPORT.md                  # P4 write-up
```

## Commands

```powershell
$env:PYTHONPATH="$PWD"

# Resume the full annotation pass (writes review_progress.json beside the queue)
.\.venv\Scripts\python.exe -m scripts.review `
  --queue experiments/v2/queues/extraction.jsonl `
  --suggestions experiments/v2/proposals/extraction.jsonl `
  --taxonomy experiments/v2/taxonomy.json `
  --reviewer Abhas

.\.venv\Scripts\python.exe -m scripts.review `
  --queue experiments/v2/queues/verification.jsonl `
  --suggestions experiments/v2/proposals/verification.jsonl `
  --taxonomy experiments/v2/taxonomy.json `
  --reviewer Abhas

# Rebuild the seeded queues from scratch (overwrites the manifest)
.\.venv\Scripts\python.exe -m scripts.seed_v2_queues --force
```

`review.py` does not overwrite completed fields, so seeded rows are skipped.
`.bak`, `review_progress.json`, and `*.log` files are gitignored.

## Guardrails

- Thresholds and the selective policy are fixed on val before the test pass.
- One test pass for the calibrated system, one for the retrieval variant.
- No dataset mutation; v2 labels live only under `experiments/v2/`.
- The fixed rule stays the serving default unless every promotion gate passes.
