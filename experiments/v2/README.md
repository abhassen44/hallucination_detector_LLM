# v2 — calibrated, coverage-aware claim verification

Track A (validity/serving) with a Track B retrieval stretch. The pre-registered
protocol is [protocol.json](protocol.json); the v1 baseline is tag `v1.0`
(commit `f79f0ff`) and its artifacts are read-only.

## Status

| Phase | Scope | Status |
|---|---|---|
| P0 | Freeze v1, scaffold, protocol, seed pilot judgments | **Done** |
| P1 | Taxonomy v2, proposals, 274/274 labels, optional human audit | **Done** — 37 human + 237 LLM-proposed labels; audit optional |
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
- **Labels are materialized, not human-annotated beyond the pilot.** 37 rows keep
  their human pilot judgment (`label_source: human_queue`); 237 take the LLM
  proposal (`label_source: llm_proposal`, `human_reviewed: false`). See
  [label_manifest.json](label_manifest.json).
- Measured label noise on the pilot: extraction κ **0.881** (17 rows), verification
  category agreement **0.75** (20 rows; the v1 figure 0.80 used the old categories).
- [audit_sheet.csv](audit_sheet.csv) is a blind 40-row stratified audit, disjoint
  from the pilot. Completing it upgrades the claim to “LLM-proposed labels,
  human-audited on a 40-row sample.” It is **not** a substitute for the full pass.

## Layout

```
experiments/v2/
  protocol.json              # frozen gates, split hashes, test budget
  queue_seed.json            # what was copied from the pilot, source hashes
  queues/
    extraction.jsonl         # 171 rows, 17 seeded verdicts
    verification.jsonl       # 103 rows, 20 seeded verdicts
  proposals/                 # taxonomy-v2 LLM proposals (P1)
  labels_llm_proposed_*.jsonl # 274 materialized labels (37 human, 237 silver)
  label_manifest.json        # label policy, counts, pilot agreement
  audit_sheet.csv            # optional blind 40-row human audit
  audit_manifest.json        # audit sample keys and hidden proposals
  audit_results.json         # scorer output, if the audit is completed
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

# Rebuild the 274 labels from the queues + proposals
.\.venv\Scripts\python.exe -m scripts.materialize_v2_labels --force

# Build the blind 40-row audit, then score it after filling the CSV
.\.venv\Scripts\python.exe -m scripts.make_v2_audit --force
.\.venv\Scripts\python.exe -m scripts.score_v2_audit

# Optional: replace silver labels with a full human pass
# See docs/REVIEW_GUIDE.md for the extraction and verification commands.
```

`review.py` does not overwrite completed fields, so seeded rows are skipped. If a
human pass is run, rerun `materialize_v2_labels --force` to promote the new human
verdicts into the label artifacts.
`.bak`, `review_progress.json`, and `*.log` files are gitignored.

## Guardrails

- Thresholds and the selective policy are fixed on val before the test pass.
- One test pass for the calibrated system, one for the retrieval variant.
- No dataset mutation; v2 labels live only under `experiments/v2/`.
- The fixed rule stays the serving default unless every promotion gate passes.
- **Label integrity:** 237 of the 274 labels are LLM proposals. Reports must not
  call them human annotations; quote the pilot noise estimate and the audit status.
