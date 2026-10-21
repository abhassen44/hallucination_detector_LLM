# v1 blind pilot (B1–B3)

Validation of the error taxonomy before annotating all 274 review rows. The pilot
is **blind**: the queues contain no LLM proposal, rationale, or confidence, and you
should not open `experiments/stage6_errors/` until both queues are finished.

- 20 extraction cases: deliberately **10 Pass / 10 Fail** across the proposals.
- 20 verification cases: stratified across proposed causes (10 retrieval,
  7 NLI, 1 each entity/multi-hop/numerical), at most one condition per example id.
- This is a **balanced validation sample, not a prevalence sample**. Agreement
  numbers describe the taxonomy, not how often the pipeline is wrong.

## Gate (declared before annotation)

| Check | Threshold |
|---|---|
| Cohen's κ, extraction Pass/Fail | ≥ 0.60 |
| Exact category agreement, verification errors | ≥ 0.50 |

Pass → annotate the remaining 274 rows. Fail → revise the taxonomy first.

## Commands

From the project root, one queue at a time:

```powershell
$env:PYTHONPATH="$PWD"
python -m scripts.review --queue experiments/pilot_v1/extraction_queue.jsonl --reviewer <your-name>
python -m scripts.review --queue experiments/pilot_v1/verification_queue.jsonl --reviewer <your-name>
```

Do **not** pass `--suggestions`. Ctrl+C saves and exits; rerun to resume — completed
rows are not offered again. Each run writes `review_progress.json` in
`experiments/pilot_v1/`.

Extraction pilot fields: open coding (optional), Pass/Fail, category. The four
quality booleans are out of pilot scope, so the pilot stays focused on the fields
that have a proposal to validate. Verification fields: open coding, Pass/Fail,
category.

## Scoring

```powershell
python -m scripts.pilot_agreement `
  --queue experiments/pilot_v1/extraction_queue.jsonl `
  --queue experiments/pilot_v1/verification_queue.jsonl
```

Writes `experiments/pilot_v1/pilot_agreement.json` with κ, exact category
agreement, disagreement ids, seconds per case (from timestamps, breaks > 10 min
excluded), and the gate verdict.

Rebuilding the queues with `--force` destroys judgments. Proposals are keyed by
`(id, condition)`; verification rows repeat ids across evidence conditions, so
never key them by id alone.

## Result (v1)

| Check | Observed | Gate |
|---|---:|---:|
| Extraction Pass/Fail κ, 17 labeled verdicts | 0.881 | ≥ 0.60 |
| Verification category agreement, 20/20 | 0.80 | ≥ 0.50 |
| Verification category κ | 0.709 | — |

**Verdict: pass with a completeness deviation.** 39/40 rows were annotated; three
extraction rows have a proposal but no binary verdict. All eight possible
completions of those verdicts give κ between 0.600 and 0.900, so the decision is
invariant to the missing judgments. Scores: `pilot_agreement.json`.

Observed disagreements: four verification rows, all proposed `nli_failure` (two
ambiguous evidence, one claim-extraction failure, one multi-hop). Extraction cause
labels diverged on 3/7 Fail rows. The gate passes; category definitions should be
tightened before the full pass.
