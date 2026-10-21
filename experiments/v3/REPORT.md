# v3 report — stopped before the test pass

**Outcome:** P1 produced a better calibrator on val, but P2 could not meet the
pre-registered NEI target. v3 stops before P4, so the v3 test pass is **unused**.
The learned detector remains offline.

Baseline: v2 at `e7f0af5` (tag `v2.0`). Branch: `ver-3`.

## TL;DR

| Question | Result |
|---|---|
| Did a Brier-targeted calibrator beat raw on val? | Yes: vector scaling fitted on Brier, **0.42161 vs 0.42303**, ECE 0.0255 |
| Did answerability reduce NEI enough? | **No.** Best NEI coverage under the verifiable constraints is **0.483**; the gate is **0.40** |
| Is there a feasible policy at all? | **No.** Even a joint tau + class-threshold search found zero points meeting all constraints |
| Test pass spent? | **No** — stopped before P4 |
| Serving decision | Learned model remains offline |

## P1 — calibration bake-off (val, out-of-fold)

| Calibrator | Brier | ECE | Beats raw? |
|---|---:|---:|---|
| Raw | 0.42303 | 0.03317 | — |
| Temperature (NLL) | 0.42210 | 0.02301 | Yes |
| Temperature (Brier) | 0.42231 | 0.02620 | Yes |
| Vector scaling (NLL) | 0.42266 | 0.01973 | Yes |
| **Vector scaling (Brier)** | **0.42161** | **0.02548** | **Yes — selected** |
| Isotonic (multiclass) | 0.42595 | 0.03153 | No |

Selection rule was pre-registered: lowest OOF Brier with ECE ≤0.05. Vector scaling
on Brier wins. This is a genuine val improvement over v2’s temperature calibrator,
but it is not enough by itself to justify a test pass without P2.

## P2 — answerability (val)

Target: `gold != INSUFFICIENT`. Features: 30 cached serving features + 3
out-of-fold base-model class probabilities.

| Model | Val AUROC | Val AP |
|---|---:|---:|
| Logistic (best C) | 0.741 | — |
| **Histogram gradient boosting** | **0.765** | **0.863** |

Policy: abstain if `P(answerable) < tau`; otherwise apply the frozen v2 class
thresholds (0.50 / 0.38 / 0.84). The selected fallback point (`tau=0.21`):

| Metric | v2 policy (val) | v3 fallback (val) | Gate |
|---|---:|---:|---:|
| Overall coverage | 0.717 | 0.695 | ≥0.55 |
| Overall precision | 0.728 | 0.741 | ≥0.75 — **fail** |
| Verifiable coverage | 0.811 | 0.800 | ≥0.80 |
| Verifiable precision | 0.961 | 0.962 | ≥0.94 |
| NEI coverage | 0.529 | **0.483** | ≤0.40 — **fail** |

The answerability model reduces NEI coverage by only 4.5 points while losing 1
point of verifiable coverage and still missing overall precision. A joint search
over `tau` and SUPPORTED/CONTRADICTED thresholds (INSUFFICIENT fixed) found **zero**
feasible points. Under the verifiable constraints, the NEI floor is 0.483.

## v3.1 diagnostic — expanded features (val only)

Before writing a new protocol, I added candidate serving features and re-ran the
answerability model:

| Feature set | Features | Val AUROC | Best NEI at verifiable gates | Feasible? |
|---|---:|---:|---:|---|
| Base (30 + raw probs) | 33 | 0.7654 | 0.483 | No |
| + uncertainty (entropy, margin, NEI mass) | 38 | **0.7690** | 0.483 | No |
| + coverage-flag score | 34 | 0.7653 | 0.477 | No |
| + uncertainty + coverage | 39 | 0.7643 | 0.480 | No |
| + interactions | 41 | 0.7666 | 0.480 | No |

The best NEI floor is **0.477**, still far from the 0.40 gate. The coverage score
adds only ~0.006 NEI reduction. The extra features do not make the gate reachable.

## Decision

v3 **stops before P4**. The pre-registered NEI and overall-precision gates are not
reachable on val, even with the expanded feature sets, so spending the single test
pass would only confirm a known failure. No v3 test predictions exist and no serving
change was made.

The learned detector remains offline; the fixed 0.7/0.95 rule stays the default.
This is a negative result for answerability, not a failure of the calibration work.

## What would change it

1. **A stronger NEI signal.** The summary features, including the coverage score
   and uncertainty, cannot separate NEI from verifiable claims well enough (AUROC
   stays 0.769; NEI floor 0.477). Remaining candidates: claim-level evidence-coverage
   labels, a retrieval-failure classifier trained with claim-level labels, or more
   NEI training examples.
2. **Relaxed targets with a pre-registered rationale.** For example NEI ≤0.48 and
   overall precision ≥0.74. That would pass on val, but it is a weaker claim; it
   should be a deliberate v3.1 protocol, not a post-hoc slide.
3. **More data.** The answerability model is trained on 3,000 FEVER claims; NEI is
   only one third of them. More NEI examples or a better label source would help.

## Artifacts

- `experiments/v3/protocol.json` — amended with the stop reason and feature choice
- `experiments/v3/calibration.json` — bake-off
- `experiments/v3/answerability.json` — model, policy search, feasibility
- `experiments/v3/answerability_diagnostic.json` — v3.1 expanded-feature diagnostic
- `experiments/v3/policy.json` — fallback policy, marked `fallback_not_promoted`
- `experiments/v3/val_predictions_v3.jsonl` — calibration output
- `experiments/v3/val_policy_v3.jsonl` — per-row answerability policy output

No test data was read. Tests cover the new calibration methods; the project suite
remains green (see `python -m pytest -q`).
