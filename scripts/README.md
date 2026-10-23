# Script layout

Research scripts are grouped by the version that introduced them. Shared tooling is
kept separate so historical experiment commands remain traceable without mixing
version-specific code.

```text
scripts/
├── reproduce.py                 # one orchestration entry point
├── v2/                          # frozen v2 calibration, policy, audit
├── v3/                          # v3 calibration and answerability diagnostics
├── v4/                          # reranker, data expansion, 9k experiments
├── v5/                          # grounding caches, ablation, freeze, test pass
└── shared/
    ├── data/                    # cache/provenance audits
    ├── inference/               # Stage 5 inference and provenance summaries
    ├── reporting/               # charts and component-ablation report
    └── review/                  # human review, pilots, recovery tooling
```

## Commands

Run modules from the repository root. The version folder is part of the module path:

```sh
python -m scripts.v2.calibrate_v2
python -m scripts.v3.answerability_v3
python -m scripts.v4.retrieval_stack_v4
python -m scripts.v5.evaluate_v5_test --check-val
python -m scripts.v5.evaluate_results_v5
python -m scripts.shared.review.review --summary
python -m scripts.shared.reporting.generate_charts
python -m scripts.shared.reporting.build_presentation
python -m scripts.reproduce --stage checks
```

`python -m scripts.v5.evaluate_v5_test` without `--check-val` is guarded: the v5
FEVER test pass has already been spent and must not be rerun.

## Version map

| Folder | Scope |
|---|---|
| `v2` | Calibration, selective policy, coverage flag, label/audit queues |
| `v3` | Brier-targeted calibration and answerability diagnostics |
| `v4` | Reranker, expanded train splits, 9k tuning, retrieval stack |
| `v5` | MiniCheck/HHEM grounding features, ablation, freeze, one test pass |
| `shared/data` | Dataset cache and extraction provenance audits |
| `shared/inference` | Stage 5/HaluEval inference summaries and optional MLflow export |
| `shared/reporting` | Results dashboard, component-ablation report, presentation deck |
| `shared/review` | Pilot, human-review queues, recovery proposals |

Cross-version imports are explicit, for example:

```python
from scripts.v2.calibrate_v2 import fit_base
from scripts.v3.answerability_v3 import answerability_policy
from scripts.shared.review.pilot_agreement import _cohen_kappa
```

Filenames retain their version suffix inside version folders to keep history and
imports unambiguous.
