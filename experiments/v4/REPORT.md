# v4 report — stopped before the test pass

**Outcome:** the reranker helped macro-F1, and more data improved NEI separation,
but the 9k system still misses the pre-registered NEI gate. v4 stops before P5.
The v4 test pass is **unused** and the learned detector remains offline.

Baseline: v3 at `e9974c6`. Branch: `ver-4`.

## TL;DR

| Experiment | Result |
|---|---|
| Reranker + refit, top-5 | Val macro-F1 **0.69296** vs 0.67636; INSUFFICIENT F1 **0.53559** vs 0.52073 |
| Reranker top-8 | 0.67732; no gain over v1 — top-5 kept |
| Train 6k | Answerability AUROC **0.781**, NEI floor **0.453** |
| Train 9k | AUROC **0.787**, NEI floor **0.423** |
| 9k tuning (NEI weight 3) | AUROC **0.794**, NEI **0.408** |
| Joint threshold search | No feasible point; best NEI **0.417** at vcov 0.809 |
| Test pass | **Not spent** |
| Serving | Learned stays offline |

## P1 — reranker + refit

| Arm | Val macro-F1 | INSUFFICIENT F1 | Keep? |
|---|---:|---:|---|
| v1 hybrid | 0.67636 | 0.52073 | baseline |
| rerank top-5 | **0.69296** | **0.53559** | **Yes** |
| rerank top-8 | 0.67732 | 0.51581 | No |

Top-5 is a real macro-F1 gain (+0.0166) and improves INSUFFICIENT F1 (+0.0149).
Top-8 dilutes it. The learned combiner absorbs most of the fixed-rule gain at top-8.

## P2 — more training data

Expanded the train set with the 6,000 unused `paper_dev` claims, balanced per class:

| Train size | Base val macro-F1 | Answerability AUROC | Best NEI | Overall precision |
|---:|---:|---:|---:|---:|
| 3,000 (v4 reranker) | 0.69296 | — | — | — |
| 6,000 | 0.69061 | 0.781 | 0.453 | 0.749 |
| 9,000 | 0.68515 | 0.787 | 0.423 | 0.761 |

More data clearly improves NEI separation (+0.021 AUROC from 3k to 9k; NEI floor
0.483 → 0.423), but it also lowers base macro-F1 on val. The macro-F1 and NEI goals
are pulling in different directions.

## P3 — answerability tuning

At 9k, a cost-sensitive HGB (NEI weight 3) reached:

- AUROC **0.794**
- NEI coverage **0.408** at verifiable coverage **0.800**, overall precision **0.768**

That is still **0.008 above** the 0.40 gate. A joint search over `tau` and the
SUPPORTED/CONTRADICTED thresholds found **no feasible point**; the best NEI under
the verifiable constraints was **0.417**. The gate is not reachable on val with
these features.

## Decision

v4 stops before P5. No test predictions exist; the v4 test pass remains unused.
The learned detector stays offline and the fixed 0.7/0.95 rule remains the default.

This is a negative serving result, but a useful one: the retrieval and data levers
both move the metrics, just not far enough to clear the NEI gate without losing
verifiable coverage.

## What would change it

1. A stronger NEI signal than the current 30 features: claim-level evidence
   labels, a dedicated retrieval-failure classifier, or more NEI examples.
2. A two-stage verdict model with hard negatives (delete/swap evidence) rather
   than more threshold tuning.
3. A deliberate v4.1 protocol with a weaker target (e.g. NEI ≤0.42) — a weaker
   claim, not an improvement.

## Artifacts

- `experiments/v4/retrieval_stack.json` — P1 arms
- `experiments/v4/data_expansion_6000.json` — 6k
- `experiments/v4/data_expansion_9000.json` — 9k
- `experiments/v4/answerability_tuning_9000.json` — tuned + joint search
- `data/splits/fever_train_v4_{6000,9000}.jsonl` — expanded splits
- Reranker caches under `data/processed/rerank/` (not committed)

No test data was read.
