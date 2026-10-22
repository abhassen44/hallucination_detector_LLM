# Claim-Level Hallucination Detection System

**Current branch:** `ver-5`, based on the v4.0 research line. v5 adds pretrained MiniCheck grounding features, passed all eight predeclared research gates, and spent its one FEVER test pass. The learned model remains offline: its verifiable precision is high only at a selective operating point, and NEI answer precision is still poor. See [v5 report](experiments/v5/VALIDATION.md), [plan/outcome](docs/PLAN_V5.md), [implementation status](docs/IMPLEMENTATION_STATUS.md), [application setup](docs/RUNNING.md), and canonical [results](docs/RESULTS.md). No serving promotion is claimed.

A research NLP system for detecting hallucinations in LLM-generated text at the **individual claim level**. Rather than classifying an entire response as binary true/false, the pipeline breaks text into atomic factual propositions, retrieves supporting/refuting evidence from Wikipedia using hybrid search (BM25 + FAISS Dense), verifies each claim with DeBERTa NLI and semantic similarity, and predicts a fine-grained 3-way label:

- **SUPPORTED** by evidence
- **CONTRADICTED** by evidence
- **INSUFFICIENT** evidence to verify

Comprehensive experimental tracking and metrics are detailed in [docs/PROGRESS.md](docs/PROGRESS.md).

---

## 🏗️ Architecture

```text
               ┌──────────────────────┐
               │    User Question     │
               └──────────┬───────────┘
                          ↓
               ┌──────────────────────┐
               │  LLM Answer Gen      │ (Ollama: gemma4:31b-cloud)
               └──────────┬───────────┘
                          ↓
               ┌──────────────────────┐
               │   Claim Extraction   │ (Decompose into atomic claims)
               └──────────┬───────────┘
                          ↓
        ┌────────────────────────────────────┐
        │        Hybrid Retrieval            │
        │   BM25 (Sparse) + FAISS (Dense)    │ (248k Wikipedia sentences, RRF)
        └─────────────────┬──────────────────┘
                          ↓
        ┌────────────────────────────────────┐
        │     Claim-Evidence Verification    │
        │   DeBERTa-v3-base-mnli + BGE Sim   │ (NLI Premise=Evidence, Hyp=Claim)
        └─────────────────┬──────────────────┘
                          ↓
           ┌──────────────┼──────────────┐
           ↓              ↓              ↓
       SUPPORTED    CONTRADICTED   INSUFFICIENT
```

---

## 📊 Pipeline Status & Key Results

| Stage | Module | Status | Key Metric / Output |
|---|---|---|---|
| **0. Setup** | Env & Ollama Cloud LLM | ✅ Done | RTX 4050 + CUDA 12.4 + Ollama SQLite cache |
| **1. Datasets** | FEVER & HaluEval QA | ✅ Done | 5.9k FEVER claims, 4k HaluEval QA, 248k sentence corpus |
| **2. Oracle NLI** | Gold-evidence Verification | ✅ Done | FEVER 2-way: **94.3% Acc / 0.943 F1**; HaluEval: 70.8% Acc |
| **3. Retrieval** | BM25 + FAISS Dense + RRF | ✅ Done | Strict R@5: **90.2%** (Dense), End-to-end FEVER Score: **60.0%** |
| **4. Claim Extraction**| LLM Claim Extractor | Extraction complete; human review pending | LLM-based atomic decomposition + quality validation + fallback |
| **5. Detector** | Learned combiner (logistic regression) | Fitted: train → val → single test pass; paired gain significant | FEVER 3-way test macro-F1 **0.702** (hybrid) vs **0.633** fixed rule |
| **6. Error Analysis** | Taxonomy Analysis | v2 taxonomy; 37 human + 237 LLM-proposed (silver) labels; 40-row audit pending | 274 cases categorised, review tooling added |
| **7. Serving & UI** | FastAPI + React Web UI | Implemented and smoke-tested; learned detector not served | Wikipedia and supplied-source modes |
| **v2. Calibration & Selective Policy** | Temperature scaling, selective policy, coverage flag | One test pass; Brier and flag gates failed, learned model stays offline | Test ECE 0.023; verifiable 84.6% coverage at 94.1% precision |
| **v5. Grounding-aware NEI** | MiniCheck features + answerability head | One test pass; all eight research gates passed; learned model stays offline | Test final-policy macro-F1 **0.7369**, NEI F1 **0.6448**; verifiable **81.8%** coverage at **95.5%** precision |

Detailed tables, ablations, and limitations are in [docs/RESULTS.md](docs/RESULTS.md); historical records remain in [docs/PROGRESS.md](docs/PROGRESS.md).

---

## Learned detector (Stage 5)

The claim-level combiner is fitted on 3,000 FEVER train claims, its `C` and feature
subset are selected on 999 val claims, and the test split (1,998 claims) is evaluated
once. Test macro-F1 is **0.702** with hybrid retrieval, **0.701** with dense, and
**0.682** with BM25 — against **0.633**, **0.612**, and **0.606** for the frozen
0.7/0.95 rule. NLI signals dominate the ablation; similarity and retrieval-score
features add little once NLI is present. The gain is statistically paired: McNemar
p ≤ 2.4e-8 and every 95% bootstrap interval for Δ macro-F1 excludes zero. On the
1,324 verifiable test claims, two thirds of hybrid errors are abstentions on claims
the oracle answers correctly — the remaining loss is evidence coverage, not the
decision rule. Two val-only studies (pretrained cross-encoder reranking,
neighbour-sentence premises) are reported in [RESULTS.md](docs/RESULTS.md) and are
**not** part of the shipped path. The serving scope is recorded in
[SERVING_DECISION.md](docs/SERVING_DECISION.md): the learned model stays offline in
v1 and the demo continues to serve the fixed rule.

## v2 — calibrated, coverage-aware verification

v2 (`feature/ver-2`, tag `v2.0`) keeps the v1 verdicts frozen and adds a validity
layer. Calibration (temperature scaling fitted on val, cross-fitted) lowers test ECE
from 0.028 to 0.023 with macro-F1 unchanged at 0.702. A frozen class-conditional
policy answers 84.6% of verifiable test claims at 94.1% precision, but overall
answered precision is only 73.6% because NEI claims are often answered wrongly. A
strict-retrieval-failure flag reaches val AUROC 0.774 but test AUROC 0.734, so it is
not served.

The pre-registered promotion gates failed: Brier rose by 0.00069, and the flag missed
its AUROC bar. **The learned detector stays offline**; the fixed rule remains the
default. Details: [v2 report](experiments/v2/REPORT.md) and
[serving decision](docs/SERVING_DECISION_V2.md). The 237 non-pilot taxonomy labels
are LLM proposals, not human annotations; a 40-row blind audit is pending.

## v3 — stopped before the test pass

A v3 attempt targeted the v2 promotion blockers. Brier-targeted vector scaling
improved val Brier (0.42303 → 0.42161), but the answerability model could not
reduce NEI coverage below 0.483 without breaking the verifiable-coverage gate. An
expanded-feature diagnostic raised AUROC only to 0.769 and left the NEI floor at
0.477, so v3 stopped before spending its test pass. The learned detector remains
offline.
Details: [v3 report](experiments/v3/REPORT.md) and
[v3 serving decision](docs/SERVING_DECISION_V3.md).

## v4 — reranker + more data, still stopped

A v4 attempt added a pretrained reranker and expanded training from 3,000 to 9,000
FEVER claims. Reranking + top-5 lifted val macro-F1 to **0.6930** (vs 0.6764), and
the larger train set improved answerability AUROC to **0.787**. Cost-sensitive tuning
reached NEI coverage **0.408**, still above the 0.40 gate, and a joint threshold
search found no feasible point. v4 stopped before spending its test pass; the learned
detector remains offline. Details: [v4 report](experiments/v4/REPORT.md) and
[v4 serving decision](docs/SERVING_DECISION_V4.md).

## v5 — grounding features, one test pass

v5 adds inference-only **MiniCheck-Flan-T5-Large** support scores for the top-5
reranked evidence sentences. The MiniCheck-only 9k candidate was selected on
validation: raw macro-F1 **0.70681**, INSUFFICIENT F1 **0.54941**, answerability
AUROC **0.81957**, and the robust `tau=0.62` policy cut NEI false-answer coverage
from v4's **0.408** to **0.363** while raising verifiable coverage to **0.814** at
**0.970** precision.

The single final FEVER test pass (after a serialization-only recovery) produced
final-policy macro-F1 **0.73694**, INSUFFICIENT F1 **0.64483**, answerability AUROC
**0.83809**, Brier **0.37519**, overall coverage **0.66266** at **0.78701**
precision, and verifiable coverage **0.81757** at **0.95500** precision. All eight
predeclared gates passed. The policy abstained on 64.7% of NEI claims; the 35.3% it
answered were wrong 99.1% of the time, so NEI is reduced but not solved. The 95.5%
figure is selective precision on the answered verifiable subset, not accuracy on
all verifiable claims. The learned path remains offline. Details:
[v5 report](experiments/v5/VALIDATION.md) and
[v5 plan/outcome](docs/PLAN_V5.md).

## Frozen-rule evaluation (HaluEval)

On the same 600 HaluEval test answers, claim decomposition with supplied knowledge
scores **73.3% accuracy with argmax**, versus **58.7% for whole-answer argmax**.
With the application's fixed rule, supplied-source accuracy is **72.0%**.
Wikipedia retrieval accuracy is only **50.5%** on this set: the limited corpus and
verification pipeline remain weak outside their FEVER setup. This section reports
no fitting and no threshold selection; the learned detector above is scored only on
FEVER, where claim-level gold labels exist. See the
[frozen-rule report](experiments/stage5_inference/REPORT.md) for all ten comparisons
and the always-hallucinated reference.

The UI now accepts optional source text; without it, the existing Wikipedia search
runs. [Human review instructions](docs/REVIEW_GUIDE.md) cover the blind pilot, the
extraction flags, the verification failure sample, and three unapplied JSON recovery
proposals.

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12, Windows 11)
- CPU supported; an NVIDIA GPU with CUDA is optional for faster inference
- [Ollama](https://ollama.ai) installed and running locally with access to `gemma4:31b-cloud` (or custom model in `config.yaml`)

For the new API and React application on macOS/Linux, follow [docs/RUNNING.md](docs/RUNNING.md). The original Windows/CUDA experiment setup is below.

### 2. Environment Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```

### 3. Reproducing Experiments

```powershell
# Stage 1: Download & prepare FEVER and HaluEval splits + Wikipedia corpus
python -m src.datasets.prepare --dataset all

# Stage 2: Oracle verification (gold evidence upper bound)
python -m src.evaluation.oracle_nli

# Stage 3: Build BM25/FAISS indexes, evaluate retrieval recall and end-to-end FEVER
python -m src.evaluation.fever_retrieval
```

---

## 📁 Repository Structure

```text
NLP_project/
├── config.yaml                    # Master project configuration
├── requirements.txt               # Python package dependencies
├── README.md                      # Project overview & quickstart
├── plan.md                        # Project specification & PRD
├── docs/
│   └── PROGRESS.md                # Detailed experimental records & milestone report
├── data/
│   ├── raw/                       # Downloaded source files (FEVER zip, HaluEval JSON)
│   ├── processed/                 # fever_corpus.jsonl, retrieval caches & indexes
│   ├── splits/                    # Train / val / test JSONL splits
│   └── cache/                     # Persistent SQLite response cache for LLM
├── experiments/                   # Experiment evaluation outputs and metrics JSONs
│   ├── stage2_oracle/
│   └── stage3_retrieval/
└── src/
    ├── datasets/                  # Dataset loaders (schema, fever, halueval, splits)
    ├── generation/                # LLM client & claim extraction
    ├── retrieval/                 # BM25, FAISS Dense, Hybrid RRF retriever
    ├── verification/              # DeBERTa NLI & BGE semantic similarity
    ├── evaluation/                # Metrics calculation & evaluation scripts
    └── utils/                     # Config and path helpers
```
