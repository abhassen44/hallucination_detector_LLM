# Claim-Level Hallucination Detection System

**Current local implementation:** see [implementation status](docs/IMPLEMENTATION_STATUS.md) and [application setup](docs/RUNNING.md). The inference API and React UI are implemented. Gemma cloud inference and pretrained retrieval/NLI are verified. All 4,000 HaluEval examples are processed, producing 6,462 claims with three flagged fallbacks. A learned Stage 5 claim-level detector is now fitted on FEVER (train → val selection → one test pass); canonical numbers are in [RESULTS.md](docs/RESULTS.md).

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
| **6. Error Analysis** | Taxonomy Analysis | Provisional LLM first pass; blind pilot passed, full pass pending | 274 cases categorised, review tooling added |
| **7. Serving & UI** | FastAPI + React Web UI | Implemented and smoke-tested | Wikipedia and supplied-source modes |

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
