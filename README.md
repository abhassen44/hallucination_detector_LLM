# Claim-Level Hallucination Detection System

A research and production-grade NLP system for detecting hallucinations in LLM-generated text at the **individual claim level**. Rather than classifying an entire response as binary true/false, the pipeline breaks text into atomic factual propositions, retrieves supporting/refuting evidence from Wikipedia using hybrid search (BM25 + FAISS Dense), verifies each claim with DeBERTa NLI and semantic similarity, and predicts a fine-grained 3-way label:

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
| **4. Claim Extraction**| LLM Claim Extractor | ✅ Done | LLM-based atomic decomposition + quality validation + fallback |
| **5. Detector** | Hybrid Multi-signal Classifier | ⏳ Next | Logistic Regression + Ablation + MLflow |
| **6. Error Analysis** | Taxonomy Analysis | ⏳ Planned | 8 failure categories |
| **7. Serving & UI** | FastAPI + React Web UI | ⏳ Planned | Interactive demo & inspection UI |

Detailed tables and ablation comparisons are documented in [docs/PROGRESS.md](docs/PROGRESS.md).

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12, Windows 11)
- NVIDIA GPU with CUDA support (e.g., RTX 4050 Laptop GPU)
- [Ollama](https://ollama.ai) installed and running locally with access to `gemma4:31b-cloud` (or custom model in `config.yaml`)

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
