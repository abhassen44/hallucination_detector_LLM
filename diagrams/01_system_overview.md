# 01 — System Overview

High-level architecture of the Claim-Level Hallucination Detection System.
The diagram shows every major component from user input to final verdict.

---

## End-to-End Architecture

```mermaid
flowchart TB
    subgraph User["🧑 User Interaction"]
        Q["User Question"]
        UI["React Claim Inspector<br/>(Vite + JSX)"]
    end

    subgraph Generation["🤖 Answer Generation"]
        OLLAMA["Ollama LLM<br/>gemma4:31b-cloud"]
        ANS["Generated Answer"]
        Q --> OLLAMA --> ANS
    end

    subgraph Extraction["📝 Claim Extraction"]
        EXT["LLM Claim Extractor<br/>(concurrent workers)"]
        FB["Sentence-Splitting<br/>Fallback"]
        CLAIMS["Atomic Claims"]
        ANS --> EXT --> CLAIMS
        EXT -. "extraction failure" .-> FB --> CLAIMS
    end

    subgraph Retrieval["🔍 Hybrid Evidence Retrieval"]
        BM25["BM25<br/>Sparse Keyword Search"]
        FAISS["FAISS + BGE-small<br/>Dense Semantic Search"]
        RRF["Reciprocal Rank Fusion<br/>(k=60)"]
        TOP5["Top-5 Evidence<br/>Passages"]
        CLAIMS --> BM25 --> RRF
        CLAIMS --> FAISS --> RRF
        RRF --> TOP5
    end

    subgraph Corpus["📚 Wikipedia Corpus"]
        WIKI["248k FEVER-derived<br/>Wikipedia Sentences"]
        BM25IDX["BM25 Index"]
        FAIDX["FAISS Vector Index"]
        WIKI --> BM25IDX --> BM25
        WIKI --> FAIDX --> FAISS
    end

    subgraph Verification["✅ Claim–Evidence Verification"]
        NLI["DeBERTa-v3-base-MNLI<br/>Entailment / Neutral / Contradiction"]
        SIM["BGE Cosine Similarity"]
        AGG["Max-Evidence Aggregation"]
        TOP5 --> NLI --> AGG
        TOP5 --> SIM --> AGG
    end

    subgraph Decision["🏷️ Verdict Decision"]
        FIXED["Fixed Rule Detector<br/>(Served)<br/>E ≥ 0.7 │ C ≥ 0.95"]
        LEARNED["Learned Logistic Combiner<br/>(Offline Research)"]
        AGG --> FIXED
        AGG --> LEARNED
    end

    subgraph Output["📊 Results"]
        SUP["✅ SUPPORTED"]
        CON["❌ CONTRADICTED"]
        INS["⚠️ INSUFFICIENT"]
        FIXED --> SUP
        FIXED --> CON
        FIXED --> INS
    end

    subgraph Serving["🌐 Application Layer"]
        API["FastAPI Server<br/>v0.3.1"]
        STATIC["Static File Mount<br/>/assets"]
        SUP --> API --> UI
        CON --> API
        INS --> API
        API --> STATIC
    end

    style User fill:#1a1a2e,stroke:#e94560,color:#fff
    style Generation fill:#16213e,stroke:#0f3460,color:#fff
    style Extraction fill:#1a1a2e,stroke:#e94560,color:#fff
    style Retrieval fill:#0f3460,stroke:#533483,color:#fff
    style Corpus fill:#16213e,stroke:#0f3460,color:#fff
    style Verification fill:#533483,stroke:#e94560,color:#fff
    style Decision fill:#1a1a2e,stroke:#e94560,color:#fff
    style Output fill:#16213e,stroke:#0f3460,color:#fff
    style Serving fill:#0f3460,stroke:#533483,color:#fff
```

---

## Component Summary

| Component | Technology | Role |
|-----------|-----------|------|
| Answer Generation | Ollama + gemma4:31b-cloud | Generate answers to user questions |
| Claim Extraction | LLM prompt + sentence fallback | Decompose answers into atomic factual propositions |
| BM25 Retrieval | rank-bm25 | Sparse keyword matching against Wikipedia corpus |
| Dense Retrieval | FAISS + BGE-small-en-v1.5 | Semantic vector search for evidence |
| Reciprocal Rank Fusion | Custom RRF (k=60) | Merge BM25 + dense candidate lists |
| NLI Verification | DeBERTa-v3-base-MNLI | Classify claim–evidence pairs as entail/neutral/contradict |
| Similarity | BGE cosine similarity | Secondary semantic similarity signal |
| Fixed Rule | Threshold detector | Deterministic served verdict (E ≥ 0.7, C ≥ 0.95) |
| Learned Combiner | Logistic regression | Offline 30–45 feature research model |
| API | FastAPI | REST endpoints: `/detect`, `/claims`, `/retrieve`, `/verify` |
| UI | React + Vite | Interactive claim inspector with evidence display |
