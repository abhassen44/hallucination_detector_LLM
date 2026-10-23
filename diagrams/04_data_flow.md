# 04 — Data Flow

Dataset preparation, splits, index construction, caching, and evaluation protocol.

---

## Dataset Preparation Pipeline

```mermaid
flowchart TD
    subgraph Sources["📥 External Data Sources"]
        FEVER_ZIP["FEVER<br/>fever.ai downloads<br/>paper_dev + paper_test + wiki-pages"]
        HALU["HaluEval QA<br/>RUCAIBox/HaluEval<br/>qa_data.json"]
    end

    subgraph Prepare["🔧 Preparation (src.datasets.prepare)"]
        direction TB
        PARSE_FEVER["Parse FEVER JSONL<br/>extract claims + gold pages"]
        PARSE_HALU["Parse HaluEval JSON<br/>normalize to schema"]
        WIKI["Build Wikipedia Corpus<br/>gold pages + 50k distractor pages"]
        BALANCE["Label-Balanced Sampling<br/>SUPPORTED / REFUTED / NEI"]
        NORM["Normalize to schema<br/>{id, claim, label, evidence_pages}"]
    end

    FEVER_ZIP --> PARSE_FEVER
    HALU --> PARSE_HALU
    PARSE_FEVER --> WIKI
    PARSE_FEVER --> BALANCE
    BALANCE --> NORM
    PARSE_HALU --> NORM

    subgraph Splits["📊 Data Splits"]
        direction LR
        subgraph FeverSplits["FEVER (official splits)"]
            FTR["Train: 3,000 claims<br/>(label-balanced)"]
            FVA["Val: 999 claims<br/>(label-balanced)"]
            FTE["Test: 1,998 claims<br/>(label-balanced, one pass)"]
        end
        subgraph HaluSplits["HaluEval (random 70/15/15)"]
            HTR["Train: 2,800 examples"]
            HVA["Val: 600 examples"]
            HTE["Test: 600 examples"]
        end
    end

    NORM --> Splits

    subgraph Storage["💾 Storage Layout"]
        RAW["data/raw/<br/>Downloaded FEVER/HaluEval sources"]
        PROC["data/processed/<br/>Corpus, indexes, retrieval caches"]
        SPLIT_DIR["data/splits/<br/>Train / val / test JSONL"]
        CACHE_DIR["data/cache/<br/>SQLite LLM + model caches"]
    end

    Splits --> SPLIT_DIR
    WIKI --> PROC

    style Sources fill:#1a1a2e,stroke:#e94560,color:#fff
    style Prepare fill:#16213e,stroke:#0f3460,color:#fff
    style Splits fill:#0f3460,stroke:#533483,color:#fff
    style Storage fill:#533483,stroke:#e94560,color:#fff
```

---

## Corpus and Index Construction

```mermaid
flowchart LR
    subgraph Corpus["📚 Wikipedia Sentence Corpus"]
        PAGES["Gold evidence pages<br/>+ 50,000 distractor pages"]
        SENTS["248,000 sentences<br/>(page_id, sent_id, text)"]
        PAGES --> SENTS
    end

    subgraph Indexes["🗂️ Index Building"]
        direction TB
        BM25_FIT["BM25Index.fit()<br/>TF-IDF sparse matrix"]
        DENSE_FIT["DenseIndex.fit()<br/>BGE-small-en-v1.5 embeddings<br/>→ FAISS inner-product index"]
        BM25_FILE["data/processed/indexes/bm25.pkl"]
        DENSE_FILE["data/processed/indexes/faiss_bge_small.index"]

        BM25_FIT --> BM25_FILE
        DENSE_FIT --> DENSE_FILE
    end

    SENTS --> BM25_FIT
    SENTS --> DENSE_FIT

    style Corpus fill:#1a1a2e,stroke:#e94560,color:#fff
    style Indexes fill:#16213e,stroke:#0f3460,color:#fff
```

---

## Caching Architecture

```mermaid
flowchart TB
    subgraph Caches["🗄️ Cache Layers"]
        LLM_CACHE["SQLite LLM Cache<br/>data/cache/llm_cache.sqlite<br/>Ollama responses keyed by<br/>(model, prompt, temperature)"]
        RET_CACHE["Retrieval Feature Cache<br/>data/processed/retrieval/<br/>fever_{split}.jsonl<br/>Pre-computed top-5 + NLI per claim"]
        GRND_CACHE["Grounding Score Cache<br/>data/processed/grounding/<br/>minicheck_{split}.jsonl<br/>hhem_{split}.jsonl"]
        MODEL_CACHE["Model Singleton Cache<br/>@lru_cache in FastAPI<br/>Detector, Extractor (in-memory)"]
    end

    subgraph Consumers["Consumers"]
        API["FastAPI /detect"]
        RESEARCH["Offline Research Scripts"]
    end

    LLM_CACHE --> API
    MODEL_CACHE --> API
    RET_CACHE --> RESEARCH
    GRND_CACHE --> RESEARCH

    style Caches fill:#0f3460,stroke:#533483,color:#fff
    style Consumers fill:#1a1a2e,stroke:#e94560,color:#fff
```

---

## Evaluation Protocol

```mermaid
sequenceDiagram
    participant Train as Train Split
    participant Val as Val Split
    participant Freeze as Freeze Artifacts
    participant Test as Test Split (one pass)

    Note over Train: Stage 2: Oracle NLI<br/>(gold evidence upper bound)
    Note over Train: Stage 3: Build indexes,<br/>retrieval recall, FEVER score

    Train->>Train: Fit logistic combiner<br/>+ calibration + answerability
    Train->>Val: Select C value, feature subset,<br/>tau, answerability config

    Val->>Freeze: Write freeze.json<br/>+ model artifacts + hashes

    Freeze->>Test: One guarded evaluation
    Test-->>Freeze: metrics, predictions, gates

    Note over Test: Test set is touched ONCE.<br/>No re-tuning after test.
```

---

## Data Shape Reference

| Artifact | Format | Key Fields | Size |
|----------|--------|------------|------|
| FEVER split | JSONL | `id, claim, label, evidence_pages` | 3k / 1k / 2k claims |
| HaluEval split | JSONL | `id, question, answer, label` | 2.8k / 600 / 600 |
| Wikipedia corpus | In-memory list | `(page_id, sent_id, text)` | ~248k sentences |
| Retrieval cache | JSONL | `id, claim, label, bm25, dense, hybrid` (each with top-5 + NLI) | Per split |
| Grounding cache | JSONL | `id, scores` (per evidence sentence) | Per model × split |
| LLM cache | SQLite | `model, prompt, temperature → response` | Grows over time |
