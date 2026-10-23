# 05 — Serving Architecture

FastAPI + React application, API routes, middleware, and operational details.

---

## Application Architecture

```mermaid
flowchart LR
    subgraph Browser["🌐 Browser"]
        USER["User"]
        REACT["React Claim Inspector<br/>Vite + JSX build"]
        USER --> REACT
    end

    subgraph FastAPI["⚡ FastAPI Server (v0.3.1)"]
        direction TB

        subgraph Middleware["Middleware Stack"]
            CORS["CORS Allowlist<br/>localhost:5173, 4173<br/>or API_ALLOWED_ORIGINS"]
            HEADERS["Operational Headers<br/>X-Request-ID (trace)<br/>X-Content-Type-Options: nosniff<br/>Cache-Control per route"]
            CORS --> HEADERS
        end

        subgraph Routes["API Endpoints"]
            HEALTH["/health GET<br/>status, models_loaded,<br/>version, ui_built"]
            DETECT["/detect POST<br/>Full pipeline:<br/>extract → retrieve → verify"]
            CLAIMS["/claims POST<br/>Claim extraction only"]
            RETRIEVE["/retrieve POST<br/>Evidence retrieval only"]
            VERIFY["/verify POST<br/>Verification only"]
            METRICS["/metrics GET<br/>Saved experiment JSON"]
            SUMMARY["/metrics/summary GET<br/>Lightweight dashboard data"]
            CHARTS["/charts GET<br/>HTML results dashboard"]
            HOME["/ GET<br/>Serve built React UI"]
        end

        subgraph Static["Static Assets"]
            MOUNT["/assets mount<br/>frontend/dist/assets/<br/>Cache: immutable, 1yr"]
            INDEX["index.html<br/>Cache: no-cache"]
        end
    end

    REACT -->|"POST /detect"| DETECT
    REACT -->|"POST /claims"| CLAIMS
    REACT -->|"POST /retrieve"| RETRIEVE
    REACT -->|"POST /verify"| VERIFY
    REACT -->|"GET /"| HOME

    style Browser fill:#1a1a2e,stroke:#e94560,color:#fff
    style FastAPI fill:#16213e,stroke:#0f3460,color:#fff
    style Middleware fill:#0f3460,stroke:#533483,color:#fff
    style Routes fill:#533483,stroke:#e94560,color:#fff
    style Static fill:#1a1a2e,stroke:#e94560,color:#fff
```

---

## Request Processing Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant M as Middleware
    participant V as Validation
    participant L as Inference Lock
    participant D as Detector

    B->>M: POST /detect {question, answer, method}
    M->>M: Assign X-Request-ID
    M->>V: Pydantic validation<br/>(≤12k chars, ≤20 passages, extra=forbid)

    alt Validation fails
        V-->>B: 422 Unprocessable Entity
    end

    V->>L: Acquire _inference_lock<br/>(serialized, one request at a time)
    L->>D: detector.detect(question, answer, method)

    D->>D: extract_with_metadata(answer, question)
    Note over D: ≤50 claims enforced

    loop For each claim
        D->>D: retriever.search(claim, method, top_k=5)
        D->>D: nli.predict(evidence, claim)
        D->>D: aggregate + decide (fixed rule)
    end

    D-->>L: {label, claims[], method, detector}
    L-->>M: Release lock
    M->>M: Set Cache-Control: no-store
    M-->>B: JSON response + X-Request-ID
```

---

## Model Loading and Locking

```mermaid
flowchart TB
    subgraph Loading["🔒 Thread-Safe Model Loading"]
        LOCK1["_model_lock<br/>(threading.Lock)"]
        LRU["@lru_cache(maxsize=1)"]

        subgraph Models["Singleton Models"]
            EXT["ClaimExtractor<br/>Ollama client + prompt"]
            DET["Detector<br/>extractor + retriever + NLI"]
            RET["Retriever (lazy)<br/>BM25 + FAISS indexes"]
            NLI["NLIModel<br/>DeBERTa-v3-base-MNLI"]
        end

        LOCK1 --> LRU --> Models
    end

    subgraph Inference["🔒 Serialized Inference"]
        LOCK2["_inference_lock<br/>(threading.Lock)"]
        CALL["One request at a time<br/>prevents GPU/memory contention"]
        LOCK2 --> CALL
    end

    Models --> Inference

    style Loading fill:#1a1a2e,stroke:#e94560,color:#fff
    style Inference fill:#0f3460,stroke:#533483,color:#fff
```

---

## Request / Response Schemas

| Endpoint | Request Body | Response Shape |
|----------|-------------|----------------|
| `POST /detect` | `{answer, question?, evidence?, method, top_k}` | `{label, claims[], method, evidence_source, detector, warnings}` |
| `POST /claims` | `{answer, question?}` | `{claims[], extraction_fallback, extraction_model}` |
| `POST /retrieve` | `{claim, method, top_k}` | `{evidence[]}` |
| `POST /verify` | `{claim, evidence[]}` | `{claim, label, signals, evidence_conflict, evidence[]}` |
| `GET /health` | — | `{status, models_loaded, mode, detector, version, ui_built}` |
| `GET /metrics` | — | `{source, results{}}` |

## Operational Hardening

| Feature | Implementation |
|---------|---------------|
| Request tracing | `X-Request-ID` header (incoming or generated UUID) |
| Security headers | `X-Content-Type-Options: nosniff` on all responses |
| Cache policy | `no-store` for API routes; `immutable` for hashed assets |
| Input limits | 12k char text, 20 evidence passages, 50 claim ceiling |
| CORS | Local dev origins by default; `API_ALLOWED_ORIGINS=*` override |
| Error handling | 422 for validation, 503 for model/index unavailability |
| Graceful degradation | 503 with actionable message if UI build or models are missing |
