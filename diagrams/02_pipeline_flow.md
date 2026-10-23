# 02 — Pipeline Flow

Detailed claim-level detection pipeline showing every data transformation
from raw user input to the final per-claim verdict. Solid arrows are the
served path; dashed arrows are fallback or offline paths.

---

## Claim-Level Detection Pipeline

```mermaid
flowchart TD
    %% ── Input ──
    Q["🧑 User Question<br/>(optional, ≤ 12k chars)"]
    A["📝 LLM Answer<br/>(text to verify)"]
    SRC["📄 Supplied Source Text<br/>(optional override)"]

    %% ── Claim Extraction ──
    subgraph Extraction["📝 Claim Extraction"]
        direction TB
        PROMPT["LLM Prompt<br/>Ollama gemma4:31b-cloud"]
        PARSE["JSON Parse<br/>atomic propositions"]
        VALIDATE["Quality Validation<br/>short / long / duplicate flags"]
        FALLBACK["Sentence Splitter<br/>(fallback on failure)"]

        PROMPT --> PARSE
        PARSE --> VALIDATE
        PARSE -. "parse failure" .-> FALLBACK
        FALLBACK --> VALIDATE
    end

    subgraph ClaimData["Claim Data Shape"]
        CLAIMS["List of atomic claims<br/>+ is_fallback flag<br/>+ extraction_model"]
    end

    Q --> Extraction
    A --> Extraction
    Extraction --> ClaimData

    %% ── Evidence Retrieval (per claim) ──
    subgraph RetrievalPhase["🔍 Evidence Retrieval (per claim)"]
        direction TB

        subgraph Sparse["BM25 Sparse"]
            TOK["Tokenize claim"]
            BM25["rank-bm25 search<br/>bm25_k=50 candidates"]
            TOK --> BM25
        end

        subgraph Dense["FAISS Dense"]
            ENC["BGE-small-en-v1.5<br/>encode claim"]
            FAISS["FAISS inner-product<br/>dense_k=50 candidates"]
            ENC --> FAISS
        end

        subgraph Fusion["Reciprocal Rank Fusion"]
            RRF["RRF(d) = Σ 1/(60+rank)<br/>merge BM25 + dense"]
            TOP5["Top-5 evidence passages<br/>with page_id, sent_id, score"]
            RRF --> TOP5
        end

        BM25 --> RRF
        FAISS --> RRF
    end

    CLAIMS --> RetrievalPhase
    SRC -. "bypass retrieval<br/>when source provided" .-> VerifyPhase

    %% ── Verification (per claim × evidence) ──
    subgraph VerifyPhase["✅ Claim–Evidence Verification"]
        direction TB

        subgraph NLIScore["NLI Scoring"]
            PAIR["Build (premise, hypothesis) pairs<br/>premise=evidence, hypothesis=claim"]
            DEB["DeBERTa-v3-base-MNLI<br/>batch_size=32, fp16"]
            PROBS["Per-pair probabilities<br/>[P(E), P(N), P(C)]"]
            PAIR --> DEB --> PROBS
        end

        subgraph Aggregation["Max-Evidence Aggregation"]
            MAXE["E_max = max P(entail)<br/>across evidence"]
            MAXC["C_max = max P(contradict)<br/>across evidence"]
            NORM["Normalize: E + N + C = 1<br/>suppress losing signal"]
            MAXE --> NORM
            MAXC --> NORM
        end

        PROBS --> MAXE
        PROBS --> MAXC
    end

    TOP5 --> VerifyPhase

    %% ── Decision ──
    subgraph DecisionPhase["🏷️ Verdict Decision"]
        RULE["Fixed Threshold Rule<br/>E ≥ 0.7 → SUPPORTED<br/>C ≥ 0.95 → CONTRADICTED<br/>else → INSUFFICIENT"]
        CONFLICT["Conflict Flag<br/>both E ≥ 0.7 and C ≥ 0.95"]
    end

    NORM --> RULE
    NORM --> CONFLICT

    %% ── Output ──
    subgraph OutputPhase["📊 Per-Claim Output"]
        RESULT["{ claim, label, signals,<br/>evidence_conflict,<br/>evidence: [...nli per pair] }"]
    end

    subgraph OverallPhase["📊 Overall Verdict"]
        OVERALL["ANY contradiction → CONTRADICTED<br/>ALL supported → SUPPORTED<br/>otherwise → INSUFFICIENT"]
    end

    RULE --> RESULT
    CONFLICT --> RESULT
    RESULT --> OVERALL

    %% ── Styles ──
    style Extraction fill:#1a1a2e,stroke:#e94560,color:#fff
    style ClaimData fill:#16213e,stroke:#0f3460,color:#fff
    style RetrievalPhase fill:#0f3460,stroke:#533483,color:#fff
    style VerifyPhase fill:#533483,stroke:#e94560,color:#fff
    style DecisionPhase fill:#1a1a2e,stroke:#e94560,color:#fff
    style OutputPhase fill:#16213e,stroke:#0f3460,color:#fff
    style OverallPhase fill:#0f3460,stroke:#533483,color:#fff
```

---

## Data Transformation Summary

| Stage | Input | Output | Shape |
|-------|-------|--------|-------|
| Claim Extraction | Answer text (+ question) | List of atomic claims | `[str, ...]` |
| Evidence Retrieval | One claim | Top-5 passages with metadata | `[{text, page_id, sent_id, score}, ...]` |
| NLI Scoring | (claim, evidence) pairs | Probability triples | `[[P(E), P(N), P(C)], ...]` per evidence |
| Aggregation | Per-evidence probabilities | Single [E, N, C] vector | `[float, float, float]` |
| Decision | Aggregated probabilities | 3-way label + conflict flag | `{label, signals, evidence_conflict}` |
| Overall | All per-claim labels | Single answer-level label | `SUPPORTED / CONTRADICTED / INSUFFICIENT` |
