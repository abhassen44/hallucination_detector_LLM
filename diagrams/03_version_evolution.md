# 03 — Version Evolution

Research lineage from v1 baseline through v5 grounding. Each version is a
self-contained research checkpoint; the served path remains the v1 fixed rule.

---

## Research Version Timeline

```mermaid
flowchart TB
    subgraph V1["v1 — Baseline (branch: main)"]
        V1_CORE["3,000 FEVER train claims<br/>30 features: NLI + similarity + retrieval + claim shape<br/>Logistic regression combiner"]
        V1_RESULT["Test macro-F1: 0.702<br/>Fixed rule: 0.633"]
        V1_SERVE["✅ Fixed 0.7/0.95 rule SERVED"]
        V1_OFFLINE["Learned combiner OFFLINE"]
        V1_CORE --> V1_RESULT
        V1_RESULT --> V1_SERVE
        V1_RESULT --> V1_OFFLINE
    end

    subgraph V2["v2 — Calibration + Selective Policy"]
        V2_ADD["+ Temperature scaling calibration<br/>+ Answerability model<br/>+ Coverage flag (diagnostic)<br/>+ Selective policy (class thresholds)"]
        V2_RESULT["Test ECE: 0.023 (improved)<br/>Brier: rose by 0.00069 ❌<br/>Flag AUROC: 0.734 ❌"]
        V2_GATE["Brier gate: FAILED<br/>Flag gate: FAILED"]
        V2_ADD --> V2_RESULT --> V2_GATE
    end

    subgraph V3["v3 — Calibration + Answerability Diagnostics"]
        V3_ADD["+ Vector scaling (Brier-targeted)<br/>+ Answerability HGB head<br/>+ Expanded uncertainty features<br/>+ Joint threshold search"]
        V3_RESULT["Val Brier: 0.42161<br/>NEI floor: 0.483 → 0.477"]
        V3_GATE["NEI gate (≤ 0.40): FAILED<br/>⛔ Stopped before test pass"]
        V3_ADD --> V3_RESULT --> V3_GATE
    end

    subgraph V4["v4 — Reranker + Larger Training"]
        V4_ADD["+ BAAI/bge-reranker-base<br/>+ 9,000 FEVER train claims<br/>+ Cost-sensitive answerability tuning"]
        V4_RESULT["Val macro-F1: 0.693<br/>Answerability AUROC: 0.787<br/>Best NEI: 0.408"]
        V4_GATE["NEI gate (≤ 0.40): FAILED<br/>⛔ Stopped before test pass"]
        V4_ADD --> V4_RESULT --> V4_GATE
    end

    subgraph V5["v5 — Grounding Features + Robust Policy"]
        V5_ADD["+ MiniCheck-Flan-T5-Large grounding<br/>+ HHEM-2.1-Open (diagnostic arm)<br/>+ 15 grounding features<br/>+ Robust policy selection (tau=0.62)"]
        V5_RESULT["Test macro-F1: 0.7369<br/>NEI F1: 0.6448<br/>Answerability AUROC: 0.838<br/>Verifiable: 81.8% cov, 95.5% prec"]
        V5_GATE["All 8 research gates: ✅ PASSED<br/>Learned model remains OFFLINE"]
        V5_ADD --> V5_RESULT --> V5_GATE
    end

    V1 -->|"v2 adds<br/>calibration layer"| V2
    V2 -->|"v3 targets<br/>v2 blockers"| V3
    V3 -->|"v4 adds reranker<br/>+ more data"| V4
    V4 -->|"v5 adds grounding<br/>+ robust policy"| V5

    style V1 fill:#1a1a2e,stroke:#4ade80,color:#fff
    style V2 fill:#16213e,stroke:#f97316,color:#fff
    style V3 fill:#1a1a2e,stroke:#ef4444,color:#fff
    style V4 fill:#16213e,stroke:#ef4444,color:#fff
    style V5 fill:#0f3460,stroke:#4ade80,color:#fff
```

---

## Cumulative Feature Stack

```mermaid
flowchart LR
    subgraph Features["Feature Evolution"]
        F1["v1: 30 features<br/>NLI (15) + Retrieval (8)<br/>+ Similarity (3) + Claim (4)"]
        F4["v4: 30 features<br/>Same groups, reranked evidence"]
        F5["v5: 45 features<br/>+15 MiniCheck grounding"]

        F1 -->|"same features,<br/>better evidence"| F4
        F4 -->|"+15 grounding<br/>features"| F5
    end

    style Features fill:#16213e,stroke:#533483,color:#fff
```

---

## Gate Summary

| Version | Test Pass | Key Gate | Outcome | Promotion |
|---------|-----------|----------|---------|-----------|
| v1 | ✅ Spent | Macro-F1 vs fixed rule | 0.702 vs 0.633 ✅ | Learned stays offline |
| v2 | ✅ Spent | Brier ≤ baseline, Flag AUROC ≥ 0.75 | Brier +0.00069 ❌, AUROC 0.734 ❌ | No promotion |
| v3 | ⛔ Not spent | NEI coverage ≤ 0.40 | NEI floor 0.477 ❌ | Stopped |
| v4 | ⛔ Not spent | NEI coverage ≤ 0.40 | NEI floor 0.408 ❌ | Stopped |
| v5 | ✅ Spent | 8 research gates | All passed ✅ | Learned still offline |

## Serving Path (All Versions)

```mermaid
flowchart LR
    REQ["Request"] --> EXT["Claim Extraction"]
    EXT --> RET["Hybrid Retrieval<br/>BM25 + FAISS + RRF"]
    RET --> NLI["DeBERTa NLI"]
    NLI --> FIX["Fixed Rule<br/>E ≥ 0.7, C ≥ 0.95"]
    FIX --> OUT["Verdict"]

    style REQ fill:#1a1a2e,stroke:#e94560,color:#fff
    style FIX fill:#4ade80,stroke:#166534,color:#000
```

> The served path has remained identical across all five versions.
> Every version's learned model is an offline research result only.
