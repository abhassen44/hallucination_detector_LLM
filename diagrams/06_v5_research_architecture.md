# 06 — v5 Research Architecture

v5 grounding features, answerability head, robust policy, and the complete
offline research pipeline. All eight predeclared research gates passed;
the learned model remains offline.

---

## v5 Full Research Pipeline

```mermaid
flowchart TB
    subgraph Data["📊 Training & Evaluation Data"]
        TR["9,000 FEVER train claims<br/>(label-balanced, expanded from 3k)"]
        VA["999 validation claims"]
        TE["1,998 test claims<br/>(one pass, never re-tuned)"]
    end

    subgraph EvidencePath["🔍 Evidence Path"]
        RET["Hybrid Retrieval<br/>BM25 top-50 + Dense top-50<br/>→ RRF merge"]
        RERANK["BAAI/bge-reranker-base<br/>Cross-encoder reranking"]
        TOP5["Reranked Top-5<br/>evidence sentences"]
        NLI["DeBERTa-v3-base-MNLI<br/>E/N/C probabilities per pair"]

        RET --> RERANK --> TOP5 --> NLI
    end

    subgraph GroundingPath["🆕 v5 Grounding Signal"]
        direction TB
        MC["MiniCheck-Flan-T5-Large<br/>(primary grounding scorer)"]
        HHEM["HHEM-2.1-Open<br/>(diagnostic arm only)"]

        subgraph MCFeatures["MiniCheck Feature Block (15 features)"]
            SCORE_F["Score features (8):<br/>max, mean, top2, margin,<br/>minimum, std, count≥0.5, count<0.2"]
            INTERACT_F["Interaction features (7):<br/>minus_nli_e, times_nli_e,<br/>minus_nli_c, times_nli_c,<br/>at_nli_e, at_top1, minus_other"]
        end

        MC --> MCFeatures
        HHEM -. "diagnostic only" .-> MCFeatures
    end

    TOP5 --> MC
    TOP5 --> HHEM

    subgraph BaseFeatures["📐 v4 Base Features (30)"]
        NLI_F["NLI group (15):<br/>e/c/n max, mean, top2, margin,<br/>count thresholds, ranks, aggregated"]
        RET_F["Retrieval group (8):<br/>top1 score, margin, mean, std,<br/>distinct pages, agreement, conflict, share"]
        SIM_F["Similarity group (3):<br/>top1, mean, max cosine"]
        CLM_F["Claim group (4):<br/>word count, has_digit,<br/>has_year, has_negation"]
    end

    NLI --> BaseFeatures

    subgraph VerdictModel["🏷️ Verdict Model"]
        CONCAT["Concatenate<br/>30 base + 15 grounding = 45 features"]
        SCALE["StandardScaler<br/>(fitted on train)"]
        LR["Logistic Regression<br/>3-way: SUP / CON / INS"]
        PROBS["Calibrated class probabilities"]

        CONCAT --> SCALE --> LR --> PROBS
    end

    BaseFeatures --> CONCAT
    MCFeatures --> CONCAT

    subgraph PolicyLayer["🎯 Sufficiency & Policy"]
        direction TB
        VCAL["Vector Scaling<br/>calibrate probabilities"]
        ANS["Answerability HGB<br/>P(claim is answerable)<br/>Features: base + verdict probs + grounding"]

        subgraph RobustSelect["Robust Policy Selection"]
            GRID["Grid search over tau<br/>Constraints:<br/>  verifiable coverage ≥ 0.81<br/>  overall coverage ≥ 0.65"]
            OBJ["Minimize NEI<br/>false-answer coverage"]
            TAU["Selected: tau = 0.62"]
            GRID --> OBJ --> TAU
        end

        DECISION{"P(answerable) ≥ tau?"}
        ANSWER["✅ Output verdict<br/>(SUP / CON)"]
        ABSTAIN["⚠️ INSUFFICIENT<br/>(abstain)"]

        VCAL --> DECISION
        ANS --> DECISION
        TAU --> DECISION
        DECISION -->|"yes"| ANSWER
        DECISION -->|"no"| ABSTAIN
    end

    PROBS --> VCAL
    PROBS --> ANS

    TR --> EvidencePath
    VA --> EvidencePath
    TE --> EvidencePath

    style Data fill:#1a1a2e,stroke:#e94560,color:#fff
    style EvidencePath fill:#16213e,stroke:#0f3460,color:#fff
    style GroundingPath fill:#0f3460,stroke:#4ade80,color:#fff
    style BaseFeatures fill:#533483,stroke:#e94560,color:#fff
    style VerdictModel fill:#1a1a2e,stroke:#e94560,color:#fff
    style PolicyLayer fill:#16213e,stroke:#0f3460,color:#fff
```

---

## Grounding Ablation Results

```mermaid
flowchart LR
    subgraph Ablation["Grounding Variant Selection (9k train)"]
        BOTH["both: mc + hhem<br/>Val F1: 0.70396"]
        MC_ONLY["mc only ✅ SELECTED<br/>Val F1: 0.70681<br/>INSUF F1: 0.54941"]
        HHEM_ONLY["hhem only<br/>Val F1: 0.69934"]
        COMPACT["compact subset<br/>Val F1: 0.70432"]
        NONE["none (v4 baseline)<br/>Val F1: 0.69296"]
    end

    MC_ONLY -->|"best NEI + AUROC"| WINNER["Selected candidate:<br/>MiniCheck-only, 9k"]

    style Ablation fill:#0f3460,stroke:#533483,color:#fff
    style WINNER fill:#4ade80,stroke:#166534,color:#000
```

---

## v5 Evaluation Flow

```mermaid
sequenceDiagram
    participant Train as 9k Train
    participant Val as 999 Val
    participant Freeze as freeze.json
    participant Test as 1,998 Test

    Train->>Train: Fit verdict model (45 features)
    Train->>Train: Fit answerability HGB
    Train->>Val: Score all 999 claims

    Val->>Val: Select grounding variant (mc-only)
    Val->>Val: Select C regularization
    Val->>Val: Run robust policy search
    Val->>Val: Select tau=0.62

    Val->>Freeze: Write freeze.json + artifacts<br/>verdict.joblib, answerability.joblib,<br/>calibration.json, SHA-256 hashes

    Freeze->>Test: One guarded evaluation
    Test-->>Freeze: All metrics + predictions

    Note over Test: 8 gates checked:<br/>macro-F1, NEI F1, AUROC,<br/>Brier, verifiable cov/prec,<br/>overall cov/prec
```

---

## Eight Research Gates

| # | Gate | Threshold | Test Result | Status |
|---|------|-----------|-------------|--------|
| 1 | Final-policy macro-F1 | ≥ val−0.02 | **0.73694** | ✅ |
| 2 | INSUFFICIENT F1 | ≥ val−0.03 | **0.64483** | ✅ |
| 3 | Answerability AUROC | ≥ val−0.02 | **0.83809** | ✅ |
| 4 | Brier score | ≤ val+0.01 | **0.37519** | ✅ |
| 5 | Verifiable coverage | ≥ 0.80 | **0.81757** | ✅ |
| 6 | Verifiable precision | ≥ 0.94 | **0.95500** | ✅ |
| 7 | Overall coverage | ≥ 0.60 | **0.66266** | ✅ |
| 8 | Overall precision | ≥ 0.70 | **0.78701** | ✅ |

## Frozen Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Freeze manifest | `experiments/v5/freeze.json` | Hashes, configs, selected params |
| Verdict model | `experiments/v5/frozen/verdict.joblib` | Trained logistic regression |
| Answerability model | `experiments/v5/frozen/answerability.joblib` | Trained HGB classifier |
| Calibration params | `experiments/v5/frozen/calibration.json` | Vector scaling weights |
| Test metrics | `experiments/v5/test_metrics.json` | Final evaluation output |
| Test predictions | `experiments/v5/test_predictions.jsonl` | Per-claim predictions |
