# Diagrams — Architecture & Flow Reference

This folder contains Mermaid-based architecture diagrams, flow charts, and summary
tables for the **Claim-Level Hallucination Detection System**. Every diagram renders
natively on GitHub.

---

## Index

| # | Diagram | Description |
|---|---------|-------------|
| 01 | [System Overview](01_system_overview.md) | High-level end-to-end architecture showing all major components |
| 02 | [Pipeline Flow](02_pipeline_flow.md) | Detailed claim-level detection pipeline with data transformations |
| 03 | [Version Evolution](03_version_evolution.md) | Research lineage from v1 baseline through v5 grounding |
| 04 | [Data Flow](04_data_flow.md) | Dataset preparation, splits, caching, and evaluation protocol |
| 05 | [Serving Architecture](05_serving_architecture.md) | FastAPI + React application, API routes, and operational details |
| 06 | [v5 Research Architecture](06_v5_research_architecture.md) | v5 grounding features, answerability head, and robust policy |
| 07 | [Results Summary](07_results_summary.md) | Key metrics across all versions (v1–v5) with comparison tables |

---

## How to read

- All diagrams use **Mermaid** syntax and render automatically on GitHub.
- Solid arrows (`-->`) represent the served data path.
- Dashed arrows (`-.->`) represent offline/research-only paths.
- Node colours and subgraph borders group related components.

## Quick links

- [Main README](../README.md)
- [Architecture docs](../docs/architecture/)
- [Results docs](../docs/results/)
- [v5 image report](../output%20version%205/summary.md)
