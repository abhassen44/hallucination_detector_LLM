# v5 grounding ablation (3000 train claims)

Created: 2026-10-22T13:59:30.818848+00:00

| Variant | Features | C | Macro-F1 | SUPP F1 | CONTRA F1 | NEI F1 | Ans AUROC | AUROC gate | NEI floor | Feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 30 | 0.1 | 0.6930 | 0.8502 | 0.6931 | 0.5356 | 0.7795 | False | 0.4744744744744745 | False |
| mc | 45 | 3 | 0.6988 | 0.8543 | 0.6990 | 0.5430 | 0.7992 | True | 0.3993993993993994 | True |
| hhem | 45 | 3 | 0.6977 | 0.8547 | 0.6973 | 0.5413 | 0.7933 | False | 0.45345345345345345 | False |
| both | 62 | 0.003 | 0.6975 | 0.8629 | 0.6957 | 0.5338 | 0.8030 | True | 0.3993993993993994 | True |
| compact | 48 | 0.003 | 0.6989 | 0.8642 | 0.7019 | 0.5306 | 0.8074 | True | 0.3933933933933934 | True |

No test data was read at ablation time; the later v5 test pass is recorded in VALIDATION.md.
Labels are silver; this is not human adjudication.
