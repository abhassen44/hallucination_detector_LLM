# v5 grounding ablation (9000 train claims)

Created: 2026-10-22T14:48:11.382993+00:00

| Variant | Features | C | Macro-F1 | SUPP F1 | CONTRA F1 | NEI F1 | Ans AUROC | AUROC gate | NEI floor | Feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 30 | 0.03 | 0.6852 | 0.8478 | 0.6855 | 0.5222 | 0.7944 | True | 0.4084084084084084 | False |
| mc | 45 | 0.3 | 0.7068 | 0.8717 | 0.6993 | 0.5494 | 0.8196 | True | 0.35735735735735735 | True |
| hhem | 45 | 0.01 | 0.6966 | 0.8633 | 0.6983 | 0.5281 | 0.8035 | True | 0.37237237237237236 | True |
| both | 62 | 0.03 | 0.7060 | 0.8701 | 0.7013 | 0.5467 | 0.8182 | True | 0.36036036036036034 | True |
| compact | 48 | 0.1 | 0.6954 | 0.8634 | 0.6920 | 0.5309 | 0.8163 | True | 0.35435435435435436 | True |

No test data was read at ablation time; the later v5 test pass is recorded in VALIDATION.md.
Labels are silver; this is not human adjudication.
