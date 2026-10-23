# Results by version

These are **claim-level FEVER** results unless stated otherwise. They are separate
from HaluEval answer-level results and from any RAGTruth or out-of-domain
response/span evaluation. Labels are silver FEVER labels; no human-validation claim
is made.

| Version | Test pass | Headline result | Serving state | Details |
|---|---|---|---|---|
| v1 | Spent | Learned test macro-F1 **0.702**; fixed rule **0.633** | Fixed rule served | [v1 results](v1.md) |
| v2 | Spent | Macro-F1 **0.70202**; ECE **0.02279**; Brier gate failed | Learned offline | [v2 results](v2.md) |
| v3 | **Not spent** | Val Brier **0.42161**; NEI floor **0.483** | Learned offline | [v3 results](v3.md) |
| v4 | **Not spent** | Val macro-F1 **0.69296**; NEI floor **0.408** | Learned offline | [v4 results](v4.md) |
| v5 | Spent | Final-policy macro-F1 **0.73694**; NEI F1 **0.64483** | Learned offline | [v5 results](v5.md) |

## Reading the numbers

- v1/v2 report the learned model's raw three-class test macro-F1.
- v3/v4 are validation-only; they never consumed a test pass.
- v5 reports a **final-policy** macro-F1 in which abstentions are mapped to
  `INSUFFICIENT`. It is not directly comparable to v1/v2 raw macro-F1.
- The fixed 0.7/0.95 rule remains the serving default in every version.
- The best research result is v5. The best serving result is still the fixed rule.
- The generated v5 image report lives in [`output version 5/`](../../output%20version%205/summary.md).

Architecture for every version is in [`../README.md`](../README.md).
