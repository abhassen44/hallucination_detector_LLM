"""Stage 2 - Oracle-evidence verification (upper bound before retrieval).

FEVER  : verifiable claims only (NEI claims have no gold evidence; 3-way is evaluated
         in Stage 3 once retrieval supplies evidence for every claim).
         * NLI-concat : premise = all gold sentences joined
         * NLI-max    : NLI per sentence, max-aggregated
         * Similarity : max cosine; threshold tuned on val
HaluEval: premise = knowledge, hypothesis = "question + answer".
         * NLI      : hallucinated if P(entail) < t   (t tuned on val)
         * Similarity: hallucinated if cos < t        (t tuned on val)

    python -m src.evaluation.oracle_nli [--nli-model NAME]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from src.datasets.schema import (
    BIN_HALLUCINATED, BIN_SUPPORTED, CONTRADICTED, INSUFFICIENT, SUPPORTED, load_examples,
)
from src.evaluation.metrics import best_threshold, binary_metrics, multiclass_metrics
from src.utils.config import PROJECT_ROOT, load_config, set_seed
from src.verification.nli import NLIModel, aggregate_max, probs_to_labels
from src.verification.similarity import SimilarityModel

SPLITS = PROJECT_ROOT / "data" / "splits"


def _fever(split: str):
    exs = [e for e in load_examples(SPLITS / f"fever_{split}.jsonl") if e.label != INSUFFICIENT and e.context]
    return exs, [e.label for e in exs]


def _two_way(probs: np.ndarray) -> list[str]:
    return [SUPPORTED if p[0] >= p[2] else CONTRADICTED for p in probs]


def run_fever(nli: NLIModel, sim: SimilarityModel) -> dict:
    res: dict = {}
    data = {s: _fever(s) for s in ("val", "test")}

    for split, (exs, y) in data.items():
        claims = [e.answer for e in exs]
        # NLI-concat
        p_cat = nli.predict([" ".join(e.context) for e in exs], claims)
        # NLI-max
        flat_p, flat_h, spans = [], [], []
        for e in exs:
            spans.append((len(flat_p), len(flat_p) + len(e.context)))
            flat_p += e.context
            flat_h += [e.answer] * len(e.context)
        p_flat = nli.predict(flat_p, flat_h)
        p_max = np.stack([aggregate_max(list(p_flat[a:b])) for a, b in spans])

        res[f"nli_concat/{split}"] = {"3way": multiclass_metrics(y, probs_to_labels(p_cat)),
                                     "2way": multiclass_metrics(y, _two_way(p_cat))}
        res[f"nli_max/{split}"] = {"3way": multiclass_metrics(y, probs_to_labels(p_max)),
                                  "2way": multiclass_metrics(y, _two_way(p_max))}
        data[split] = (exs, y, sim.max_similarity(claims, [e.context for e in exs]))

    # Similarity: "contradicted" is positive, low similarity => contradicted
    _, yv, sv = data["val"]
    t = best_threshold(sv, np.array([l == CONTRADICTED for l in yv]))
    _, yt, st = data["test"]
    pred = [CONTRADICTED if s < t else SUPPORTED for s in st]
    res["similarity/test"] = {"threshold": t, "2way": multiclass_metrics(yt, pred)}
    return res


def run_halueval(nli: NLIModel, sim: SimilarityModel) -> dict:
    res: dict = {}
    scores = {}
    for split in ("val", "test"):
        exs = load_examples(SPLITS / f"halueval_{split}.jsonl")
        y = [e.binary_label for e in exs]
        hyp = [f"{e.question} {e.answer}" for e in exs]
        probs = nli.predict([e.context[0] if e.context else "" for e in exs], hyp)
        s = sim.max_similarity(hyp, [e.context for e in exs])
        scores[split] = (y, probs, s)

    yv, pv, sv = scores["val"]
    yt, pt, st = scores["test"]
    is_h = np.array([l == BIN_HALLUCINATED for l in yv])

    argmax_pred = [BIN_SUPPORTED if l == SUPPORTED else BIN_HALLUCINATED for l in probs_to_labels(pt)]
    res["nli_argmax/test"] = binary_metrics(yt, argmax_pred)

    t = best_threshold(pv[:, 0], is_h)
    res["nli_entail_thresh/test"] = {"threshold": t, **binary_metrics(
        yt, [BIN_HALLUCINATED if p < t else BIN_SUPPORTED for p in pt[:, 0]])}

    t = best_threshold(sv, is_h)
    res["similarity/test"] = {"threshold": t, **binary_metrics(
        yt, [BIN_HALLUCINATED if s < t else BIN_SUPPORTED for s in st])}
    return res


def _print(results: dict) -> None:
    print("\n=== FEVER (verifiable claims, gold evidence) ===")
    print(f"{'method':28s} {'acc':>6s} {'macroF1':>8s}")
    for k, v in results["fever"].items():
        for mode in ("3way", "2way"):
            if mode in v:
                m = v[mode]
                print(f"{k + ' ' + mode:28s} {m['accuracy']:6.3f} {m['macro_f1']:8.3f}")
    print("\n=== HaluEval QA (knowledge as evidence) ===")
    print(f"{'method':28s} {'acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for k, m in results["halueval"].items():
        print(f"{k:28s} {m['accuracy']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", default=None)
    args = ap.parse_args()
    set_seed()
    cfg = load_config()

    nli = NLIModel(args.nli_model)
    sim = SimilarityModel()
    results = {
        "nli_model": nli.model_name,
        "embedding_model": sim.model_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fever": run_fever(nli, sim),
        "halueval": run_halueval(nli, sim),
    }
    out_dir = PROJECT_ROOT / cfg["paths"]["experiments"] / "stage2_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = nli.model_name.split("/")[-1]
    with open(out_dir / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    _print(results)
    print(f"\nSaved -> {out_dir / f'metrics_{tag}.json'}")


if __name__ == "__main__":
    main()
