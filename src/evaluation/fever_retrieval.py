"""Stage 3 - Retrieval + end-to-end 3-way FEVER verification.

Part A  Retrieval quality on verifiable claims (gold evidence known):
        * strict R@k : some complete gold evidence set is inside the top-k sentences
        * hit@k      : at least one gold sentence in top-k
        * page R@k   : at least one gold page among the top-k sentences' pages
Part B  Full pipeline on ALL claims (incl. NOT ENOUGH INFO), top_k retrieved sentences:
        NLI(sentence, claim) per sentence -> max-aggregation -> label via
        * argmax
        * thresholds (t_entail, t_contra) tuned on val for macro-F1
        Reports 3-way acc/macro-F1, FEVER score, and binary hallucination metrics.

Retrieved hits + per-sentence NLI probs are cached to data/processed/retrieval/ so the
Stage-5 detector can build features without re-running retrieval/NLI.

    python -m src.evaluation.fever_retrieval [--rebuild-index]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from src.datasets.schema import (
    CONTRADICTED, INSUFFICIENT, SUPPORTED, load_examples, to_binary, write_jsonl,
)
from src.evaluation.metrics import binary_metrics, multiclass_metrics
from src.retrieval.hybrid import METHODS, Retriever
from src.utils.config import PROJECT_ROOT, load_config, set_seed
from src.verification.nli import NLIModel, aggregate_max

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
KS = (1, 5, 10, 20)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- retrieval metrics
def _gold(ex) -> list[set[tuple[str, int]]]:
    return [{(p, int(s)) for p, s in es} for es in ex.meta.get("evidence_sets", [])]


def retrieval_metrics(exs, hits_per_q, keys) -> dict:
    out = {}
    ver = [(e, h) for e, h in zip(exs, hits_per_q) if e.label != INSUFFICIENT and _gold(e)]
    for k in KS:
        strict = hit = page = 0
        for e, h in ver:
            top = {keys[i] for i, _ in h[:k]}
            pages = {p for p, _ in top}
            gold = _gold(e)
            strict += any(g <= top for g in gold)
            hit += any(g & top for g in gold)
            page += any({p for p, _ in g} & pages for g in gold)
        n = max(len(ver), 1)
        out[f"@{k}"] = {"strict_recall": strict / n, "hit": hit / n, "page_recall": page / n}
    out["n"] = len(ver)
    return out


# --------------------------------------------------------------------------- decision rules
def decide(p: np.ndarray, te: float | None = None, tc: float | None = None) -> str:
    e, n, c = p
    if te is None:                                   # plain argmax
        return (SUPPORTED, INSUFFICIENT, CONTRADICTED)[int(np.argmax(p))]
    if e >= te and e >= c:
        return SUPPORTED
    if c >= tc and c > e:
        return CONTRADICTED
    return INSUFFICIENT


def tune_thresholds(P: np.ndarray, y: list[str]) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    best, best_f = (0.5, 0.5), -1.0
    for te in grid:
        for tc in grid:
            f = multiclass_metrics(y, [decide(p, te, tc) for p in P])["macro_f1"]
            if f > best_f:
                best, best_f = (float(te), float(tc)), f
    return best


def fever_score(exs, preds, hits_per_q, keys, k) -> float:
    """Label correct AND (for verifiable claims) a full gold evidence set in the top-k."""
    ok = 0
    for e, p, h in zip(exs, preds, hits_per_q):
        if p != e.label:
            continue
        if e.label == INSUFFICIENT:
            ok += 1
        else:
            top = {keys[i] for i, _ in h[:k]}
            ok += any(g <= top for g in _gold(e))
    return ok / len(exs)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--nli-model", default=None)
    args = ap.parse_args()
    set_seed()
    cfg = load_config()
    top_k = cfg["retrieval"]["top_k"]
    cache_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "retrieval"

    retriever = Retriever(cfg, rebuild=args.rebuild_index)
    keys, texts = retriever.corpus.keys, retriever.corpus.texts
    print(f"[stage3] corpus sentences: {len(texts)}")

    data: dict[str, dict] = {}
    results: dict = {"retrieval": {}, "pipeline": {}}

    # ---- Part A: retrieval
    for split in SPLITS:
        exs = load_examples(SPLITS_DIR / f"fever_{split}.jsonl")
        hits = retriever.search_all([e.answer for e in exs], k=max(KS))
        data[split] = {"exs": exs, "hits": hits}
        if split != "train":
            results["retrieval"][split] = {m: retrieval_metrics(exs, hits[m], keys) for m in METHODS}

    # ---- Part B: NLI on top-k retrieved sentences (unique pairs only)
    nli = NLIModel(args.nli_model)
    pairs: dict[tuple[int, int], int] = {}
    prem, hyp = [], []
    offset = 0
    for split in SPLITS:
        exs, hits = data[split]["exs"], data[split]["hits"]
        for qi, e in enumerate(exs):
            for m in METHODS:
                for doc, _ in hits[m][qi][:top_k]:
                    key = (offset + qi, doc)
                    if key not in pairs:
                        pairs[key] = len(prem)
                        prem.append(texts[doc])
                        hyp.append(e.answer)
        data[split]["offset"] = offset
        offset += len(exs)
    print(f"[stage3] NLI on {len(prem)} unique (sentence, claim) pairs")
    probs = nli.predict(prem, hyp)

    agg: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        exs, hits, off = data[split]["exs"], data[split]["hits"], data[split]["offset"]
        agg[split] = {}
        rows = []
        for m in METHODS:
            agg[split][m] = np.stack([
                aggregate_max([probs[pairs[(off + qi, d)]] for d, _ in hits[m][qi][:top_k]])
                for qi in range(len(exs))
            ])
        for qi, e in enumerate(exs):
            rows.append({
                "id": e.id, "claim": e.answer, "label": e.label,
                **{m: [{"doc": d, "page_id": keys[d][0], "sent_id": keys[d][1], "score": s,
                        **({"nli": probs[pairs[(off + qi, d)]].tolist()} if r < top_k else {})}
                       for r, (d, s) in enumerate(hits[m][qi])] for m in METHODS},
            })
        write_jsonl(cache_dir / f"fever_{split}.jsonl", rows)

    for m in METHODS:
        Pv, Pt = agg["val"][m], agg["test"][m]
        yv = [e.label for e in data["val"]["exs"]]
        exs_t = data["test"]["exs"]
        yt = [e.label for e in exs_t]
        te, tc = tune_thresholds(Pv, yv)
        for rule, kw in (("argmax", {}), ("thresh", {"te": te, "tc": tc})):
            pred = [decide(p, **kw) for p in Pt]
            results["pipeline"][f"{m}/{rule}"] = {
                **({"thresholds": {"entail": te, "contra": tc}} if kw else {}),
                "3way": multiclass_metrics(yt, pred),
                "fever_score": fever_score(exs_t, pred, data["test"]["hits"][m], keys, top_k),
                "binary": binary_metrics([e.binary_label for e in exs_t], [to_binary(p) for p in pred]),
            }

    results.update({"nli_model": nli.model_name, "top_k": top_k,
                    "timestamp": datetime.now().isoformat(timespec="seconds")})
    out_dir = PROJECT_ROOT / cfg["paths"]["experiments"] / "stage3_retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{nli.model_name.split('/')[-1]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    _print(results)
    print(f"\nSaved -> {out_path}\nCached retrieval -> {cache_dir}")


def _print(r: dict) -> None:
    print("\n=== Retrieval on FEVER test (verifiable claims) ===")
    print(f"{'method':8s} " + " ".join(f"{'R@' + str(k):>7s} {'hit@' + str(k):>7s}" for k in KS))
    for m, v in r["retrieval"]["test"].items():
        print(f"{m:8s} " + " ".join(f"{v[f'@{k}']['strict_recall']:7.3f} {v[f'@{k}']['hit']:7.3f}" for k in KS))
    print(f"\n=== End-to-end 3-way FEVER test (top-{r['top_k']} evidence) ===")
    print(f"{'method/rule':18s} {'acc':>6s} {'mF1':>6s} {'FEVER':>6s} {'binF1':>6s}")
    for k, v in r["pipeline"].items():
        print(f"{k:18s} {v['3way']['accuracy']:6.3f} {v['3way']['macro_f1']:6.3f} "
              f"{v['fever_score']:6.3f} {v['binary']['f1']:6.3f}")


if __name__ == "__main__":
    main()
