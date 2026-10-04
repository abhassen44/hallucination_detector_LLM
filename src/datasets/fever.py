"""FEVER loader.

* Claims come from the official ``paper_dev`` / ``paper_test`` files.
* Gold evidence sentences are resolved from ``wiki-pages.zip``, which is streamed
  (never fully extracted). In the same single pass we also keep a random sample of
  non-gold "distractor" pages, giving a realistic yet laptop-sized retrieval corpus.
"""
from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from src.datasets.download import download
from src.datasets.schema import (
    INSUFFICIENT, LABELS, Example, read_jsonl, to_binary, write_jsonl,
)

LABEL_MAP = {"SUPPORTS": "SUPPORTED", "REFUTES": "CONTRADICTED", "NOT ENOUGH INFO": INSUFFICIENT}
TOTAL_WIKI_PAGES = 5_416_537

_PTB = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
}
_PTB_RE = re.compile("|".join(map(re.escape, _PTB)))
_ID_RE = re.compile(r'^\{"id": "((?:[^"\\]|\\.)*)"')


def clean_text(s: str) -> str:
    return _PTB_RE.sub(lambda m: _PTB[m.group(0)], s).strip()


def page_title(page_id: str) -> str:
    return clean_text(page_id.replace("_", " "))


def parse_lines(lines_field: str) -> list[str]:
    """FEVER 'lines' = 'idx\\tsentence\\tlink\\t...\\n...'. Returns list indexed by sentence id."""
    out: dict[int, str] = {}
    for row in lines_field.split("\n"):
        parts = row.split("\t")
        if parts and parts[0].isdigit():
            out[int(parts[0])] = clean_text(parts[1]) if len(parts) > 1 else ""
    return [out.get(i, "") for i in range(max(out) + 1)] if out else []


# --------------------------------------------------------------------------- download
def download_fever(raw_dir: Path, base_url: str) -> dict[str, Path]:
    files = ["paper_dev.jsonl", "paper_test.jsonl", "wiki-pages.zip"]
    return {f: download(f"{base_url}/{f}", raw_dir / "fever" / f) for f in files}


# --------------------------------------------------------------------------- claims
def _balanced_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    per = n // len(by_label)
    out: list[dict] = []
    for lab in sorted(by_label):
        bucket = by_label[lab][:]
        rng.shuffle(bucket)
        out.extend(bucket[:per])
    rng.shuffle(out)
    return out


def load_fever_claims(paths: dict[str, Path], max_per_split: dict[str, int], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    dev = list(read_jsonl(paths["paper_dev.jsonl"]))
    test = list(read_jsonl(paths["paper_test.jsonl"]))

    dev = _balanced_sample(dev, max_per_split["train"] + max_per_split["val"], rng)
    # stratified train/val cut from the balanced dev sample
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in dev:
        by_label[r["label"]].append(r)
    val_frac = max_per_split["val"] / (max_per_split["train"] + max_per_split["val"])
    train, val = [], []
    for bucket in by_label.values():
        k = round(len(bucket) * val_frac)
        val.extend(bucket[:k])
        train.extend(bucket[k:])
    rng.shuffle(train), rng.shuffle(val)

    return {"train": train, "val": val, "test": _balanced_sample(test, max_per_split["test"], rng)}


def _gold_sets(row: dict) -> list[list[tuple[str, int]]]:
    """Evidence sets as [(page_id, sent_id), ...]; empty for NOT ENOUGH INFO."""
    sets = []
    for ev_set in row.get("evidence", []):
        s = [(e[2], int(e[3])) for e in ev_set if e[2] is not None and e[3] is not None]
        if s:
            sets.append(s)
    return sets


# --------------------------------------------------------------------------- wiki
def scan_wiki(
    zip_path: Path, needed: set[str], n_distractors: int, seed: int
) -> tuple[dict[str, list[str]], list[dict]]:
    """Single streaming pass over wiki-pages.zip.

    Returns (gold page_id -> sentences, corpus rows incl. gold + distractor pages).
    """
    rng = random.Random(seed)
    p_keep = min(1.0, 1.3 * n_distractors / TOTAL_WIKI_PAGES)  # oversample, cap later
    gold: dict[str, list[str]] = {}
    distractors: list[dict] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".jsonl"))
        for name in tqdm(names, desc="Scanning wiki shards"):
            with zf.open(name) as fh:
                for raw in fh:
                    line = raw.decode("utf-8", errors="replace")
                    m = _ID_RE.match(line)
                    if not m:
                        continue
                    try:
                        pid = json.loads(f'"{m.group(1)}"')
                    except Exception:
                        continue
                    is_gold = pid in needed
                    if not pid or (not is_gold and rng.random() >= p_keep):
                        continue
                    try:
                        page = json.loads(line)
                    except Exception:
                        continue
                    sents = parse_lines(page.get("lines", ""))
                    if not any(sents):
                        continue
                    if is_gold:
                        gold[pid] = sents
                    else:
                        distractors.append({"page_id": pid, "title": page_title(pid),
                                            "sentences": sents, "is_gold_page": False})

    rng.shuffle(distractors)
    corpus = [{"page_id": p, "title": page_title(p), "sentences": s, "is_gold_page": True}
              for p, s in gold.items()]
    corpus.extend(distractors[:n_distractors])
    return gold, corpus


# --------------------------------------------------------------------------- public API
def to_example(row: dict, split: str, wiki: dict[str, list[str]]) -> Example:
    label = LABEL_MAP[row["label"]]
    sets = _gold_sets(row)
    context, seen, missing = [], set(), 0
    for s in sets:
        for pid, sid in s:
            sents = wiki.get(pid)
            if sents is None or sid >= len(sents) or not sents[sid]:
                missing += 1
                continue
            if (pid, sid) not in seen:
                seen.add((pid, sid))
                context.append(f"{page_title(pid)}: {sents[sid]}")
    claim = clean_text(row["claim"])
    return Example(
        id=f"fever-{row['id']}", source="fever", question=None, answer=claim, claims=[claim],
        context=context, label=label, binary_label=to_binary(label), group=f"fever-{row['id']}",
        meta={"split": split, "evidence_sets": sets, "missing_evidence": missing},
    )


def load_fever(cfg: dict, raw_dir: Path, processed_dir: Path, splits_dir: Path, seed: int) -> dict[str, list[Example]]:
    fcfg = cfg["datasets"]["fever"]
    paths = download_fever(raw_dir, fcfg["base_url"])
    claims = load_fever_claims(paths, fcfg["max_per_split"], seed)

    needed = {pid for rows in claims.values() for r in rows for s in _gold_sets(r) for pid, _ in s}
    print(f"[fever] {sum(map(len, claims.values()))} claims need {len(needed)} gold pages")

    wiki, corpus = scan_wiki(paths["wiki-pages.zip"], needed, fcfg["n_distractor_pages"], seed)
    n = write_jsonl(processed_dir / "fever_corpus.jsonl", corpus)
    print(f"[fever] corpus: {n} pages ({len(wiki)} gold, {n - len(wiki)} distractors)")

    out = {}
    for split, rows in claims.items():
        exs = [to_example(r, split, wiki) for r in rows]
        write_jsonl(splits_dir / f"fever_{split}.jsonl", exs)
        out[split] = exs
        dist = Counter(e.label for e in exs)
        print(f"[fever] {split}: {len(exs)} | " + ", ".join(f"{l}={dist[l]}" for l in LABELS))

    if fcfg.get("delete_zip_after"):
        paths["wiki-pages.zip"].unlink(missing_ok=True)
    return out
