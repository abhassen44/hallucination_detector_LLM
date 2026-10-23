"""Build the v4 expanded FEVER train split from unused paper_dev claims.

The 999-row val split is untouched. Stage A adds 3,000 balanced claims (train
3k -> 6k); stage B adds the remaining 3,000 (train 6k -> 9k).

    python -m scripts.v4.make_v4_train_split
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.datasets.schema import (CONTRADICTED, INSUFFICIENT, SUPPORTED, load_examples,
                                 to_binary, write_jsonl)
from src.utils.config import PROJECT_ROOT

RAW = PROJECT_ROOT / 'data/raw/fever/paper_dev.jsonl'
TRAIN = PROJECT_ROOT / 'data/splits/fever_train.jsonl'
VAL = PROJECT_ROOT / 'data/splits/fever_val.jsonl'
OUT = PROJECT_ROOT / 'data/splits'
SEED = 42

LABEL_MAP = {'SUPPORTS': SUPPORTED, 'REFUTES': CONTRADICTED, 'NOT ENOUGH INFO': INSUFFICIENT}


def _evidence_sets(raw: dict) -> list[list[list[object]]]:
    sets = []
    for group in raw.get('evidence', []):
        evidence = [[item[2], int(item[3])] for item in group
                    if item[2] is not None and item[3] is not None]
        if evidence:
            sets.append(evidence)
    return sets


def _to_row(raw: dict, split: str) -> dict:
    claim = str(raw['claim']).strip()
    label = LABEL_MAP[raw['label']]
    return {
        'id': f"fever-{raw['id']}",
        'source': 'fever', 'question': None, 'answer': claim, 'claims': [claim],
        'context': [], 'label': label, 'binary_label': to_binary(label),
        'group': f"fever-{raw['id']}",
        'meta': {'split': split, 'evidence_sets': _evidence_sets(raw),
                 'missing_evidence': 0},
    }


def main() -> None:
    used = {example.id for example in load_examples(TRAIN)}
    used |= {example.id for example in load_examples(VAL)}
    raw_rows = [json.loads(line) for line in RAW.read_text(encoding='utf-8').splitlines()
                if line.strip()]
    unused = [row for row in raw_rows if f"fever-{row['id']}" not in used]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in unused:
        by_label[row['label']].append(row)

    rng = random.Random(SEED)
    stage_a_raw, stage_b_raw = [], []
    for label in sorted(by_label):
        bucket = by_label[label][:]
        rng.shuffle(bucket)
        stage_a_raw.extend(bucket[:1000])
        stage_b_raw.extend(bucket[1000:2000])
    rng.shuffle(stage_a_raw)
    rng.shuffle(stage_b_raw)
    stage_a = [_to_row(row, 'train') for row in stage_a_raw]
    stage_b = [_to_row(row, 'train') for row in stage_b_raw]

    current = load_examples(TRAIN)
    combined_6k = [example.to_dict() for example in current] + stage_a
    combined_9k = combined_6k + stage_b
    write_jsonl(OUT / 'fever_train_v4_6000.jsonl', combined_6k)
    write_jsonl(OUT / 'fever_train_v4_9000.jsonl', combined_9k)
    write_jsonl(OUT / 'fever_train_v4_extra_3000.jsonl', stage_a)
    write_jsonl(OUT / 'fever_train_v4_extra_6000.jsonl', stage_a + stage_b)
    print(json.dumps({'unused': len(unused), 'extra_3000': len(stage_a),
                      'extra_6000': len(stage_a) + len(stage_b),
                      'train_6000': len(combined_6k), 'train_9000': len(combined_9k)},
                     indent=2))


if __name__ == '__main__':
    main()
