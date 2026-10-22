"""Recount frozen v2 predictions with correctly named reliability metrics."""
import hashlib
import json

from src.datasets.schema import read_jsonl
from src.evaluation.reliability import reliability_metrics
from src.utils.config import PROJECT_ROOT


def main():
    policy_path = PROJECT_ROOT / 'experiments/v2/policy.json'
    thresholds = json.loads(policy_path.read_text(encoding='utf-8'))['policy']['thresholds']
    inputs = {str(policy_path.relative_to(PROJECT_ROOT)):
              hashlib.sha256(policy_path.read_bytes()).hexdigest()}
    results = {}
    for split in ('val', 'test'):
        path = PROJECT_ROOT / f'experiments/v2/{split}_predictions_v2.jsonl'
        rows = list(read_jsonl(path))
        accepted = [r['confidence'] >= thresholds[r['prediction']] for r in rows]
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate prediction IDs')
        if any('answered' in r and r['answered'] != a for r, a in zip(rows, accepted)):
            raise ValueError('Frozen acceptance decisions differ')
        results[split] = reliability_metrics([r['gold'] for r in rows],
                                              [r['prediction'] for r in rows], accepted)
        inputs[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = PROJECT_ROOT / 'experiments/improvements/metrics_frozen_policy_reliability.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'scope': 'diagnostic recount of frozen v2 outputs; not a new evaluation',
        'training_performed': False, 'inference_performed': False,
        'threshold_selection_performed': False, 'input_sha256': inputs,
        'results': results,
    }, indent=2) + '\n', encoding='utf-8')
    print(output.relative_to(PROJECT_ROOT))


if __name__ == '__main__':
    main()
