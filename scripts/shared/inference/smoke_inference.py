"""Exercise the running local API with real pretrained inference, never training."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--output', type=Path, default=Path('experiments/inference_setup/live_smoke.json'))
args = parser.parse_args()
report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'training_performed': False, 'checks': []}


def post(route, body):
    started = time.monotonic()
    response = requests.post(args.url.rstrip('/')+route, json=body, timeout=300)
    response.raise_for_status()
    result = response.json()
    report['checks'].append({'route': route, 'input': body, 'output': result,
                             'seconds': round(time.monotonic()-started, 3)})
    return result


try:
    evidence = ['The Eiffel Tower was completed in 1889.']
    for claim, expected, ev in [
        ('The Eiffel Tower was completed in 1889.', 'SUPPORTED', evidence),
        ('The Eiffel Tower was completed in 1895.', 'CONTRADICTED', evidence),
        ('The Eiffel Tower was completed in 1889.', 'INSUFFICIENT', []),
    ]:
        result = post('/verify', {'claim': claim, 'evidence': ev})
        assert result['label'] == expected, (expected, result)
    result = post('/detect', {'answer': 'Chris Paul is a four-time NBA assist leader and six-time steal leader.'})
    assert result['claims'], 'No extracted claims'
    assert not result['extraction_fallback'], 'LLM extraction fell back to sentence splitting'
    assert any(c['evidence'] for c in result['claims']), 'No retrieved evidence'
    supplied = post('/detect', {
        'question': 'How many times has Chris Paul led the NBA in assists and steals?',
        'answer': 'Chris Paul is a four-time NBA assist leader and six-time steal leader.',
        'evidence': ['Chris Paul has led the NBA in assists four times and steals six times.']})
    assert supplied['evidence_source'] == 'provided' and supplied['label'] == 'SUPPORTED'
    assert all(e['source'] == 'provided' for c in supplied['claims'] for e in c['evidence'])
    report['passed'] = True
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': report.get('passed', False), 'checks': len(report['checks']),
                      'report': str(args.output)}, indent=2))
