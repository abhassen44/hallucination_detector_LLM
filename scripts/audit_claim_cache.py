"""Report extraction progress from cached responses; makes no model requests."""
import json

from src.datasets.schema import load_examples
from src.generation.claim_extraction import ClaimExtractor, EXTRACTION_SYSTEM_PROMPT
from src.generation.llm import parse_json_loose, OllamaLLM
from src.utils.config import PROJECT_ROOT

llm = OllamaLLM()
extractor = ClaimExtractor(llm)
report = {'model': llm.model, 'expected_examples': 4000, 'splits': {}}
for split in ('train', 'val', 'test'):
    examples = load_examples(PROJECT_ROOT / f'data/splits/halueval_{split}.jsonl')
    cached = valid = total_claims = 0
    for e in examples:
        key = llm._key(extractor._build_prompt(e.answer, e.question), EXTRACTION_SYSTEM_PROMPT, None)
        text = llm.cache.get(key)
        if text is None:
            continue
        cached += 1
        try:
            parsed = parse_json_loose(text)
            claims = extractor._parse_response(parsed)
        except (ValueError, TypeError):
            claims, parsed = [], None
        if claims or extractor._is_valid_empty(parsed):
            valid += 1
            total_claims += len(claims)
    report['splits'][split] = {'expected': len(examples), 'cached': cached,
                               'valid_cached': valid, 'claims': total_claims}
report['cached_examples'] = sum(s['cached'] for s in report['splits'].values())
report['all_responses_cached'] = report['cached_examples'] == report['expected_examples']
report['valid_cached_examples'] = sum(s['valid_cached'] for s in report['splits'].values())
out = PROJECT_ROOT / 'experiments/stage4_claims/cache_progress.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
