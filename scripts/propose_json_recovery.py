"""Create reviewable syntax proposals for the three fallbacks, without changing data."""
import json

from src.claims.recovery import propose_terminal_quote_recovery
from src.datasets.schema import load_examples
from src.generation.claim_extraction import ClaimExtractor, EXTRACTION_SYSTEM_PROMPT
from src.generation.llm import OllamaLLM
from src.utils.config import PROJECT_ROOT


def main():
    llm = OllamaLLM()
    extractor = ClaimExtractor(llm)
    proposals = []
    for split in ('train','val','test'):
        for e in load_examples(PROJECT_ROOT/f'data/splits/halueval_{split}.jsonl'):
            if not e.meta['claim_extraction']['is_fallback']:
                continue
            raw=llm.cache.get(llm._key(extractor._build_prompt(e.answer,e.question),EXTRACTION_SYSTEM_PROMPT,None))
            proposals.append({'id':e.id,'split':split,'question':e.question,'answer':e.answer,
                              'current_fallback_claims':e.claims,'raw_model_response':raw,
                              'syntax_proposal':propose_terminal_quote_recovery(raw) if raw else None,
                              'human_accepted':None})
    out=PROJECT_ROOT/'experiments/stage4_claims/json_recovery_proposals.json'
    if out.exists() and any(r.get('human_accepted') is not None for r in json.loads(out.read_text())):
        raise ValueError('Refusing to overwrite reviewed proposals')
    out.write_text(json.dumps(proposals,indent=2)+'\n')
    print('Syntax proposals:',sum(r['syntax_proposal'] is not None for r in proposals),'/',len(proposals))


if __name__=='__main__':
    main()
