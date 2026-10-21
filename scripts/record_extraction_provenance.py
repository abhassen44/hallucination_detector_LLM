"""Record exact cached extraction dates and input fingerprints; no model calls."""
import hashlib
import json
from datetime import datetime, timezone

from src.datasets.schema import load_examples
from src.generation.claim_extraction import ClaimExtractor, EXTRACTION_SYSTEM_PROMPT
from src.generation.llm import OllamaLLM
from src.utils.config import PROJECT_ROOT

llm=OllamaLLM()
extractor=ClaimExtractor(llm)
created=[]
files={}
fallbacks=0
for split in ('train','val','test'):
    path=PROJECT_ROOT/f'data/splits/halueval_{split}.jsonl'
    files[str(path.relative_to(PROJECT_ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for e in load_examples(path):
        assert e.meta['claim_extraction']['model']==llm.model
        key=llm._key(extractor._build_prompt(e.answer,e.question),EXTRACTION_SYSTEM_PROMPT,None)
        row=llm.cache._conn.execute('select created from cache where key=?',(key,)).fetchone()
        if row is None:
            raise ValueError(f'No cached response for {e.id}')
        created.append(row[0])
        fallbacks+=e.meta['claim_extraction']['is_fallback']
report={'model':llm.model,'temperature':llm.temperature,'cached_examples':len(created),
        'generation_window_utc':{'first':datetime.fromtimestamp(min(created),timezone.utc).isoformat(),
                                 'last':datetime.fromtimestamp(max(created),timezone.utc).isoformat()},
        'system_prompt_sha256':hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode()).hexdigest(),
        'dataset_sha256':files,'fallback_examples':fallbacks,
        'syntax_proposals_applied':False,'training_performed':False,
        'cloud_model_revision':'not exposed; cached responses freeze this extraction run'}
(PROJECT_ROOT/'experiments/stage4_claims/provenance.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
print('Recorded extraction provenance for',len(created),'examples')
