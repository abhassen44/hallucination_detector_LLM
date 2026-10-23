"""Attach local corpus/index fingerprints to the completed frozen-rule report."""
import hashlib
import json

from src.utils.config import PROJECT_ROOT, load_config
from src.evaluation.halueval_claims import cached_revision
import os
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    out=PROJECT_ROOT/'experiments/stage5_inference/metrics.json'
    r=json.loads(out.read_text(encoding='utf-8'))
    if len(r['results'])!=5:
        raise ValueError('Evaluation is not complete')
    if digest(PROJECT_ROOT/'data/splits/halueval_test.jsonl')!=r['dataset_sha256']:
        raise ValueError('Dataset changed during evaluation')
    cache_meta=PROJECT_ROOT/'data/cache/halueval_nli_pairs.metadata.json'
    cache_provenance={'versions':r['versions'], 'unresolved_snapshot':
                       cached_revision(r['nli_fingerprint'][0]) if r['nli_fingerprint'][1] is None else None}
    if cache_meta.exists() and json.loads(cache_meta.read_text(encoding='utf-8'))!=cache_provenance:
        raise ValueError('Cache runtime provenance does not match this evaluation')
    cache_meta.write_text(json.dumps(cache_provenance,indent=2)+'\n', encoding='utf-8')
    cfg=load_config()
    paths=[PROJECT_ROOT/'data/processed/fever_corpus.jsonl']
    paths.extend(sorted((PROJECT_ROOT/'data/processed/indexes').glob('*')))
    r['retrieval_provenance']={'configuration':cfg['retrieval'], 'embedding_configuration':cfg['embeddings'],
        'files':{str(p.relative_to(PROJECT_ROOT)): {'sha256':digest(p),'bytes':p.stat().st_size} for p in paths if p.is_file()}}
    revision = cached_revision(r['nli_fingerprint'][0])
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub')))
    snapshot = hub/('models--'+r['nli_fingerprint'][0].replace('/','--'))/'snapshots'/str(revision)
    r['nli_cache_provenance'] = {'cached_revision':revision,'files':{}}
    if snapshot.exists():
        for p in snapshot.iterdir():
            if p.is_file() and (p.suffix in ('.json','.model','.bin','.safetensors')):
                r['nli_cache_provenance']['files'][p.name] = {'sha256':digest(p),'bytes':p.stat().st_size}
    out.write_text(json.dumps(r,indent=2)+'\n', encoding='utf-8')
    print('Dataset unchanged; corpus and index fingerprints saved.')


if __name__=='__main__':
    main()
