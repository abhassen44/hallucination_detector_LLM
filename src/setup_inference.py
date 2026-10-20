"""Download pretrained models and build evidence indexes; never trains models.

Usage: python -m src.setup_inference
Run Ollama separately. This command does not pull or invoke a generative model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.utils.config import PROJECT_ROOT


def main():
    from src.retrieval.hybrid import Retriever
    from src.retrieval.dense import INDEX_FILENAME
    from src.verification.nli import NLIModel

    retriever = Retriever()
    print(f'Index ready: {len(retriever.corpus)} sentences ({INDEX_FILENAME})', flush=True)
    nli = NLIModel()
    probabilities = nli.predict(['Paris is the capital of France.'],
                                ['Paris is in France.'], show_progress=False)
    if probabilities.shape != (1, 3):
        raise RuntimeError('Unexpected NLI output shape')
    report = {'timestamp': datetime.now(timezone.utc).isoformat(),
              'corpus_sentences': len(retriever.corpus), 'index': INDEX_FILENAME,
              'nli_model': nli.model_name, 'embedding_model': retriever.dense.encoder.model_name,
              'smoke_probabilities': probabilities.tolist(), 'training_performed': False}
    out = PROJECT_ROOT / 'experiments' / 'inference_setup'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
