import json
import pytest

from src.evaluation.cached_baseline import evaluate
from src.utils.config import PROJECT_ROOT


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_committed_predictions_reproduce_original_metrics(tmp_path, method):
    actual = evaluate(method=method, output=tmp_path)
    original = json.loads((PROJECT_ROOT / 'experiments/stage3_retrieval/metrics_DeBERTa-v3-base-mnli.json').read_text())['pipeline'][f'{method}/thresh']['3way']
    assert actual['n'] == original['n']
    assert actual['accuracy'] == original['accuracy']
    assert abs(actual['macro_f1'] - original['macro_f1']) < 1e-12
    assert actual['confusion_matrix'] == original['confusion_matrix']
    review = json.loads((tmp_path / f'{method}_review.json').read_text())
    assert len(review['examples']) == 60
    assert all(r['category'] is None for r in review['examples'])
