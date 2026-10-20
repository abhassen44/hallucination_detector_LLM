from types import SimpleNamespace

import numpy as np
import pytest

from src.datasets.schema import Example
from src.evaluation.halueval_claims import PairCache, answer_prediction, provided_rows, score_rows


class FakeNLI:
    model_name = 'test-model'
    max_length = 512
    use_fp16 = False
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash='revision-one'))

    def __init__(self):
        self.calls = []

    def predict(self, premises, hypotheses, **kwargs):
        self.calls.append(list(zip(premises, hypotheses)))
        return np.tile([.9,.05,.05], (len(premises),1))


@pytest.mark.parametrize('rule', ['argmax','fixed_threshold'])
def test_all_claims_must_be_supported_and_empty_is_not_supported(rule):
    assert answer_prediction([[.9,.05,.05]],rule) == 'supported'
    assert answer_prediction([[.9,.05,.05],[.01,.01,.98]],rule) == 'hallucinated'
    assert answer_prediction([],rule) == 'hallucinated'


def test_decision_rules_are_frozen_and_distinct():
    assert answer_prediction([[.5,.3,.2]],'argmax') == 'supported'
    assert answer_prediction([[.5,.3,.2]],'fixed_threshold') == 'hallucinated'
    with pytest.raises(ValueError):
        answer_prediction([[1,0,0]],'tuned')


def test_pair_cache_deduplicates_and_isolates_model_revisions(tmp_path):
    model = FakeNLI()
    path = tmp_path/'pairs.jsonl'
    cache = PairCache(model,path)
    pairs = [('evidence','claim'),('evidence','claim')]
    actual = cache.predict(pairs)
    assert model.calls == [[('evidence','claim')]]
    assert actual.shape == (2,3)
    reloaded = PairCache(model,path)
    np.testing.assert_equal(reloaded.predict(pairs),actual)
    assert len(model.calls) == 1
    model.model = SimpleNamespace(config=SimpleNamespace(_commit_hash='revision-two'))
    PairCache(model,path).predict(pairs)
    assert len(model.calls) == 2


def test_provided_context_evaluation_preserves_examples_without_claims(tmp_path):
    examples = [Example('1','halueval','question','answer',[],['knowledge'],None,
                        'hallucinated','g',{'claim_extraction':{'is_fallback':False}})]
    model = FakeNLI()
    rows = provided_rows(examples,PairCache(model,tmp_path/'pairs.jsonl'))
    assert model.calls == [[('knowledge','question answer')]]
    assert rows[0]['claims'] == []
    whole = score_rows(rows,whole=True)
    claims = score_rows(rows)
    assert whole['argmax']['accuracy'] == 0
    assert claims['argmax']['accuracy'] == 1
    assert rows[0]['binary_gold'] == 'hallucinated'


def test_pair_cache_recovers_only_an_interrupted_final_line(tmp_path):
    model = FakeNLI()
    path = tmp_path/'pairs.jsonl'
    PairCache(model,path).predict([('p','h')])
    with path.open('a') as out:
        out.write('{"key": "unfinished')
    cache = PairCache(model,path)
    cache.predict([('p','h'),('new','hypothesis')])
    assert len(model.calls) == 2
    assert model.calls[-1] == [('new','hypothesis')]
    import json
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 2


def test_pair_cache_does_not_ignore_completed_corrupt_records(tmp_path):
    model = FakeNLI()
    path = tmp_path/'pairs.jsonl'
    PairCache(model,path).predict([('p','h')])
    with path.open('a') as out:
        out.write('not json\n')
    original = path.read_bytes()
    import json
    with pytest.raises(json.JSONDecodeError):
        PairCache(model,path)
    assert path.read_bytes() == original
