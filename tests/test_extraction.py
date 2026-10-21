import json
from unittest.mock import Mock

import pytest

from src.claims.extractor import assess_claims
from src.datasets.schema import Example, load_examples, write_jsonl
from src.generation import claim_extraction as extraction


def test_failed_llm_preserves_fallback_provenance():
    llm = Mock()
    llm.model = 'test-model'
    llm.generate_json.side_effect = RuntimeError('offline')
    extractor = extraction.ClaimExtractor(llm)
    result = extractor.extract_with_metadata('Paris is in France. It is a city.')
    assert result.is_fallback
    assert len(result.claims) == 2
    report = assess_claims('a', 'answer', result.claims, is_fallback=result.is_fallback)
    assert report.is_fallback and report.has_issues


def test_successful_batch_retains_order_and_rejects_nontext():
    llm = Mock()
    llm.model = 'test-model'
    llm.generate_json.return_value = ['A valid factual statement.', None, {'bad': 'format'}, 12, '']
    extractor = extraction.ClaimExtractor(llm)
    results = extractor.extract_batch_results([('first', None), ('second', None)], False)
    assert [r.claims for r in results] == [['A valid factual statement.']] * 2
    assert not any(r.is_fallback for r in results)


def test_limited_extraction_preserves_remaining_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, 'PROJECT_ROOT', tmp_path)
    path = tmp_path / 'data/splits/halueval_test.jsonl'
    rows = [Example(str(i), 'halueval', 'q', 'an answer', [], [], None,
                    'supported', str(i)) for i in range(3)]
    write_jsonl(path, rows)
    llm = Mock()
    llm.model = 'test-model'
    llm.generate_json.return_value = ['An extracted factual claim.']
    extraction.process_dataset('test', extraction.ClaimExtractor(llm), limit=1)
    saved = load_examples(path)
    assert len(saved) == 3
    assert saved[0].claims == ['An extracted factual claim.']
    assert saved[0].meta['claim_extraction'] == {'is_fallback': False, 'model': 'test-model'}
    assert saved[1:] == rows[1:]
    with pytest.raises(ValueError):
        extraction.process_dataset('test', extraction.ClaimExtractor(llm), limit=-1)


def test_interrupted_write_does_not_destroy_original(tmp_path):
    path = tmp_path / 'split.jsonl'
    write_jsonl(path, [{'id': 'original'}])
    def broken_rows():
        yield {'id': 'new'}
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        write_jsonl(path, broken_rows())
    assert json.loads(path.read_text(encoding='utf-8')) == {'id': 'original'}
    assert not list(tmp_path.glob('*.tmp'))


def test_valid_empty_claims_are_not_an_llm_failure():
    llm = Mock()
    llm.generate_json.return_value = []
    result = extraction.ClaimExtractor(llm).extract_with_metadata("It's hard to tell.")
    assert result.claims == [] and not result.is_fallback
