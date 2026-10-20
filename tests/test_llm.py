from unittest.mock import Mock, patch

from src.generation.llm import OllamaLLM


def config(tmp_path, model):
    return dict(base_url='http://localhost:11434', model=model, temperature=0,
                timeout=10, max_retries=1, retry_backoff=1,
                cache_db=str(tmp_path / 'cache.sqlite'))


def test_cloud_requests_json_in_prompt_without_unsupported_format(tmp_path):
    llm = OllamaLLM(config(tmp_path, 'gemma4:31b-cloud'))
    response = Mock()
    response.json.return_value = {'response': '```json\n["Paris is in France."]\n```'}
    with patch('src.generation.llm.requests.post', return_value=response) as post:
        assert llm.generate_json('Return JSON claims', schema={'type': 'array'}) == ['Paris is in France.']
        body = post.call_args.kwargs['json']
        assert 'format' not in body
        assert body['model'] == 'gemma4:31b-cloud'
        # Cached raw text is parsed on subsequent calls without another request.
        assert llm.generate_json('Return JSON claims', schema={'type': 'array'})
        assert post.call_count == 1


def test_local_schema_and_disabled_thinking_are_sent(tmp_path):
    llm = OllamaLLM(config(tmp_path, 'qwen3:4b'))
    response = Mock()
    response.json.return_value = {'response': '["Paris is in France."]'}
    schema = {'type': 'array', 'items': {'type': 'string'}}
    with patch('src.generation.llm.requests.post', return_value=response) as post:
        llm.generate_json('Return claims', schema=schema)
        body = post.call_args.kwargs['json']
        assert body['format'] == schema and body['think'] is False
        assert llm._key('p', None, schema) != llm._key('p', None, 'json')


def test_rate_limit_is_not_silently_cached_or_fallback(tmp_path):
    import pytest
    from src.generation.llm import ProviderUnavailableError
    from src.generation.claim_extraction import ClaimExtractor
    llm = OllamaLLM(config(tmp_path, 'gemma4:31b-cloud'))
    response = Mock(status_code=429, headers={'Retry-After': '30'})
    with patch('src.generation.llm.requests.post', return_value=response):
        with pytest.raises(ProviderUnavailableError, match='rate limit'):
            ClaimExtractor(llm).extract_with_metadata('Paris is in France.')
    assert llm.cache._conn.execute('select count(*) from cache').fetchone()[0] == 0


def test_rate_limit_retry_honors_shared_delay(tmp_path):
    llm = OllamaLLM(dict(config(tmp_path, 'gemma4:31b-cloud'), max_retries=2))
    limited = Mock(status_code=429, headers={'Retry-After': '12'})
    ok = Mock(status_code=200)
    ok.json.return_value = {'response': '["Paris is in France."]'}
    with patch('src.generation.llm.requests.post', side_effect=[limited, ok]), \
         patch('src.generation.llm.time.monotonic', return_value=100), \
         patch('src.generation.llm.time.sleep') as sleep:
        assert llm.generate_json('Return JSON') == ['Paris is in France.']
        sleep.assert_called_once_with(12.)


def test_adjacent_json_arrays_preserve_all_claims():
    from src.generation.llm import parse_json_loose
    raw = '```json\n["First claim."]\n["Second claim."]\n```'
    assert parse_json_loose(raw) == ['First claim.', 'Second claim.']
