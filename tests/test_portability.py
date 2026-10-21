"""Guard against platform-dependent text encoding (Windows defaults to cp1252)."""
import re
from pathlib import Path

from src.utils.config import PROJECT_ROOT

READ_OR_WRITE_WITHOUT_ENCODING = re.compile(r"\.(?:read|write)_text\(\)")
TEXT_APPEND_WITHOUT_ENCODING = re.compile(r"\.open\((['\"])[wa]\+?\1\)")
SOURCES = ('src', 'scripts', 'tests')


def _python_files():
    for folder in SOURCES:
        yield from (PROJECT_ROOT / folder).rglob('*.py')


def test_every_text_io_call_declares_encoding():
    offenders = []
    for path in _python_files():
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if READ_OR_WRITE_WITHOUT_ENCODING.search(line) or TEXT_APPEND_WITHOUT_ENCODING.search(line):
                offenders.append(f'{path.relative_to(PROJECT_ROOT)}:{number}')
    assert not offenders, f'Missing utf-8 encoding: {offenders}'


def test_non_ascii_jsonl_round_trip(tmp_path):
    from src.datasets.schema import read_jsonl, write_jsonl

    path = tmp_path / 'unicode.jsonl'
    write_jsonl(path, [{'id': 'x', 'text': 'Café — naïve façade’s 北京'}])
    assert list(read_jsonl(path))[0]['text'] == 'Café — naïve façade’s 北京'
