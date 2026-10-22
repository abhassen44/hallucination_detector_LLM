"""Verify frozen text artifacts across Git LF/CRLF checkouts without changing them."""
import hashlib
from pathlib import Path


def matches_text_sha256(path: Path, expected: str) -> bool:
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    return any(hashlib.sha256(data).hexdigest() == expected
               for data in (raw, lf, lf.replace(b'\n', b'\r\n')))
