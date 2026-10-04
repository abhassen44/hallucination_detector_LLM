"""Resumable HTTP download with progress bar."""
from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm


def download(url: str, dest: str | Path, chunk: int = 1 << 20) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    total = int(requests.head(url, allow_redirects=True, timeout=30).headers.get("Content-Length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        return dest

    headers = {"Range": f"bytes={have}-"} if have and total else {}
    mode = "ab" if headers else "wb"
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        if headers and r.status_code != 206:      # server ignored Range -> restart
            mode, have = "wb", 0
        with open(dest, mode) as f, tqdm(
            total=total or None, initial=have, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))
    return dest
