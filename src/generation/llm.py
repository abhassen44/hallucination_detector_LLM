"""Ollama LLM client with persistent caching and robust JSON parsing.

Caching guarantees that generated answers / extracted claims stay fixed across
experiments (plan §5), and avoids re-hitting the cloud model.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from typing import Any

import requests

from src.utils.config import load_config, resolve_path

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_loose(text: str) -> Any:
    """Parse JSON even if wrapped in ```json fences or surrounded by prose."""
    s = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced [...] or {...} block.
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = s.find(open_c), s.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]!r}")


class LLMCache:
    def __init__(self, db_path: str):
        self.path = resolve_path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, "
            "prompt TEXT, response TEXT, created REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, model: str, prompt: str, response: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (key, model, prompt, response, time.time()),
            )
            self._conn.commit()


class OllamaLLM:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()["llm"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = cfg["temperature"]
        self.timeout = cfg["timeout"]
        self.max_retries = cfg["max_retries"]
        self.backoff = cfg["retry_backoff"]
        self.cache = LLMCache(cfg["cache_db"])

    def _key(self, prompt: str, system: str | None, fmt: str | None) -> str:
        payload = json.dumps(
            [self.model, self.temperature, system, fmt, prompt], ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> str:
        fmt = "json" if json_mode else None
        key = self._key(prompt, system, fmt)
        if use_cache and (hit := self.cache.get(key)) is not None:
            return hit

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            body["system"] = system
        if fmt:
            body["format"] = fmt

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
                r.raise_for_status()
                text = r.json()["response"]
                self.cache.set(key, self.model, prompt, text)
                return text
            except (requests.RequestException, KeyError) as e:
                last_err = e
                time.sleep(self.backoff ** attempt)
        raise RuntimeError(f"Ollama request failed after {self.max_retries} retries: {last_err}")

    def generate_json(self, prompt: str, system: str | None = None, use_cache: bool = True) -> Any:
        return parse_json_loose(self.generate(prompt, system, json_mode=True, use_cache=use_cache))


if __name__ == "__main__":
    llm = OllamaLLM()
    print(f"Model: {llm.model}")
    out = llm.generate_json(
        "Extract atomic factual claims as a JSON list of strings: "
        "'The Eiffel Tower was completed in 1889 and is located in London.'"
    )
    print(out)
