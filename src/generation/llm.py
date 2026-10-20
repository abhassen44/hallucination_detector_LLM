"""Ollama LLM client with persistent caching and robust JSON parsing.

Caching guarantees that generated answers / extracted claims stay fixed across
experiments (plan §5), and avoids re-hitting the cloud model.
"""
from __future__ import annotations

import os
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
    # Some models emit adjacent JSON arrays instead of one array. Recover all
    # items only if the entire remaining response consists of valid arrays.
    if s.startswith('['):
        decoder = json.JSONDecoder()
        arrays, pos = [], 0
        try:
            while pos < len(s):
                while pos < len(s) and s[pos].isspace():
                    pos += 1
                if pos == len(s):
                    break
                value, pos = decoder.raw_decode(s, pos)
                if not isinstance(value, list):
                    raise ValueError('Expected adjacent arrays')
                arrays.append(value)
            if len(arrays) > 1:
                return [item for array in arrays for item in array]
        except ValueError:
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


class ProviderUnavailableError(RuntimeError):
    """Authentication/rate limits must not become ordinary extraction fallback."""


class OllamaLLM:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()["llm"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", cfg["model"])
        self.temperature = cfg["temperature"]
        self.timeout = cfg["timeout"]
        self.max_retries = cfg["max_retries"]
        self.backoff = cfg["retry_backoff"]
        self.cache = LLMCache(cfg["cache_db"])
        self._retry_lock = threading.Lock()
        self._retry_at = 0.0

    def _key(self, prompt: str, system: str | None, fmt: str | dict | None) -> str:
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
        json_schema: dict | None = None,
    ) -> str:
        fmt = json_schema if json_schema is not None else ("json" if json_mode else None)
        key = self._key(prompt, system, fmt)
        if use_cache and (hit := self.cache.get(key)) is not None:
            return hit

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.model.startswith("qwen3"):
            body["think"] = False
        if system:
            body["system"] = system
        if fmt:
            body["format"] = fmt

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            with self._retry_lock:
                pause = max(0., self._retry_at - time.monotonic())
            if pause:
                time.sleep(pause)
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
                if r.status_code in (401, 403):
                    raise ProviderUnavailableError(f"Ollama authentication failed (HTTP {r.status_code})")
                if r.status_code == 429:
                    try:
                        delay = max(10., float(r.headers.get('Retry-After', 10 * 2 ** attempt)))
                    except (TypeError, ValueError):
                        delay = 10 * 2 ** attempt
                    with self._retry_lock:
                        self._retry_at = max(self._retry_at, time.monotonic() + delay)
                    if attempt == self.max_retries - 1:
                        raise ProviderUnavailableError('Ollama rate limit persists; resume later using the cache.')
                    continue
                r.raise_for_status()
                text = r.json()["response"]
                self.cache.set(key, self.model, prompt, text)
                return text
            except (requests.RequestException, KeyError) as e:
                last_err = e
                time.sleep(self.backoff ** attempt)
        raise RuntimeError(f"Ollama request failed after {self.max_retries} retries: {last_err}")

    def generate_json(self, prompt: str, system: str | None = None, use_cache: bool = True,
                      schema: dict | None = None) -> Any:
        # Ollama cloud does not support structured output formats. The extraction
        # prompt still requests JSON; parse and validate the returned text locally.
        cloud = self.model.endswith('-cloud') or ':cloud' in self.model
        return parse_json_loose(self.generate(prompt, system, json_mode=not cloud,
                                             use_cache=use_cache,
                                             json_schema=None if cloud else schema))


if __name__ == "__main__":
    llm = OllamaLLM()
    print(f"Model: {llm.model}")
    out = llm.generate_json(
        "Extract atomic factual claims as a JSON list of strings: "
        "'The Eiffel Tower was completed in 1889 and is located in London.'"
    )
    print(out)
