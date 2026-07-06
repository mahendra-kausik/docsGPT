"""Single LLM call path: Groq routing + backoff/jitter + on-disk response cache.

CLAUDE.md §4 and D-008 require every LLM call to go through one wrapper with
exponential backoff + jitter on 429s and a response cache — never unbounded
parallel calls against a free-tier quota. This is the lean version built for
Layer 3 synthetic-gold generation; the Layer 5/6 agent and the RAGAS judge reuse
it (adding Gemini routing for synthesis on top).

The cache is keyed by the full request (model + messages + params), so re-running
generation or eval never re-spends quota and makes those runs reproducible against
a fixed set of completions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)


def _strip_provider(model: str) -> str:
    """`groq/llama-3.1-8b-instant` -> `llama-3.1-8b-instant` (config keeps the prefix)."""
    return model.split("/", 1)[1] if "/" in model else model


class LLMGateway:
    """Cached, backed-off Groq chat completions (config-driven model + retries)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        cache_dir: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.s = get_settings()
        self.model = _strip_provider(model or self.s.cheap_model)
        self.temperature = temperature
        self.max_retries = self.s.llm_max_retries if max_retries is None else max_retries
        cache = Path(cache_dir or self.s.llm_cache_dir)
        self.cache_dir = cache if cache.is_absolute() else PROJECT_ROOT / cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazily created so importing this module needs no key

    @property
    def client(self):
        """Lazily construct the Groq client (fails clearly if the key is missing)."""
        if self._client is None:
            from groq import Groq

            if not self.s.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is empty — add it to .env (console.groq.com).")
            self._client = Groq(api_key=self.s.groq_api_key, timeout=self.s.llm_timeout_s)
        return self._client

    def _cache_path(self, payload: dict) -> Path:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 1024,
        response_json: bool = False,
    ) -> str:
        """Return the assistant text for a system+user prompt, cached and retried.

        ``response_json=True`` asks Groq for a JSON object (the caller still parses).
        Identical requests hit the on-disk cache and never re-call the API.
        """
        temp = self.temperature if temperature is None else temperature
        payload = {
            "model": self.model,
            "system": system,
            "user": user,
            "temperature": temp,
            "max_tokens": max_tokens,
            "response_json": response_json,
            "seed": self.s.llm_seed,
        }
        cache_path = self._cache_path(payload)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["content"]

        content = self._call_with_backoff(payload)
        cache_path.write_text(
            json.dumps({"payload": payload, "content": content}), encoding="utf-8"
        )
        return content

    def _call_with_backoff(self, payload: dict) -> str:
        """Call Groq, retrying transient/429 errors with exponential backoff + jitter."""
        from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        kwargs = {
            "model": payload["model"],
            "messages": [
                {"role": "system", "content": payload["system"]},
                {"role": "user", "content": payload["user"]},
            ],
            "temperature": payload["temperature"],
            "max_tokens": payload["max_tokens"],
            "seed": payload["seed"],
        }
        if payload["response_json"]:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except retryable as exc:
                if attempt == self.max_retries:
                    raise
                # exponential base (2^attempt) capped, plus full jitter
                delay = min(2.0**attempt, 30.0) + random.uniform(0, 1.0)
                logger.warning(
                    "Groq %s (attempt %d/%d) — backing off %.1fs",
                    type(exc).__name__, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # loop either returns or raises
