"""Single LLM call path: Groq + Gemini routing + backoff/jitter + on-disk cache.

CLAUDE.md §4 and D-008 require every LLM call to go through one wrapper with
exponential backoff + jitter on 429s and a response cache — never unbounded
parallel calls against a free-tier quota. Model ids are provider-prefixed
(`groq/…`, `gemini/…`); the prefix selects the backend so the agent routes cheap
high-volume nodes to Groq 8B and final synthesis to Gemini Flash (D-008/D-033)
through one `complete()` surface.

The cache is keyed by the full request (model + messages + params); re-running
generation or eval never re-spends quota and keeps those runs reproducible. The
stripped model name is the cache key, so Groq and Gemini never collide and the
existing Groq cache stays valid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.llm.metrics import record_cache_hit, record_call

logger = logging.getLogger(__name__)


def _split_provider(model: str) -> tuple[str, str]:
    """`gemini/gemini-2.5-flash` -> ('gemini', 'gemini-2.5-flash'); bare -> ('groq', model)."""
    if "/" in model:
        provider, name = model.split("/", 1)
        return provider, name
    return "groq", model  # backwards-compatible default (config keeps the prefix)


class LLMGateway:
    """Cached, backed-off chat completions over Groq or Gemini (provider from model id)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        cache_dir: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.s = get_settings()
        self.provider, self.model = _split_provider(model or self.s.cheap_model)
        if self.provider not in ("groq", "gemini"):
            raise ValueError(f"Unknown LLM provider {self.provider!r} (use groq/ or gemini/).")
        self.temperature = temperature
        self.max_retries = self.s.llm_max_retries if max_retries is None else max_retries
        cache = Path(cache_dir or self.s.llm_cache_dir)
        self.cache_dir = cache if cache.is_absolute() else PROJECT_ROOT / cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazily created so importing this module needs no key

    @property
    def client(self):
        """Lazily construct the provider client (fails clearly if its key is missing)."""
        if self._client is None:
            if self.provider == "gemini":
                from google import genai

                if not self.s.gemini_api_key:
                    raise RuntimeError("GEMINI_API_KEY is empty — add it to .env (AI Studio).")
                self._client = genai.Client(api_key=self.s.gemini_api_key)
            else:
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
            # Served without spending quota — counted as a cache hit, not an LLM call (D-043).
            record_cache_hit()
            return json.loads(cache_path.read_text(encoding="utf-8"))["content"]

        content, usage = self._call_with_backoff(payload)
        # Record real-call accounting for the active request scope, if any (Layer 6).
        record_call(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        cache_path.write_text(
            json.dumps({"payload": payload, "content": content}), encoding="utf-8"
        )
        return content

    def _call_with_backoff(self, payload: dict) -> tuple[str, dict]:
        """Call the provider, retrying transient/429 errors with backoff + full jitter.

        Returns ``(content, usage)`` where usage carries prompt/completion token counts
        (best-effort — providers may omit them), so the caller can record it (D-043).
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(payload)
            except Exception as exc:  # noqa: BLE001 — narrowed by _is_retryable below
                if attempt == self.max_retries or not self._is_retryable(exc):
                    raise
                # exponential base (2^attempt) capped, plus full jitter
                delay = min(2.0**attempt, 30.0) + random.uniform(0, 1.0)
                logger.warning(
                    "%s %s (attempt %d/%d) — backing off %.1fs",
                    self.provider, type(exc).__name__, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # loop either returns or raises

    def _call_once(self, payload: dict) -> tuple[str, dict]:
        """One provider call returning ``(assistant_text, usage)``."""
        return self._gemini_call(payload) if self.provider == "gemini" else self._groq_call(payload)

    def _groq_call(self, payload: dict) -> tuple[str, dict]:
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
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
        }
        return content, usage

    def _gemini_call(self, payload: dict) -> tuple[str, dict]:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=payload["system"],
            temperature=payload["temperature"],
            max_output_tokens=payload["max_tokens"],
            seed=payload["seed"],
            # Disable "thinking" for synthesis: deterministic, cheaper, and avoids empty
            # replies when thinking tokens would consume the output budget (D-033).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if payload["response_json"] else None,
        )
        resp = self.client.models.generate_content(
            model=payload["model"], contents=payload["user"], config=cfg
        )
        u = getattr(resp, "usage_metadata", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_token_count", None),
            "completion_tokens": getattr(u, "candidates_token_count", None),
        }
        return resp.text or "", usage

    def _is_retryable(self, exc: Exception) -> bool:
        """Provider-specific: retry 5xx + 429/connection/timeout, never 4xx bad requests."""
        if self.provider == "gemini":
            from google.genai import errors

            if isinstance(exc, errors.ServerError):
                return True
            if isinstance(exc, errors.ClientError):
                return getattr(exc, "code", None) == 429
            return False
        from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        return isinstance(exc, retryable)
