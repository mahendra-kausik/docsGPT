"""Single typed configuration surface for DocsGPT-Agent.

Why: CLAUDE.md §6 mandates config-over-constants so every ablation is a one-line
change and every tunable is defensible in an interview. Secrets come from the
environment (.env, git-ignored); tunables come from config.yaml. Both are exposed
as one cached, typed Settings object the whole codebase imports.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML = PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read tunables from config.yaml; a missing/empty file yields no overrides."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    """All runtime config: secrets from .env, tunables from config.yaml.

    Why one object: keeps model ids and retrieval knobs in a single place so
    ablations and audits are trivial. config.yaml values are layered on top of
    the environment (see get_settings), making config.yaml the authority for
    tunables and .env the authority for secrets.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # permit field names such as embedding_model
    )

    # --- Secrets / keys (from .env; blank until the user provides them) ---
    groq_api_key: str = ""
    gemini_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    github_token: str = ""

    # --- Tunables (defaults mirror config.yaml; config.yaml overrides at load) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    synthesis_model: str = "gemini-flash"
    cheap_model: str = "groq/llama-3.1-8b-instant"
    rrf_k: int = 60
    retrieve_top_k: int = 50
    rerank_top_n: int = 6
    chunk_size: int = 1000
    chunk_overlap: int = 150
    corpus_jsonl: str = "data/corpus/chunks.jsonl"
    results_dir: str = "results"


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings with config.yaml tunables layered over the env.

    Passing the yaml values as init kwargs gives them highest precedence in
    pydantic-settings, so config.yaml wins for tunables while secrets still load
    from the environment.
    """
    return Settings(**_load_yaml(CONFIG_YAML))
