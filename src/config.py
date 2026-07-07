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
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML = PROJECT_ROOT / "config.yaml"


class ForumConfig(BaseModel):
    """LangChain Forum (Discourse) ingestion -> GOLD-EVAL SEEDS ONLY (Layer 1b).

    GitHub Discussions migrated to forum.langchain.com in mid-2025 (D-018).
    Solved topics = natural relevance labels (D-003). Their questions become eval
    items whose gold targets are MIT docs chunks (mapped at Layer 3); forum answer
    text is NOT added to the retrieval corpus (D-019).
    """

    base_url: str = "https://forum.langchain.com"
    categories: list[str] = ["OSS Product Help"]  # by display name; solved topics only
    max_pages_per_category: int = 20
    request_delay_s: float = 0.2                   # be polite to the public forum
    raw_dir: str = "data/raw/forum"                # cached topic JSON (git-ignored)
    # committed gold seeds: question + metadata, no answer body (D-019)
    seeds_jsonl: str = "data/gold/forum_seeds.jsonl"


class CorpusConfig(BaseModel):
    """Where the docs corpus comes from and which slice to ingest (Layer 1a).

    Why nested: keeps the several corpus-sourcing knobs grouped and readable in
    config.yaml while staying part of the single Settings surface.
    """

    repo: str = "https://github.com/langchain-ai/docs.git"
    ref: str = "main"
    raw_dir: str = "data/raw/langchain-docs"
    snippets_dir: str = "src/snippets"
    docs_base_url: str = "https://docs.langchain.com"
    include_globs: list[str] = [
        "src/oss/python/**",
        "src/oss/langchain/**",
        "src/oss/langgraph/**",
        "src/oss/deepagents/**",
        "src/oss/concepts/**",
        "src/oss/integrations/**",
        "src/oss/*.mdx",
    ]


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
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # MiniLM for CPU latency (D-030)
    synthesis_model: str = "gemini-flash"
    cheap_model: str = "groq/llama-3.1-8b-instant"
    sparse_model: str = "Qdrant/bm25"  # fastembed BM25 for hybrid (Layer 4a, D-026)
    rrf_k: int = 60                     # client-side RRF constant, tunable (D-027)
    retrieve_top_k: int = 50
    rerank_top_n: int = 6
    rerank_max_chars: int = 1200        # cap passage length for the cross-encoder (D-032)
    # --- Indexing (Layer 2: dense baseline, D-021/D-022) ---
    qdrant_collection: str = "docs_dense"
    qdrant_hybrid_collection: str = "docs_hybrid"  # dense+sparse named vectors (D-026)
    vector_distance: str = "Cosine"
    embed_batch_size: int = 64
    upsert_batch_size: int = 256
    embed_max_chars: int = 8000
    query_instruction: str = "Represent this sentence for searching relevant passages:"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_chars: int = 24
    corpus_jsonl: str = "data/corpus/chunks.jsonl"
    results_dir: str = "results"
    # --- LLM gateway (Layer 3+, D-008): single wrapper, backoff + on-disk cache ---
    llm_cache_dir: str = ".cache/llm"
    llm_max_retries: int = 6
    llm_timeout_s: float = 60.0
    llm_seed: int = 13  # Groq seed for as-deterministic-as-possible generation
    # --- Eval / gold set (Layer 3, D-024) ---
    eval_candidate_pool: int = 20              # dense candidates shown per Q during gold review
    eval_snippet_chars: int = 240              # candidate-chunk preview length in review.md
    gold_candidates_jsonl: str = "data/gold/candidates.jsonl"  # machine rank->chunk_id map
    gold_review_md: str = "data/gold/review.md"                # human-edited decision file
    gold_forum_jsonl: str = "data/gold/gold_forum.jsonl"       # compiled forum (real) slice
    gold_jsonl: str = "data/gold/gold.jsonl"                   # final merged gold set (committed)
    # --- Synthetic gold generation (Layer 3, D-025): Groq-8B questions from docs chunks ---
    gold_synth_jsonl: str = "data/gold/gold_synth.jsonl"       # generated gold (source=synthetic)
    synth_min_chunk_chars: int = 300      # skip thin chunks; generate from substantive ones
    synth_multi_group_size: int = 3       # chunks combined for a multi-hop question
    synth_seed: int = 7                   # seeds chunk sampling for reproducible selection
    corpus: CorpusConfig = CorpusConfig()
    forum: ForumConfig = ForumConfig()


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings with config.yaml tunables layered over the env.

    Passing the yaml values as init kwargs gives them highest precedence in
    pydantic-settings, so config.yaml wins for tunables while secrets still load
    from the environment.
    """
    return Settings(**_load_yaml(CONFIG_YAML))
