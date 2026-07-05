"""Data model for a corpus chunk (Layer 1).

Why a typed model: every later layer (indexing, retrieval, eval) depends on a
stable chunk schema; the plan's Layer 1 gate requires each chunk to carry
{id, text, source_url, heading_path, version, type}.
"""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    """One retrievable unit of documentation with provenance for citations."""

    id: str                      # deterministic hash of (source_path, heading_path, index)
    text: str                    # cleaned chunk text (code blocks kept intact)
    source_url: str              # public docs URL, used for per-claim citations
    heading_path: str            # breadcrumb, e.g. "Streaming > Stream tokens"
    version: str                 # corpus snapshot id (short git SHA of the docs repo)
    type: str = "doc"            # source type; "discussion"/"issue" arrive in Layer 1b

    # --- Extra provenance (useful downstream; not required by the gate) ---
    title: str = ""              # page title from frontmatter
    section: str = ""            # oss subtree: python/langchain/langgraph/...
    source_path: str = ""        # repo-relative path, e.g. src/oss/python/...
    n_chars: int = 0             # length of text (cheap size signal)
    has_code: bool = False       # whether the chunk contains a fenced code block
