"""Integrity checks on the committed corpus (Layer 1a gate as a regression guard).

Runs against data/corpus/chunks.jsonl if present (it is committed as the durable
source of truth, D-004). Skips cleanly if the corpus hasn't been built yet.
"""

import json

import pytest

from src.config import PROJECT_ROOT, get_settings

REQUIRED = ("id", "text", "source_url", "heading_path", "version", "type")


def _load_chunks():
    path = (PROJECT_ROOT / get_settings().corpus_jsonl).resolve()
    if not path.exists():
        pytest.skip(f"corpus not built yet: {path}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_corpus_schema_and_integrity():
    chunks = _load_chunks()
    assert len(chunks) > 1000, "expected a substantial corpus"

    ids = set()
    for c in chunks:
        for key in REQUIRED:
            assert key in c, f"chunk {c.get('id')} missing field {key}"
        # heading_path may be empty (page-intro chunks before any heading); others must not be.
        for key in ("id", "text", "source_url", "version", "type"):
            assert c[key], f"chunk {c.get('id')} has empty {key}"
        assert c["id"] not in ids, f"duplicate id {c['id']}"
        ids.add(c["id"])
        assert c["source_url"].startswith("https://docs.langchain.com/oss/")
        # Guard against the base64/unclosed-fence blow-up (a whole file as one chunk).
        assert c["n_chars"] < 200_000, f"suspiciously large chunk in {c['source_path']}"
