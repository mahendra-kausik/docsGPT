"""Unit tests for Layer 2 retrieval helpers (embedding hygiene + point ids).

Kept offline/pure so `./tasks.ps1 test` needs neither torch nor a live cluster;
the deployed dense-search path is exercised by the Layer 2 gate (search command).
"""

import uuid

from src.retrieval.embedder import prepare_text, scrub_binary
from src.retrieval.index import point_id


def test_scrub_removes_data_uri():
    text = "before ![x](data:image/png;base64,AAAABBBBCCCCDDDDEEEEFFFF==) after"
    out = scrub_binary(text)
    assert "base64" not in out
    assert "[binary omitted]" in out
    assert out.startswith("before ") and out.endswith(" after")


def test_scrub_removes_long_base64_run():
    blob = "QUJD" * 400  # 1600-char base64-looking run
    out = scrub_binary(f"code = '{blob}'")
    assert blob not in out
    assert "[binary omitted]" in out


def test_scrub_keeps_ordinary_code_and_prose():
    text = "def add(a, b):\n    return a + b  # short tokens stay"
    assert scrub_binary(text) == text


def test_prepare_text_caps_length():
    # spaces keep this out of the base64-run scrubber; only the length cap applies
    assert len(prepare_text("word " * 2000, max_chars=100)) == 100


def test_point_id_is_deterministic_uuid():
    a1 = point_id("e579080d8c81ef75")
    a2 = point_id("e579080d8c81ef75")
    b = point_id("f69276bfbac96f67")
    assert a1 == a2  # stable across calls -> idempotent upserts
    assert a1 != b
    uuid.UUID(a1)  # raises if not a valid UUID string
