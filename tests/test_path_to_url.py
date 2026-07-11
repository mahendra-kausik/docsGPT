"""Unit test for path_to_url (D-051): language-neutral repo paths must get the
/python/ segment the live docs site requires (verified live: /oss/langchain/X
404s, /oss/python/langchain/X 200s); already-prefixed paths pass through."""
from src.ingest.corpus import path_to_url

BASE = "https://docs.langchain.com"


def test_inserts_python_segment_for_language_neutral_paths():
    assert path_to_url("src/oss/langchain/tools.mdx", BASE) == f"{BASE}/oss/python/langchain/tools"
    assert (
        path_to_url("src/oss/langgraph/streaming/index.mdx", BASE)
        == f"{BASE}/oss/python/langgraph/streaming"
    )
    assert path_to_url("src/oss/common-errors.mdx", BASE) == f"{BASE}/oss/python/common-errors"


def test_leaves_already_prefixed_python_paths_unchanged():
    assert (
        path_to_url("src/oss/python/langchain/agents.mdx", BASE)
        == f"{BASE}/oss/python/langchain/agents"
    )
