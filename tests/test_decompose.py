"""Unit tests for query-decomposition parsing (Layer 5b-i, D-037).

Pure + offline: the Groq decomposition call returns a JSON object of sub-queries;
this pins how we parse/clean it (cap, dedupe, fall back to the original question)
without an LLM call. The live decompose() path is exercised by the eval gate.
"""

from src.agent.decompose import parse_subqueries


def test_parses_json_queries():
    raw = '{"queries": ["stream tokens from a chat model", "async streaming callback"]}'
    assert parse_subqueries(raw, "orig") == [
        "stream tokens from a chat model",
        "async streaming callback",
    ]


def test_caps_at_three_subqueries():
    raw = '{"queries": ["a", "b", "c", "d", "e"]}'
    assert parse_subqueries(raw, "orig") == ["a", "b", "c"]


def test_dedupes_and_drops_blanks():
    raw = '{"queries": ["a", "a", "  ", "b"]}'
    assert parse_subqueries(raw, "orig") == ["a", "b"]


def test_falls_back_to_original_on_empty():
    assert parse_subqueries('{"queries": []}', "orig") == ["orig"]


def test_falls_back_to_original_on_bad_json():
    assert parse_subqueries("not json at all", "orig") == ["orig"]


def test_falls_back_when_queries_key_missing():
    assert parse_subqueries('{"foo": ["a"]}', "orig") == ["orig"]
