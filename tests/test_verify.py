"""Unit tests for grounding-verdict parsing (Layer 5b, D-038).

Pure + offline: the Groq 8B verifier returns {"grounded": true/false}; this pins the
parse. It FAILS OPEN (a malformed verdict trusts the answer) so a verifier glitch never
wrongly refuses a good answer — the verifier is a best-effort faithfulness net, not a
hard gate. The live verify path (which actually catches "Paris") is the layer's gate.
"""

from src.agent.verify import parse_grounded


def test_true_verdict():
    assert parse_grounded('{"grounded": true}') is True


def test_false_verdict():
    assert parse_grounded('{"grounded": false}') is False


def test_string_false_is_false():
    assert parse_grounded('{"grounded": "false"}') is False


def test_bad_json_fails_open_to_grounded():
    assert parse_grounded("not json") is True


def test_missing_key_fails_open_to_grounded():
    assert parse_grounded('{"foo": 1}') is True
