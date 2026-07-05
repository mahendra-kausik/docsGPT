"""Tests for LangChain Forum (Discourse) parsing (Layer 1b), no network required."""

from src.ingest.forum import html_to_text, parse_topic

# Minimal Discourse topic JSON with an accepted answer (post_number 3).
TOPIC = {
    "id": 3877,
    "slug": "proxy-authentication-required-407",
    "title": "Proxy Authentication Required 407",
    "tags": ["chatopenai", "proxy"],
    "created_at": "2025-09-01T00:00:00Z",
    "has_accepted_answer": True,
    "post_stream": {
        "posts": [
            {
                "post_number": 1,
                "accepted_answer": False,
                "cooked": "<p>Please help with <code>ChatOpenAI</code>.</p><p>I get error 407.</p>",
            },
            {
                "post_number": 2,
                "accepted_answer": False,
                "cooked": "<p>Have you tried a proxy env var?</p>",
            },
            {
                "post_number": 3,
                "accepted_answer": True,
                "cooked": (
                    "<p>This fixed it:</p>"
                    "<pre><code>import truststore\ntruststore.inject_into_ssl()</code></pre>"
                ),
            },
        ]
    },
}


def test_html_to_text_preserves_code():
    out = html_to_text("<p>Do this:</p><pre><code>x = 1\ny = 2</code></pre>")
    assert "Do this:" in out
    assert "```" in out
    assert "x = 1" in out and "y = 2" in out


def test_parse_topic_extracts_question_and_accepted_answer():
    seed = parse_topic(TOPIC, "OSS Product Help", "https://forum.langchain.com")
    assert seed is not None
    assert seed["id"] == 3877
    assert seed["url"] == "https://forum.langchain.com/t/proxy-authentication-required-407/3877"
    assert seed["category"] == "OSS Product Help"
    assert seed["tags"] == ["chatopenai", "proxy"]
    assert "ChatOpenAI" in seed["question"]
    assert seed["accepted_answer_post"] == 3
    assert seed["accepted_answer_url"].endswith("/3877/3")
    assert "truststore" in seed["accepted_answer"]  # answer kept for local mapping only


def test_parse_topic_without_accepted_answer_returns_none():
    no_answer = {
        "id": 1, "slug": "x", "title": "X", "tags": [], "created_at": "",
        "post_stream": {
            "posts": [{"post_number": 1, "accepted_answer": False, "cooked": "<p>hi</p>"}]
        },
    }
    assert parse_topic(no_answer, "OSS Product Help", "https://forum.langchain.com") is None
