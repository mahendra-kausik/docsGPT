"""Shared test fixtures.

Keep the suite hermetic: force Langfuse tracing OFF for every test (Layer 7, D-045),
so no test ships a span to the network even when the developer's .env carries real
LANGFUSE_* keys. Tests that exercise the *enabled* tracing path opt back in by patching
`src.obs.langfuse._client` with a fake client themselves (a later patch wins).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch):
    import src.obs.langfuse as lf

    monkeypatch.setattr(lf, "_client", lambda: None)
