"""Layer 0 smoke test: the package wires up and config loads (Acceptance Gate)."""

import importlib


def test_all_subpackages_import():
    """Every src subpackage imports cleanly — verifies the module skeleton."""
    for mod in ("src", "src.ingest", "src.retrieval", "src.agent", "src.eval", "src.api"):
        importlib.import_module(mod)


def test_config_loads_with_expected_tunables():
    """get_settings() loads and exposes known tunables read from config.yaml."""
    from src.config import get_settings

    settings = get_settings()
    assert settings.rrf_k == 60
    assert settings.retrieve_top_k == 50
    assert settings.rerank_top_n == 6
    assert settings.embedding_model.startswith("BAAI/")
