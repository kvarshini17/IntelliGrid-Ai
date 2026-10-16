"""
tests/conftest.py
---------------------
Keeps the whole suite hermetic: no test may pick up a developer's real API key
or make a real (billable) API request. Tests that need a key set a fake one
explicitly via monkeypatch, and every Anthropic call in the tests goes through
an injected fake client.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm_credentials(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    # config.LLM_PROVIDER was read from the environment at import time; reset to the default.
    import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
