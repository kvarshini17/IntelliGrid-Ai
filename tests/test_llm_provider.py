"""
tests/test_llm_provider.py
------------------------------
Tests for llm/provider.py (Phase 8).
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.provider import (
    LLMProviderError,
    NullProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)


# ---------------------------------------------------------------------------
# NullProvider -- the default, no-key-required path
# ---------------------------------------------------------------------------
def test_null_provider_requires_no_configuration():
    provider = NullProvider()
    result = provider.generate(
        prompt="RETRIEVED KNOWLEDGE:\nSome retrieved text here.\n\nUSER QUESTION:\nWhy?",
        system_prompt="irrelevant",
    )
    assert "Some retrieved text here." in result
    assert "No LLM provider is configured" in result


def test_null_provider_extracts_only_knowledge_section():
    provider = NullProvider()
    result = provider.generate(
        prompt="SYSTEM RESULTS:\nSecret internal stuff\n\nRETRIEVED KNOWLEDGE:\nPublic knowledge text\n\nUSER QUESTION:\nQ?",
    )
    assert "Public knowledge text" in result
    # Should not include the trailing question text
    assert "Q?" not in result


def test_null_provider_handles_prompt_without_markers_gracefully():
    provider = NullProvider()
    result = provider.generate(prompt="just some plain text")
    assert "just some plain text" in result


def test_null_provider_name():
    assert NullProvider().name == "null"


# ---------------------------------------------------------------------------
# get_provider() factory
# ---------------------------------------------------------------------------
def test_get_provider_defaults_to_null(monkeypatch):
    monkeypatch.setattr("config.LLM_PROVIDER", "null")
    provider = get_provider()
    assert isinstance(provider, NullProvider)


def test_get_provider_unrecognized_name_falls_back_to_null():
    provider = get_provider("not_a_real_provider")
    assert isinstance(provider, NullProvider)


def test_get_provider_explicit_override():
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaProvider)


def test_get_provider_openai():
    provider = get_provider("openai")
    assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# OpenAIProvider -- must never hardcode or require a key
# ---------------------------------------------------------------------------
def test_openai_provider_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        provider.generate(prompt="test")


def test_openai_provider_never_hardcodes_a_key():
    import inspect
    from llm import provider as provider_module
    source = inspect.getsource(provider_module)
    # No literal-looking API key patterns (sk-...) anywhere in source.
    assert "sk-" not in source


# ---------------------------------------------------------------------------
# OllamaProvider -- must fail gracefully when unreachable, never crash the app
# ---------------------------------------------------------------------------
def test_ollama_provider_raises_clear_error_when_unreachable():
    provider = OllamaProvider(host="http://localhost:1")  # nothing listens here
    with pytest.raises(LLMProviderError):
        provider.generate(prompt="test")


def test_ollama_provider_uses_env_vars_not_hardcoded_values(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://custom-host:1234")
    provider = OllamaProvider()
    assert provider.model == "custom-model"
    assert provider.host == "http://custom-host:1234"


# ---------------------------------------------------------------------------
# Provider interface uniformity (any provider is swappable)
# ---------------------------------------------------------------------------
def test_all_providers_implement_the_same_interface():
    for provider in (NullProvider(), OllamaProvider(), OpenAIProvider()):
        assert hasattr(provider, "generate")
        assert hasattr(provider, "name")
