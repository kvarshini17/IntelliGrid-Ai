"""
tests/test_anthropic_provider.py
------------------------------------
Mocked tests for the Claude (Anthropic) provider and its RAG wiring.

NO test here contacts the Anthropic API: every call goes through an injected
fake client (or a fake `anthropic.Anthropic` class). The API key used is the
obviously fake FAKE_KEY. tests/conftest.py clears any real key from the
environment before each test.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from llm import provider as provider_module
from llm.provider import (
    AnthropicProvider,
    LLMProviderError,
    NullProvider,
    get_provider,
    resolve_anthropic_api_key,
)
from rag import rag_pipeline
from rag.rag_pipeline import SYSTEM_PROMPT, answer_with_rag

FAKE_KEY = "FAKE-test-key-0123456789"

CHUNKS = [
    {"title": "HVAC Management", "text": "Reduce HVAC setpoints when a room is unoccupied.", "score": 0.61},
    {"title": "Lighting Control", "text": "Turn lighting off in unoccupied rooms.", "score": 0.42},
]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClient:
    """Stands in for anthropic.Anthropic. Records every request; never touches the network."""

    def __init__(self, text="Fake Claude answer [HVAC Management].", error=None, content=None):
        self.calls = []
        self._text, self._error, self._content = text, error, content
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        content = self._content if self._content is not None else [SimpleNamespace(type="text", text=self._text)]
        return SimpleNamespace(content=content)


class ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


@pytest.fixture
def fixed_chunks(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "retrieve_relevant_chunks", lambda q, top_k=None: list(CHUNKS))


def _use_client(monkeypatch, client):
    """Make the pipeline's get_provider() return an AnthropicProvider backed by `client`."""
    monkeypatch.setattr(rag_pipeline, "get_provider", lambda: AnthropicProvider(client=client))


# ---------------------------------------------------------------------------
# 1. Missing API key -> retrieval-only fallback
# ---------------------------------------------------------------------------
def test_missing_key_auto_selects_null_provider():
    assert resolve_anthropic_api_key() == ""
    assert isinstance(get_provider(), NullProvider)


def test_missing_key_pipeline_returns_retrieved_text_labelled_not_llm(fixed_chunks):
    result = answer_with_rag("Why reduce HVAC?")
    assert result["mode"] == "null"
    assert "No LLM provider is configured" in result["answer"]
    assert "Reduce HVAC setpoints" in result["answer"]
    assert result["sources"] == CHUNKS


def test_explicit_anthropic_without_key_raises_clear_error_and_reports_not_ready():
    provider = AnthropicProvider()
    ready, detail = provider.status()
    assert not ready and "ANTHROPIC_API_KEY" in detail
    with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
        provider.generate("hello")


def test_explicit_anthropic_without_key_falls_back_in_pipeline(monkeypatch, fixed_chunks):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    result = answer_with_rag("Why reduce HVAC?")
    assert result["mode"] == "fallback"
    assert "ANTHROPIC_API_KEY" in result["answer"]
    assert "Reduce HVAC setpoints" in result["answer"]


# ---------------------------------------------------------------------------
# 2. Successful Claude response
# ---------------------------------------------------------------------------
def test_generate_returns_text_from_text_blocks_only():
    client = FakeClient(content=[
        SimpleNamespace(type="thinking", text="private reasoning"),
        SimpleNamespace(type="text", text="Part one. "),
        SimpleNamespace(type="text", text="Part two."),
    ])
    assert AnthropicProvider(client=client).generate("q", "sys") == "Part one. Part two."


def test_pipeline_uses_claude_when_available(monkeypatch, fixed_chunks):
    client = FakeClient("Reduce HVAC when unoccupied [HVAC Management].")
    _use_client(monkeypatch, client)
    result = answer_with_rag("Why reduce HVAC?")
    assert result["mode"] == "anthropic"
    assert result["grounded"] is True
    assert result["answer"] == "Reduce HVAC when unoccupied [HVAC Management]."
    assert len(client.calls) == 1
    assert [s["title"] for s in result["sources"]] == ["HVAC Management", "Lighting Control"]  # citations kept


def test_empty_model_response_is_an_error_not_a_blank_answer():
    with pytest.raises(LLMProviderError, match="empty"):
        AnthropicProvider(client=FakeClient(content=[])).generate("q")


# ---------------------------------------------------------------------------
# 3. Retrieved context + instructions reach Claude
# ---------------------------------------------------------------------------
def test_retrieved_context_question_and_system_results_are_sent_to_claude(monkeypatch, fixed_chunks):
    client = FakeClient()
    _use_client(monkeypatch, client)
    answer_with_rag("Why reduce HVAC?", system_context="Record 7 -- potential wastage: True")
    req = client.calls[0]
    user_msg = req["messages"][0]
    assert user_msg["role"] == "user"
    assert "Reduce HVAC setpoints when a room is unoccupied." in user_msg["content"]
    assert "[Lighting Control]" in user_msg["content"]
    assert "Why reduce HVAC?" in user_msg["content"]
    assert "SYSTEM RESULTS" in user_msg["content"] and "Record 7" in user_msg["content"]
    assert req["system"] == SYSTEM_PROMPT


def test_system_prompt_encodes_grounding_rules():
    low = SYSTEM_PROMPT.lower()
    for phrase in ("retrieved evidence", "do not invent", "cost savings", "assumption", "not proven wastage",
                   "does not contain enough information", "square brackets"):
        assert phrase in low, phrase


def test_model_and_limits_come_from_config_and_environment(monkeypatch):
    client = FakeClient()
    AnthropicProvider(client=client).generate("q")
    assert client.calls[0]["model"] == config.ANTHROPIC_MODEL_NAME
    assert client.calls[0]["max_tokens"] == config.ANTHROPIC_MAX_TOKENS

    monkeypatch.setenv("ANTHROPIC_MODEL", "my-custom-model")
    assert AnthropicProvider(client=client).model == "my-custom-model"
    assert AnthropicProvider(model="explicit", client=client).model == "explicit"


# ---------------------------------------------------------------------------
# 4. Empty retrieval -> Claude is never called
# ---------------------------------------------------------------------------
def test_empty_retrieval_never_calls_claude(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(rag_pipeline, "retrieve_relevant_chunks", lambda q, top_k=None: [])
    _use_client(monkeypatch, client)
    result = answer_with_rag("What is the airspeed of a swallow?")
    assert result["mode"] == "no_evidence" and result["grounded"] is False and result["sources"] == []
    assert "No sufficiently relevant information" in result["answer"]
    assert client.calls == []


def test_blank_question_never_calls_claude(monkeypatch):
    client = FakeClient()
    _use_client(monkeypatch, client)
    assert answer_with_rag("   ")["mode"] == "no_evidence"
    assert client.calls == []


# ---------------------------------------------------------------------------
# 5. API errors and timeouts -> labelled, non-sensitive fallback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("error, expected", [
    (APITimeoutError("Request timed out."), "timed out"),
    (APIConnectionError("Connection error."), "Could not reach"),
    (ApiError("invalid x-api-key", status_code=401), "HTTP 401"),
    (ApiError("model not found", status_code=404), "ANTHROPIC_MODEL"),
    (ApiError("rate limited", status_code=429), "billing"),
    (ApiError("server exploded", status_code=500), "HTTP 500"),
])
def test_api_failures_become_provider_errors(error, expected):
    with pytest.raises(LLMProviderError) as info:
        AnthropicProvider(client=FakeClient(error=error)).generate("q")
    assert expected in str(info.value)


def test_api_failure_falls_back_to_retrieved_text_and_hides_secrets(monkeypatch, fixed_chunks):
    leaky = ApiError(f"401 invalid key {FAKE_KEY}", status_code=401)
    _use_client(monkeypatch, FakeClient(error=leaky))
    result = answer_with_rag("Why reduce HVAC?")
    assert result["mode"] == "fallback"
    assert "Reduce HVAC setpoints" in result["answer"]
    assert "unavailable" in result["answer"]
    assert FAKE_KEY not in result["answer"]


def test_unexpected_provider_exception_is_reduced_to_class_name(monkeypatch, fixed_chunks):
    class Boom(LLMProviderError.__mro__[1]):  # plain Exception subclass, not LLMProviderError
        pass

    class BadProvider:
        name = "bad"

        def generate(self, prompt, system_prompt=""):
            raise Boom(f"secret detail {FAKE_KEY}")

    monkeypatch.setattr(rag_pipeline, "get_provider", lambda: BadProvider())
    result = answer_with_rag("Why reduce HVAC?")
    assert result["mode"] == "fallback" and FAKE_KEY not in result["answer"] and "Boom" in result["answer"]


# ---------------------------------------------------------------------------
# 6. Missing SDK or configuration
# ---------------------------------------------------------------------------
def test_missing_sdk_is_reported_with_install_hint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` raise ImportError
    provider = AnthropicProvider()
    ready, detail = provider.status()
    assert not ready and "pip install anthropic" in detail
    with pytest.raises(LLMProviderError, match="pip install anthropic"):
        provider.generate("q")


def test_missing_sdk_does_not_crash_the_pipeline(monkeypatch, fixed_chunks):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = answer_with_rag("Why reduce HVAC?")
    assert result["mode"] == "fallback"
    assert "pip install anthropic" in result["answer"] and "Reduce HVAC setpoints" in result["answer"]


def test_real_sdk_client_is_built_with_key_and_timeout_without_network(monkeypatch):
    """Exercises the real construction path with a fake `anthropic.Anthropic` -- no request is made."""
    created = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.messages = SimpleNamespace(
                create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")]))

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert AnthropicProvider().generate("q") == "ok"
    assert created["api_key"] == FAKE_KEY
    assert created["timeout"] == config.ANTHROPIC_TIMEOUT_SECONDS
    assert created["max_retries"] == 1


# ---------------------------------------------------------------------------
# 7. API-key secrecy
# ---------------------------------------------------------------------------
def test_no_key_like_literals_in_source_or_config():
    import re
    key_like = re.compile(r"sk-(ant-)?[A-Za-z0-9_\-]{20,}")
    root = Path(__file__).resolve().parent.parent
    for path in [*root.glob("*.py"), *root.glob("*/*.py"), root / ".env.example", root / "README.md"]:
        assert not key_like.search(path.read_text(encoding="utf-8")), f"key-like string in {path}"


def test_key_never_appears_in_repr_or_error_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    provider = AnthropicProvider()
    assert FAKE_KEY not in repr(provider) and FAKE_KEY not in str(provider)
    assert FAKE_KEY not in provider.status()[1]

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = SimpleNamespace(create=self._create)

        @staticmethod
        def _create(**kw):
            raise ApiError(f"bad key {FAKE_KEY}", status_code=401)

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    with pytest.raises(LLMProviderError) as info:
        provider.generate("q")
    assert FAKE_KEY not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__  # raw SDK error not chained


def test_redact_helper():
    assert provider_module._redact(f"x {FAKE_KEY} y", FAKE_KEY) == "x [redacted] y"
    assert provider_module._redact("nothing", None) == "nothing"


def test_prompt_sent_to_claude_does_not_contain_the_key(monkeypatch, fixed_chunks):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = FakeClient()
    _use_client(monkeypatch, client)
    answer_with_rag("Why reduce HVAC?")
    assert FAKE_KEY not in repr(client.calls)


def test_dotenv_files_and_streamlit_secrets_are_git_ignored():
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and ".streamlit/" in ignore
    assert (Path(__file__).resolve().parent.parent / ".env.example").exists()
    example = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY=\n" in example  # placeholder only, no value


# ---------------------------------------------------------------------------
# 8. Provider selection and fallback behaviour
# ---------------------------------------------------------------------------
def test_auto_selects_anthropic_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert isinstance(get_provider(), AnthropicProvider)
    assert isinstance(get_provider("auto"), AnthropicProvider)


def test_explicit_null_wins_over_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setattr(config, "LLM_PROVIDER", "null")
    assert isinstance(get_provider(), NullProvider)


def test_whitespace_only_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert isinstance(get_provider(), NullProvider)


def test_streamlit_secrets_are_supported(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "secrets", {"ANTHROPIC_API_KEY": FAKE_KEY})
    assert resolve_anthropic_api_key() == FAKE_KEY
    assert isinstance(get_provider(), AnthropicProvider)


def test_environment_variable_takes_precedence_over_secrets(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "secrets", {"ANTHROPIC_API_KEY": "from-secrets"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert resolve_anthropic_api_key() == "from-env"


def test_anthropic_provider_is_registered_and_swappable():
    assert isinstance(get_provider("anthropic"), AnthropicProvider)
    provider = AnthropicProvider(client=FakeClient())
    assert provider.name == "anthropic" and callable(provider.generate)
    assert provider.status() == (True, "")


def test_null_provider_status_is_always_ready():
    assert NullProvider().status() == (True, "")
