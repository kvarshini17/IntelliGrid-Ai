"""
llm/provider.py
------------------
Purpose : LLM provider abstraction (Phase 8). Lets the project switch
          between "no LLM" (default, deterministic, no key required),
          a local Ollama model, or a hosted API, without any calling code
          depending on which one is active.

Design:
    - `LLMProvider` is the interface: `generate(prompt, system_prompt) -> str`.
    - `get_provider()` is the single factory every caller should use; it
      reads `config.LLM_PROVIDER` ("null" | "ollama" | "openai") and never
      requires an API key for the default ("null") path.
    - The LLM is used ONLY for explanation/natural-language framing (Phase
      7/9). It never performs condition classification, wastage detection,
      anomaly detection, or optimization -- those remain the deterministic
      modules' responsibility and are passed to the provider as authoritative
      "SYSTEM RESULTS" context it must not override (see rag/rag_pipeline.py).

SECURITY: no API key is ever hard-coded here. OpenAIProvider reads
OPENAI_API_KEY from the environment only; AnthropicProvider reads
ANTHROPIC_API_KEY from the environment (or Streamlit secrets). If a key is
absent, a clear error is raised (caught by the caller, e.g. rag_pipeline's
fallback) rather than silently using a placeholder. Keys are never included
in error messages, logs or UI text (see `_redact`).
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import config


class LLMProviderError(Exception):
    """Raised when a provider is misconfigured or unreachable."""


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response given a (already-grounded) prompt."""
        raise NotImplementedError

    def status(self) -> Tuple[bool, str]:
        """
        Cheap, offline readiness check (never makes a network request).
        Returns (ready, human-readable detail). The UI uses this to say, for
        example, that the API key is missing, without spending an API call.
        """
        return True, ""


class NullProvider(LLMProvider):
    """
    Default provider. Requires no API key, no network access, and no extra
    dependency. Does not perform natural-language generation -- it
    deterministically extracts and returns the "RETRIEVED KNOWLEDGE" section
    of the prompt built by rag.rag_pipeline, prefixed with a clear notice
    that this is unprocessed retrieved text, not an LLM-generated answer.
    This keeps the system fully functional (per the project requirement
    that no LLM/API key is required for the system to work) while never
    pretending to be something it is not.
    """

    name = "null"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        knowledge_section = prompt
        if "RETRIEVED KNOWLEDGE:" in prompt:
            knowledge_section = prompt.split("RETRIEVED KNOWLEDGE:", 1)[1]
        if "USER QUESTION:" in knowledge_section:
            knowledge_section = knowledge_section.split("USER QUESTION:", 1)[0]
        knowledge_section = knowledge_section.strip()

        return (
            "No LLM provider is configured, so here is the most relevant information "
            "retrieved directly from the knowledge base (not LLM-generated text):\n\n"
            f"{knowledge_section}"
        )


class OllamaProvider(LLMProvider):
    """
    Calls a local Ollama server. Requires no API key, but does require
    Ollama to be installed and running locally (OLLAMA_HOST, default
    http://localhost:11434). Raises LLMProviderError if unreachable --
    callers (rag_pipeline) handle this by falling back to raw retrieved text.
    """

    name = "ollama"

    def __init__(self, model: str = None, host: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        try:
            import requests
        except ImportError as exc:
            raise LLMProviderError("The 'requests' package is required for OllamaProvider.") from exc

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": f"{system_prompt}\n\n{prompt}", "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc


class OpenAIProvider(LLMProvider):
    """
    Calls an OpenAI-compatible hosted API. Requires OPENAI_API_KEY to be set
    as an environment variable -- NEVER hard-coded. Raises LLMProviderError
    if the key or the `openai` package is missing.
    """

    name = "openai"

    def __init__(self, model: str = None):
        self.model = model or config.LLM_MODEL_NAME
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key:
            raise LLMProviderError(
                "OPENAI_API_KEY is not set. Set it as an environment variable to use OpenAIProvider."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError("The 'openai' package is required for OpenAIProvider.") from exc

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc


def _redact(text: str, secret: Optional[str]) -> str:
    """Remove an API key (if present) from text destined for the UI or logs."""
    if secret and secret in text:
        text = text.replace(secret, "[redacted]")
    return text


def resolve_anthropic_api_key() -> str:
    """
    Return the Anthropic API key, or "" if none is configured.

    Lookup order: the ANTHROPIC_API_KEY environment variable (which a
    git-ignored .env file can populate -- see config.py), then Streamlit
    secrets (.streamlit/secrets.toml, git-ignored). Never raises.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
    except Exception:  # noqa: BLE001 -- no secrets file / not running in Streamlit
        return ""


class AnthropicProvider(LLMProvider):
    """
    Calls Claude through the official `anthropic` Python SDK.

    Configuration (no value is hard-coded):
        ANTHROPIC_API_KEY  -- required (env var, .env file or Streamlit secrets)
        ANTHROPIC_MODEL    -- optional; defaults to config.ANTHROPIC_MODEL_NAME

    Claude API access is billed separately from, and is NOT included in, a
    Claude.ai subscription: an API key from the Anthropic Console is required.

    Every failure (missing key, missing SDK, network error, timeout, API
    error, empty response) is raised as LLMProviderError with a message that
    is safe to show to the user -- it never contains the API key. Callers
    (rag.rag_pipeline) turn that into a labelled retrieval-only fallback.

    `client` can be injected (used by the tests, which never call the real API).
    """

    name = "anthropic"

    def __init__(self, model: str = None, client=None):
        self.model = (model or os.getenv("ANTHROPIC_MODEL") or config.ANTHROPIC_MODEL_NAME).strip()
        self._client = client
        self._api_key = None if client is not None else resolve_anthropic_api_key()

    def __repr__(self) -> str:  # never expose the key via repr/logging
        return f"AnthropicProvider(model={self.model!r})"

    def status(self) -> Tuple[bool, str]:
        if self._client is not None:
            return True, ""
        if not self._api_key:
            return False, "ANTHROPIC_API_KEY is not set."
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "The 'anthropic' package is not installed (pip install anthropic)."
        return True, ""

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMProviderError(
                "ANTHROPIC_API_KEY is not set. Set it as an environment variable (or in a git-ignored "
                ".env file / Streamlit secrets) to use the Anthropic provider."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMProviderError(
                "The 'anthropic' package is not installed. Install it with: pip install anthropic"
            ) from exc
        # max_retries=1 keeps worst-case latency bounded; the UI falls back to retrieved text on failure.
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=config.ANTHROPIC_TIMEOUT_SECONDS,
            max_retries=1,
        )
        return self._client

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=config.ANTHROPIC_MAX_TOKENS,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            # Report only the error class and HTTP status -- not the raw message, which could
            # echo request details. Timeouts and connection errors are named plainly.
            status_code = getattr(exc, "status_code", None)
            kind = type(exc).__name__
            hint = ""
            if status_code in (401, 403):
                hint = " Check that ANTHROPIC_API_KEY is valid and that the account has API access."
            elif status_code == 404:
                hint = f" Check that the model name '{self.model}' (ANTHROPIC_MODEL) is available to your account."
            elif status_code == 429:
                hint = " Rate limit or quota reached; check usage and billing in the Anthropic Console."
            elif "timeout" in kind.lower():
                hint = " The request timed out."
            elif "connection" in kind.lower():
                hint = " Could not reach the Anthropic API."
            detail = f"{kind}" + (f", HTTP {status_code}" if status_code else "")
            raise LLMProviderError(_redact(f"Anthropic request failed ({detail}).{hint}", self._api_key)) from None

        text = "".join(
            getattr(block, "text", "") for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMProviderError("Anthropic returned an empty response.")
        return text


_PROVIDERS = {
    "null": NullProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(provider_name: str = None) -> LLMProvider:
    """
    Factory: return the configured LLMProvider instance.

    Input:  optional override; defaults to config.LLM_PROVIDER.
            "auto" (the default) selects AnthropicProvider when an Anthropic
            API key is available and NullProvider otherwise, so the app works
            with no configuration and upgrades itself when a key is supplied.
    Output: an LLMProvider instance. Unrecognized names fall back to
            NullProvider (never silently requires a key the user didn't ask for).
    """
    name = (provider_name or config.LLM_PROVIDER or "auto").strip().lower()
    if name == "auto":
        name = "anthropic" if resolve_anthropic_api_key() else "null"
    provider_cls = _PROVIDERS.get(name, NullProvider)
    return provider_cls()
