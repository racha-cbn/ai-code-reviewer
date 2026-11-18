"""Provider-agnostic LLM interface.

Keep this boundary thin: Phase 3 ablations will swap models, and the eval
harness needs a deterministic fake. Anything that implements `complete` works.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = os.environ.get("REVIEWER_MODEL", "claude-sonnet-5-5")
DEFAULT_OLLAMA_MODEL = os.environ.get("REVIEWER_OLLAMA_MODEL", "qwen2.5-coder:7b")
PROVIDERS = ("anthropic", "ollama")


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the raw text of the model's reply."""
        ...


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        import anthropic  # lazy: tests don't need the SDK or a key

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        # anthropic>=2.3 Messages.create() has no `temperature` parameter.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class MockClient:
    """Replays a canned JSON response. Use for tests, demos without a key, and CI."""

    model = "mock"

    def __init__(self, response: str | dict | Path):
        if isinstance(response, Path):
            response = response.read_text()
        self._response = json.dumps(response) if isinstance(response, dict) else response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response


class OllamaError(RuntimeError):
    """Ollama unreachable, model missing, or request failed. Message says how to fix it."""


def _ollama_base_url() -> str:
    # Same variable the Ollama server uses; it may be a bare "host:port".
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"http://{host}"


class OllamaClient:
    """Local models via Ollama's /api/chat. Free, no API key, stdlib only."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 16384,  # Ollama defaults to 4096, which silently truncates our prompt
        timeout: float = 300.0,  # first call loads the model into memory, can be slow
    ):
        self.model = model
        self.base_url = (base_url or _ollama_base_url()).rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = _error_detail(e)
            if e.code == 404:
                raise OllamaError(
                    f"model '{self.model}' is not available in Ollama ({detail}). "
                    f"Download it with: ollama pull {self.model}"
                ) from e
            raise OllamaError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise OllamaError(f"Ollama timed out after {self.timeout:.0f}s") from e
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url} ({e.reason}). "
                "Is it running? Open the Ollama app, or run: ollama serve"
            ) from e
        except TimeoutError as e:
            raise OllamaError(f"Ollama timed out after {self.timeout:.0f}s") from e

        return data.get("message", {}).get("content", "")


def _error_detail(e: urllib.error.HTTPError) -> str:
    try:
        return json.loads(e.read()).get("error", "") or e.reason
    except (ValueError, AttributeError):
        return str(e.reason)


def get_client(
    model: str | None = None,
    mock_path: str | None = None,
    provider: str = "anthropic",
) -> LLMClient:
    if mock_path:
        return MockClient(Path(mock_path))
    if provider == "ollama":
        return OllamaClient(model=model or DEFAULT_OLLAMA_MODEL)
    if provider == "anthropic":
        return AnthropicClient(model=model or DEFAULT_MODEL)
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
