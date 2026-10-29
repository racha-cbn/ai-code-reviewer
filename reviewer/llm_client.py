"""Provider-agnostic LLM interface.

Keep this boundary thin: Phase 3 ablations will swap models, and the eval
harness needs a deterministic fake. Anything that implements `complete` works.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = os.environ.get("REVIEWER_MODEL", "claude-sonnet-5-5")


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the raw text of the model's reply."""
        ...


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096, temperature: float = 0.0):
        import anthropic  # lazy: tests don't need the SDK or a key

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
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


def get_client(model: str | None = None, mock_path: str | None = None) -> LLMClient:
    if mock_path:
        return MockClient(Path(mock_path))
    return AnthropicClient(model=model or DEFAULT_MODEL)
