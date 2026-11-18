from pathlib import Path

import pytest

from reviewer.cli import main
from reviewer.diff_parser import parse_diff
from reviewer.llm_client import MockClient
from reviewer.pipeline import ReviewParseError, review_diff

FX = Path(__file__).parent / "fixtures"
DIFF = (FX / "sample.diff").read_text()
MOCK = FX / "mock_response.json"


# --- diff parser ---------------------------------------------------------

def test_parser_keeps_only_python_files():
    files = parse_diff(DIFF)
    assert [f.path for f in files] == ["app/users.py"]


def test_parser_line_numbers_are_new_file_numbers():
    f = parse_diff(DIFF)[0]
    assert 16 in f.added_lines          # the f-string query
    assert 26 in f.added_lines          # bare except
    assert 9 in f.commentable_lines     # context line
    assert 9 not in f.added_lines
    assert max(f.commentable_lines) == 27


def test_render_shows_numbers_and_markers():
    text = parse_diff(DIFF)[0].render()
    assert "### FILE: app/users.py" in text
    assert "  16 +     query = f\"SELECT" in text


def test_empty_when_no_python_changes():
    only_md = DIFF.split("diff --git a/app")[0]
    assert parse_diff(only_md) == []


# --- pipeline -------------------------------------------------------------

def test_pipeline_filters_hallucinated_and_low_confidence():
    result = review_diff(DIFF, MockClient(MOCK))
    kept = {(c.file, c.line) for c in result.review.comments}
    assert kept == {("app/users.py", 14), ("app/users.py", 16), ("app/users.py", 26)}

    reasons = sorted(d["reason"] for d in result.dropped)
    assert reasons == ["confidence 0.3 < 0.5", "file not in diff", "line 40 not in diff"]


def test_comments_sorted_by_location():
    lines = [c.line for c in review_diff(DIFF, MockClient(MOCK)).review.comments]
    assert lines == sorted(lines)


def test_malformed_single_comment_is_dropped_not_fatal():
    client = MockClient({"summary": "x", "comments": [
        {"file": "app/users.py", "line": 16, "severity": "blocker",  # invalid enum
         "category": "bug", "message": "something long enough", "confidence": 0.9},
    ]})
    result = review_diff(DIFF, client)
    assert result.review.comments == []
    assert result.dropped[0]["reason"].startswith("schema:")


def test_repair_loop_recovers_from_bad_json():
    class Flaky:
        model = "flaky"
        def __init__(self):
            self.n = 0
        def complete(self, system, user):
            self.n += 1
            return "not json at all" if self.n == 1 else '{"summary": "ok", "comments": []}'

    client = Flaky()
    result = review_diff(DIFF, client)
    assert client.n == 2 and result.review.summary == "ok"


def test_gives_up_after_repairs():
    with pytest.raises(ReviewParseError):
        review_diff(DIFF, MockClient("still not json"), max_repairs=1)


def test_no_llm_call_when_nothing_to_review():
    client = MockClient({"summary": "", "comments": []})
    review_diff(DIFF.split("diff --git a/app")[0], client)
    assert client.calls == []


# --- CLI --------------------------------------------------------------------

def test_cli_fail_on_returns_1(capsys):
    code = main(["--diff", str(FX / "sample.diff"), "--mock", str(MOCK),
                 "--format", "text", "--fail-on", "major"])
    out = capsys.readouterr().out
    assert code == 1
    assert "app/users.py:16 [critical/security]" in out


def test_cli_json_ok(capsys):
    assert main(["--diff", str(FX / "sample.diff"), "--mock", str(MOCK)]) == 0
    assert '"files_reviewed"' in capsys.readouterr().out


# --- Ollama client (HTTP is faked; Ollama does not need to be installed) ----

import io
import json
import urllib.error
import urllib.request

from reviewer.llm_client import AnthropicClient, OllamaClient, OllamaError, get_client


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(monkeypatch, *, reply=None, error=None):
    sent = {}

    def fake(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        if error is not None:
            raise error
        return _FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return sent


def test_ollama_sends_prompts_and_returns_content(monkeypatch):
    sent = _fake_urlopen(monkeypatch, reply={"message": {"role": "assistant", "content": '{"x": 1}'}})
    client = OllamaClient(model="qwen2.5-coder:7b", base_url="http://localhost:11434")

    assert client.complete("SYS", "USER") == '{"x": 1}'
    assert sent["url"] == "http://localhost:11434/api/chat"
    body = sent["body"]
    assert body["model"] == "qwen2.5-coder:7b"
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["stream"] is False and body["format"] == "json"
    assert body["options"]["num_ctx"] > 4096   # Ollama's default would truncate our prompt


def test_ollama_end_to_end_through_pipeline(monkeypatch):
    _fake_urlopen(monkeypatch, reply={"message": {"content": MOCK.read_text()}})
    result = review_diff(DIFF, OllamaClient(model="qwen2.5-coder:7b"))
    assert result.model == "qwen2.5-coder:7b"
    assert len(result.review.comments) == 3


def test_ollama_not_running(monkeypatch):
    _fake_urlopen(monkeypatch, error=urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")))
    with pytest.raises(OllamaError, match="Is it running"):
        OllamaClient().complete("s", "u")


def test_ollama_model_not_downloaded(monkeypatch):
    err = urllib.error.HTTPError("http://localhost:11434/api/chat", 404, "Not Found", {},
                                 io.BytesIO(b'{"error": "model \'nope:1b\' not found"}'))
    _fake_urlopen(monkeypatch, error=err)
    with pytest.raises(OllamaError, match=r"ollama pull nope:1b"):
        OllamaClient(model="nope:1b").complete("s", "u")


def test_ollama_timeout(monkeypatch):
    _fake_urlopen(monkeypatch, error=TimeoutError("timed out"))
    with pytest.raises(OllamaError, match="timed out"):
        OllamaClient(timeout=1).complete("s", "u")


def test_get_client_selects_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(get_client(provider="ollama"), OllamaClient)   # no key needed
    assert get_client(provider="ollama", model="qwen2.5-coder:14b").model == "qwen2.5-coder:14b"
    assert isinstance(get_client(provider="ollama", mock_path=str(MOCK)), MockClient)  # --mock still wins
    with pytest.raises(ValueError):
        get_client(provider="openai")


def test_get_client_default_is_still_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    assert isinstance(get_client(), AnthropicClient)


def test_ollama_host_env_without_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    assert OllamaClient().base_url == "http://127.0.0.1:11500"


def test_cli_reports_ollama_error_cleanly(monkeypatch, capsys):
    _fake_urlopen(monkeypatch, error=urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")))
    code = main(["--diff", str(FX / "sample.diff"), "--provider", "ollama"])
    assert code == 2
    assert "Is it running" in capsys.readouterr().err
