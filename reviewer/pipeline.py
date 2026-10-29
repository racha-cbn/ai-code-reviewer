"""diff -> prompt -> LLM -> parse -> validate -> filtered review."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .diff_parser import FileDiff, parse_diff, render_for_prompt
from .llm_client import LLMClient
from .prompts import REPAIR_TEMPLATE, build_system_prompt, build_user_prompt
from .schema import Review, ReviewComment, ReviewResult

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ReviewParseError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    """Tolerate code fences and leading chatter; take the outermost {...}."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start : end + 1])


def _parse_raw(raw: str) -> tuple[str, list[dict]]:
    """Validate the envelope; keep comments as raw dicts so one bad comment
    doesn't sink the whole review."""
    data = _extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("comments", []), list):
        raise ValueError("expected an object with a 'comments' list")
    return str(data.get("summary", "")), data.get("comments", [])


def _validate_comments(
    raw_comments: list[dict],
    files: list[FileDiff],
    min_confidence: float,
) -> tuple[list[ReviewComment], list[dict]]:
    by_path = {f.path: f for f in files}
    kept, dropped = [], []

    for rc in raw_comments:
        try:
            c = ReviewComment.model_validate(rc)
        except ValidationError as e:
            dropped.append({"comment": rc, "reason": f"schema: {e.errors()[0]['msg']}"})
            continue

        fd = by_path.get(c.file)
        if fd is None:
            dropped.append({"comment": rc, "reason": "file not in diff"})
        elif c.line not in fd.commentable_lines:
            dropped.append({"comment": rc, "reason": f"line {c.line} not in diff"})
        elif c.confidence < min_confidence:
            dropped.append({"comment": rc, "reason": f"confidence {c.confidence} < {min_confidence}"})
        else:
            kept.append(c)

    return kept, dropped


def review_diff(
    diff_text: str,
    client: LLMClient,
    pr_description: str | None = None,
    min_confidence: float = 0.5,
    max_repairs: int = 1,
) -> ReviewResult:
    files = parse_diff(diff_text)
    if not files:
        return ReviewResult(
            review=Review(summary="No reviewable Python changes in this diff."),
            model=client.model,
            files_reviewed=[],
        )

    system = build_system_prompt()
    user = build_user_prompt(render_for_prompt(files), pr_description)

    raw = client.complete(system, user)
    for attempt in range(max_repairs + 1):
        try:
            summary, raw_comments = _parse_raw(raw)
            break
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == max_repairs:
                raise ReviewParseError(f"model output unparseable after {max_repairs} repair(s): {e}") from e
            repair = f"{user}\n\nYour reply:\n{raw}\n\n{REPAIR_TEMPLATE.format(error=e)}"
            raw = client.complete(system, repair)

    kept, dropped = _validate_comments(raw_comments, files, min_confidence)
    kept.sort(key=lambda c: (c.file, c.line))

    return ReviewResult(
        review=Review(summary=summary, comments=kept),
        dropped=dropped,
        model=client.model,
        files_reviewed=[f.path for f in files],
    )
