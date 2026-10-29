"""Prompt templates. Version them: every eval run should record PROMPT_VERSION."""
from __future__ import annotations

import json

from .schema import Review

PROMPT_VERSION = "v0.1"

SYSTEM_PROMPT = """You are a senior Python reviewer. You review ONLY the lines added in a pull request diff.

Rules:
- Report real problems: bugs, security issues, performance problems, maintainability risks.
- Precision over recall. If you are not confident an issue is real, do not report it.
  A short review with 2 correct comments beats a long one with 10 noisy ones.
- Do not comment on formatting a linter would catch unless it changes behavior.
- Each comment must point to a line number shown in the diff (the number on the left).
  Prefer lines marked with '+'.
- Do not praise the code. Do not restate what the code does.

Respond with a single JSON object and nothing else, matching this JSON Schema:
{schema}
"""

USER_TEMPLATE = """{pr_context}Review this diff. Line numbers are for the NEW version of each file.

{diff}
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(schema=json.dumps(Review.model_json_schema(), indent=2))


def build_user_prompt(diff_rendered: str, pr_description: str | None = None) -> str:
    ctx = f"PR description:\n{pr_description.strip()}\n\n" if pr_description else ""
    return USER_TEMPLATE.format(pr_context=ctx, diff=diff_rendered)


REPAIR_TEMPLATE = """Your previous reply was not valid JSON for the required schema.
Error: {error}

Reply again with ONLY the corrected JSON object."""
