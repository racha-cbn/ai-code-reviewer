"""Structured output contract for the reviewer.

Every comment the LLM produces must validate against `ReviewComment`.
Anything that doesn't is dropped before it reaches the user.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    critical = "critical"  # will break prod / security hole
    major = "major"        # real bug or clear perf problem
    minor = "minor"        # maintainability, edge case
    nit = "nit"            # style; candidates for suppression


class Category(str, Enum):
    bug = "bug"
    security = "security"
    performance = "performance"
    maintainability = "maintainability"
    style = "style"


class ReviewComment(BaseModel):
    file: str = Field(description="Path of the changed file, as in the diff (new path).")
    line: int = Field(ge=1, description="Line number in the NEW version of the file.")
    severity: Severity
    category: Category
    message: str = Field(min_length=10, description="What is wrong and why it matters.")
    suggested_fix: str | None = Field(default=None, description="Concrete replacement code or action.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's self-reported confidence.")


class Review(BaseModel):
    summary: str = Field(description="One or two sentences on the change overall.")
    comments: list[ReviewComment] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """What the CLI emits: the review plus what was filtered out (for debugging/eval)."""
    review: Review
    dropped: list[dict] = Field(default_factory=list)
    model: str
    files_reviewed: list[str]
