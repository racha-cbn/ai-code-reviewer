"""Command-line entry point.

Examples:
    git diff main...HEAD | reviewer
    reviewer --git main...HEAD --format text
    reviewer --diff tests/fixtures/sample.diff --mock tests/fixtures/mock_response.json
    reviewer --diff tests/fixtures/sample.diff --provider ollama --format text
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .llm_client import PROVIDERS, OllamaError, get_client
from .pipeline import ReviewParseError, review_diff
from .schema import ReviewResult, Severity

SEVERITY_ORDER = [s.value for s in Severity]  # critical first
ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡", "nit": "⚪"}


def _read_diff(args: argparse.Namespace) -> str:
    if args.diff:
        with open(args.diff, encoding="utf-8") as f:
            return f.read()
    if args.git:
        return subprocess.run(
            ["git", "diff", "--no-color", "-U5", args.git],
            check=True, capture_output=True, text=True,
        ).stdout
    if not sys.stdin.isatty():
        return sys.stdin.read()
    sys.exit("error: pass --diff FILE, --git RANGE, or pipe a diff on stdin")


def _format_text(result: ReviewResult) -> str:
    r = result.review
    lines = [f"Model: {result.model} | files: {len(result.files_reviewed)} | "
             f"comments: {len(r.comments)} (dropped {len(result.dropped)})",
             "", r.summary, ""]
    for c in r.comments:
        lines.append(f"{ICON[c.severity.value]} {c.file}:{c.line} [{c.severity.value}/{c.category.value}] "
                     f"(conf {c.confidence:.2f})")
        lines.append(f"   {c.message}")
        if c.suggested_fix:
            lines.append("   fix: " + c.suggested_fix.replace("\n", "\n        "))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reviewer", description="LLM code reviewer for Python diffs.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--diff", help="path to a unified diff file")
    src.add_argument("--git", metavar="RANGE", help="git range, e.g. main...HEAD")
    p.add_argument("--pr-description", help="PR description text (extra context)")
    p.add_argument("--provider", choices=PROVIDERS, default="anthropic",
                   help="LLM backend: anthropic (needs ANTHROPIC_API_KEY) or ollama (local, free)")
    p.add_argument("--model", help="override model id (default: $REVIEWER_MODEL for anthropic, "
                                   "$REVIEWER_OLLAMA_MODEL or qwen2.5-coder:7b for ollama)")
    p.add_argument("--mock", metavar="JSON", help="replay a canned model response (no API call)")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--format", choices=["json", "text"], default="json")
    p.add_argument("--show-dropped", action="store_true", help="include dropped comments in text output")
    p.add_argument("--fail-on", choices=SEVERITY_ORDER,
                   help="exit 1 if any comment at or above this severity (for CI)")
    args = p.parse_args(argv)

    diff_text = _read_diff(args)
    client = get_client(model=args.model, mock_path=args.mock, provider=args.provider)

    try:
        result = review_diff(diff_text, client, args.pr_description, args.min_confidence)
    except (ReviewParseError, OllamaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(result.model_dump_json(indent=2))
    else:
        print(_format_text(result))
        if args.show_dropped and result.dropped:
            print("Dropped:")
            for d in result.dropped:
                c = d["comment"]
                print(f"  - {c.get('file')}:{c.get('line')} -> {d['reason']}")

    if args.fail_on:
        threshold = SEVERITY_ORDER.index(args.fail_on)
        if any(SEVERITY_ORDER.index(c.severity.value) <= threshold for c in result.review.comments):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
