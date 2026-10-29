# code-reviewer

LLM code reviewer for Python pull requests. Takes a unified diff, returns **structured, line-anchored review comments**, and drops anything it can't verify.

> Phase 0: skeleton. Next phases add static-analysis grounding, an evaluation harness on SWR-Bench, and ablations.

## How it works

```
git diff ──► diff_parser ──► prompts ──► LLM ──► pipeline (parse + validate) ──► JSON / text
              │                                     │
              └── commentable line set ─────────────┘  (hallucination filter)
```

Every comment must:
1. validate against the Pydantic schema (`reviewer/schema.py`)
2. point to a file that is in the diff
3. point to a line that is in the diff (added or context, new-file numbering)
4. have `confidence >= --min-confidence` (default 0.5)

Comments that fail are kept in `dropped` with a reason, so you can measure how often the model hallucinates.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...        # only needed for real runs
```

## Usage

```bash
# Review your branch against main
reviewer --git main...HEAD --format text

# Pipe any diff
git diff HEAD~1 | reviewer

# Offline demo (no API key): replays a canned model response
reviewer --diff tests/fixtures/sample.diff --mock tests/fixtures/mock_response.json --format text --show-dropped

# CI gate: exit 1 if anything major or worse
reviewer --git origin/main...HEAD --fail-on major
```

Options: `--model`, `--pr-description`, `--min-confidence`, `--format json|text`.
Default model comes from `$REVIEWER_MODEL`.

## Tests

```bash
pytest -q
```

## Layout

```
reviewer/
  schema.py       # output contract (ReviewComment, Review, ReviewResult)
  diff_parser.py  # unified diff -> numbered prompt text + commentable line sets
  prompts.py      # versioned prompt templates (PROMPT_VERSION)
  llm_client.py   # LLMClient protocol, AnthropicClient, MockClient
  pipeline.py     # orchestration, JSON repair loop, validation
  cli.py          # entry point
tests/
  fixtures/       # sample.diff + mock_response.json (includes deliberate bad comments)
```

## Known limitations (Phase 0)

- Python files only; diff context only (no repo-level retrieval yet).
- Self-reported `confidence` is not calibrated. Treat it as a weak signal until Phase 2 measures it.
- No evaluation numbers yet. Do not claim quality until the eval harness exists.
