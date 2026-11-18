# code-reviewer

LLM code reviewer for Python pull requests. Takes a unified diff, returns **structured, line-anchored review comments**, and drops anything it can't verify.

> Phase 0: skeleton. Next phases add static-analysis grounding, an evaluation harness on SWR-Bench, and ablations.

## Quick start

The tool sends the Python changes of a diff to an LLM, keeps only comments that point to real lines of the diff, and prints them as JSON or text. To try it offline, no API key needed:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
reviewer --diff tests/fixtures/sample.diff --mock tests/fixtures/mock_response.json --format text --show-dropped
```

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

Options: `--provider anthropic|ollama`, `--model`, `--pr-description`, `--min-confidence`, `--format json|text`.
Default model comes from `$REVIEWER_MODEL` (Anthropic) or `$REVIEWER_OLLAMA_MODEL` (Ollama).

## Providers

| `--provider` | Cost | Needs | Default model |
|---|---|---|---|
| `anthropic` (default) | paid per token | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `ollama` | free, runs on your machine | Ollama app + a downloaded model | `qwen2.5-coder:7b` |
| `--mock FILE` | free | nothing | replays a canned reply (overrides `--provider`) |

## Running locally with Ollama (free, no API key)

### 1. Check your Mac

```bash
sysctl -n machdep.cpu.brand_string   # Apple M-series = GPU acceleration
sysctl -n hw.memsize                 # RAM in bytes; divide by 1073741824 for GB
df -h ~                              # free disk space
sw_vers -productVersion              # Ollama needs macOS 14 (Sonoma) or newer
```

Pick a model that fits your RAM. Download sizes are from [ollama.com/library/qwen2.5-coder](https://ollama.com/library/qwen2.5-coder/tags); RAM in use is a rough estimate (weights + context), and grows with longer diffs.

| Model | Download | RAM while running (approx.) | Suggested for |
|---|---|---|---|
| `qwen2.5-coder:3b` | 1.9 GB | ~3–4 GB | 8 GB Macs; plumbing tests only |
| `qwen2.5-coder:7b` | 4.7 GB | ~6–8 GB | 16 GB+ Macs (**default**) |
| `qwen2.5-coder:14b` | 9.0 GB | ~11–13 GB | 24 GB+ Macs, better quality, slower |
| `qwen2.5-coder:32b` | 20 GB | ~22 GB+ | 32 GB+ Macs only |

### 2. Install Ollama

Download the macOS app from <https://ollama.com/download>, open `Ollama.dmg`, and drag Ollama into Applications. Launch it once; it will offer to install the `ollama` command-line tool. Then check:

```bash
ollama --version
curl http://localhost:11434/api/version   # the app runs this server in the background
```

### 3. Download a model

```bash
ollama pull qwen2.5-coder:7b
ollama list                               # confirm it's there
```

Models are stored in `~/.ollama`. Remove one with `ollama rm <name>`.

### 4. Review a diff

```bash
# The bundled sample diff
reviewer --diff tests/fixtures/sample.diff --provider ollama --format text --show-dropped

# Your own branch
reviewer --git main...HEAD --provider ollama --format text

# A different local model
reviewer --git main...HEAD --provider ollama --model qwen2.5-coder:14b --format text

# Make Ollama the default model without typing --model
export REVIEWER_OLLAMA_MODEL=qwen2.5-coder:14b
```

The first run is slow (the model is loaded into memory); later runs within ~5 minutes are faster. If Ollama listens somewhere other than `localhost:11434`, set `OLLAMA_HOST` (e.g. `export OLLAMA_HOST=127.0.0.1:11500`).

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `cannot reach Ollama at http://localhost:11434 ... Is it running?` | Ollama app not running | Open the Ollama app, or run `ollama serve` in another terminal |
| `model 'X' is not available in Ollama ... ollama pull X` | Model not downloaded, or name typo | Run the `ollama pull` command shown; check names with `ollama list` |
| `Ollama timed out after 300s` | Model too big/slow for your machine, or a huge diff | Try a smaller model (`qwen2.5-coder:3b`) or a smaller diff |
| `model output unparseable after 1 repair(s)` | The model didn't return valid JSON | Retry, or use a larger model |
| Many comments in `dropped` with `line N not in diff` | Small models get line numbers wrong | Expected; this is what the filter is for. Try a larger model |
| Mac becomes very slow / swaps | Model doesn't fit in RAM | Use a smaller model; check with `ollama ps` |

The reviewer calls Ollama with `num_ctx=16384` because Ollama's default context (4096 tokens) is too small for the prompt + schema + diff and would silently truncate it.

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
  llm_client.py   # LLMClient protocol, AnthropicClient, OllamaClient, MockClient
  pipeline.py     # orchestration, JSON repair loop, validation
  cli.py          # entry point
tests/
  fixtures/       # sample.diff + mock_response.json (includes deliberate bad comments)
```

## Known limitations (Phase 0)

- Python files only; diff context only (no repo-level retrieval yet).
- Self-reported `confidence` is not calibrated. Treat it as a weak signal until Phase 2 measures it.
- No evaluation numbers yet. Do not claim quality until the eval harness exists.
- Local 7B/14B models are useful for exercising the pipeline for free, but expect lower precision and more
  wrong line numbers than frontier models. Don't compare them to Anthropic results without the eval harness.
