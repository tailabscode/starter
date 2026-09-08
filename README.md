# TAI Labs AI Starters

A collection of ~10 clonable, runnable starter codebases for modern AI engineering patterns — RAG, agents, memory, evals, and Claude Code workflows — built for the TAI Labs community.

> **Status:** All 10 starters are complete, tested, and runnable offline.

## Choose a starter

| # | Starter | Demonstrates | Difficulty | Primary tech | Ideal use case | Status |
|---|---|---|---|---|---|---|
| 01 | [Basic RAG](starters/01-basic-rag/) | Ingestion, chunking, hybrid retrieval, grounded generation with citations | Beginner | Python, Anthropic SDK | Q&A over your own docs | Complete |
| 02 | [Multimodal RAG](starters/02-multimodal-rag/) | RAG across text + images | Intermediate | Python, Anthropic SDK (vision) | Retrieval over mixed media | Complete |
| 03 | [Agentic RAG](starters/03-agentic-rag/) | An agent that decides when/how to retrieve | Intermediate | Python, Anthropic SDK, tool use | Chat agents with optional grounding | Complete |
| 04 | [Multi-Agent Research](starters/04-multi-agent-research/) | Coordinator/researcher/critic/synthesizer pipeline | Advanced | Python, Anthropic SDK | Deep research over a topic | Complete |
| 05 | [Long-Running Agent](starters/05-long-running-agent/) | Checkpointed, resumable, retryable background jobs | Advanced | Python, SQLite | Long async processing tasks | Complete |
| 06 | [Memory Agent](starters/06-memory-agent/) | Session vs. persistent memory, retrieval-augmented recall | Intermediate | Python, SQLite | Agents that remember users across sessions | Complete |
| 07 | [Tool-Using Agent](starters/07-tool-using-agent/) | Multi-tool routing, validation, bounded execution | Beginner | Python, Anthropic SDK | Function-calling agents | Complete |
| 08 | [Agent Evals & Observability](starters/08-agent-evals-observability/) | Eval datasets, deterministic runners, traces, regression testing | Intermediate | Python | Measuring and guarding agent quality | Complete |
| 09 | [Claude Code Project Starter](starters/09-claude-code-project-starter/) | A well-structured repo for working with Claude Code | Beginner | Claude Code, CLAUDE.md | Bootstrapping a new Claude Code project | Complete |
| 10 | [Claude Code Skills](starters/10-claude-code-skills/) | Reusable skills: review, tests, debugging, docs | Intermediate | Claude Code Skills | Extending Claude Code for a team | Complete |

## Quick start

Each starter is fully standalone — clone the repo, `cd` into the one you want, and follow its own README. General shape:

```bash
git clone https://github.com/tailabscode/starter.git
cd starter/starters/01-basic-rag
cp .env.example .env        # add your ANTHROPIC_API_KEY, or skip and use --offline
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
uv run <starter-cli> --offline   # see the starter's own README for exact commands
```

Every starter ships an offline "stub" mode (`--offline`) so you can see it run end-to-end before you own an API key.

## Environment & setup expectations

- **Python 3.11+** for every starter. Node/TypeScript is used only where it materially improves a starter (called out explicitly in that starter's README if so).
- Dependency management via [`uv`](https://docs.astral.sh/uv/) and `pyproject.toml`. `pip install -e ".[dev]"` also works if you don't use `uv`.
- Credentials live in a per-starter `.env` (never committed — see `.env.example` in each folder).
- An [Anthropic API key](https://console.anthropic.com/settings/keys) is required to exercise the real model path; it is **not** required to run the tests or the offline demo.

## Common prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`
- An Anthropic API key for the live (non-offline) path
- Some starters optionally use a [Voyage AI](https://www.voyageai.com/) key for real embeddings — the default offline embedder needs no key

## Contributing a new starter

1. Copy the layout of an existing starter (`pyproject.toml`, `src/<pkg>/`, `tests/`, `.env.example`, `README.md`).
2. Keep dependencies minimal and the happy path runnable offline via a stub client.
3. Follow the README structure used across the other starters (architecture, setup, run commands, limitations, troubleshooting).
4. Add tests that exercise real logic, not `assert True`.
5. Open a PR — include what you validated (tests, lint, a real run) in the description.

## Security & secrets

- Never commit a real API key. `.env` is git-ignored everywhere in this repo; only commit `.env.example` with placeholder values.
- If you accidentally commit a secret, rotate it immediately at [console.anthropic.com](https://console.anthropic.com/settings/keys) — do not just delete the commit.
- Review any PR diff for accidental key material before merging.

## Disclaimer

These are **educational starter implementations** meant to teach patterns clearly, not production systems. Before shipping any of this to real users, add proper authentication, rate limiting, monitoring, input validation hardening, and a security review appropriate to your use case.

## License

[MIT](LICENSE)
