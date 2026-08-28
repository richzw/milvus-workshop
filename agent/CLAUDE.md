# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Milvus 3.0 "Agentic RAG" workshop repo. The root is a Markdown-first spec workspace (`readme.md`, `specs/`); the only runnable code is the Python demo under `demo/`. See `AGENTS.md` for full repo-guideline detail and `specs/index.md` for the design-doc reading order.

The demo simulates an enterprise knowledge assistant (Streamlit UI) exercising query understanding, permission-aware grounded caching, tool planning, hybrid retrieval, evidence evaluation, answer verification, and multi-tier memory. It uses synthetic data only — this is a teaching project, not production: no real auth/ACL, no tenant isolation, never load real secrets or PII.

## Commands

Run all of these from the **repo root**, not from `demo/`:

```bash
PYTHONPATH=demo/src python3 -m unittest discover -t demo -s demo/tests -v  # test suite (deterministic, no live services needed)
PYTHONPATH=demo/src python3 demo/scripts/run_eval.py                      # golden-question retrieval/citation eval
PYTHONPATH=demo/src python3 demo/scripts/run_tier_eval.py                 # retrieval tier comparison (T0/T1/T2 arms)
ruff check demo/src demo/scripts demo/tests                               # lint (default config, no ruff.toml)
mypy demo/src demo/scripts demo/tests                                     # strict type check (settings in mypy.ini)
```

Milvus-backed setup, in order: `create_collections.py` → `create_indexes.py` → `ingest_demo.py`, then `streamlit run demo/src/agent_workshop_demo/streamlit_app.py`.

`demo/scripts/cleanup_milvus.py` only ever targets 8 fixed demo collection names and requires `--confirm-drop-demo-data`; run without it first to preview. `demo/scripts/evolve_schema.py` subcommands default to dry-run and require `--apply`.

No CI is configured (no `.github/workflows/`); ruff/mypy above are the only automated checks.

The `-t demo` root matters: it makes `demo/tests` a package so its `__init__.py` runs first and
strips `demo/.env` plus every provider selector and credential from the environment. Discovering
with `discover demo/tests` imports the test modules flat, skips that hook, and lets a configured
provider turn the suite into live network calls.

## Architecture gotchas

- **Deterministic by default**: setting `OPENAI_API_KEY` alone does not trigger network calls. Each stage needs its own explicit provider env var (`EMBEDDING_PROVIDER=openai`, `ANSWER_GENERATOR=openai`, `QUERY_CLASSIFIER=openai`, `RERANKER=openai`, etc.).
- Embeddings are fixed at 1024 dimensions; switching embedding provider requires a full collection re-ingest — providers/dimensions cannot be mixed within one collection.
- `pymilvus==3.0.1` is pinned exactly in `demo/requirements.txt`; the code relies on Milvus 3.0-specific features (BM25 function, MINHASH function, TIMESTAMPTZ TTL field, SINDI sparse index).
- `RETRIEVAL_TIER` selects the spec 15 retrieval ladder rung: `hybrid_dense` (default, T2), `lexical_rewrite` (T1) and `lexical_only` (T0) run BM25-only baseline arms and refuse to start alongside a non-disabled `STRUCT_ARRAY_RETRIEVAL` profile.
- Answers use "validated-buffered streaming": tokens are released only after passing citation/evidence self-checks, not streamed raw token-by-token.
- Conversation Memory, Selective Memory, and Grounded Response Cache persistence is intentionally sequential (not parallel) pending adapter thread-safety guarantees.
- `demo/.env` is gitignored and auto-loaded by every entrypoint (copy from `demo/.env.example`); real process env vars take precedence over the file, and `AGENT_WORKSHOP_SKIP_ENV_FILE=1` suppresses the default auto-load entirely (the test suite sets it). `MEMORY_CLEANUP_CURSOR_SECRET` must be a real random secret (≥32 bytes), not the placeholder.
- DINOv3 image embeddings are gated on HuggingFace (requires license acceptance + `HF_TOKEN`); the real-checkpoint smoke test is opt-in via `RUN_DINOV3_SMOKE=1`.
- Streamlit binds to loopback by default — do not expose port 8501 publicly; use an SSH tunnel or restricted security group for remote access.

## Code style

- Markdown: ATX headings, short sections, fenced code blocks with a language tag. Lowercase filenames for new general docs; keep existing mixed-case names unchanged.
- The repo mixes Chinese workshop prose with English technical terms by design — preserve this, do not translate terminology, product names, or library/protocol names.

## Commits

Conventional Commits with scope, one topic per commit, e.g. `docs(agent-workshop): update UI demo outline`, `fix(workflow): align runtime transition dispatch`. Do not invent project commands or dependencies; verify files/symbols before referencing them; avoid touching unrelated files outside the workshop directory.
