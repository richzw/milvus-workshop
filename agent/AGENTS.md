# Repository Guidelines

## Project Structure & Module Organization

This directory is a Markdown-first workshop planning workspace. Keep the main workshop overview at the top level and authoritative, numbered design documents in `specs/`.

- `readme.md`: main workshop outline and research synthesis.
- `specs/index.md`: spec entry point, reading order, and build-order graph.
- `specs/00-prd.md`: product scope, goals, non-goals, and success measures.
- `specs/10-*.md` through `specs/70-*.md`: dependency-ordered component and quality contracts.
- `specs/90-roadmap.md`: stakeholder-facing, user-visible milestones.
- `specs/91-impl-plan.md`: engineer-facing implementation phases.
- `specs/99-key-decisions.md`: load-bearing decisions and rationale.
- `specs/archive/`: pre-reorganization source notes retained for traceability; not authoritative.

If runnable demo code or notebooks are added, place them in clear folders such as `demo/`, `notebooks/`, or `assets/` and update this guide.

## Build, Test, and Development Commands

The runnable Python demo lives under `demo/`. Useful local checks are:

- `rg --files`: list tracked workshop files quickly.
- `git diff -- *.md`: review Markdown-only changes before commit.
- `open readme.md`: preview the main document in the default macOS app.
- `PYTHONPATH=demo/src python3 -m unittest discover demo/tests -v`: run the deterministic test suite.
- `PYTHONPATH=demo/src python3 demo/scripts/run_eval.py`: run the golden-question retrieval/citation evaluation.
- `ruff check demo/src demo/scripts demo/tests`: run the available Python lint checks.
- `mypy --strict demo/src demo/tests demo/scripts`: run strict type checks once optional dependencies are installed.

Install optional UI, API, LangGraph, and pymilvus dependencies with `pip install -r demo/requirements.txt`; the core fallback and unit tests require only Python.

## Coding Style & Naming Conventions

Use Markdown with ATX headings (`#`, `##`, `###`) and short sections. Prefer descriptive filenames in lowercase for new general docs, for example `architecture.md`; keep existing mixed-case names unchanged unless doing a deliberate rename. Use fenced code blocks with a language tag when possible, such as `json`, `text`, or `python`.

This repository contains Chinese workshop notes with English technical terms. Preserve that style: keep terminology precise, and avoid translating product names, libraries, or protocol names.

## Testing Guidelines

For documentation changes, verify that headings render correctly, code fences close, links are still valid, and examples are copy-pasteable. If notebooks or demos are added, run them from a clean checkout and document required services such as Milvus, MinIO, or Streamlit.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit-style messages such as `docs(tips): update tips`. Prefer:

```text
docs(agent-workshop): update UI demo outline
```

Keep commits focused on one topic. Pull requests should include a short summary, changed files or sections, validation performed, and screenshots only when UI assets or rendered pages change.

## Agent-Specific Instructions

Do not invent project commands or dependencies. Verify files and symbols before referencing them, keep changes incremental, and avoid touching unrelated modified files outside this workshop directory.
