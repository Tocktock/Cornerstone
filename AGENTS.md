# Cornerstone Agents

This is the critical one‑pager for agents working in this repo. Deep architecture lives in `docs/architecture.md`.

## TRM Loop

- OUTER_MAX_CYCLES 128, INNER_CRITIQUES 6, CONFIDENCE_THRESHOLD ≥ 0.8.
- Draft → critique (6 roles) → revise → self‑rate; repeat until threshold or cap.

## Scope & Precedence

- Applies repo‑wide. Direct system/developer/user instructions override. Nested AGENTS.md (if any) takes precedence within its subtree.

## Coding Conventions (loose)

- Follow existing style; add type hints when helpful; keep diffs small.
- Log pragmatically via the `cornerstone` logger; adjust/add tests when behavior changes.

## Quick Start

- Run API: `uvicorn src.cornerstone.app:create_app --factory --host 0.0.0.0 --port 8000`
- Env: copy `.env` or `env.example.local`
- Tests: `pytest -q` (integration tests use marker `integration`)

## Critical Behaviors

- Privacy: conversation logs mask emails/phones; don’t log secrets; retention controlled by settings.
- Metrics: keep tag keys stable; Prometheus exposed on `/metrics` when enabled.
- Backends: chat via OpenAI/Ollama/vLLM; vectors via Qdrant; lexical via SQLite FTS.

## Components (map only)

- Chat service: `src/cornerstone/chat.py:1` — retrieval, prompt, backends, streaming.
- App wiring: `src/cornerstone/app.py:1` — FastAPI routes, schedulers, queues.
- Ingestion: `src/cornerstone/ingestion.py:1`; chunking `src/cornerstone/chunker.py:1`; vectors `src/cornerstone/vector_store.py:1`; FTS `src/cornerstone/fts.py:1`.
- Personas/projects: `src/cornerstone/personas.py:1`, `src/cornerstone/projects.py:1`.
- Glossary/hints: `src/cornerstone/glossary.py:1`, `src/cornerstone/query_hints.py:1`, scheduler `src/cornerstone/query_hint_scheduler.py:1`.
- Keywords/insights: `src/cornerstone/keyword_runner.py:1`, `src/cornerstone/keywords.py:1`, jobs `src/cornerstone/keyword_jobs.py:1`, insights `src/cornerstone/insights.py:1`, auto‑refresh `src/cornerstone/keyword_refresh.py:1`.
- Logging/analytics: `src/cornerstone/conversations.py:1`; observability `src/cornerstone/observability.py:1`.

## Extending

- Swap chat/embedding backends via `Settings`; implement `Reranker` via protocol; extend personas via `PersonaStore.update_persona`.

## PR Checklist

- `pytest -q` passes; don’t break metric tag keys.
- Update templates in `templates/` when adding endpoints.
- Persist glossary/hints consistently; schedule refresh if fields change.
