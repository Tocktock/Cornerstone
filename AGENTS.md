# Cornerstone Agent Guide

## System Rule for Chat

Follow this principle **for the entire session** until I explicitly say "reset rule".
Planning is the most important part.
Spend most of your time on planning.

Process :

- OUTER_MAX_CYCLES=512, INNER_CRITIQUES=8, CONFIDENCE_THRESHOLD≥0.85
- Loop: Draft Plan First → private scratchpad reasoning → 8 critiques[Owner, Backend, AI, UI/UX, Frontend, Data, Others... ] → Revise → Assess.

Policy:

- If info may be outdated, verify with fresh sources and cite.
- Prefer primary sources; include dates/units and link-quality notes.
- If under-specified but solvable, proceed with explicit assumptions and partial result.
- Deliverable must include:

1. Solution (concise, structured)
2. Assumptions
3. Key Checks (how to validate)
4. Residual Risks
5. Confidence (0–1)

## Project Snapshot

- **Purpose:** Retrieval-augmented support workspace combining ingestion, hybrid search, guided chat, keyword discovery, and analytics.
- **Core Service (`src/cornerstone/app.py`):** FastAPI factory wiring embeddings, vector/Qdrant storage, SQLite FTS, persona/project stores, support chat endpoints, knowledge upload UI, keyword run APIs, analytics dashboard, Prometheus metrics, asyncio diagnostics.
- **State & Stores:** `ProjectStore` (JSON manifests for documents, keyword runs, glossary, personas) and `PersonaStore` (catalog + overrides) feed templates under `templates/`.
- **Chat & Retrieval (`chat.py`):** `SupportAgentService` augments queries with glossary/hints, blends FTS + vector hits, optional reranks, and hits OpenAI/Ollama/vLLM backends. `conversations.py` logs sanitized transcripts and aggregates analytics.
- **Ingestion (`ingestion.py`, `chunker.py`, `vector_store.py`, `fts.py`):** Converts uploads/URLs/local directories to chunks, embeds via `EmbeddingService`, upserts to Qdrant (`QdrantVectorStore`) and SQLite FTS, records manifests, and optionally triggers keyword auto-refresh.
- **Keyword Pipeline (`keywords.py`, `keyword_runner.py`, `keyword_jobs.py`, `keyword_refresh.py`, `insights.py`):** Seven-stage concept extraction, clustering, ranking, LLM harmonization, and insight summarization, with async queues serving `/keywords/*` endpoints.
- **Query Hints & Glossary (`glossary.py`, `query_hints.py`, `query_hint_scheduler.py`):** YAML-backed glossary + hint generation jobs (LLM-assisted) supporting multilingual retrieval.
- **Observability (`observability.py`):** Structured logging and optional Prometheus export; enable with `OBSERVABILITY_METRICS_ENABLED`.

## Runtime Data Layout

- `data/` (gitignored): live projects (`projects.json`), personas, manifests (`data/manifests/*.json`), ingested documents metadata (`data/documents/*.json`), keyword run outputs (`data/keyword_runs/**`), conversation logs (`data/conversations/*.jsonl`), SQLite FTS (`data/fts.sqlite`).
- `samples/`: anonymized fixtures to bootstrap demos; copy into `data/` for local previews.

## Dev Runbook

- **Start app:** `uvicorn cornerstone.app:create_app --factory --reload`
- **Tests:** `pytest`
- **Regenerate query hints:** `python scripts/run_hints.py`
- **Seed demo vectors:** `python scripts/seed_ag_news.py --limit 200` or `python scripts/seed_wikipedia.py --limit 500`
- **Manage vLLM env:** `python scripts/setup_vllm_env.py [start|status|stop]`
- **Cleanup Qdrant collections:** `python scripts/cleanup_collections.py --dry-run`

## Testing Focus

- Retrieval + support endpoints (`tests/test_support_api.py`, `tests/test_support_retrieval.py`, `tests/test_chat_backends.py`)
- Ingestion + chunking (`tests/test_document_ingestion.py`, `tests/test_chunker.py`, `tests/test_local_ingest.py`)
- Keyword jobs + pipeline (`tests/test_keywords.py`, `tests/test_keyword_jobs.py`, `tests/test_keyword_run_api.py`)
- Analytics + observability (`tests/test_analytics.py`, `tests/test_conversations.py`, `tests/test_observability.py`)

## Collaboration Notes

- Respect JSON/YAML stores under `data/`; do not commit runtime artifacts.
- Hybrid retrieval depends on consistent embedding dimensions when switching backends via `Settings` (OpenAI, HuggingFace, Ollama, vLLM).
- Background queues (keyword runs, insights, hint scheduler) require event loop startup; guard async mode with env flags (`KEYWORD_RUN_SYNC_MODE`, `KEYWORD_RUN_MAX_*`).
