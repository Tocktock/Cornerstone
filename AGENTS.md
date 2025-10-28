# Cornerstone Agents

## Core Principle: TRM Reasoning Loop

- Parameters: `OUTER_MAX_CYCLES` 128, `INNER_CRITIQUES` 6, `CONFIDENCE_THRESHOLD` ≥ 0.8 self-rating.
- Loop: Draft a full-pass solution, reason privately to map assumptions and risks, run six role-specific critiques (Project Owner, Backend Engineer, AI Engineer, UI/UX Engineer, Frontend Engineer, Data Engineer), revise with strengthened logic, then assess confidence; repeat until the threshold or cycle cap is reached.

## Repo Ground Rules

- Scope & precedence: This file applies to the whole repo. Direct system/developer/user instructions override. If a more-nested AGENTS.md exists, it takes precedence for files in its tree.
- Coding conventions (loose): Follow existing style; use type hints where helpful; keep diffs small; prefer pragmatic logging via the `cornerstone` logger; add/adjust tests when behavior changes.
- Local runbook: `uvicorn src.cornerstone.app:create_app --factory --host 0.0.0.0 --port 8000` to run; env vars from `.env`/`env.example.local`; tests via `pytest -q` (integration tests marked `integration`).
- Security & privacy: Conversation logging masks emails and phone numbers; avoid logging secrets; retention and enablement controlled by settings.

## High-Level Flow

- Ingestion: normalize + chunk uploads, embed, write vectors to Qdrant, mirror text into SQLite FTS, persist project metadata.
- Retrieval: resolve persona settings, expand the query with glossary and hint tokens, run hybrid search, optionally rerank, assemble prompt context.
- Response: invoke configured chat backend (OpenAI, Ollama, or vLLM) to produce full answers or streaming deltas with cited sources and definitions.
- Logging & metrics: sanitize + persist JSONL transcripts, update analytics counters, and emit retrieval/chat/ingestion metrics.

## Components

### SupportAgentService (`src/cornerstone/chat.py`)

- Orchestrates retrieval, prompt assembly, and chat invocation (`generate`, `stream_generate`), with metrics.
- Computes per-request persona options (temperature, token caps, glossary depth, retrieval top‑k).
- `_build_context` fuses dense + lexical hits, optional `Reranker`, glossary definitions, and formats snippets.
- Query augmentation: normalization, multilingual hint tokens, glossary synonyms for cross‑language recall.

### Persona & Project Context (`src/cornerstone/personas.py`, `src/cornerstone/projects.py`)

- `PersonaStore` maintains persona profiles; per‑project overrides adjust any field at runtime.
- `ProjectStore` persists project metadata, persona assignments, ingestion manifests, glossary extensions, and keyword run history.
- `_resolve_persona` merges base persona data with overrides into a `PersonaSnapshot` for prompting/runtime tuning.
- `ProjectVectorStoreManager` and `ConversationLogStore` isolate per‑project vectors, documents, query hints, and conversation logs.

### Retrieval & Storage (`src/cornerstone/ingestion.py`, `src/cornerstone/vector_store.py`, `src/cornerstone/fts.py`)

- `DocumentIngestor` extracts from PDF/DOCX/HTML/MD/TXT, chunks (`chunker.py`), embeds via `EmbeddingService`.
- `QdrantVectorStore` ensures collections, payload indexes, upsert/search/delete; `ProjectVectorStoreManager` caches per‑project stores.
- `FTSIndex` provides lexical fallback and keyword signals merged during context building.
- Ingestion metrics track embedding, vector upsert, FTS timings; auto‑refresh hooks can enqueue keyword runs.

### Query Enrichment & Glossary (`src/cornerstone/glossary.py`, `src/cornerstone/query_hints.py`, `src/cornerstone/query_hint_scheduler.py`)

- Global and per‑project glossary YAML load into `Glossary` (top‑match retrieval + prompt section generation).
- Query hints expand tokens across English/Korean; `QueryHintGenerator` can LLM‑generate hints; `QueryHintScheduler` schedules refreshes.
- Glossary synonyms feed both prompt context and retrieval augmentation.
- Settings expose glossary top‑k defaults, hint batch sizes, cron expressions, and storage paths.

### Keyword & Insight Loop (`src/cornerstone/keyword_runner.py`, `src/cornerstone/keywords.py`, `src/cornerstone/keyword_jobs.py`, `src/cornerstone/insights.py`, `src/cornerstone/keyword_refresh.py`)

- `execute_keyword_run`: chunk prep → concept extraction → clustering → ranking → harmonization → fallback frequency keywords.
- `KeywordLLMFilter` reuses the active chat backend to refine/harmonize concepts with guardrails (candidate/token/chunk limits, debug reporting).
- `KeywordRunQueue` manages async jobs; `KeywordInsightQueue` spawns insight summarization off the main loop.
- `KeywordRunAutoRefresher` listens for ingestion updates and enqueues fresh runs (deduped) when enabled.

### Conversation Logging & Analytics (`src/cornerstone/conversations.py`)

- `ConversationLogger` sanitizes responses, sources, and history (masks emails/phones) before appending per‑project JSONL.
- Records include persona identity, token counts, backend id, resolution flags, latency, and source metadata.
- `AnalyticsService` aggregates stats (daily volumes, resolution rate, unanswered queries, token usage) for dashboards/APIs.
- Retention policies prune aged logs; rewrites keep file‑backed stores compact with thread‑safe appends.

### Observability & Metrics (`src/cornerstone/observability.py`)

- `MetricsRecorder` emits structured metrics and can publish Prometheus counters/histograms/gauges.
- Retrieval, ingestion, chat, reranking, keyword runs, and queue wait times report tagged measurements per project/backend.
- Settings control metric namespace/enablement/Prometheus export; `/metrics` surfaces scrape output.
- Keep metric tag keys stable to preserve dashboard compatibility.

## FastAPI Surface (`src/cornerstone/app.py`)

- `create_app` wires settings, embeddings, vector stores, personas, glossary, chat service, ingestion, queues, hint generator, and analytics.
- Endpoints: global search (`/search`), support chat (`/support`, `/support/chat`, `/support/chat/stream`), knowledge mgmt (uploads/deletions/persona assignment), keyword flows (runs, insights, candidates), personas API, dashboards (Jinja in `templates/`).
- Background schedulers: query hints + keyword jobs start on app startup; streaming responses emit metadata frames before token deltas.
- Prometheus metrics and asyncio diagnostics endpoints available when configured.

## Extending the Agent

- Swap chat/embedding backends via `Settings`; `SupportAgentService` and `EmbeddingService` handle backends (OpenAI/Ollama/vLLM).
- Implement custom rerankers via the `Reranker` protocol; configure strategy to tune dense/lexical fusion.
- Extend personas or project overrides; `PersonaStore.update_persona` persists changes; `SupportAgentService` surfaces them in prompts.
- Customize ingestion (new file types, chunk heuristics) in `DocumentIngestor._extract_document` and `chunker.py`.
- Integrate external monitoring by providing a `MetricsRecorder` with custom loggers or a Prometheus registry.

## Tests & Verification

- Retrieval/prompt assembly: `tests/test_support_retrieval.py`, `tests/test_support_api.py`.
- Ingestion/chunking/vector store: `tests/test_document_ingestion.py`, `tests/test_chunker.py`, `tests/test_qdrant_store.py`.
- Keywords/pipelines/queues: `tests/test_keywords.py`, `tests/test_keyword_run_api.py`, `tests/test_keyword_jobs.py`.
- Conversation logging/analytics/personas: `tests/test_conversations.py`, `tests/test_analytics.py`, `tests/test_persona_prompts.py`.
- Observability/config: `tests/test_observability.py`, `tests/test_embeddings.py`, `tests/test_query_hints.py`.

## PR Checklist (Quick)

- Tests pass locally (`pytest -q`); integration tests guarded by `integration` marker.
- Metrics tag keys unchanged unless explicitly migrating dashboards.
- Jinja templates and `/templates/` updated if adding endpoints.
- Glossary/hints persisted consistently if fields change; consider scheduling refresh.

