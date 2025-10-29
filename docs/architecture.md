# Cornerstone Architecture

This document provides a detailed overview of the Cornerstone repository. It mirrors current code paths and should be updated alongside implementation changes.

## Core Principle: TRM Reasoning Loop

- Parameters: `OUTER_MAX_CYCLES` 128, `INNER_CRITIQUES` 6, `CONFIDENCE_THRESHOLD` ≥ 0.8 self‑rating.
- Loop: Draft a full‑pass solution, reason privately to map assumptions and risks, run six role‑specific critiques (Project Owner, Backend Engineer, AI Engineer, UI/UX Engineer, Frontend Engineer, Data Engineer), revise with strengthened logic, then assess confidence; repeat until the threshold or cycle cap is reached.

## High‑Level Flow

- Document ingestion converts uploads into normalized chunks, embeds them, writes vectors to Qdrant, mirrors text into SQLite FTS, and records metadata for each project.
- Support requests resolve persona settings, expand the query with glossary and hint terms, retrieve hybrid matches, optionally rerank, and assemble a structured prompt.
- Response generation delegates to the configured chat backend (OpenAI Responses, Ollama chat, or vLLM), returning full answers or streaming deltas alongside cited sources and definitions.
- Conversation logging sanitizes transcripts, persists JSONL records, updates analytics counters, and emits metrics for retrieval, chat, and ingestion stages.

## Components

### SupportAgentService (`src/cornerstone/chat.py`)

- `generate` and `stream_generate` orchestrate retrieval, prompt assembly, and chat backend invocation while tracking metrics.
- Persona runtime options (temperature, token caps, glossary depth, retrieval top‑k) are computed per request via `PersonaStore` overrides or project defaults.
- `_build_context` fuses dense and lexical hits, applies the optional `Reranker`, gathers glossary definitions, and formats contextual snippets for the prompt.
- Query augmentation normalizes user input, injects multilingual hint tokens, and adds glossary‑derived synonyms to stabilize cross‑language recall.

### Persona & Project Context (`src/cornerstone/personas.py`, `src/cornerstone/projects.py`)

- `PersonaStore` maintains reusable persona profiles with tone, system prompt, and retrieval defaults; per‑project overrides adjust any field at runtime.
- `ProjectStore` persists project metadata, assigned persona ids, ingestion manifests, glossary extensions, and keyword run history on disk.
- `SupportAgentService._resolve_persona` merges base persona data with overrides to produce a `PersonaSnapshot` used in prompting and runtime tuning.
- `ProjectVectorStoreManager` and `ConversationLogStore` operate per‑project to isolate vector collections, documents, query hints, and conversation logs.

### Retrieval & Storage (`src/cornerstone/ingestion.py`, `src/cornerstone/vector_store.py`, `src/cornerstone/fts.py`)

- `DocumentIngestor` extracts text from PDF, DOCX, HTML, Markdown, and plain files, chunks content (`chunker.py`), and embeds via `EmbeddingService`.
- `QdrantVectorStore` wraps Qdrant APIs (ensure collection, payload indexes, upsert/search/delete) while `ProjectVectorStoreManager` caches per‑project stores.
- SQLite‑backed `FTSIndex` provides lexical fallback search and keyword signals merged with vector results during context building.
- Ingestion metrics capture embedding, vector upsert, and FTS timings; keyword auto‑refresh hooks mark projects dirty for asynchronous reruns.

### Query Enrichment & Glossary (`src/cornerstone/glossary.py`, `src/cornerstone/query_hints.py`, `src/cornerstone/query_hint_scheduler.py`)

- Global and per‑project glossary YAML files load into a `Glossary` that supports top‑match retrieval and prompt section generation.
- Query hints expand tokens across English and Korean; hints can be generated via LLM (`QueryHintGenerator`) and scheduled for refresh with `QueryHintScheduler`.
- Glossary‑derived synonyms feed both prompt context and retrieval augmentation, keeping multilingual terminology consistent.
- Settings expose glossary top‑k defaults, hint batch sizes, cron expressions, and resolve storage paths for glossary and hint catalogs.

### Keyword & Insight Loop (`src/cornerstone/keyword_runner.py`, `src/cornerstone/keywords.py`, `src/cornerstone/keyword_jobs.py`, `src/cornerstone/insights.py`, `src/cornerstone/keyword_refresh.py`)

- `execute_keyword_run` orchestrates the multi‑stage pipeline: chunk prep, concept extraction, clustering, ranking, harmonization, and fallback frequency keywords.
- `KeywordLLMFilter` reuses the chat backend to refine or harmonize concepts, with safeguards for candidate, token, and chunk limits plus debug reporting.
- `KeywordRunQueue` manages asynchronous execution with concurrency limits, while `KeywordInsightQueue` spawns insight summarization jobs off the main loop.
- `KeywordRunAutoRefresher` listens for ingestion updates and enqueues fresh runs when auto‑refresh is enabled, collapsing duplicate triggers per project.

### Conversation Logging & Analytics (`src/cornerstone/conversations.py`)

- `ConversationLogger` sanitizes responses, sources, and history (masking emails/phones) before appending JSONL records per project.
- Stored records include persona identity, token counts, backend id, resolution flags, latency, and source metadata for later audits.
- `AnalyticsService` aggregates conversation stats (daily volumes, resolution rate, unanswered queries, token usage) for admin dashboards and APIs.
- Retention policies prune aged logs, and rewrite operations keep file‑backed stores compact while retaining thread‑safe appends.

### Observability & Metrics (`src/cornerstone/observability.py`)

- `MetricsRecorder` emits structured log metrics and optionally publishes Prometheus counters, histograms, and gauges.
- Retrieval, ingestion, chat, reranking, keyword runs, and queue wait times all report tagged measurements for each project/backend.
- Settings control metric namespace, enablement, and Prometheus export; `/metrics` surfaces scrape output when configured.
- Analytics dashboards and structured logs consume the same metric namespace, so instrumentation changes should preserve tag keys for compatibility.

## FastAPI Surface (`src/cornerstone/app.py`)

- `create_app` wires settings, embedding, vector stores, personas, glossary, chat service, ingestion, keyword queues, hint generator, and analytics.
- Endpoints cover global search (`/search`), support chat (`/support`, `/support/chat`, streaming endpoints), knowledge management (uploads, deletions, persona assignment), and keyword flows (runs, insights, candidates).
- Conversation analytics, glossary editing, query hint generation, and keyword dashboards render via Jinja templates in `templates/`.
- Background schedulers (query hints, keyword jobs) attach during startup, and streaming responses emit metadata frames before delta tokens.
- Prometheus metrics and asyncio diagnostics endpoints expose operational insight for deployments.

## Extending the Agent

- Add or swap chat/embedding backends by extending `Settings` defaults and updating `EmbeddingService` or `SupportAgentService` hooks.
- Implement custom rerankers via the `Reranker` protocol and configure `RERANKER_STRATEGY` to tune dense/lexical fusion.
- Extend personas or project overrides with new fields; `PersonaStore.update_persona` persists changes and `SupportAgentService` will surface them in prompts.
- Customize ingestion (new file types, chunk heuristics) by enhancing `DocumentIngestor._extract_document` and `chunker.py`.
- Integrate external monitoring by providing a `MetricsRecorder` with bespoke loggers or Prometheus registries.

## Tests & Verification

- Retrieval and prompt assembly: `tests/test_support_retrieval.py`, `tests/test_support_api.py`.
- Ingestion, chunking, and vector store: `tests/test_document_ingestion.py`, `tests/test_chunker.py`, `tests/test_qdrant_store.py`.
- Keyword pipelines and queues: `tests/test_keywords.py`, `tests/test_keyword_run_api.py`, `tests/test_keyword_jobs.py`.
- Conversation logging, analytics, and personas: `tests/test_conversations.py`, `tests/test_analytics.py`, `tests/test_persona_prompts.py`.
- Observability and configuration edge cases: `tests/test_observability.py`, `tests/test_embeddings.py`, `tests/test_query_hints.py`.

