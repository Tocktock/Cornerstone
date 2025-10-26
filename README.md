# Cornerstone

Cornerstone is a retrieval-augmented support workspace that lets operations teams turn scattered manuals, meeting notes, and product specs into a searchable, conversational knowledge base. It was built to de-risk frontline support for fast-growing businesses that need hybrid deployment options (cloud APIs when available, local models when data must stay on-prem) and want the same stack to power self-service search, guided chat responses, and post-incident analytics.

## Why We Built It
- **Support fire drills waste expert time.** Cornerstone ingests PDFs, DOCX files, HTML exports, and free-form notes so agents no longer dig through shared drives during incidents.
- **Multilingual teams need guidance in context.** A glossary-driven hint system bridges English and Korean terminology, making retrieval consistent across languages.
- **Compliance constraints change model choices.** The embedding service can flip between OpenAI, sentence-transformers, or Ollama-hosted models without code changes, keeping one code base viable for both SaaS and fully offline deployments.
- **Leaders want visibility, not dashboards pasted from spreadsheets.** Every chat is sanitized, persisted, and summarized for analytics so product and support managers can see what customers ask and which answers fail.

## What Cornerstone Delivers
- **Full document ingestion pipeline.** `DocumentIngestor` chunks uploads, extracts text (PDF, DOCX, HTML), stores vectors in Qdrant, and syncs metadata to a lightweight SQLite FTS index for lexical fallbacks.
- **Retrieval-augmented support agent.** `SupportAgentService` blends dense search, optional reranking, glossary snippets, and persona overrides to generate final answers via OpenAI or Ollama chat backends.
- **Semantic search web experience.** A FastAPI + Jinja UI surfaces global search, support chat, knowledge browsers, persona management, and analytics dashboards out of the box.
- **Keyword explorer & insights pipeline.** Multi-stage concept extraction (frequency heuristics, embedding similarity, clustering, optional LLM summarization) now runs through a background job system that caches results, exposes run metadata, and pushes insights onto an async queue.
- **Operational guardrails.** Conversation logging strips emails/phone numbers, retention windows are enforced, and optional Prometheus metrics emit timings and counters for everything from ingestion throughput to chat latency.
- **Language-aware query hints.** Glossary definitions, manually curated hints, and LLM-generated bridge tokens keep search usable across English/Korean terminology or other mixed-language corpora.

## Development Goals
1. **Unified RAG foundation.** One service app orchestrates ingestion, retrieval, chat, analytics, and keyword discovery so new support tools share the same data contracts.
2. **Config-first flexibility.** Environment-driven `Settings` enable embedding/back-end swaps, tuning retrieval depth, and toggling observability without touching code.
3. **Background-managed keyword runs.** Offload heavy keyword scans to asynchronous workers, surface job status, and reuse cached results in the UI.
4. **Privacy by default.** Keep sensitive datasets out of the repo (`data/` is ignored), anonymize transcripts, and make it easy to run fully offline with local embeddings.
5. **Actionable insights loop.** Feed conversation logs and keyword summaries back into product planning via the analytics endpoints and queued insight jobs.

## Architecture at a Glance
- **FastAPI application (`cornerstone.app`).** Wires together the embedding service, project/persona stores, ingestion pipeline, chat service, analytics, and the scheduled query-hint generator.
- **Embedding backends (`cornerstone.embeddings`).** Supports OpenAI, SentenceTransformers, or Ollama models with consistent APIs and validation of dimensionality.
- **Storage layer.** Qdrant collections per project hold dense vectors; SQLite powers FTS fallbacks; JSONL logs persist conversations; YAML files maintain glossary and hint catalogs.
- **Ingestion & chunking (`cornerstone.ingestion`, `cornerstone.chunker`).** Handles throttling, multi-file jobs, and rich metadata capture for every chunk.
- **Support intelligence (`cornerstone.chat`, `cornerstone.conversations`, `cornerstone.insights`).** Provides persona-aware chat generation, token accounting, analytics queries, and asynchronous insight publication.
- **UI templates (`templates/`).** Semantic search, support chat, knowledge browsers, persona editors, analytics dashboards, and keyword explorer pages share the same backend services.

## Getting Started
1. **Prerequisites:** Python 3.14+, Docker (for Qdrant), and optionally access to OpenAI or an Ollama instance.
2. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```
3. **Launch Qdrant locally:**
   ```bash
   docker compose up -d qdrant
   ```
4. **Configure environment:** copy `env.example.local` to `.env` (or export variables) and set your embedding/chat backend (`EMBEDDING_MODEL`, `OPENAI_API_KEY` or Ollama options, `QDRANT_URL`, etc.). For remote vLLM embeddings point `VLLM_EMBEDDING_BASE_URL` at your server (e.g. `https://llm.example.com:8000`), set `EMBEDDING_MODEL=vllm:<alias>`, and adjust `VLLM_EMBEDDING_CONCURRENCY` / `VLLM_EMBEDDING_BATCH_SIZE` if you need more throughput. Keep real secrets out of version control.
5. **Run the app:**
   ```bash
   uvicorn cornerstone.app:create_app --factory --reload
   ```
   Visit `http://localhost:8000` for the search interface and `/support` for the agent UI.
6. **(Optional) Load sample data:**
   ```bash
   cp -R samples/* data/
   ```
   The anonymized fixtures will populate the Search, Support, and Analytics views.
7. **(Optional) Enable background keyword runs:** set `KEYWORD_RUN_SYNC_MODE=0` to process keyword scans asynchronously, surface job status in the UI, and reuse cached results. The default (`1`) keeps the synchronous pipeline handy for local tests.
8. **Ingest your own content:** use the UI upload flow or the helper script `python -m cornerstone.scripts.ingest_local <path>` to populate project knowledge bases.

## Testing & Tooling
- Run the automated suite with `pytest`. Integration tests that hit a live Qdrant instance are marked with `@pytest.mark.integration`.
- Script utilities in `scripts/` seed demo datasets (AG News, Wikipedia), generate glossary hints, and clean stale vector collections.
- Observability endpoints expose Prometheus-formatted metrics when `OBSERVABILITY_METRICS_ENABLED=1`.
- When a coroutine hangs, visit `/admin/asyncio` to stream Python 3.14’s asyncio call-graph output (backed by the new `asyncio.format_call_graph` APIs), or run `python -m asyncio ps <pid>` from another shell for the same view against a running process.

## Security & Secrets
- Real credentials should live in environment variables or your secrets manager—keep `.env` out of version control.
- Rotate any API key that may have been exposed; do not store production keys in sample fixtures.
- Toggle conversation logging, retention windows, and anonymization rules via `Settings` to match your compliance posture.

## Deployment Notes
- **Local dev:** `uvicorn` + Docker-hosted Qdrant is enough for experimentation.
- **Container image:** package the FastAPI app together with a reverse proxy (e.g., Traefik or Nginx) and point it at managed Qdrant or the bundled container.
- **Observability:** enable Prometheus metrics and forward logs to your platform (metrics keys follow the `cornerstone.*` namespace).
- **Scaling:** separate ingestion workers from the API instance if you expect heavy document uploads; both depend only on Qdrant and the shared `data/` directory.
- **Keyword guardrails:** tune `KEYWORD_LLM_MAX_CANDIDATES`, `KEYWORD_LLM_MAX_TOKENS`, and `KEYWORD_LLM_MAX_CHUNKS` to cap LLM-heavy stages on very large projects—the UI automatically falls back to frequency-based results when limits are exceeded.

## Community
- Read the [Code of Conduct](./CODE_OF_CONDUCT.md) before participating.
- Issues and PRs are welcome—share feature ideas, bug reports, and localization improvements.

## Repository Layout
- `src/cornerstone/` – FastAPI application, services, and domain logic.
- `tests/` – Unit and integration coverage for ingestion, chat, retrieval, keywords, and analytics flows.
- `docs/` – Design notes on embedding strategy and keyword pipeline improvements.
- `templates/` – Jinja-powered UI views for search, support, knowledge, personas, and analytics.
- `scripts/` – Utility scripts for seeding data and managing Qdrant collections.
- `samples/` – Sanitized demo datasets to explore the UI without real customer content.

## Next Steps
- Produce anonymized sample datasets (outside of `data/`) for public demos.
- Harden deployment automation (container image, CI cadence, infrastructure as code).
- Expand multilingual coverage by extending glossary YAML files or swapping in locale-specific embedding models.

Cornerstone is now ready to be shared publicly—just remember to keep real customer data and API keys in private storage and use this repository as the blueprint for building compliant, high-signal support intelligence systems.
