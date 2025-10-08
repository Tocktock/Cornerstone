# Cornerstone Improvement Roadmap

This document translates the audit findings into a sequenced backlog for the next improvement cycle. Tasks are grouped by theme, sorted in the order we should tackle them. Each item calls out owners, dependencies, and measurable outcomes whenever possible.

## Guiding Principles

- Preserve the working prototype while layering upgrades iteratively.
- Ship user-facing wins early (persona controls, UX polish) to validate value quickly.
- Instrument the pipeline so later analytics and scaling work rest on real metrics.

## Phase 0 – Preparation (Day 0-1)

1. **Confirm environment prerequisites**
   - Owners: Eng
   - Steps: Verify `.env` completeness, ensure Qdrant + Ollama optional dependencies documented.
2. **Establish baseline metrics**
   - Owners: Eng + PM
   - Steps: Capture current ingestion latency, chat response latency, vector count per project.

## Phase 1 – UX & Persona Quick Wins (Week 1)

1. **Persona catalog and project assignment**
   - Owners: Backend + Frontend
   - Steps:
     1. Introduce a global persona store (id, name, tone, system prompt, avatar, tags) and seed “Chatty”, “Sleepy”, “Analyze Agent” examples.
     2. Build persona management UI (list/create/edit) with guidance for best-practice fields (clear persona goal, tone descriptors, escalation guidance).
     3. Update project schema to reference a persona id plus optional project-level overrides; hydrate chat prompts by resolving the assigned persona.
     4. Allow multiple projects to reuse the same persona and surface assignment status in both persona and project views.
2. **Persona best-practice playbook**
   - Owners: PM + Design
   - Steps:
     1. Document guardrails for persona design (inclusive language, escalation rules, brand tone alignment) and weave them into the UI as helper copy.
     2. Establish a lightweight approval checklist so newly created personas get reviewed before production use.
     3. Capture analytics hooks needed later (e.g., persona id on conversations) to evaluate persona performance.
3. **Session management controls**
   - Owners: Frontend
   - Steps:
     1. Add "New Chat" button to reset conversation state.
     2. Provide confirmation modal before clearing history.
4. **Chat accessibility + feedback polish**
   - Owners: Frontend
   - Steps:
     1. Improve focus management after send/receive events.
     2. Show inline error banners for failed sends or retrieval issues.

## Phase 2 – Ingestion & Retrieval Enhancements (Week 2)

1. **Multi-file upload + drag-and-drop**
   - Owners: Frontend + Backend
   - Steps:
     1. Update upload UI for multi-select and drag targets.
     2. Batch backend ingestion by file, return per-file status payload.
2. **Expanded file type support**
   - Owners: Backend
   - Steps:
     1. Add DOCX parsing (python-docx) and HTML ingestion.
     2. Surface user-friendly errors for unsupported formats.
3. **Asynchronous ingestion jobs**
   - Owners: Backend
   - Steps:
     1. Move heavy ingestion to background worker (Celery/RQ or FastAPI background task with queue).
     2. Provide progress polling endpoint; show status indicator in UI.
4. **Metadata enrichment**
   - Owners: Backend
   - Steps:
     1. Capture document title, author, section headers when available.
     2. Include metadata in retrieval payload and source display.

## Phase 3 – Local Corpus Ingestion & Hybrid Retrieval (Week 3)

1. **Local data ingestion workflow**
   - Owners: Backend + Frontend
   - Steps:
     1. Build Knowledge Base UI picker for directories under `data/local/**` with file count/size preview and job status.
     2. Extend ingestion API to enqueue local-file jobs (stream from disk, reuse job manager) and surface status in UI.
     3. Deliver CLI (`python -m cornerstone.ingest_local`) with resumable manifest support for bulk indexing.
      4. Persist ingestion manifests/checkpoints so multi-day jobs resume cleanly and apply per-tenant throttles during imports.
      - [done] Local directory imports emit file/byte progress; UI renders live progress bar and regression tests cover `/knowledge/uploads` payload.
      - [done] Ollama embedding backend reuses connections and batches concurrent requests to reduce bulk ingest time.
2. **Advanced chunking pipeline**
   - Owners: Backend
   - Steps:
     1. Implement hierarchical splitter (headings → paragraphs) targeting 200–500 token chunks with ~10% overlap.
     2. Normalize text and capture rich metadata (path, section, language, timestamps) per chunk.
     3. Generate optional section summaries and store alongside chunk metadata for faster preview.
3. **Hybrid search enablement**
   - Owners: Backend
   - Steps:
     1. Create FTS index (e.g., SQLite FTS5) for chunk text + metadata and populate during ingestion.
     2. Fuse BM25 + vector hits via Reciprocal Rank Fusion; update SupportAgentService retrieval path.
     3. Add regression tests ensuring keyword-only and semantic-only matches surface correctly.
4. **Qdrant tuning for scale**
   - Owners: Infrastructure
   - Steps:
     1. Enable on-disk vectors/payloads, tune HNSW parameters, and index frequently filtered payload fields.
     2. Configure monitoring for ingestion throughput, query latency, and recall sampling.
      - [done] Qdrant tuning switches exposed via `QDRANT_ON_DISK_*` and `QDRANT_HNSW_*` environment variables.
      - [done] Baseline metrics logger enabled with `OBSERVABILITY_METRICS_ENABLED`/`OBSERVABILITY_NAMESPACE` capturing ingestion and retrieval timings.

## Phase 4 – Retrieval Quality & Personalization (Week 4)

1. **Hybrid + rerank retrieval**
   - Owners: Backend
     - Steps:
       1. Introduce keyword BM25 (e.g., Qdrant full-text) alongside vector search.
       2. Experiment with rerankers (Cohere Rerank or open-source) and A/B evaluate answer quality.
      - [done] Configurable embedding-based reranker with `RERANKER_STRATEGY=embedding`; optional cross-encoder support.
2. **Dynamic prompt tuning**
   - Owners: Backend + PM
   - Steps:
     1. Parameterize glossary depth, top-K snippets per persona.
     2. Add temperature and max tokens controls surfaced in UI.
3. **Glossary management UX**
   - Owners: Frontend
   - Steps:
     1. Provide inline glossary editor per project.
     2. Allow tagging glossary entries with keywords for retrieval weighting.

## Phase 5 – Analytics & Observability (Week 5)

1. **Conversation logging service**
   - Owners: Backend
   - Steps:
     1. Persist chat transcripts with project + persona metadata.
     2. Anonymize sensitive fields; enforce retention policy.
2. **Admin analytics dashboard MVP**
   - Owners: Frontend + Data
   - Steps:
     1. Display daily conversation counts, resolution rate (heuristic), token usage.
     2. Surface top queries and unanswered questions list.
3. **Telemetry instrumentation**
   - Owners: Engineering
   - Steps:
     1. Add tracing/timing around ingestion + query pipeline.
     2. Forward metrics to Prometheus/Grafana or hosted alternative.

## Phase 6 – Scalability & Multi-Tenancy Hardening (Week 6+)

1. **Project storage refactor**
   - Owners: Backend
   - Steps:
     1. Migrate project + document metadata from JSON to SQLite/PostgreSQL.
     2. Implement migrations and data seeding scripts.
2. **Vector store optimization**
   - Owners: Backend
   - Steps:
     1. Evaluate consolidating collections vs. metadata filtering under load.
     2. Add automated cleanup routines for stale projects.
3. **Role-based access control**
   - Owners: Backend + Frontend
   - Steps:
     1. Introduce user accounts with roles (admin, editor, viewer).
     2. Gate project CRUD + ingestion endpoints by role.
4. **Massive corpus operations**
   - Owners: Backend + Infrastructure
   - Steps:
     1. Promote the ingestion queue to a horizontally scalable broker (Redis Streams baseline; evaluate Kafka when throughput exceeds 10k docs/hr) with autoscaled workers and backpressure safeguards.
     2. Implement tiered storage policies (hot SSD, warm object store) with deduplication, versioning, and purge/archival tooling for compliance-driven deletes.
     3. Benchmark hybrid retrieval on corpora exceeding 1M chunks, tuning ANN + RRF parameters, warming caches for top intents, and alerting on relevance drift.
     4. Track per-tenant storage, compute, and token spend with capacity alerts; publish scale-event runbooks covering bulk imports, capacity adds, and partial outages.

## Supporting Tracks

- **Documentation**: Maintain updated setup + architecture guides after each major feature drop.
- **Testing**: Expand integration tests to cover persona prompts, new file types, analytics endpoints.
- **Design Review**: Host weekly UX review to align on polish and copy updates.

## Acceptance Criteria for Completion

- Persona-aware chats, multi-file ingestion, and analytics dashboard are demonstrable with automated tests where applicable.
- Observability baseline established (dashboards + alerts).
- Migration pathways documented for future contributors.
