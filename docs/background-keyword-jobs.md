# Background Keyword Job Architecture

## Context

The `/keywords/{project}/candidates` endpoint currently executes the entire keyword pipeline inline. For large tenants, Stage 2 (concept extraction + embeddings) and Stage 3/4 (clustering, ranking, LLM filtering) can run for minutes or hours, exceeding browser timeouts and tying up FastAPI workers. Even when the server finishes, the UI shows “Unable to load keywords…” because the fetch call aborted. Every page load re-runs the expensive pipeline, and only Stage 7 (insight summarisation) uses the lightweight `KeywordInsightQueue`.

## Goals

- **Responsive UI** – Show the latest keyword set instantly, regardless of project size.
- **Asynchronous execution** – Offload heavy keyword stages to background workers with progress reporting.
- **Result caching** – Persist successful runs and reuse them until new content is ingested or a refresh is requested.
- **Operational visibility** – Expose job status, metrics, and failures to operators.
- **Configurable guardrails** – Retain candidate/token/chunk limits to prevent runaway jobs.

## Proposed Architecture

### 1. Keyword Run Data Model

Create a `KeywordRun` record stored per project (JSON files managed by `ProjectStore`):

```json
{
  "id": "20250117T105500Z",
  "projectId": "full-document-test",
  "status": "success",
  "requestedAt": "2025-01-17T10:55:00Z",
  "startedAt": "2025-01-17T10:55:03Z",
  "completedAt": "2025-01-17T11:01:40Z",
  "requestedBy": "user@example.com",
  "stats": {
    "chunkTotal": 37945,
    "tokenTotal": 3108835,
    "candidateTotal": 103019,
    "bypassReason": "chunk-limit",
    "durationSeconds": 397,
    "llmBackend": "ollama"
  },
  "keywords": [...],
  "insights": [...],
  "debug": {...},
  "error": null
}
```

Persist each run in `data/keyword_runs/<project_id>/<run_id>.json` and maintain a `latest.json` pointer for quick access. On startup, load the newest successful run into memory for fast responses.

### 2. Keyword Job Queue

Introduce `KeywordRunQueue` mirroring `KeywordInsightQueue` but handling full Stage 1–6 execution:

- Backed by `asyncio.Queue` plus disk metadata so restarts can recover queued/running jobs.
- Configurable concurrency (`KEYWORD_RUN_MAX_CONCURRENCY`, default 1) and queue size (`KEYWORD_RUN_MAX_QUEUE`).
- Each job is an instance of `KeywordRunJob` with asyncio Event signaling, status transitions (`pending → running → success/error`).
- Worker coroutine launched from `create_app()` consumes jobs, runs the pipeline inside `asyncio.to_thread` (or a thread pool), updates run metadata, and persists results.
- On failure, capture exception message and store partial debug payload.

### 3. API Surface

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/keywords/{project}/runs` | `POST` | Start a new keyword run. Returns `{jobId, latest}` where `latest` is the most recent cached result (if any). Supports `force=true` to bypass freshness checks. |
| `/keywords/{project}/runs/latest` | `GET` | Fetch the latest successful run (keywords, stats, insights). Fast path for initial page render. |
| `/keywords/{project}/runs/{jobId}` | `GET` | Poll job status. Returns metadata (`status`, timestamps, progress stats, bypass flags). |
| `/keywords/{project}/runs/history` | `GET` | Paginated history for admin/debug UI. |

Existing `/keywords/{project}/candidates` becomes a compatibility wrapper:

- `GET` → returns the latest run; when async mode is enabled and no run exists it responds with `409` so the client can enqueue a job.
- `POST` → proxy to `runs`.

During migration, keep the env flag (`KEYWORD_RUN_SYNC_MODE`) to fall back to inline execution if needed.

### 4. Worker Execution Flow

1. Worker dequeues job, sets `status=running`, records `startedAt`.
2. Runs existing pipeline (chunking → concept extraction → clustering → ranking → LLM filtering) via a reusable function `generate_keyword_run(...)` extracted from the current route.
3. Applies guardrails (candidate/token/chunk limits). If bypass triggered, propagate reason in `stats`.
4. Runs Stage 7 insight summarisation inline by instantiating `KeywordLLMFilter` (or by reusing `KeywordInsightQueue` synchronously).
5. Persists run JSON with keywords, insights, debug payload.
6. Updates `status=success` and notifies waiters. On exception, set `status=error` and capture `error` + debug snippet.

### 5. Frontend Changes

- Surface a `keywordRunAsyncEnabled` flag so the template swaps between synchronous and async flows.
- When async mode is on, `/keywords/{project}/candidates` serves cached results (and returns 409 when no run exists) so the UI can show the previous keywords immediately.
- Replace the legacy “Search keywords” button with “Run Keyword Scan”, which enqueues a job (`POST /keywords/{project}/runs`), disables itself while pending, and polls `/runs/{jobId}` for status updates.
- Reuse the existing stage-7 insight card: any `insightJob` payload triggers the existing polling logic so summarisation completes asynchronously.
- Preserve synchronized fallback (when `KEYWORD_RUN_SYNC_MODE=1`) for local tests and lightweight installs.

### 6. Integration with Ingestion

- `DocumentIngestor` now calls the `KeywordRunAutoRefresher` after each successful ingest so projects marked “dirty” are queued automatically.
- With `KEYWORD_RUN_AUTO_REFRESH=1`, the refresher deduplicates in-flight jobs and schedules a follow-up run if more content lands while a scan is running.
- Maintain TTL (`KEYWORD_RUN_CACHE_TTL`, default 24h) so stale results trigger automatic refresh on next visit.

### 7. Metrics & Observability

- Emit metrics via `MetricsRecorder`:
  - `keyword.run.enqueued` (counter by project).
  - `keyword.run.duration` (histogram, tags: project, status, bypass_reason, llm_backend).
  - `keyword.run.errors` (counter for failed runs).
  - `keyword.run.queue_time` (time from request to start).
  - `keyword.run.active` (gauge: number of running jobs).
- Keyword run stats include `batch_total`, `batches_completed`, `candidates_processed`, `poll_after_ms`, and `keywords_total`, enabling the UI and operators to surface mid-run progress and tune polling cadence.
- Log transitions with structured metadata for advanced tracing.

### 8. Configuration

Add to `Settings` and `env.example.local`:

- `KEYWORD_RUN_MAX_CONCURRENCY` (default `1`).
- `KEYWORD_RUN_MAX_QUEUE` (default `8`).
- `KEYWORD_RUN_CACHE_TTL` in seconds (default `86400`).
- `KEYWORD_RUN_AUTO_REFRESH` (`0`/`1`).
- `KEYWORD_RUN_SYNC_MODE` (escape hatch—set to `0` to enable the async queue).

### 9. Testing Strategy

- **Unit tests** for queue lifecycle: enqueue → running → success/error persistence.
- **Integration tests** with fake embedding service to ensure cached results served instantly and background job completes via event loop tasks (use `pytest-asyncio`).
- **Frontend tests** (Jest or Playwright if added) verifying polling logic handles status states.
- **Load test** using synthetic 40 k chunk dataset to validate worker throughput and guardrail triggers.

### 10. Migration Plan

1. Implement queue + storage + API behind feature flag.
2. Update frontend to consume new endpoints while keeping fallback to old route when flag disabled.
3. Deploy and monitor metrics; once stable, retire synchronous mode.
4. Document new workflow in README and admin guides.

## Open Questions

- Do we need multi-worker support across processes? Initial version can run single worker per FastAPI instance; later we can move to external worker (e.g., an async service launched via separate process or Celery/RQ).
- Should partial progress (Stage 2 frequency list) be streamed before job completion? Optional enhancement after initial rollout.
- Where to store large debug payloads? We can keep them truncated or separate from main run JSON if size becomes an issue.

## Next Steps

1. Stress-test the queue with large corpora (40 k+ chunks) and tune concurrency/guardrails based on `keyword.run.*` metrics.
2. Explore multi-worker or out-of-process execution so keyword runs can scale beyond a single FastAPI instance.
3. Evaluate streaming partial keyword results (e.g., Stage 2 frequency list) for exceptionally large projects.
4. Consider trimming or externalising very large debug payloads to keep run records lightweight.

This architecture keeps the keyword explorer responsive while scaling to tens of thousands of documents, and provides operators with the observability needed to manage long-running runs.
