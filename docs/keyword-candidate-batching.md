# Keyword Candidate Batching Design

## Context

- Large datasets (e.g., `full-document-test3`) produce 100k+ Stage 2 candidates.
- `KeywordLLMFilter` currently caps input at 25k entries; excess triggers the `candidate-limit` bypass.
- When refinement is skipped, Stage 3 clustering collapses to a single mega-cluster and downstream ranking degrades.
- Job status lacks mid-run progress, hiding long-running activity and making operations hard to monitor.

## Goals

1. **No candidate loss:** Every Stage 2 candidate must traverse refinement, clustering, and ranking.
2. **Deterministic outputs:** Multi-batch runs should match existing single-batch scoring semantics.
3. **Observable progress:** Background jobs must expose batch progress (counts, durations) to `/keywords/*` APIs and UI polling.
4. **Configurable throughput:** Allow tuning without code changes to handle different LLM limits or cost envelopes.

## Non-Goals

- Replacing the seven-stage keyword pipeline.
- Altering ranking heuristics beyond batched aggregation.
- Implementing distributed workers or resumable partial runs.

## Proposed Changes

### 1. Configuration

- Add to `Settings`:
  - `KEYWORD_CANDIDATE_BATCH_SIZE` (default: 5000).
  - `KEYWORD_CANDIDATE_BATCH_MIN_SIZE` (default: 500) for adaptive retry.
  - Optional `KEYWORD_CANDIDATE_BATCH_OVERLAP` to carry the top N across batches (default: 0).
- Surface env vars in `.env`, `env.example.local`, and docs.

### 2. Batch Orchestration

- Introduce `iter_candidate_batches(candidates, batch_size, overlap)` in `keywords.py`.
- Ensure deterministic ordering:
  1. Sort candidates by Stage 2 score + document fingerprint (existing order).
  2. Slice contiguous windows with optional overlap.
- Track `batch_index`, `batch_total`, `candidates_processed` during iteration.

### 3. Candidate Accumulator

- New helper class (e.g., `_CandidateAccumulator`) keyed by normalized concept ID.
- Aggregates:
  - Occurrence counts (chunks, documents).
  - Score components (frequency, chunk, embedding, LLM, etc.).
  - Representative aliases/sections (bounded list, e.g., top 5).
- After each batch, merge into accumulator and drop per-batch structures to limit memory usage.
- After all batches, run existing Stage 4 ranking logic on the merged concepts to preserve deterministic ordering.

### 4. LLM Integration

- Wrap `KeywordLLMFilter.filter_candidates` with batch context:
  - Include `batch_index`, `batch_size`, `candidates_in_batch` in logs/metrics.
  - On `candidate-limit`, halve the batch (down to `MIN_SIZE`) and retry.
  - If minimum size still fails, mark batch with `filter-stage2.candidate-limit` and continue (preserving today’s bypass messaging).
- Reuse `LLMFilter` prompts/client to avoid reconnect cost.

### 5. Progress Reporting & Metrics

- Extend `KeywordRunResult.stats` in `ProjectStore` to include:
  - `batch_total`, `batch_completed`, `candidates_total`, `candidates_processed`.
  - `last_batch_duration_ms`.
- Update `_serialize_keyword_run` (FastAPI) and UI polling (JS in `templates/keywords.html`) to render incremental progress.
- Metrics (`observability.MetricsRecorder`):
  - Counter: `keyword.batch.processed{project_id}`.
  - Histogram: `keyword.batch.duration_seconds`.
  - Gauge: `keyword.batch.inflight`.

### 6. API & UI

- `/keywords/{project}/runs/{id}` response includes new stats.
- `/keywords/{project}/runs` initial payload surfaces `batch_total`.
- UI progress card shows e.g., “Processing batch 3/18 (15,000 / 103,021 candidates)”.
- Update `docs/background-keyword-jobs.md` with batching behaviour.

### 7. Testing

- Unit tests:
  - Batch iterator deterministic slicing (with/without overlap).
  - Accumulator merges produce identical results as single-batch baseline.
  - LLM retry path reduces batch size and flags bypass correctly.
- Integration tests:
  - Multi-batch run on synthetic dataset verifying API stats.
  - UI polling rendering progress JSON (JS unit test or Playwright scenario).
- Regression: confirm existing small dataset runs remain unchanged.

## Open Questions

1. Should overlap be enabled by default to aid clustering, or is deterministic slice sufficient?
2. Do we need to persist batch-level debug payloads for post-mortem analysis?
3. How granular should UI charts be (per batch timeline vs. textual stats)?

## Rollout

1. Implement batching behind settings with defaults that mimic current single-batch behaviour for small corpora.
2. Ship with metrics + logging to validate in test environments.
3. Update operator runbooks and UI copy.
4. Monitor large-project runs (e.g., `full-document-test3`) for improved quality; adjust batch size based on telemetry.
