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
  - `KEYWORD_CANDIDATE_BATCH_SIZE` (default: 160).
  - `KEYWORD_CANDIDATE_BATCH_MIN_SIZE` (default: 80) for adaptive retry.
  - Optional `KEYWORD_CANDIDATE_BATCH_OVERLAP` to carry the top N across batches (default: 0).
- Surface env vars in `.env`, `env.example.local`, and docs.

### 2. Batch Orchestration

- Introduce `iter_candidate_batches(candidates, batch_size, overlap)` in `keywords.py`.
- Maintain existing Stage 2 ordering; slicing is deterministic to keep results reproducible.
- Track `batch_index`, `batch_total`, and `candidates_processed` while iterating.
- Accumulate refined candidates from each batch into a single list used by the downstream clustering/ranking stages. This keeps Stage 3/4 behaviour aligned with the current single-pass pipeline while still respecting LLM limits.

### 3. LLM Integration

- Update `KeywordLLMFilter.refine_concepts` usage so refinement happens per batch rather than in one large call.
- Include `batch_index`, `batch_size`, and `candidate_count` in logs/metrics.
- Clamp the per-request batch size so it never exceeds `KEYWORD_LLM_MAX_CANDIDATES`, avoiding the historical `candidate-limit` bypass.
- When LLM support is disabled or bypassed (e.g., token/chunk limit), the pipeline falls back to the unrefined candidates but still reports batch stats.

### 4. Candidate Aggregation

- Accumulate refined candidates from each batch and feed the combined list to `cluster_concepts` and `rank_concept_clusters` exactly once.
- This keeps scoring deterministic relative to the single-batch pipeline while allowing full refinement coverage.
- Deduplicate overlapped entries (same phrase + chunk/document ids) prior to Stage 3 so collisions from overlap do not inflate scores.

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
  - Progress callback and stats wiring for batched execution.
- Integration tests:
  - UI polling rendering batched progress from `/keywords/{project}/runs`.
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
