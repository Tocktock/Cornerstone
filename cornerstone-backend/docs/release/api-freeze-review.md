# API Freeze Review

## Purpose

Before backend v1.0.0, freeze the MVP API surface that supports the proven loop.

## Frozen MVP surfaces

```text
GET  /healthz
GET  /v1/sources
POST /v1/sources                         # manual sources only
POST /v1/manual-sources/{sourceId}/sync
GET  /v1/connectors
GET  /v1/connectors/{provider}
POST /v1/connections/intents
GET  /v1/connections/intents/{intentId}
GET  /v1/oauth/{provider}/authorize
GET  /v1/oauth/{provider}/callback
POST /v1/sources/{sourceId}/test
POST /v1/sources/{sourceId}/discover
GET  /v1/sources/{sourceId}/objects
GET  /v1/sources/{sourceId}/selections
PUT  /v1/sources/{sourceId}/selections
POST /v1/sources/{sourceId}/sync-jobs
GET  /v1/sources/{sourceId}/sync-jobs
GET  /v1/sync-jobs/{syncJobId}
POST /v1/sync-jobs/{syncJobId}/run
POST /v1/sync-jobs/{syncJobId}/cancel
POST /v1/sync-jobs/{syncJobId}/heartbeat
POST /v1/sync-worker/run
GET  /v1/artifacts
GET  /v1/evidence
GET  /v1/evidence/review-queue
POST /v1/evidence/{evidenceFragmentId}/review
POST /v1/evidence/{evidenceFragmentId}/concept-candidates
GET  /v1/concepts
POST /v1/concepts
GET  /v1/concepts/{conceptId}
POST /v1/concepts/{conceptId}/officialize
GET  /v1/concept-relations
POST /v1/concept-relations
POST /v1/concept-relations/{relationId}/officialize
GET  /v1/decision-records
POST /v1/decision-records
GET  /v1/context/query
POST /v1/evaluations/tasks
POST /v1/evaluations/tasks/{taskId}/run
GET  /v1/evaluations/summary
GET  /v1/audit-events
```

## Removed or intentionally unavailable surfaces

```text
POST /v1/sources/{sourceId}/sync                 # removed; legacy bypass
POST /v1/sources/{sourceId}/oauth/complete       # removed; fake OAuth path
POST /v1/sources with type=notion                # rejected; provider source must use connector flow
POST /v1/manual-sources/{notionSourceId}/sync    # rejected; manual sync is manual-only
```

## Snapshot protection

OpenAPI snapshots currently protect:

```text
- grounded serving contract
- evaluation contract
```

Before v1.0.0, run:

```bash
./scripts/run_tests.sh
```

and confirm the OpenAPI contract snapshot tests pass.

