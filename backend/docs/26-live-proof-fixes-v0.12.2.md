# v0.12.2 — Live Proof Fixes and Release-Gate Record

## Goal

Package the fixes discovered during the manual live proof so the release artifact matches the code that passed the backend proof gate.

The proof gate validates the product loop:

```text
Live PostgreSQL
→ live Notion page
→ Artifact
→ EvidenceFragment
→ evidence review
→ official Concept
→ grounded context response
→ evaluation result
→ grounded_context_task_success_rate
```

## Fixes

1. Widened `alembic_version.version_num` to `VARCHAR(128)` in the first migration so descriptive revision IDs are accepted by live PostgreSQL.
2. Updated live PostgreSQL concurrency fixtures to use valid UUID strings for source IDs.
3. Updated live Notion runner and gated live Notion test to use `Settings.from_env()` so sourced environment variables are honored.
4. Replaced the fixed mock Notion `last_edited_time` with a current UTC Notion-style timestamp to avoid time-dependent freshness failures.
5. Added a live-proof change log under `docs/live-proof-records/2026-04-27-change-log.md`.

## Verification status

The user-run live proof passed against the locally patched v0.12.1 workspace before this package was created. The packaged v0.12.2 artifact incorporates those local fixes.

See: `docs/live-proof-records/2026-04-27-change-log.md`.

## Known limitations

- Notion page ingestion is proven for the pilot page.
- Notion database/data_source ingestion remains intentionally deferred.
- Slack, Google Docs, and GitHub connectors remain post-1.0.0 scope.
- Production KMS/secret-manager integration remains a future hardening item unless required by deployment policy.
