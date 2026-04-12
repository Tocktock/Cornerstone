# 2026-04-10 - Local startup backfills additive schema columns

## Context

Recent P0 iterations added required persisted fields to existing local tables:
- `decision_records.public_slug` for presentable decision routes and workspace-home cards
- `source_connections.provider_credential_ref`
- `source_connections.selected_scope_json`
- `source_connections.sync_checkpoint_json`
- `source_connections.next_scheduled_sync_at`

Local Docker development still relies on a persisted PostgreSQL volume, and backend startup only called `Base.metadata.create_all()`. That creates missing tables, but it does not add new columns to an existing table. Reusing an older local volume therefore left `decision_records` and `source_connections` in a partially upgraded shape, and the first `/api/v1/workspace-home` request failed with `UndefinedColumn`.

## Decision

Local backend startup now performs narrow compatibility repairs after `create_all()`:
- add the missing additive columns to `decision_records` and `source_connections`
- backfill deterministic decision slugs from existing titles
- suffix colliding decision slugs within the same workspace
- restore the workspace-scoped decision slug uniqueness guarantee and the direct connector credential index
- normalize missing JSON connector state to empty objects before restoring `NOT NULL`

## Why it matters

- local Docker upgrades should survive additive schema changes that are immediately required by member-facing routes
- persisted dev volumes remain usable without forcing a destructive reset for every additive field
- `./run-all.sh up --reset-db` remains the recovery path for larger schema drift, but it is no longer required for these additive column changes
