# Connector Framework + Notion Skeleton v0.4.0

## Goal

Build the backend connector foundation before adding live Notion page/block ingestion.

The goal of this slice is to make Source Studio easy to connect and honest to operate:

```text
connector catalog
→ connection intent
→ OAuth redirect/callback
→ encrypted credential
→ source test
→ source selection
→ sync job lifecycle
→ audit/log visibility
```

This slice intentionally does **not** fabricate Notion content. The Notion adapter can complete mocked OAuth/test flows and returns zero normalized objects until the next slice implements live page/block fetching.

## Product UX contract

The UI should be able to render these screens directly from backend state:

```text
1. Connector catalog
2. Notion connector detail
3. OAuth redirect/loading
4. OAuth success/failure
5. Test connection result
6. Source selection
7. First sync job progress
8. Sync history/events
9. Disconnect/reconnect
```

The backend must never say real content is fresh when the connector has not ingested content yet. In v0.4.0, Notion sync jobs can succeed with zero artifacts while leaving `contentFreshnessState=unknown`.

## New API endpoints

```http
GET  /v1/connectors
GET  /v1/connectors/{provider}

POST /v1/connections/intents
GET  /v1/connections/intents/{intent_id}

GET  /v1/oauth/{provider}/authorize?intentId=...
GET  /v1/oauth/{provider}/callback?state=...&code=...
GET  /v1/oauth/{provider}/callback?state=...&error=...

GET  /v1/sources/{source_id}
POST /v1/sources/{source_id}/test

GET  /v1/sources/{source_id}/selections
PUT  /v1/sources/{source_id}/selections

POST /v1/sources/{source_id}/sync-jobs
GET  /v1/sources/{source_id}/sync-jobs
GET  /v1/sync-jobs/{sync_job_id}
GET  /v1/sync-jobs/{sync_job_id}/events
POST /v1/sync-jobs/{sync_job_id}/cancel
POST /v1/sync-jobs/{sync_job_id}/retry

POST /v1/sources/{source_id}/disconnect
```

## Connector catalog

`GET /v1/connectors` returns provider cards with:

```text
provider
displayName
description
authType
availability
productionReady
requiredScopes
optionalScopes
supportedObjects
setupSteps
limitations
```

Notion is marked available. Slack, Google Docs, and GitHub are cataloged as coming soon so the UI can show roadmap-ready cards without implementation hooks.

## Connection intent

`POST /v1/connections/intents` creates a short-lived OAuth intent with:

```text
stateNonce
redirectUri
requestedScopes
authorizationUrl
expiresAt
status
```

The state nonce is persisted and validated on callback. Expired, completed, failed, or provider-mismatched intents are rejected.

## Credential boundary

Connector credentials are encrypted before persistence:

```text
connector_credentials.encrypted_access_token
connector_credentials.encrypted_refresh_token
```

Plaintext provider tokens are only available inside adapter calls. API responses expose `ConnectorCredentialPublic`, which omits encrypted and plaintext token fields.

For local development this uses a Fernet key derived from `CONNECTOR_ENCRYPTION_SECRET`. Production should replace or wrap this with managed KMS/secrets infrastructure.

## Source selection

Sync jobs require source selection before first connector sync when `syncMode=selected_only`.

```json
{
  "syncMode": "selected_only",
  "selectedExternalObjectIds": ["notion-page-1"]
}
```

This prevents the product from silently syncing too much data after OAuth.

## Sync jobs

Connector sync uses durable jobs and events:

```text
queued → running → succeeded | failed | cancelled
```

Events include:

```text
sync.job_queued
sync.job_started
sync.job_succeeded
sync.job_failed
sync.job_cancelled
```

The job model already includes counts for created artifacts, reused artifacts, evidence fragments, error payloads, and future cursor storage.

## Persistence additions

Migration `0002_connector_framework.py` adds:

```text
connection_intents
connector_credentials
source_selections
sync_jobs
sync_job_events
```

These tables use PostgreSQL-compatible UUIDs and JSONB fields while still supporting SQLite contract tests.

## Notion skeleton

Implemented now:

```text
- OAuth authorization URL builder
- Mock OAuth token exchange for tests/local dev
- Live OAuth token exchange boundary
- Mock connection test
- Live /search connection test boundary
- Provider error mapping
- Notion-Version header in live requests
- Bearer Authorization header in live requests
- Rate-limit error mapping to wait_and_retry
```

Deferred to the next connector slice:

```text
- Notion page/database discovery
- page/block content fetching
- block tree flattening
- database property extraction
- incremental cursor/checkpointing
- webhook handling
- provider retry/backoff runtime
```

## Tests added

```text
Unit:
- connector catalog exposes Notion and future providers
- token encryption round trip
- wrong secret fails decrypt
- Notion OAuth URL includes required state
- Notion provider errors map to actionable errors
- connector persistence tables exist
- credential table exposes only encrypted token fields

Integration:
- connector catalog API
- connection intent + OAuth redirect state
- bad OAuth state rejected
- provider OAuth error marks intent failed
- OAuth success creates source + public credential
- test connection marks source connected
- sync job requires selection
- source selection save/read
- sync job lifecycle with no fake content freshness
- disconnect revokes credentials
- SQLAlchemy persistence survives new store instance
```

## Local usage

```bash
export CONNECTOR_ENCRYPTION_SECRET=local-dev-only-change-me-secret
export NOTION_MOCK_EXTERNAL_API=true
uvicorn cornerstone.main:app --reload
```

Live Notion mode requires:

```bash
export NOTION_MOCK_EXTERNAL_API=false
export NOTION_CLIENT_ID=...
export NOTION_CLIENT_SECRET=...
export CONNECTOR_OAUTH_CALLBACK_URL=http://localhost:8000/v1/oauth/notion/callback
```

## Exit criteria

```text
- UI can render connector cards from backend catalog.
- OAuth success/failure is stateful and persisted.
- Tokens are encrypted and never returned by API responses.
- Source test clearly distinguishes authorization from actual provider readability.
- Source selection is required before first sync.
- Sync jobs and events are persisted and observable.
- Notion skeleton does not fabricate content or freshness.
```
