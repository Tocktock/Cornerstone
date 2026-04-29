# Backend v0.6.1 — SDK-backed Notion Gateway

## Goal

v0.6.1 corrects the connector implementation strategy: Cornerstone should not hand-write provider API plumbing when stable provider SDKs exist.

The backend now uses this boundary:

```text
Notion SDK / provider API
→ NotionGateway
→ NotionConnector adapter
→ SourceObject
→ Artifact
→ EvidenceFragment
```

The Notion-specific API access is isolated behind gateway and mapper modules. Cornerstone-specific trust behavior remains in the connector runtime, source state model, sync jobs, persistence, Artifact/Evidence creation, review gates, and officialization gates.

## What changed

```text
src/cornerstone/connectors/providers/notion/
  adapter.py   # Cornerstone-facing Notion connector
  gateway.py   # NotionGateway protocol, OAuth boundary, SDK-backed gateway
  mapper.py    # Notion payload to ProviderObjectSnapshot/SourceObject mapping
```

The compatibility import remains:

```text
src/cornerstone/connectors/notion.py
```

so existing registry/routes continue to resolve `NotionConnector` without route changes.

## Provider access strategy

- OAuth token exchange remains isolated in `NotionOAuthClient` because OAuth exchange is not the core job of `notion-client`.
- Authenticated Notion API calls are routed through `NotionGateway`.
- Live mode defaults to `NotionSdkGateway`.
- Mock mode defaults to `MockNotionGateway`.
- The page markdown endpoint remains a narrow HTTP fallback inside `NotionSdkGateway` because SDK coverage may lag newer Notion endpoints.

## Why this matters

This protects the backend from two opposite risks:

1. Rebuilding provider API clients from scratch.
2. Outsourcing Cornerstone's product-specific trust model to a generic integration tool.

The SDK handles Notion API mechanics. Cornerstone still owns:

```text
Source state
Source selection
Sync jobs
Artifact idempotency
Evidence provenance
Freshness
Review state
Officialization gates
Grounded serving
Audit logs
```

## Tests added

```text
- Default live Notion gateway is SDK-backed.
- NotionGateway is protocol-checked.
- OAuth exchange is isolated behind its own boundary.
- OAuth failures map to actionable ConnectorError values.
- Connection test uses the gateway search contract.
- Discovery maps pages/databases/data_sources via the mapper.
- Rate limits map to retryable wait_and_retry errors.
- Selected page ingestion uses gateway page + markdown retrieval.
- Block traversal fallback works without leaking Notion logic into core sync.
- SDK gateway calls notion-client for search/page/block APIs.
- SDK gateway converts SDK exceptions into NotionAPIResponseError.
- Markdown fallback maps HTTP success, missing endpoint, and rate-limit responses.
```

## Verification

```text
117 passed
coverage: 86%
ruff: passed
mypy: passed
compileall: passed
Alembic offline SQL rendered
```

## Still not done

```text
- Real Notion workspace E2E test
- Durable background sync worker
- Retry scheduling execution
- Token refresh/revoke lifecycle hardening
- PostgreSQL live CI
- Notion database/data_source ingestion
```
