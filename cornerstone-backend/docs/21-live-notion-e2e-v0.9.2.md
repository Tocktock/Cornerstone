# v0.9.2 — Live Notion E2E Pilot Path

## Goal

v0.9.2 adds a backend-only live Notion pilot path so the first real-source connector can be verified end to end before building reviewer workflow and grounded serving.

The path is intentionally gated. Normal local tests do not contact Notion. The operator must explicitly provide a real Notion token, a specific page ID shared with that integration, and a non-default connector encryption secret.

## Scope

Implemented in this version:

```text
- NotionConnector.retrieve_page_snapshot(...)
- src/cornerstone/verification/notion_e2e.py
- scripts/run_live_notion_e2e.py
- scripts/run_notion_e2e_ci.sh
- tests/live_notion/test_live_notion_page_e2e.py
- tests/unit/test_live_notion_e2e_verification.py
```

The pilot flow is:

```text
real Notion token + page ID
→ DataSource
→ ConnectorCredential
→ connection test
→ page snapshot lookup
→ ProviderObjectSnapshot
→ SourceSelection
→ SyncJob
→ worker ingestion
→ Artifact
→ EvidenceFragment
```

## Required environment

```bash
export RUN_NOTION_E2E=1
export NOTION_MOCK_EXTERNAL_API=false
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-32-char-or-longer-secret'
export NOTION_E2E_ACCESS_TOKEN='secret_...'
export NOTION_E2E_PAGE_ID='notion-page-id-shared-with-the-integration'
```

Optional:

```bash
export NOTION_E2E_REQUIRE_EVIDENCE=1
export NOTION_E2E_SOURCE_NAME='Live Notion E2E Source'
export NOTION_E2E_WORKER_ID='notion-e2e-worker'
export NOTION_E2E_LEASE_SECONDS=300
```

## Commands

Local one-shot runner:

```bash
PYTHONPATH=src python scripts/run_live_notion_e2e.py
```

Live pytest path:

```bash
RUN_NOTION_E2E=1 python -m pytest tests/live_notion -m live_notion
```

Default suite remains safe:

```bash
./scripts/run_tests.sh
```

`run_tests.sh` ignores `tests/live_notion` unless `RUN_NOTION_E2E=1` or `RUN_LIVE_NOTION_TESTS=1` is set.

## What this does not claim

```text
- It does not prove every Notion workspace configuration.
- It does not ingest Notion databases or data_sources.
- It does not replace the OAuth browser callback flow.
- It does not prove production KMS integration.
```

## Exit criteria

The live pilot passes only if:

```text
- The supplied token can read Notion.
- The supplied page ID retrieves an accessible page.
- The page can be selected for sync.
- The production sync worker path succeeds.
- At least one Artifact is created.
- EvidenceFragments are created when NOTION_E2E_REQUIRE_EVIDENCE=1.
```
