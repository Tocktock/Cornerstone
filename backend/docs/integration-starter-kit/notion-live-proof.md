# Notion Live Proof Starter

## Purpose

Prove that a real Notion page can become:

```text
ProviderObjectSnapshot
-> Artifact
-> EvidenceFragment
```

Then the normal reviewer/product loop can turn that evidence into official grounded context.

## Prepare Notion

Create a safe test Notion page and share it with the integration.

Suggested content:

```text
Cornerstone is a shared organizational context layer.
Cornerstone preserves provenance, freshness, trust labels, review state, and limitations.
```

## Environment

Do not commit tokens. Use environment variables only:

```bash
export RUN_NOTION_E2E=1
export NOTION_MOCK_EXTERNAL_API=false
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-long-local-proof-secret-32chars-plus'
export NOTION_E2E_ACCESS_TOKEN='<your-notion-access-token>'
export NOTION_E2E_PAGE_ID='<your-shared-notion-page-id>'
export NOTION_E2E_REQUIRE_EVIDENCE=1
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone_live_proof'
```

## Run

```bash
alembic upgrade head
cornerstone live notion
RUN_NOTION_E2E=1 python -m pytest tests/live_notion -m live_notion -vv --color=no
```

Pass condition:

```text
status = passed
sync_job_status = succeeded
artifact_count >= 1
evidence_fragment_count >= 1
live_notion pytest = 1 passed, 0 skipped
```

## After proof

Start the API and continue with:

```text
evidence review
Concept candidate creation
officialization
grounded context query
evaluation task
```

Use the backend operator runbook for the full product-loop proof.
