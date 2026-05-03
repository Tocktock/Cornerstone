# v1.1.2 — One-command Proof Runner

## Goal

`v1.1.2` makes the CLI useful as a release and pilot proof tool.

The backend MVP loop is already proven:

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

The CLI now lets an operator generate a consolidated proof report with one command instead of collecting many curl/script outputs manually.

## Command

```bash
cornerstone proof run --all --continue-on-failure --markdown --save reports/proof.json
```

When no scope flag is provided, `--all` is implied.

## Proof categories

The report can include:

```text
local
local_tests
live_postgres
live_notion
product_loop
safety
secret_scan
```

## Common scoped runs

Dry-run the plan:

```bash
cornerstone proof run --dry-run --all --save reports/proof-plan.json
```

Run local release checks only:

```bash
cornerstone proof run --local --local-tests --secret-scan
```

Run live source/database gates:

```bash
cornerstone proof run --postgres --notion
```

Run API product and safety gates:

```bash
cornerstone proof run --product-loop --safety-checks --base-url http://localhost:8000
```

## Report shape

JSON reports use schema version 2:

```json
{
  "schemaVersion": 2,
  "version": "1.1.2",
  "summary": {
    "status": "passed",
    "passed": 12,
    "failed": 0,
    "planned": 0,
    "skipped": 0,
    "total": 12
  },
  "categories": {
    "product_loop": {
      "passed": 7,
      "failed": 0,
      "planned": 0,
      "skipped": 0,
      "total": 7
    }
  },
  "steps": []
}
```

Optional Markdown reports can be generated with:

```bash
cornerstone proof run --all --markdown
```

## Product-loop checks

The product-loop category verifies:

```text
healthz
real Notion source presence
Artifact presence
EvidenceFragment presence
official grounded answer with valid citation
unsupported query behavior
evaluation summary success rate
```

## Safety checks

The safety category verifies:

```text
direct Notion source creation returns 409
fake OAuth completion returns 404
legacy source sync returns 404
manual sync on Notion source returns 409
weak evaluation task returns 422
```

## Secret scan

The secret scan checks source, docs, scripts, tests, reports, pyproject, and `.env.example` for Notion token patterns.

It intentionally does not read local `.env` files. Tokens must remain environment-only.

## Exit criteria

`v1.1.2` succeeds when a release/pilot proof can be generated with one command and the report clearly shows pass/fail status for local, live, product-loop, safety, and secret-scan checks.

## Limitations

```text
The proof runner orchestrates existing backend/API gates.
It does not create sources or review evidence by itself.
It assumes the API and environment are prepared for product-loop checks.
Live Notion proof still requires a real token and shared page ID.
```
