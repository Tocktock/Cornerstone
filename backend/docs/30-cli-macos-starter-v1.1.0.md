# v1.1.0 — CLI and macOS Integration Starter Kit

## Goal

Make the v1.0 backend MVP easy to run from a terminal before building a UI.

This version keeps product scope small:

```text
backend API
+ CLI wrapper
+ macOS setup scripts
+ integration starter docs
```

It does not add new connectors or new trust semantics.

## Why CLI First

The backend MVP is live-proven, but curl-only operation is too hard for a pilot.
A CLI gives operators and reviewers a repeatable path for:

```text
local setup
PostgreSQL startup
migrations
worker execution
live proof
source status inspection
grounded context query
evaluation summary
```

This directly supports the MVP loop:

```text
source -> Artifact -> EvidenceFragment -> review -> official context -> grounded response -> evaluation
```

## Added

```text
src/cornerstone/cli.py
scripts/macos_setup.sh
scripts/macos_start_local.sh
scripts/macos_run_live_proof.sh
docs/integration-starter-kit/cli-guide.md
docs/integration-starter-kit/macos-quickstart.md
docs/integration-starter-kit/notion-live-proof.md
```

## CLI Commands

```bash
cornerstone version
cornerstone doctor
cornerstone env init
cornerstone stack up --migrate
cornerstone db migrate
cornerstone db check-extensions
cornerstone api --reload
cornerstone worker --once --run-scheduler --max-jobs 10
cornerstone live postgres
cornerstone live notion
cornerstone status
cornerstone context query "What is Cornerstone?"
cornerstone eval summary
```

## macOS Starter Scripts

```bash
./scripts/macos_setup.sh
./scripts/macos_start_local.sh
./scripts/macos_run_live_proof.sh
```

## Non-goals

```text
No Slack connector.
No Google Docs connector.
No GitHub connector.
No UI.
No new sync semantics.
No production secret-manager integration.
```

## Verification

The CLI is intentionally stdlib-only. It can be tested without starting PostgreSQL or Notion by using dry-run commands.

