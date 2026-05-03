# Linux Quickstart

This guide runs the Cornerstone backend locally on Linux.

## Prerequisites

```text
Python 3.13+
Docker Engine
Docker Compose plugin
Git
```

## Setup

```bash
./scripts/setup_local.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cornerstone setup linux --fix
```

## Start

```bash
./scripts/start_local.sh
```

## Live proof

```bash
export RUN_POSTGRES_TESTS=1
export RUN_NOTION_E2E=1
export NOTION_MOCK_EXTERNAL_API=false
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-long-local-proof-secret-32chars-plus'
export NOTION_E2E_ACCESS_TOKEN='<your-token>'
export NOTION_E2E_PAGE_ID='<your-page-id>'

./scripts/run_live_proof.sh
```

Do not write the Notion token to `.env` or commit it.
