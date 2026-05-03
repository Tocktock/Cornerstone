# Cornerstone CLI Guide

## Install

From the backend project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The package installs the `cornerstone` command.

```bash
cornerstone version
cornerstone doctor
```

## Local environment

Create a local `.env` file:

```bash
cornerstone env init
```

The generated `.env` is for local development only. Do not commit it.

## PostgreSQL stack

Start local PostgreSQL and run migrations:

```bash
cornerstone stack up --migrate
cornerstone db check-extensions
```

Stop it:

```bash
cornerstone stack down
```

## API and worker

Run API:

```bash
cornerstone api --reload
```

Run one worker pass:

```bash
cornerstone worker --once --run-scheduler --max-jobs 10
```

## Read backend state

After the API is running:

```bash
cornerstone status
cornerstone context query "What is Cornerstone?"
cornerstone eval summary
```

## Live proof helpers

PostgreSQL:

```bash
export RUN_POSTGRES_TESTS=1
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone_live_proof'
cornerstone live postgres
```

Notion:

```bash
export RUN_NOTION_E2E=1
export NOTION_MOCK_EXTERNAL_API=false
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-long-local-proof-secret-32chars-plus'
export NOTION_E2E_ACCESS_TOKEN='<your-notion-access-token>'
export NOTION_E2E_PAGE_ID='<your-shared-notion-page-id>'
export NOTION_E2E_REQUIRE_EVIDENCE=1
cornerstone live notion
```

Do not paste tokens into chat, reports, or committed files.

## Dry-run examples

```bash
cornerstone stack up --migrate --dry-run
cornerstone db migrate --dry-run
cornerstone worker --once --run-scheduler --max-jobs 1 --dry-run
```

## v1.1.1 Product-Loop Commands

The CLI now supports the release proof loop without curl.

```bash
cornerstone source list
cornerstone evidence queue
cornerstone evidence review <evidence-id> --reviewer reviewer@example.com
cornerstone concept create-from-evidence <evidence-id> --name "Cornerstone" --definition "Cornerstone is a shared organizational context layer." --created-by reviewer@example.com
cornerstone concept officialize <concept-id> --reviewer reviewer@example.com
cornerstone ask "What is Cornerstone?"
cornerstone eval summary
```

Use `--json` on read commands when running from CI or scripts.


## v1.1.2 proof runner

Run a full proof plan:

```bash
cornerstone proof run --dry-run --all --save reports/proof-plan.json
```

Run and save a consolidated proof:

```bash
cornerstone proof run --all --continue-on-failure --markdown --save reports/proof.json
```

The report includes local checks, live PostgreSQL, live Notion, product-loop checks, safety negative checks, and secret scanning.


## v1.1.3 cross-platform setup

```bash
cornerstone setup --fix
cornerstone setup windows --json
cornerstone doctor --fix
cornerstone local reset --yes --start-after --migrate
```

Use `cornerstone setup` for OS-agnostic planning. It detects macOS, Linux, or Windows and prints the right quickstart, scripts, prerequisites, and next commands.

Safe fixes never overwrite `.env` and never delete data. Destructive local reset requires `--yes`.

## v1.1.4 maintainability helpers

The CLI is now split into modules under `src/cornerstone/cli/` so command parsing, command handlers, proof helpers, config, completion, and output helpers are easier to review.

Local non-secret profile values can be configured with:

```bash
cornerstone config get
cornerstone config set baseUrl http://localhost:8000
cornerstone config set defaultReviewer reviewer@example.com
cornerstone config set defaultReportsDir reports
cornerstone config unset defaultReviewer
cornerstone config path
```

The config file is stored at `~/.cornerstone/config.json` unless `CORNERSTONE_CONFIG_DIR` is set. Do not store Notion tokens or provider credentials in this config file.

Shell completion helpers:

```bash
cornerstone completion zsh
cornerstone completion bash
cornerstone completion powershell
```


## Google Drive commands

Google Drive uses the same connector/source workflow as Notion once the source is connected.

```bash
cornerstone live google-drive
cornerstone source list
cornerstone source objects <google-drive-source-id>
cornerstone source sync <google-drive-source-id>
cornerstone worker --once --run-scheduler --max-jobs 1
```

v1.2.0 supports selected Google Docs and plain text files. Sheets, Slides, PDFs, folders, and unsupported binary files are discovery-only until extraction policy is defined.
