# macOS Quickstart

> v1.1.3 note: macOS now uses the shared Unix starter scripts. See `local-quickstart.md` for the OS-agnostic flow.

# Cornerstone macOS Quickstart

## Requirements

```text
macOS
Python 3.13+
Docker Desktop for Mac
jq, recommended
```

Install optional tooling with Homebrew:

```bash
brew install python@3.13 jq
```

Install Docker Desktop from Docker, then start it.

## Setup

From the project root:

```bash
./scripts/setup_local.sh
```

This will:

```text
check Python and Docker
create .venv
install dev dependencies
create .env if missing
run cornerstone doctor
```

## Start local database

```bash
./scripts/start_local.sh
```

This will:

```text
start PostgreSQL with pgvector
run Alembic migrations
check pgcrypto/citext/vector extensions
print next commands
```

## Start API

In a new terminal:

```bash
source .venv/bin/activate
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone'
cornerstone api --reload
```

Check:

```bash
cornerstone status
```

## Run worker

In another terminal:

```bash
source .venv/bin/activate
cornerstone worker --once --run-scheduler --max-jobs 10
```

## Run local tests

```bash
./scripts/run_tests.sh
```

## Run live proof

For PostgreSQL only:

```bash
./scripts/run_live_proof.sh
```

For PostgreSQL + Notion, set Notion environment variables first. See `notion-live-proof.md`.

## Troubleshooting

### Docker not running

Start Docker Desktop and rerun the script.

### Python too old

Install Python 3.13+ and recreate `.venv`.

### vector extension missing

Use the included `docker-compose.yml`, which uses the pgvector PostgreSQL image.

### API cannot import cornerstone

Activate the virtual environment and run `pip install -e '.[dev]'`.
