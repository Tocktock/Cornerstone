#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/compose.yml"
ENV_FILE="$ROOT_DIR/.env"
ENV_EXAMPLE="$ROOT_DIR/.env.example"
LAUNCHER_NAME="${CORNERSTONE_LAUNCHER_NAME:-$(basename "$0")}"
LAUNCHER_PROFILE="${CORNERSTONE_LAUNCHER_PROFILE:-generic}"
COMPOSE_PROJECT_NAME="${CORNERSTONE_COMPOSE_PROJECT_NAME:-cornerstone}"

apply_launcher_profile() {
  case "$LAUNCHER_PROFILE" in
    dev)
      export CORNERSTONE_RUNTIME_MODE="${CORNERSTONE_RUNTIME_MODE:-mock}"
      export CORNERSTONE_AUTO_SEED_DEMO="${CORNERSTONE_AUTO_SEED_DEMO:-true}"
      export CORNERSTONE_NOTION_DEMO_OAUTH_MODE="${CORNERSTONE_NOTION_DEMO_OAUTH_MODE:-true}"
      ;;
    prod)
      export CORNERSTONE_RUNTIME_MODE="${CORNERSTONE_RUNTIME_MODE:-production}"
      export CORNERSTONE_AUTO_SEED_DEMO="${CORNERSTONE_AUTO_SEED_DEMO:-false}"
      export CORNERSTONE_NOTION_DEMO_OAUTH_MODE="${CORNERSTONE_NOTION_DEMO_OAUTH_MODE:-false}"
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: ./$LAUNCHER_NAME [up|down|logs|ps|check] [options]

Commands:
  up            Start the full Cornerstone stack. Default command.
  down          Stop and remove the stack.
  logs          Follow Compose logs for all services.
  ps            Show Compose service status.
  check         Run the full local quality gate once.

Options:
  --sample-data Use sample-data instead of demo_sources.
  --demo-data   Use demo_sources. Default.
  --ollama      Enable Ollama-backed answering.
  --reset-db    Recreate the local dev database volume before startup.
  --with-corpus Include the opt-in full corpus smoke during check.
  --no-build    Skip docker image rebuild on startup.
  -d, --detach  Run startup in the background.
  -h, --help    Show this help text.

Launchers:
  ./run-dev.sh  Force the local mock/dev runtime profile.
  ./run-prod.sh Force the local production-like runtime profile.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

ensure_prerequisites() {
  [[ -f "$COMPOSE_FILE" ]] || fail "missing compose file: $COMPOSE_FILE"
  [[ -f "$ENV_EXAMPLE" ]] || fail "missing env template: $ENV_EXAMPLE"

  command -v docker >/dev/null 2>&1 || fail "docker is required"
  docker compose version >/dev/null 2>&1 || fail "docker compose is required"
  docker info >/dev/null 2>&1 || fail "docker daemon is not running"

  if [[ "$command_name" != "check" && ! -f "$ENV_FILE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    printf 'Created %s from %s\n' "$ENV_FILE" "$ENV_EXAMPLE"
  fi

  if [[ "$data_mode" == "sample" && ! -d "$ROOT_DIR/sample-data" ]]; then
    fail "sample-data directory not found: $ROOT_DIR/sample-data"
  fi

  if [[ "$ollama_enabled" == "true" ]] && command -v curl >/dev/null 2>&1; then
    if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      printf '%s\n' \
        "warning: Ollama is not responding on http://127.0.0.1:11434; the stack will start, but /answers will fail until Ollama is available." \
        >&2
    fi
  fi
}

run_compose() {
  local source_root
  if [[ "$data_mode" == "sample" ]]; then
    source_root="/workspace/sample-data"
  else
    source_root="/workspace/demo_sources"
  fi

  (
    cd "$ROOT_DIR"
    env \
      CORNERSTONE_SOURCE_ROOT="$source_root" \
      CORNERSTONE_OLLAMA_ENABLED="$ollama_enabled" \
      COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT_NAME" \
      docker compose \
      --project-directory "$ROOT_DIR" \
      --env-file "$ENV_FILE" \
      -f "$COMPOSE_FILE" \
      "$@"
  )
}

run_make() {
  (
    cd "$ROOT_DIR"
    make "$@"
  )
}

run_checks() {
  command -v make >/dev/null 2>&1 || fail "make is required"

  cleanup_stack="true"
  trap 'if [[ "$cleanup_stack" == "true" ]]; then run_make test-stack-down >/dev/null 2>&1 || true; fi' EXIT

  printf 'Running Cornerstone quality gate\n'
  printf '  lint\n'
  run_make lint

  printf '  typecheck\n'
  run_make typecheck

  printf '  backend-fast\n'
  run_make backend-fast

  printf '  backend-integration\n'
  run_make backend-integration

  printf '  symptoms\n'
  run_make symptoms

  if [[ "$with_corpus" == "true" ]]; then
    printf '  corpus-smoke\n'
    run_make corpus-smoke
  fi

  printf 'Cornerstone quality gate passed.\n'
}

repair_stale_compose_state() {
  local default_network service cid networks image
  default_network="${COMPOSE_PROJECT_NAME}_default"

  for service in db backend frontend; do
    while IFS= read -r cid; do
      [[ -n "$cid" ]] || continue

      networks="$(docker inspect "$cid" --format '{{json .NetworkSettings.Networks}}' 2>/dev/null || printf '{}')"
      if [[ "$networks" != *"\"$default_network\""* ]]; then
        printf 'Removing stale %s container %s missing Compose network %s\n' "$service" "$cid" "$default_network"
        docker rm -f "$cid" >/dev/null 2>&1 || true
        continue
      fi

      if [[ "$service" == "db" ]]; then
        image="$(docker inspect "$cid" --format '{{.Config.Image}}' 2>/dev/null || true)"
        if [[ "$image" == postgres:16* ]]; then
          printf 'Removing stale db container %s using %s so Compose can recreate it with postgres:17-alpine\n' "$cid" "$image"
          docker rm -f "$cid" >/dev/null 2>&1 || true
        fi
      fi
    done < <(
      docker ps -aq \
        --filter "label=com.docker.compose.project=$COMPOSE_PROJECT_NAME" \
        --filter "label=com.docker.compose.service=$service"
    )
  done
}

reset_dev_database() {
  printf 'Resetting local Cornerstone database volume\n'
  (
    cd "$ROOT_DIR"
    docker compose \
      --project-name "$COMPOSE_PROJECT_NAME" \
      --project-directory "$ROOT_DIR" \
      --env-file "$ENV_FILE" \
      -f "$COMPOSE_FILE" \
      down -v
  )
}

command_name="up"
data_mode="demo"
ollama_enabled="false"
build_enabled="true"
detach_enabled="false"
with_corpus="false"
reset_db="false"

apply_launcher_profile

while [[ $# -gt 0 ]]; do
  case "$1" in
    up | down | logs | ps | check)
      command_name="$1"
      ;;
    --sample-data)
      data_mode="sample"
      ;;
    --demo-data)
      data_mode="demo"
      ;;
    --ollama)
      ollama_enabled="true"
      ;;
    --reset-db)
      reset_db="true"
      ;;
    --with-corpus)
      with_corpus="true"
      ;;
    --no-build)
      build_enabled="false"
      ;;
    -d | --detach)
      detach_enabled="true"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown argument: $1"
      ;;
  esac
  shift
done

ensure_prerequisites

case "$command_name" in
  up)
    printf 'Starting Cornerstone stack\n'
    printf '  launcher: %s\n' "$LAUNCHER_NAME"
    printf '  launcher-profile: %s\n' "$LAUNCHER_PROFILE"
    printf '  compose-project: %s\n' "$COMPOSE_PROJECT_NAME"
    printf '  runtime-mode: %s\n' "${CORNERSTONE_RUNTIME_MODE:-mock}"
    printf '  demo-seed: %s\n' "${CORNERSTONE_AUTO_SEED_DEMO:-true}"
    printf '  notion-demo-oauth: %s\n' "${CORNERSTONE_NOTION_DEMO_OAUTH_MODE:-true}"
    printf '  data: %s\n' "$data_mode"
    printf '  ollama: %s\n' "$ollama_enabled"
    printf '  detach: %s\n' "$detach_enabled"
    printf '  reset-db: %s\n' "$reset_db"

    if [[ "$reset_db" == "true" ]]; then
      reset_dev_database
    else
      repair_stale_compose_state
      cat <<EOF
Note:
- If startup fails with missing columns like 'context_spaces.kind' or 'decision_records.public_slug', your local Postgres volume is from the old schema.
- If startup fails with a Postgres data-directory compatibility error after the Postgres 17 upgrade, your local volume was initialized by the old major version.
- Recover with: ./$LAUNCHER_NAME up --reset-db
EOF
    fi

    compose_args=(up)
    if [[ "$build_enabled" == "true" ]]; then
      compose_args+=(--build)
    fi
    if [[ "$detach_enabled" == "true" ]]; then
      compose_args+=(-d)
    fi

    run_compose "${compose_args[@]}"

    if [[ "$detach_enabled" == "true" ]]; then
      cat <<'EOF'
Cornerstone is starting in the background.
- frontend: http://localhost:5173
- backend: http://localhost:8000
- docs: http://localhost:8000/docs

Use ./$LAUNCHER_NAME logs to follow logs.
Use ./$LAUNCHER_NAME down to stop the stack.
EOF
    fi
    ;;
  down)
    run_compose down
    ;;
  logs)
    run_compose logs -f
    ;;
  ps)
    run_compose ps
    ;;
  check)
    run_checks
    ;;
esac
