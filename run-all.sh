#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/compose.yml"
ENV_FILE="$ROOT_DIR/.env"
ENV_EXAMPLE="$ROOT_DIR/.env.example"

usage() {
  cat <<'EOF'
Usage: ./run-all.sh [up|down|logs|ps] [options]

Commands:
  up            Start the full Cornerstone stack. Default command.
  down          Stop and remove the stack.
  logs          Follow Compose logs for all services.
  ps            Show Compose service status.

Options:
  --sample-data Use sample-data instead of demo_sources.
  --demo-data   Use demo_sources. Default.
  --ollama      Enable Ollama-backed answering.
  --no-build    Skip docker image rebuild on startup.
  -d, --detach  Run startup in the background.
  -h, --help    Show this help text.
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

  if [[ ! -f "$ENV_FILE" ]]; then
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
      docker compose \
      --project-directory "$ROOT_DIR" \
      --env-file "$ENV_FILE" \
      -f "$COMPOSE_FILE" \
      "$@"
  )
}

command_name="up"
data_mode="demo"
ollama_enabled="false"
build_enabled="true"
detach_enabled="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    up | down | logs | ps)
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
    printf '  data: %s\n' "$data_mode"
    printf '  ollama: %s\n' "$ollama_enabled"
    printf '  detach: %s\n' "$detach_enabled"

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

Use ./run-all.sh logs to follow logs.
Use ./run-all.sh down to stop the stack.
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
esac
