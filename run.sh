#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${CORNERSTONE_HOST:-127.0.0.1}"
PORT="${CORNERSTONE_PORT:-8787}"
STATE_DIR="${CORNERSTONE_STATE_DIR:-data/local}"
GENERATION_MODEL="${CORNERSTONE_GENERATION_MODEL:-ornith:9b}"
EMBEDDING_MODEL="${CORNERSTONE_EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
OLLAMA_URL="${CORNERSTONE_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_URL="${OLLAMA_URL%/}"
STARTUP_TIMEOUT="${CORNERSTONE_STARTUP_TIMEOUT:-30}"

APP_PID=""
OLLAMA_PID=""
OLLAMA_LOG=""

usage() {
  cat <<'EOF'
Usage: ./run.sh

Starts the complete local CornerStone server with its Ollama generation and
embedding models. Press Ctrl-C to stop it.

Optional environment variables:
  CORNERSTONE_HOST                 Web server host (default: 127.0.0.1)
  CORNERSTONE_PORT                 Web server port (default: 8787)
  CORNERSTONE_STATE_DIR            Runtime state directory (default: data/local)
  CORNERSTONE_GENERATION_MODEL     Ollama generation model (default: ornith:9b)
  CORNERSTONE_EMBEDDING_MODEL      Ollama embedding model (default: qwen3-embedding:0.6b)
  CORNERSTONE_OLLAMA_BASE_URL      Loopback Ollama URL (default: http://127.0.0.1:11434)
  CORNERSTONE_STARTUP_TIMEOUT      Startup timeout in seconds (default: 30)
EOF
}

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

stop_process() {
  local pid="$1"
  local killer_pid
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    (sleep 5; kill -KILL "$pid" 2>/dev/null || true) &
    killer_pid=$!
    wait "$pid" 2>/dev/null || true
    kill -TERM "$killer_pid" 2>/dev/null || true
    wait "$killer_pid" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  stop_process "$APP_PID"
  stop_process "$OLLAMA_PID"
  if [[ -n "$OLLAMA_LOG" ]]; then
    rm -f "$OLLAMA_LOG"
  fi
  exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

validate_ollama_url() {
  python3 - "$OLLAMA_URL" <<'PY' >/dev/null 2>&1
import ipaddress
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    raise SystemExit(1)
if parsed.username or parsed.password or parsed.query or parsed.fragment:
    raise SystemExit(1)
if parsed.path not in {"", "/"}:
    raise SystemExit(1)
try:
    parsed.port
except ValueError:
    raise SystemExit(1)
if parsed.hostname.lower() == "localhost":
    raise SystemExit(0)
try:
    if ipaddress.ip_address(parsed.hostname).is_loopback:
        raise SystemExit(0)
except ValueError:
    pass
raise SystemExit(1)
PY
}

ollama_ready() {
  python3 - "$OLLAMA_URL/api/tags" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    payload = json.load(response)
if not isinstance(payload.get("models"), list):
    raise SystemExit(1)
PY
}

missing_models() {
  python3 - "$OLLAMA_URL/api/tags" "$GENERATION_MODEL" "$EMBEDDING_MODEL" <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    payload = json.load(response)
available = {
    value
    for model in payload.get("models", [])
    if isinstance(model, dict)
    for value in (model.get("name"), model.get("model"))
    if isinstance(value, str)
}
print(" ".join(model for model in sys.argv[2:] if model not in available))
PY
}

runtime_ready() {
  python3 - "$RUNTIME_URL/health" "$GENERATION_MODEL" "$EMBEDDING_MODEL" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    payload = json.load(response)
runtime = payload.get("model_runtime", {})
if not (
    payload.get("status") == "success"
    and runtime.get("model_provider") == "ollama"
    and runtime.get("generation_model") == sys.argv[2]
    and runtime.get("embedding_model") == sys.argv[3]
):
    raise SystemExit(1)
PY
}

runtime_port_available() {
  python3 - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
sock = socket.socket(family, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_until() {
  local description="$1"
  local check="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  local process_status

  while ((SECONDS < deadline)); do
    if "$check"; then
      return 0
    fi
    if [[ "$description" == "CornerStone" && -n "$APP_PID" ]] && ! kill -0 "$APP_PID" 2>/dev/null; then
      set +e
      wait "$APP_PID" 2>/dev/null
      process_status=$?
      set -e
      APP_PID=""
      fail "CornerStone exited with status $process_status before becoming healthy."
    fi
    if [[ "$description" == "Ollama" && -n "$OLLAMA_PID" ]] && ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
      if [[ -s "$OLLAMA_LOG" ]]; then
        tail -n 20 "$OLLAMA_LOG" >&2
      fi
      set +e
      wait "$OLLAMA_PID" 2>/dev/null
      process_status=$?
      set -e
      OLLAMA_PID=""
      fail "Ollama exited with status $process_status before becoming ready."
    fi
    sleep 0.25
  done
  fail "$description did not become ready within ${STARTUP_TIMEOUT}s."
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    usage >&2
    fail "run.sh does not accept command-line arguments; use the environment variables above."
    ;;
  esac

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "CORNERSTONE_PORT must be an integer from 1 to 65535."
PORT=$((10#$PORT))
((PORT >= 1 && PORT <= 65535)) || fail "CORNERSTONE_PORT must be an integer from 1 to 65535."
[[ "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] || fail "CORNERSTONE_STARTUP_TIMEOUT must be a positive integer."
STARTUP_TIMEOUT=$((10#$STARTUP_TIMEOUT))
((STARTUP_TIMEOUT >= 1)) || fail "CORNERSTONE_STARTUP_TIMEOUT must be a positive integer."
[[ -n "$HOST" ]] || fail "CORNERSTONE_HOST must not be empty."
[[ "$HOST" != *:* ]] || fail "CORNERSTONE_HOST must be an IPv4 address or hostname; this runtime does not support IPv6 binding."
[[ -n "$STATE_DIR" ]] || fail "CORNERSTONE_STATE_DIR must not be empty."
[[ -n "$GENERATION_MODEL" ]] || fail "CORNERSTONE_GENERATION_MODEL must not be empty."
[[ -n "$EMBEDDING_MODEL" ]] || fail "CORNERSTONE_EMBEDDING_MODEL must not be empty."

case "$HOST" in
  0.0.0.0)
    RUNTIME_URL="http://127.0.0.1:$PORT"
    ;;
  *)
    RUNTIME_URL="http://$HOST:$PORT"
    ;;
esac

command -v python3 >/dev/null 2>&1 || fail "python3 is required."
python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || fail "Python 3.11 or newer is required."
[[ -x "$ROOT/cornerstone" ]] || fail "The CornerStone CLI is missing or is not executable: $ROOT/cornerstone"
validate_ollama_url || fail "CORNERSTONE_OLLAMA_BASE_URL must be an HTTP(S) loopback URL without credentials, a path, query, or fragment."

if ! ollama_ready; then
  command -v ollama >/dev/null 2>&1 || fail "Ollama is required. Install it, then run: ollama pull $GENERATION_MODEL && ollama pull $EMBEDDING_MODEL"
  if [[ "$OLLAMA_URL" != "http://127.0.0.1:11434" ]]; then
    fail "No Ollama server is reachable at $OLLAMA_URL. Start that loopback server and retry."
  fi

  OLLAMA_LOG="${TMPDIR:-/tmp}/cornerstone-ollama-$$.log"
  printf 'Starting Ollama...\n'
  OLLAMA_HOST="127.0.0.1:11434" ollama serve >"$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!
  wait_until "Ollama" ollama_ready
fi

missing="$(missing_models)" || fail "Could not inspect the models available from $OLLAMA_URL."
if [[ -n "$missing" ]]; then
  fail "Missing required Ollama model(s): $missing. Run: ollama pull $GENERATION_MODEL && ollama pull $EMBEDDING_MODEL"
fi

runtime_port_available || fail "Cannot bind CornerStone to $HOST:$PORT; the address is already in use or unavailable."

printf 'Starting CornerStone...\n'
"$ROOT/cornerstone" runtime serve \
  --host "$HOST" \
  --port "$PORT" \
  --state-dir "$STATE_DIR" \
  --model-provider ollama \
  --generation-model "$GENERATION_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --ollama-url "$OLLAMA_URL" &
APP_PID=$!

wait_until "CornerStone" runtime_ready

cat <<EOF

CornerStone is ready.
  UI:         $RUNTIME_URL/
  API health: $RUNTIME_URL/health
  Generation: $GENERATION_MODEL
  Embeddings: $EMBEDDING_MODEL

Press Ctrl-C to stop.
EOF

ollama_failures=0
while kill -0 "$APP_PID" 2>/dev/null; do
  sleep 1
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    break
  fi
  if ollama_ready; then
    ollama_failures=0
  else
    ((ollama_failures += 1))
    if ((ollama_failures >= 3)); then
      fail "Ollama is no longer reachable at $OLLAMA_URL; stopping CornerStone to avoid silent model fallback."
    fi
  fi
done

set +e
wait "$APP_PID"
exit_code=$?
set -e
APP_PID=""
exit "$exit_code"
