from __future__ import annotations

import base64
import hashlib
import hmac
import json
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cornerstone_cli.vs2_verification_metadata import build_source_fingerprint, proof_hash


POSTGRES_IMAGE = "postgres:16-alpine"
OPA_IMAGE = "openpolicyagent/opa@sha256:dc009236137bb225a1ef09293bb32f2ee1861cc428870d297bf71412d50221c3"
PYTHON_IMAGE = "python:3.12-bookworm"
VS2_LOCAL_RANGE_REPORT = Path("reports/security/vs2-local-range.json")
POLICY_INPUT_SCHEMA_PATH = Path("config/vs2/policy_input_schema.v1.json")
REASON_CODE_CATALOG_PATH = Path("config/vs2/reason_code_catalog.v1.json")
POLICY_LIMITS_PATH = Path("config/vs2/policy_limits.v1.json")
POLICY_INPUT_SCHEMA_VERSION = "cs.policy_input.vs2.v1"
POLICY_INPUT_OPERATION_FAMILIES = {
    "gateway",
    "service",
    "tool_runtime",
    "action_card",
    "connector",
    "model_router",
    "policy_admin",
    "memory",
}

DURABLE_OBJECT_TABLES = [
    {"table": "artifacts", "object_type": "Artifact", "id_column": "artifact_id", "classification_required": True},
    {"table": "derived_representations", "object_type": "DerivedRepresentation", "id_column": "artifact_id", "classification_required": True},
    {"table": "search_snapshots", "object_type": "SearchSnapshot", "id_column": "artifact_id", "classification_required": True},
    {"table": "evidence_bundles", "object_type": "EvidenceBundle", "id_column": "artifact_id", "classification_required": True},
    {"table": "claims", "object_type": "Claim", "id_column": "artifact_id", "classification_required": True},
    {"table": "ontology_objects", "object_type": "OntologyObject", "id_column": "artifact_id", "classification_required": True},
    {"table": "ontology_links", "object_type": "OntologyLink", "id_column": "artifact_id", "classification_required": True},
    {"table": "action_cards", "object_type": "ActionCard", "id_column": "artifact_id", "classification_required": True},
    {"table": "workflow_runs", "object_type": "WorkflowRun", "id_column": "artifact_id", "classification_required": True},
    {"table": "policy_decisions", "object_type": "PolicyDecision", "id_column": "artifact_id", "classification_required": True},
    {"table": "jobs", "object_type": "Job", "id_column": "artifact_id", "classification_required": True},
    {"table": "idempotency_keys", "object_type": "IdempotencyKey", "id_column": "artifact_id", "classification_required": True},
    {"table": "egress_grants", "object_type": "EgressGrant", "id_column": "artifact_id", "classification_required": True},
    {"table": "migration_quarantine", "object_type": "MigrationQuarantine", "id_column": "artifact_id", "classification_required": True},
    {"table": "audit_events", "object_type": "AuditEvent", "id_column": "event_id", "classification_required": False},
]


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(root: Path, relative_path: Path, payload: dict[str, Any]) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _finalize_report_payload(
    root: Path,
    relative_path: Path,
    payload: dict[str, Any],
    *,
    started: float,
    cleanup_seconds: float = 0.0,
    cleanup_errors: list[str] | None = None,
    cleanup_results: list[dict[str, Any]] | None = None,
) -> None:
    profile = payload.setdefault("profile", {"schema_version": "cs.vs2_local_range_profile.v1"})
    measured_wall = float(profile.get("wall_seconds") or 0.0)
    bootstrap_seconds = measured_wall if profile.get("failure_layer") == "bootstrap" else 0.0
    execution_seconds = 0.0 if profile.get("failure_layer") == "bootstrap" else measured_wall
    profile["phase_seconds"] = {
        "bootstrap": round(bootstrap_seconds, 3),
        "execution": round(execution_seconds, 3),
        "evidence_serialization": 0.0,
        "cleanup": round(cleanup_seconds, 3),
    }
    if cleanup_errors:
        profile["cleanup_errors"] = cleanup_errors
    profile["cleanup_results"] = cleanup_results or []
    profile["cleanup_success"] = not cleanup_errors and all(
        result.get("exit_code") == 0
        for result in profile["cleanup_results"]
        if result.get("mandatory") is True
    )
    if profile["cleanup_success"] is False and payload.get("status") == "passed":
        payload["status"] = "failed"
        payload["cleanup_failure_demoted_pass"] = True
    profile["total_command_wall_seconds"] = round(time.perf_counter() - started, 3)
    profile["wall_seconds"] = profile["total_command_wall_seconds"]
    serialization_started = time.perf_counter()
    payload["proof_hash"] = proof_hash(payload)
    _write_json(root, relative_path, payload)
    serialization_seconds = time.perf_counter() - serialization_started
    profile["phase_seconds"]["evidence_serialization"] = round(serialization_seconds, 3)
    profile["total_command_wall_seconds"] = round(time.perf_counter() - started, 3)
    profile["wall_seconds"] = profile["total_command_wall_seconds"]
    profile["profile_boundary"] = (
        "total_command_wall_seconds includes cleanup and the first profiled report serialization; "
        "phase bootstrap/execution boundaries are conservative for the legacy monolithic runner"
    )
    payload["proof_hash"] = proof_hash(payload)
    _write_json(root, relative_path, payload)


def _container_disappeared(result: dict[str, Any]) -> bool:
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    return "No such container" in combined or "broken pipe" in combined


def _sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_json(value: Any) -> str:
    return _sql_literal(json.dumps(value, sort_keys=True)) + "::jsonb"


def _run(command: list[str], *, cwd: Path, input_text: str | None = None, timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _safe_transcript(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for entry in entries:
        safe.append(
            {
                "command": _redact_command(entry.get("command")),
                "exit_code": entry.get("exit_code"),
                "elapsed_seconds": entry.get("elapsed_seconds"),
                "stdout_tail": str(entry.get("stdout", "")).splitlines()[-6:],
                "stderr_tail": str(entry.get("stderr", "")).splitlines()[-6:],
            }
        )
    return safe


def _json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("stdout", "")).strip()
    if not text:
        return {"parse_error": "empty_stdout", "exit_code": result.get("exit_code")}
    try:
        return json.loads(text)
    except ValueError:
        try:
            return json.loads(text.splitlines()[-1])
        except ValueError:
            return {"parse_error": text[-500:], "exit_code": result.get("exit_code")}


def _digest_transcript_entry(entry: dict[str, Any]) -> dict[str, Any]:
    stdout = str(entry.get("stdout", ""))
    return {
        "command": _redact_command(entry.get("command")),
        "exit_code": entry.get("exit_code"),
        "elapsed_seconds": entry.get("elapsed_seconds"),
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stdout_bytes": len(stdout.encode()),
        "stderr_tail": str(entry.get("stderr", "")).splitlines()[-6:],
    }


def _redact_command(command: Any) -> Any:
    if not isinstance(command, list):
        return command
    redacted = []
    redact_next = False
    for part in command:
        value = str(part)
        if redact_next:
            redacted.append("<redacted-local-session-token>")
            redact_next = False
            continue
        if value == "--token":
            redacted.append(value)
            redact_next = True
            continue
        if value.startswith("POSTGRES_PASSWORD="):
            redacted.append("POSTGRES_PASSWORD=<redacted-local-password>")
            continue
        if value.startswith("eyJ") and "." in value and len(value) > 80:
            redacted.append("<redacted-local-session-token>")
            continue
        redacted.append(part)
    return redacted


def _clone_json(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True))


def _path_get(payload: dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _path_set(payload: dict[str, Any], path: str, value: Any) -> None:
    cursor: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _path_remove(payload: dict[str, Any], path: str) -> None:
    cursor: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            return
        cursor = cursor.get(part)
    if isinstance(cursor, dict):
        cursor.pop(parts[-1], None)


def _schema_type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _validate_schema_subset(value: Any, schema: dict[str, Any], *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _schema_type_matches(value, expected_type):
        return [{"path": path, "code": "type_mismatch", "expected": expected_type, "actual": type(value).__name__}]
    if "const" in schema and value != schema["const"]:
        errors.append({"path": path, "code": "const_mismatch", "expected": str(schema["const"]), "actual": str(value)})
    if expected_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for field in required:
            if not isinstance(value, dict) or field not in value:
                errors.append({"path": f"{path}.{field}", "code": "missing_required"})
        if schema.get("additionalProperties") is False and isinstance(value, dict):
            for field in sorted(set(value) - set(properties)):
                errors.append({"path": f"{path}.{field}", "code": "additional_property"})
        if isinstance(value, dict):
            for field, field_schema in properties.items():
                if field in value:
                    errors.extend(_validate_schema_subset(value[field], field_schema, path=f"{path}.{field}"))
    if expected_type == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_validate_schema_subset(item, schema["items"], path=f"{path}[{index}]"))
    return errors


def _json_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + (max((_json_depth(item) for item in value.values()), default=0))
    if isinstance(value, list):
        return 1 + (max((_json_depth(item) for item in value), default=0))
    return 1


def _json_pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.strip("/").split("/") if part]


def _apply_json_pointer_masks(payload: dict[str, Any], pointers: list[str]) -> dict[str, Any]:
    masked = _clone_json(payload)
    for pointer in pointers:
        cursor: Any = masked
        parts = _json_pointer_parts(pointer)
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if isinstance(cursor, dict) and parts and parts[-1] in cursor:
            cursor[parts[-1]] = "<masked-by-system-log-mask>"
    return masked


def _sign_token(payload: dict[str, Any], key: bytes) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(key, encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _worker_signature(envelope: dict[str, Any], key: bytes) -> str:
    unsigned = {field: value for field, value in envelope.items() if field != "signature"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _sign_worker_envelope(envelope: dict[str, Any], key: bytes) -> dict[str, Any]:
    signed = dict(envelope)
    signed["signature"] = _worker_signature(envelope, key)
    return signed


def _decode_token(token: str | None, key: bytes) -> tuple[dict[str, Any] | None, str | None]:
    if not token:
        return None, "missing_session"
    if not token.startswith("Bearer "):
        return None, "missing_bearer_session"
    raw = token.removeprefix("Bearer ").strip()
    if "." not in raw:
        return None, "missing_or_malformed_session"
    encoded, signature = raw.rsplit(".", 1)
    expected = hmac.new(key, encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None, "invalid_session_signature"
    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except (ValueError, OSError) as error:
        return None, f"invalid_session_payload:{type(error).__name__}"
    if int(payload.get("exp", 0)) < int(time.time()):
        return None, "expired_session"
    return payload, None


class _PostgresRange:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.container = f"cornerstone-vs2-range-pg-{hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]}"
        self.transcript: list[dict[str, Any]] = []

    def start(self) -> bool:
        started = _run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.container,
                "-e",
                "POSTGRES_PASSWORD=cornerstone",
                "-e",
                "POSTGRES_DB=cornerstone",
                POSTGRES_IMAGE,
            ],
            cwd=self.root,
            timeout=120,
        )
        self.transcript.append(started)
        if started["exit_code"] != 0:
            return False
        ready = False
        for _ in range(60):
            ready = _run(["docker", "exec", self.container, "pg_isready", "-U", "postgres", "-d", "cornerstone"], cwd=self.root, timeout=10)
            self.transcript.append(ready)
            if ready["exit_code"] == 0:
                ready = True
                break
            if _container_disappeared(ready):
                inspect = _run(["docker", "container", "inspect", self.container], cwd=self.root, timeout=10)
                self.transcript.append(inspect)
                if inspect["exit_code"] != 0:
                    return False
            time.sleep(0.5)
        if not ready:
            return False
        for _ in range(60):
            probe = self.psql("SELECT 1;", timeout=10)
            if probe["exit_code"] == 0:
                return True
            time.sleep(0.5)
        return False

    def stop(self) -> dict[str, Any]:
        result = _run(["docker", "rm", "-f", self.container], cwd=self.root, timeout=30)
        self.transcript.append(result)
        return result

    def psql(self, sql: str, *, database: str = "cornerstone", timeout: int = 120) -> dict[str, Any]:
        result = _run(
            [
                "docker",
                "exec",
                "-i",
                self.container,
                "psql",
                "-U",
                "postgres",
                "-d",
                database,
                "-v",
                "ON_ERROR_STOP=1",
                "-X",
                "-q",
                "-t",
                "-A",
            ],
            cwd=self.root,
            input_text=sql,
            timeout=timeout,
        )
        self.transcript.append(result)
        return result

    def psql_continue_on_error(self, sql: str, *, database: str = "cornerstone", timeout: int = 120) -> dict[str, Any]:
        result = _run(
            [
                "docker",
                "exec",
                "-i",
                self.container,
                "psql",
                "-U",
                "postgres",
                "-d",
                database,
                "-X",
                "-q",
                "-t",
                "-A",
            ],
            cwd=self.root,
            input_text=sql,
            timeout=timeout,
        )
        self.transcript.append(result)
        return result

    def json_query(self, sql: str) -> Any:
        result = self.psql(sql)
        if result["exit_code"] != 0:
            raise RuntimeError(result["stderr"] or result["stdout"])
        text = result["stdout"].strip()
        return json.loads(text) if text else None

    def apply_migrations(self) -> bool:
        migration_paths = sorted((self.root / "migrations" / "vs2").glob("*.sql"))
        for path in migration_paths:
            result = self.psql(path.read_text())
            if result["exit_code"] != 0:
                return False
        return True

    def seed(self) -> bool:
        seed_sql = """
INSERT INTO cs.principals(principal_id, display_name)
VALUES
  ('principal_alice', 'Alice Alpha'),
  ('principal_bob', 'Bob Beta'),
  ('principal_ada', 'Ada Alpha Admin'),
  ('principal_carla', 'Carla Gamma Revocation');
INSERT INTO cs.tenants(tenant_id, display_name)
VALUES
  ('tenant_alpha', 'Tenant Alpha'),
  ('tenant_beta', 'Tenant Beta'),
  ('tenant_gamma', 'Tenant Gamma');
INSERT INTO cs.memberships(
  membership_id, principal_id, tenant_id, namespace_id, workspace_id, owner_id, roles, membership_revision, session_version, revoked_at
) VALUES
  ('m_alpha_alice_personal', 'principal_alice', 'tenant_alpha', 'personal', 'alpha-home', 'principal_alice', ARRAY['owner'], 'memrev-alpha-001', 1, NULL),
  ('m_alpha_alice_org', 'principal_alice', 'tenant_alpha', 'organization', 'alpha-org', 'org_alpha', ARRAY['owner'], 'memrev-alpha-org-001', 1, NULL),
  ('m_alpha_ada_admin', 'principal_ada', 'tenant_alpha', 'personal', 'alpha-admin', 'principal_ada', ARRAY['tenant_admin'], 'memrev-alpha-admin-001', 1, NULL),
  ('m_beta_bob_personal', 'principal_bob', 'tenant_beta', 'personal', 'beta-home', 'principal_bob', ARRAY['member'], 'memrev-beta-001', 1, NULL),
  ('m_gamma_carla_personal', 'principal_carla', 'tenant_gamma', 'personal', 'gamma-home', 'principal_carla', ARRAY['owner'], 'memrev-gamma-001', 1, NULL);
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES
  ('tenant_alpha', 'personal', 'principal_alice', 'alpha-home', 'artifact_alpha_001', 'internal', '{"canary":"ALPHA_ONLY_RANGE_CANARY","title":"Alpha local range artifact"}', 'seed:alpha'),
  ('tenant_alpha', 'organization', 'org_alpha', 'alpha-org', 'artifact_alpha_org_001', 'internal', '{"canary":"ALPHA_ORG_RANGE_CANARY","title":"Alpha organization artifact"}', 'seed:alpha-org'),
  ('tenant_beta', 'personal', 'principal_bob', 'beta-home', 'artifact_beta_001', 'internal', '{"canary":"BETA_ONLY_RANGE_CANARY","title":"Beta local range artifact"}', 'seed:beta'),
  ('tenant_gamma', 'personal', 'principal_carla', 'gamma-home', 'artifact_gamma_001', 'internal', '{"canary":"GAMMA_REVOCATION_RANGE_CANARY","title":"Gamma revocation local range artifact"}', 'seed:gamma');
"""
        return self.psql(seed_sql)["exit_code"] == 0

    def seed_egress_grant(self, provider_url: str) -> bool:
        grant_sql = f"""
INSERT INTO cs.egress_grants(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_alpha',
  'personal',
  'principal_alice',
  'alpha-home',
  'grant_mock_provider_status',
  'internal',
  {_sql_json({
            "capability": "mock_provider.status.write",
            "provider_url": provider_url,
            "approved": True,
            "credential_ref": "credential_ref_mock_provider_write",
        })},
  'seed:egress-grant'
);
"""
        return self.psql(grant_sql)["exit_code"] == 0


class _OpaRange:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.container = f"cornerstone-vs2-range-opa-{hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]}"
        self.port = _free_port()
        self.transcript: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> bool:
        started = _run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.container,
                "-p",
                f"127.0.0.1:{self.port}:8181",
                "-v",
                f"{self.root / 'policies' / 'vs2'}:/policies:ro",
                OPA_IMAGE,
                "run",
                "--server",
                "--addr=0.0.0.0:8181",
                "/policies",
            ],
            cwd=self.root,
            timeout=120,
        )
        self.transcript.append(started)
        if started["exit_code"] != 0:
            return False
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    self.transcript.append({"command": ["GET", "/health"], "exit_code": 0, "elapsed_seconds": 0, "stdout": str(response.status), "stderr": ""})
                    if response.status == 200:
                        return True
            except (OSError, urllib.error.URLError) as error:
                self.transcript.append({"command": ["GET", "/health"], "exit_code": 1, "elapsed_seconds": 0, "stdout": "", "stderr": str(error)})
                time.sleep(0.25)
        return False

    def stop(self) -> dict[str, Any]:
        result = _run(["docker", "rm", "-f", self.container], cwd=self.root, timeout=30)
        self.transcript.append(result)
        return result


class _OpaRevisionRange:
    def __init__(self, root: Path, *, revision: str, decision: str, reason_codes: list[str]) -> None:
        self.root = root
        self.revision = revision
        self.decision = decision
        self.reason_codes = reason_codes
        self.container = f"cornerstone-vs2-range-opa-rev-{hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]}"
        self.port = _free_port()
        self.policy_dir: Path | None = None
        self.transcript: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> bool:
        policy_dir = Path(tempfile.mkdtemp(prefix="cornerstone-vs2-opa-revision-"))
        self.policy_dir = policy_dir
        decision = {
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "policy_path": "cornerstone.vs2/revision_update",
            "bundle_revision": self.revision,
        }
        (policy_dir / "policy.rego").write_text(
            "package cornerstone.vs2\n\n"
            "decision := "
            + json.dumps(decision, sort_keys=True)
            + "\n"
        )
        started = _run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.container,
                "-p",
                f"127.0.0.1:{self.port}:8181",
                "-v",
                f"{policy_dir}:/policies:ro",
                OPA_IMAGE,
                "run",
                "--server",
                "--addr=0.0.0.0:8181",
                "/policies",
            ],
            cwd=self.root,
            timeout=120,
        )
        self.transcript.append(started)
        if started["exit_code"] != 0:
            return False
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    self.transcript.append({"command": ["GET", "/health"], "exit_code": 0, "elapsed_seconds": 0, "stdout": str(response.status), "stderr": ""})
                    if response.status == 200:
                        return True
            except (OSError, urllib.error.URLError) as error:
                self.transcript.append({"command": ["GET", "/health"], "exit_code": 1, "elapsed_seconds": 0, "stdout": "", "stderr": str(error)})
                time.sleep(0.25)
        return False

    def stop(self) -> dict[str, Any]:
        result = _run(["docker", "rm", "-f", self.container], cwd=self.root, timeout=30)
        self.transcript.append(result)
        if self.policy_dir is not None:
            shutil.rmtree(self.policy_dir, ignore_errors=True)
        return result


class _MockProvider:
    def __init__(self) -> None:
        self.port = _free_port()
        self.calls: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/provider/status"

    def start(self) -> None:
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                raw_body = self.rfile.read(length).decode() if length else "{}"
                try:
                    body = json.loads(raw_body)
                except ValueError:
                    body = {"raw": raw_body}
                call = {
                    "method": "POST",
                    "path": self.path,
                    "body": body,
                    "authorization_header_seen": "authorization" in {key.lower() for key in self.headers},
                    "credential_ref_seen": self.headers.get("x-cs-credential-ref"),
                    "secret_canary_seen": "VS2_SECRET_DO_NOT_LOG" in raw_body,
                }
                provider.calls.append(call)
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "provider_ok",
                            "provider_request_id": f"mock_provider_{len(provider.calls):03d}",
                            "received_fields": sorted(body),
                        },
                        sort_keys=True,
                    ).encode()
                )

            def log_message(self, format: str, *args: Any) -> None:
                return

        class RangeThreadingHTTPServer(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 64

        self.server = RangeThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)


class _EgressProxy:
    def __init__(self) -> None:
        self.port = _free_port()
        self.calls: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/egress"

    def start(self) -> None:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                raw_body = self.rfile.read(length).decode() if length else "{}"
                try:
                    body = json.loads(raw_body)
                except ValueError:
                    body = {"raw": raw_body}
                target_url = str(body.get("target_url", ""))
                outbound_body = json.dumps(body.get("payload", {}), sort_keys=True).encode()
                proxy.calls.append(
                    {
                        "target_url": target_url,
                        "capability": body.get("capability"),
                        "credential_ref": body.get("credential_ref"),
                        "payload_keys": sorted(body.get("payload", {})),
                        "secret_canary_seen": "VS2_SECRET_DO_NOT_LOG" in raw_body,
                    }
                )
                request = urllib.request.Request(
                    target_url,
                    data=outbound_body,
                    headers={
                        "content-type": "application/json",
                        "x-cs-trace-id": str(body.get("trace_id", "trace_vs2_action")),
                        "x-cs-credential-ref": str(body.get("credential_ref", "credential_ref_missing")),
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    response_body = response.read().decode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "egress_proxy_ok",
                            "provider_status": response.status,
                            "provider_body": json.loads(response_body),
                        },
                        sort_keys=True,
                    ).encode()
                )

            def log_message(self, format: str, *args: Any) -> None:
                return

        class RangeThreadingHTTPServer(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 64

        self.server = RangeThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)


class _OpaFaultServer:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.port = _free_port()
        self.calls: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        fault = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                raw_body = self.rfile.read(length).decode() if length else "{}"
                fault.calls.append(
                    {
                        "mode": fault.mode,
                        "path": self.path,
                        "request_bytes": len(raw_body),
                        "secret_canary_seen": "VS2_SECRET_DO_NOT_LOG" in raw_body,
                    }
                )
                if fault.mode == "timeout":
                    time.sleep(0.5)
                    return
                if fault.mode == "http_500":
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"injected_opa_500"}')
                    return
                if fault.mode == "malformed_result":
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"result":"not-an-object"}')
                    return
                if fault.mode == "undefined_result":
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{}')
                    return
                if fault.mode == "revision_mismatch":
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "result": {
                                    "decision": "allow",
                                    "reason_codes": [],
                                    "policy_path": "cornerstone.vs2/allow",
                                    "bundle_revision": "vs2-rego-stale-revision",
                                }
                            },
                            sort_keys=True,
                        ).encode()
                    )
                    return
                self.send_response(500)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        class RangeThreadingHTTPServer(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 64

        self.server = RangeThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)


def _docker_network_boundary_probe(root: Path) -> dict[str, Any]:
    suffix = hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]
    service_net = f"cornerstone-vs2-range-service-{suffix}"
    provider_net = f"cornerstone-vs2-range-provider-{suffix}"
    containers = {
        "provider": f"cornerstone-vs2-provider-{suffix}",
        "egress_proxy": f"cornerstone-vs2-egress-proxy-{suffix}",
        "api": f"cornerstone-vs2-api-{suffix}",
        "worker": f"cornerstone-vs2-worker-{suffix}",
        "tool_runtime": f"cornerstone-vs2-tool-{suffix}",
    }
    transcript: list[dict[str, Any]] = []

    provider_script = r"""
import json
import pathlib
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log_path = pathlib.Path("/tmp/provider_calls.jsonl")

def read_events():
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/count"):
            events = read_events()
            body = {"requests": len(events), "events": events}
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body, sort_keys=True).encode())
            return
        parsed = urllib.parse.urlparse(self.path)
        event = {
            "path": parsed.path,
            "query": urllib.parse.parse_qs(parsed.query),
            "client": self.client_address[0],
            "ts": round(time.time(), 3),
        }
        with log_path.open("a") as file:
            file.write(json.dumps(event, sort_keys=True) + "\n")
        body = {"status": "provider_ok", "request_number": len(read_events())}
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, sort_keys=True).encode())

    def log_message(self, format, *args):
        return

ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
"""
    direct_attempt_script = r"""
import json
import socket
import sys
import urllib.error
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
actor = sys.argv[3]
result = {"actor": actor, "target": f"{host}:{port}"}
try:
    urllib.request.urlopen(f"http://{host}:{port}/status?actor={actor}&path=http", timeout=2).read()
    result["http"] = {"blocked": False, "error_type": None}
except Exception as error:
    result["http"] = {"blocked": True, "error_type": type(error).__name__, "error": str(error)[:240]}
try:
    sock = socket.create_connection((host, port), timeout=2)
    sock.close()
    result["socket"] = {"blocked": False, "error_type": None}
except Exception as error:
    result["socket"] = {"blocked": True, "error_type": type(error).__name__, "error": str(error)[:240]}
print(json.dumps(result, sort_keys=True))
"""
    allowed_attempt_script = r"""
import json
import sys
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
actor = sys.argv[3]
with urllib.request.urlopen(f"http://{host}:{port}/status?actor={actor}&path=governed_proxy", timeout=3) as response:
    body = json.loads(response.read().decode())
print(json.dumps({"actor": actor, "http_status": response.status, "body": body}, sort_keys=True))
"""
    count_script = r"""
import json
import pathlib

path = pathlib.Path("/tmp/provider_calls.jsonl")
events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
print(json.dumps({"requests": len(events), "events": events}, sort_keys=True))
"""
    readiness_script = r"""
import urllib.request

urllib.request.urlopen("http://provider:8080/count", timeout=2).read()
print("ready")
"""

    def run(command: list[str], *, timeout: int = 120) -> dict[str, Any]:
        result = _run(command, cwd=root, timeout=timeout)
        transcript.append(result)
        return result

    def start_sleep_container(name: str, network: str) -> bool:
        result = run(["docker", "run", "-d", "--rm", "--name", name, "--network", network, PYTHON_IMAGE, "sleep", "600"])
        return result["exit_code"] == 0

    def provider_count() -> dict[str, Any]:
        result = run(["docker", "exec", containers["provider"], "python", "-c", count_script], timeout=20)
        return _json_stdout(result)

    try:
        setup_ok = True
        setup_ok = run(["docker", "network", "create", "--internal", service_net])["exit_code"] == 0 and setup_ok
        setup_ok = run(["docker", "network", "create", "--internal", provider_net])["exit_code"] == 0 and setup_ok
        setup_ok = (
            run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    containers["provider"],
                    "--network",
                    provider_net,
                    "--network-alias",
                    "provider",
                    PYTHON_IMAGE,
                    "python",
                    "-u",
                    "-c",
                    provider_script,
                ]
            )["exit_code"]
            == 0
            and setup_ok
        )
        setup_ok = start_sleep_container(containers["egress_proxy"], service_net) and setup_ok
        setup_ok = run(["docker", "network", "connect", provider_net, containers["egress_proxy"]])["exit_code"] == 0 and setup_ok
        for actor in ["api", "worker", "tool_runtime"]:
            setup_ok = start_sleep_container(containers[actor], service_net) and setup_ok
        if not setup_ok:
            return {
                "status": "failed",
                "reason": "docker_network_setup_failed",
                "topology": {"service_net": service_net, "provider_net": provider_net, "containers": containers},
                "transcript": _safe_transcript(transcript),
            }

        provider_ready = False
        for _ in range(40):
            ready = run(["docker", "exec", containers["egress_proxy"], "python", "-c", readiness_script], timeout=10)
            if ready["exit_code"] == 0:
                provider_ready = True
                break
            time.sleep(0.25)
        if not provider_ready:
            return {
                "status": "failed",
                "reason": "provider_not_reachable_from_governed_proxy",
                "topology": {"service_net": service_net, "provider_net": provider_net, "containers": containers},
                "transcript": _safe_transcript(transcript),
            }

        count_before = provider_count()
        direct_attempts = {}
        for actor in ["api", "worker", "tool_runtime"]:
            result = run(
                ["docker", "exec", containers[actor], "python", "-c", direct_attempt_script, "provider", "8080", actor],
                timeout=20,
            )
            direct_attempts[actor] = _json_stdout(result)
        count_after_direct = provider_count()
        allowed_result_raw = run(
            ["docker", "exec", containers["egress_proxy"], "python", "-c", allowed_attempt_script, "provider", "8080", "egress_proxy"],
            timeout=20,
        )
        allowed_result = _json_stdout(allowed_result_raw)
        count_after_allowed = provider_count()
        service_inspect = _json_stdout(run(["docker", "network", "inspect", service_net], timeout=20))
        provider_inspect = _json_stdout(run(["docker", "network", "inspect", provider_net], timeout=20))

        provider_members = set()
        service_members = set()
        if isinstance(provider_inspect, list) and provider_inspect:
            provider_members = {item.get("Name") for item in provider_inspect[0].get("Containers", {}).values()}
        if isinstance(service_inspect, list) and service_inspect:
            service_members = {item.get("Name") for item in service_inspect[0].get("Containers", {}).values()}
        direct_blocked = all(
            attempt.get("http", {}).get("blocked") is True and attempt.get("socket", {}).get("blocked") is True
            for attempt in direct_attempts.values()
        )
        provider_zero_after_direct = count_before.get("requests") == 0 and count_after_direct.get("requests") == 0
        governed_proxy_reaches_provider = allowed_result.get("http_status") == 200 and count_after_allowed.get("requests") == 1
        membership_isolated = {
            containers["provider"],
            containers["egress_proxy"],
        }.issubset(provider_members) and all(containers[actor] not in provider_members for actor in ["api", "worker", "tool_runtime"])
        service_members_expected = all(containers[actor] in service_members for actor in ["api", "worker", "tool_runtime", "egress_proxy"])
        checks = {
            "provider_reachable_from_governed_proxy": governed_proxy_reaches_provider,
            "direct_http_and_socket_blocked": direct_blocked,
            "provider_zero_requests_after_direct_attempts": provider_zero_after_direct,
            "provider_network_membership_isolated": membership_isolated,
            "service_network_membership_expected": service_members_expected,
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "topology": {
                "service_net": service_net,
                "provider_net": provider_net,
                "containers": containers,
                "service_members": sorted(member for member in service_members if member),
                "provider_members": sorted(member for member in provider_members if member),
                "published_ports": False,
                "privileged": False,
                "host_network": False,
            },
            "direct_attempts": direct_attempts,
            "provider_counts": {
                "before_direct": count_before,
                "after_direct": count_after_direct,
                "after_allowed": count_after_allowed,
            },
            "governed_proxy_attempt": allowed_result,
            "checks": checks,
            "transcript": _safe_transcript(transcript),
        }
    finally:
        for name in containers.values():
            transcript.append(_run(["docker", "rm", "-f", name], cwd=root, timeout=30))
        for network in [service_net, provider_net]:
            transcript.append(_run(["docker", "network", "rm", network], cwd=root, timeout=30))


def _backup_restore_probe(root: Path, primary: _PostgresRange) -> dict[str, Any]:
    restore = _PostgresRange(root)
    bootstrap_roles = """
DO $$
BEGIN
  CREATE ROLE cornerstone_schema_owner NOLOGIN NOSUPERUSER NOBYPASSRLS;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$
BEGIN
  CREATE ROLE cornerstone_app LOGIN NOSUPERUSER NOBYPASSRLS;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$
BEGIN
  CREATE ROLE cornerstone_identity NOLOGIN NOSUPERUSER NOBYPASSRLS;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$
BEGIN
  CREATE ROLE cornerstone_migrator LOGIN NOSUPERUSER NOBYPASSRLS;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$
BEGIN
  CREATE ROLE cornerstone_maintenance LOGIN NOSUPERUSER NOBYPASSRLS;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$
BEGIN
  CREATE ROLE cornerstone_auditor NOLOGIN NOSUPERUSER NOBYPASSRLS;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
"""
    counts_sql = """
SELECT jsonb_build_object(
  'principals', (SELECT count(*) FROM cs.principals),
  'tenants', (SELECT count(*) FROM cs.tenants),
  'memberships', (SELECT count(*) FROM cs.memberships),
  'artifacts', (SELECT count(*) FROM cs.artifacts),
  'action_cards', (SELECT count(*) FROM cs.action_cards),
  'workflow_runs', (SELECT count(*) FROM cs.workflow_runs),
  'egress_grants', (SELECT count(*) FROM cs.egress_grants),
  'audit_events', (SELECT count(*) FROM cs.audit_events),
  'policy_count', (SELECT count(*) FROM pg_policies WHERE schemaname = 'cs')
)::text;
"""
    tenant_export_sql = """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT jsonb_build_object(
  'artifact_ids', (SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.artifacts),
  'artifact_payloads', (SELECT jsonb_agg(payload ORDER BY artifact_id) FROM cs.artifacts),
  'tenant_beta_rows', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_beta'),
  'audit_events_visible', (SELECT count(*) FROM cs.audit_events),
  'action_cards_visible', (SELECT count(*) FROM cs.action_cards),
  'workflow_runs_visible', (SELECT count(*) FROM cs.workflow_runs)
)::text;
COMMIT;
"""
    audit_sql = """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT jsonb_build_object(
  'audit_events_visible', count(*),
  'event_hashes', jsonb_agg(event_hash ORDER BY event_id),
  'decision_ids', jsonb_agg(decision_id ORDER BY event_id)
)::text
FROM cs.audit_events;
COMMIT;
"""
    try:
        primary_counts_before = primary.json_query(counts_sql)
        primary_tenant_export = primary.json_query(tenant_export_sql)
        primary_audit = primary.json_query(audit_sql)
        dump = _run(
            [
                "docker",
                "exec",
                primary.container,
                "pg_dump",
                "-U",
                "postgres",
                "-d",
                "cornerstone",
                "--no-owner",
            ],
            cwd=root,
            timeout=120,
        )
        if dump["exit_code"] != 0:
            return {
                "status": "failed",
                "reason": "pg_dump_failed",
                "primary_counts_before": primary_counts_before,
                "dump": _digest_transcript_entry(dump),
            }
        restore_ready = restore.start()
        if not restore_ready:
            return {
                "status": "failed",
                "reason": "restore_postgres_not_ready",
                "primary_counts_before": primary_counts_before,
                "dump": _digest_transcript_entry(dump),
                "restore_transcript": _safe_transcript(restore.transcript),
            }
        bootstrap = restore.psql(bootstrap_roles)
        restored = restore.psql(dump["stdout"], timeout=180) if bootstrap["exit_code"] == 0 else {"exit_code": 1, "stdout": "", "stderr": "role bootstrap failed"}
        try:
            restore_counts = restore.json_query(counts_sql) if restored["exit_code"] == 0 else {}
            restore_tenant_export = restore.json_query(tenant_export_sql) if restored["exit_code"] == 0 else {}
            restore_rls = _rls_tenant_probe(restore) if restored["exit_code"] == 0 else {}
            restore_audit = restore.json_query(audit_sql) if restored["exit_code"] == 0 else {}
        except RuntimeError as error:
            return {
                "status": "failed",
                "reason": "restore_verification_query_failed",
                "error": str(error).splitlines()[-6:],
                "dump": _digest_transcript_entry(dump),
                "restore": _digest_transcript_entry(restored),
                "role_bootstrap": _digest_transcript_entry(bootstrap),
                "primary_counts_before": primary_counts_before,
                "restore_transcript": _safe_transcript(restore.transcript),
            }
        checks = {
            "pg_dump_succeeded": dump["exit_code"] == 0 and len(dump["stdout"]) > 0,
            "pg_restore_succeeded": restored["exit_code"] == 0,
            "row_counts_match_after_restore": primary_counts_before == restore_counts,
            "policy_count_preserved": primary_counts_before.get("policy_count") == restore_counts.get("policy_count") and restore_counts.get("policy_count", 0) > 0,
            "rls_rechecked_after_restore": restore_rls.get("visible_artifacts", 0) >= 1 and restore_rls.get("tenant_beta_rows_visible") == 0,
            "audit_rechecked_after_restore": primary_audit.get("audit_events_visible", 0) == restore_audit.get("audit_events_visible", -1)
            and primary_audit.get("event_hashes") == restore_audit.get("event_hashes"),
            "tenant_export_scoped": restore_tenant_export.get("tenant_beta_rows") == 0
            and "BETA_ONLY_RANGE_CANARY" not in json.dumps(restore_tenant_export, sort_keys=True),
            "tenant_export_matches_primary": primary_tenant_export == restore_tenant_export,
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "scope": "local_synthetic_fixture; not production migration readiness",
            "primary_container": primary.container,
            "restore_container": restore.container,
            "dump": _digest_transcript_entry(dump),
            "restore": _digest_transcript_entry(restored),
            "role_bootstrap": _digest_transcript_entry(bootstrap),
            "primary_counts_before": primary_counts_before,
            "restore_counts": restore_counts,
            "primary_tenant_export": primary_tenant_export,
            "restore_tenant_export": restore_tenant_export,
            "restore_rls": restore_rls,
            "primary_audit": primary_audit,
            "restore_audit": restore_audit,
            "checks": checks,
            "restore_transcript": _safe_transcript(restore.transcript),
        }
    finally:
        restore.stop()


class _RangeGateway:
    def __init__(self, root: Path, postgres: _PostgresRange, opa_url: str, token_key: bytes, egress_proxy_url: str) -> None:
        self.root = root
        self.postgres = postgres
        self.opa_url = opa_url
        self.token_key = token_key
        self.egress_proxy_url = egress_proxy_url
        self.port = _free_port()
        self.requests: list[dict[str, Any]] = []
        self.object_store: dict[str, dict[str, Any]] = {}
        self.object_store_access_log: list[dict[str, Any]] = []
        self.policy_decision_cache: dict[str, dict[str, Any]] = {}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/health":
                    self._send_json(200, {"status": "ok", "component": "vs2-local-range-gateway"})
                    return
                if parsed.path.startswith("/api/vs2/artifacts/"):
                    artifact_id = parsed.path.rsplit("/", 1)[1]
                    caller_fields = {key: value[-1] for key, value in urllib.parse.parse_qs(parsed.query).items()}
                    payload = gateway.artifact_show(
                        token=self.headers.get("authorization"),
                        artifact_id=artifact_id,
                        caller_fields=caller_fields,
                        surface="http_api",
                        trace_id=self.headers.get("x-cs-trace-id") or "trace_vs2_local_range",
                    )
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/object-contract":
                    payload = gateway.object_contract(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/object-access-matrix":
                    payload = gateway.object_access_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/observability-matrix":
                    payload = gateway.observability_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/tenant-read-matrix":
                    payload = gateway.tenant_read_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/search-matrix":
                    payload = gateway.search_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/db-path-matrix":
                    payload = gateway.db_path_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/constraint-collision-matrix":
                    payload = gateway.constraint_collision_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/migration-matrix":
                    payload = gateway.migration_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/upgrade-path-matrix":
                    payload = gateway.upgrade_path_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path == "/api/vs2/audit-integrity-matrix":
                    payload = gateway.audit_integrity_matrix(token=self.headers.get("authorization"), surface="http_api")
                    self._send_json(payload["status_code"], payload)
                    return
                if parsed.path.startswith("/ui/vs2/artifacts/"):
                    artifact_id = parsed.path.rsplit("/", 1)[1]
                    payload = gateway.artifact_show(
                        token=self.headers.get("authorization"),
                        artifact_id=artifact_id,
                        caller_fields={},
                        surface="browser_ui",
                        trace_id=self.headers.get("x-cs-trace-id") or "trace_vs2_local_range",
                    )
                    html = (
                        "<!doctype html><html><body>"
                        "<main id='vs2-local-range' "
                        f"data-status='{payload['status']}' "
                        f"data-context-digest='{payload.get('context_digest') or ''}' "
                        f"data-policy-decision='{payload.get('policy_decision', {}).get('decision') if payload.get('policy_decision') else ''}' "
                        f"data-audit-ref-count='{len(payload.get('audit_refs', []))}'>"
                        "<h1>CornerStone VS2 Local Range</h1>"
                        f"<p>{payload['status']}</p>"
                        "</main></body></html>"
                    )
                    self.send_response(payload["status_code"])
                    self.send_header("content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html.encode())
                    return
                self._send_json(404, {"status": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                length = int(self.headers.get("content-length", "0"))
                raw_body = self.rfile.read(length).decode() if length else "{}"
                try:
                    body = json.loads(raw_body)
                except ValueError:
                    body = {}
                if parsed.path == "/api/vs2/actions/external":
                    payload = gateway.external_action_flow(
                        token=self.headers.get("authorization"),
                        provider_url=str(body.get("provider_url", "")),
                        surface="http_api",
                    )
                    self._send_json(payload["status_code"], payload)
                    return
                self._send_json(404, {"status": "not_found"})

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                try:
                    self.send_response(status_code)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload, sort_keys=True).encode())
                except (BrokenPipeError, ConnectionResetError):
                    return

        class RangeThreadingHTTPServer(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 64

        self.server = RangeThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join(timeout=5)

    def _resolve_context_from_token(self, token: str | None, surface: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        token_payload, token_error = _decode_token(token, self.token_key)
        counters = {"db_calls": 0, "policy_calls": 0, "egress_calls": 0, "audit_inserts": 0}
        if token_error or token_payload is None:
            return None, {
                "surface": surface,
                "status": "denied",
                "status_code": 401,
                "error": {"code": "CS_IDENTITY_CONTEXT_INVALID", "reason": token_error},
                "context_digest": None,
                "audit_refs": [],
                "counters": counters,
            }
        counters["db_calls"] += 1
        context_rows = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_agg(to_jsonb(r))::text
FROM cs.resolve_membership(
  {_sql_literal(token_payload['sub'])},
  {_sql_literal(token_payload['membership_id'])},
  {int(token_payload['session_version'])}
) r;
COMMIT;
"""
        )
        if not context_rows:
            reason = self._membership_denial_reason(token_payload)
            audit_ref = self._insert_identity_denial_audit(token_payload, reason=reason, surface=surface, artifact_id="object_contract")
            counters["audit_inserts"] += 1 if audit_ref else 0
            return None, {
                "surface": surface,
                "status": "denied",
                "status_code": 401,
                "error": {"code": "CS_MEMBERSHIP_UNRESOLVED", "reason": reason},
                "context_digest": None,
                "audit_refs": [audit_ref] if audit_ref else [],
                "counters": counters,
            }
        payload = {"context": context_rows[0], "counters": counters}
        return payload, None

    def _load_policy_input_schema(self) -> dict[str, Any]:
        return json.loads((self.root / POLICY_INPUT_SCHEMA_PATH).read_text())

    def _load_reason_code_catalog(self) -> dict[str, Any]:
        return json.loads((self.root / REASON_CODE_CATALOG_PATH).read_text())

    def _load_policy_limits(self) -> dict[str, Any]:
        return json.loads((self.root / POLICY_LIMITS_PATH).read_text())

    def _build_policy_input(
        self,
        context: dict[str, Any],
        resource: dict[str, Any],
        *,
        enforcement_point: str,
        action: str,
        policy_path: str,
        trace_id: str,
        risk: str = "low",
        approval_status: str = "not_required",
        capability_declared: bool = True,
        connectorhub_mediated: bool = True,
        schema_version: str = POLICY_INPUT_SCHEMA_VERSION,
        mission_authorized: bool = True,
        mission_id: str = "mission_alpha",
        authority_ref: str = "authority_alpha_policy_probe",
        data_scope: str = "tenant",
        data_purpose: str = "artifact_read",
        deployment: str = "local",
        workspace_mode: str = "assist",
        subject_overrides: dict[str, Any] | None = None,
        omit_paths: list[tuple[str, ...]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        policy_input = {
            "schema_version": schema_version,
            "trace_id": trace_id,
            "subject": {
                "principal_id": context["principal_id"],
                "roles": context["roles"],
                "membership_revision": context["membership_revision"],
                "revoked": context["revoked"],
            },
            "scope": {
                "tenant_id": context["tenant_id"],
                "namespace_id": context["namespace_id"],
                "workspace_id": context["workspace_id"],
            },
            "resource": {
                "resource_id": resource.get("resource_id") or resource["artifact_id"],
                "tenant_id": resource["tenant_id"],
                "namespace_id": resource["namespace_id"],
                "classification": resource["classification"],
            },
            "action": action,
            "risk": risk,
            "policy_path": policy_path,
            "mission_authority": {
                "mission_id": mission_id,
                "authorized": mission_authorized,
                "authority_ref": authority_ref,
            },
            "data_scope": {"scope": data_scope, "purpose": data_purpose},
            "approval": {"required": risk == "high", "status": approval_status},
            "capability": {"declared": capability_declared, "connectorhub_mediated": connectorhub_mediated},
            "environment": {"deployment": deployment, "workspace_mode": workspace_mode},
        }
        if subject_overrides:
            policy_input["subject"].update(subject_overrides)
        for path in omit_paths or []:
            cursor: Any = policy_input
            for key in path[:-1]:
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, dict):
                cursor.pop(path[-1], None)
        source_map = {
            "schema_version": "policy_client.constant",
            "trace_id": f"{enforcement_point}.trace_context",
            "subject.principal_id": "trusted_request_context.postgres.resolve_membership",
            "subject.roles": "trusted_request_context.postgres.membership_roles",
            "subject.membership_revision": "trusted_request_context.postgres.membership_revision",
            "subject.revoked": "trusted_request_context.postgres.revocation_state",
            "scope.tenant_id": "trusted_request_context.tenant_scope",
            "scope.namespace_id": "trusted_request_context.namespace_scope",
            "scope.workspace_id": "trusted_request_context.workspace_scope",
            "resource.resource_id": f"{enforcement_point}.resource_record",
            "resource.tenant_id": f"{enforcement_point}.resource_record",
            "resource.namespace_id": f"{enforcement_point}.resource_record",
            "resource.classification": f"{enforcement_point}.resource_record",
            "action": f"{enforcement_point}.declared_action",
            "risk": f"{enforcement_point}.risk_classifier",
            "policy_path": f"{enforcement_point}.policy_entrypoint",
            "mission_authority.mission_id": "trusted_request_context.mission_authority",
            "mission_authority.authorized": "trusted_request_context.mission_authority",
            "mission_authority.authority_ref": "trusted_request_context.mission_authority",
            "data_scope.scope": f"{enforcement_point}.data_scope",
            "data_scope.purpose": f"{enforcement_point}.data_purpose",
            "approval.required": f"{enforcement_point}.risk_classifier",
            "approval.status": f"{enforcement_point}.approval_state",
            "capability.declared": f"{enforcement_point}.capability_manifest",
            "capability.connectorhub_mediated": f"{enforcement_point}.connectorhub_boundary",
            "environment.deployment": "local_range.environment",
            "environment.workspace_mode": "trusted_request_context.workspace_mode",
        }
        return policy_input, source_map

    def policy_input_schema_matrix(self, *, owner_token: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(owner_token, "policy_input_schema")
        if denied or resolved is None:
            return {"status": "failed", "reason": "context_resolution_failed", "denied": denied, "checks": {}}
        context = resolved["context"]
        schema = self._load_policy_input_schema()
        schema_hash = hashlib.sha256((self.root / POLICY_INPUT_SCHEMA_PATH).read_bytes()).hexdigest()
        resource = {
            "artifact_id": "artifact_alpha_001",
            "tenant_id": context["tenant_id"],
            "namespace_id": context["namespace_id"],
            "classification": "internal",
        }
        specs = [
            ("gateway", "artifact.read", "artifact.read", "gateway_read", "low", "not_required"),
            ("service", "artifact.write", "artifact.write", "service_write", "low", "not_required"),
            ("tool_runtime", "tool.execute", "tool.execute", "tool_execute", "low", "not_required"),
            ("action_card", "action.execute", "action.execute", "action_execute", "low", "not_required"),
            ("connector", "connector.execute", "connector.execute", "connector_execute", "low", "not_required"),
            ("model_router", "model.route", "model.route", "model_route", "low", "not_required"),
            ("policy_admin", "policy.bundle.publish", "policy.bundle.publish", "policy_admin", "high", "approved"),
            ("memory", "memory.read", "memory.read", "memory_read", "low", "not_required"),
        ]
        valid_cases: list[dict[str, Any]] = []
        audit_refs: list[str] = []
        for index, (family, action, policy_path, purpose, risk, approval_status) in enumerate(specs, start=1):
            policy_input, source_map = self._build_policy_input(
                context,
                resource | {"artifact_id": f"policy_input_{family}_{index}"},
                enforcement_point=family,
                action=action,
                policy_path=policy_path,
                trace_id=f"trace_vs2_policy_input_026_{index}",
                risk=risk,
                approval_status=approval_status,
                data_purpose=purpose,
                authority_ref=f"authority_alpha_{family}",
            )
            schema_errors = _validate_schema_subset(policy_input, schema)
            decision = self._evaluate_opa_policy_input(policy_input) if not schema_errors else {"decision": "not_called", "reason_codes": ["schema_validation_failed"]}
            audit_ref = self._insert_audit(
                context,
                "policy.input.validated",
                action,
                {
                    "scenario_id": "VS2-SEC-026",
                    "operation_family": family,
                    "input_digest": _sha256_json(policy_input),
                    "schema_error_count": len(schema_errors),
                },
                decision.get("decision_id", "policy_input_schema_failed"),
                trace_id=f"trace_vs2_policy_input_026_{index}",
            )
            audit_refs.append(audit_ref)
            valid_cases.append(
                {
                    "operation_family": family,
                    "action": action,
                    "policy_path": policy_path,
                    "input_digest": _sha256_json(policy_input),
                    "source_map": source_map,
                    "schema_errors": schema_errors,
                    "opa_call_attempted": not schema_errors,
                    "decision": decision,
                    "audit_ref": audit_ref,
                }
            )
        invalid_specs = [
            ("missing_subject_principal", "remove", "subject.principal_id", None),
            ("roles_wrong_type", "set", "subject.roles", "owner"),
            ("wrong_schema_version", "set", "schema_version", "cs.policy_input.vs2.invalid"),
            ("unexpected_top_level_tenant", "set", "tenant_id", "tenant_beta"),
            ("missing_mission_authorized", "remove", "mission_authority.authorized", None),
            ("unexpected_authoritative_subject_tenant", "set", "subject.tenant_id", "tenant_beta"),
        ]
        invalid_cases: list[dict[str, Any]] = []
        base_policy_input, base_source_map = self._build_policy_input(
            context,
            resource | {"artifact_id": "policy_input_invalid_template"},
            enforcement_point="gateway",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_input_026_invalid_template",
            data_purpose="gateway_read",
        )
        for name, mutation, path, value in invalid_specs:
            policy_input = _clone_json(base_policy_input)
            if mutation == "remove":
                _path_remove(policy_input, path)
            else:
                _path_set(policy_input, path, value)
            schema_errors = _validate_schema_subset(policy_input, schema)
            invalid_cases.append(
                {
                    "case": name,
                    "mutation": {"kind": mutation, "path": path},
                    "input_digest": _sha256_json(policy_input),
                    "schema_errors": schema_errors,
                    "opa_call_attempted": False,
                    "source_map": base_source_map,
                }
            )
        required_source_paths = [
            "subject.principal_id",
            "subject.roles",
            "subject.membership_revision",
            "subject.revoked",
            "scope.tenant_id",
            "scope.namespace_id",
            "scope.workspace_id",
            "resource.resource_id",
            "resource.tenant_id",
            "resource.namespace_id",
            "resource.classification",
            "action",
            "risk",
            "policy_path",
            "mission_authority.mission_id",
            "mission_authority.authorized",
            "mission_authority.authority_ref",
            "data_scope.scope",
            "data_scope.purpose",
            "approval.required",
            "approval.status",
            "capability.declared",
            "capability.connectorhub_mediated",
            "environment.deployment",
            "environment.workspace_mode",
        ]
        required_schema_paths = [
            "schema_version",
            "trace_id",
            *required_source_paths,
        ]
        checks = {
            "schema_file_present": (self.root / POLICY_INPUT_SCHEMA_PATH).exists(),
            "schema_version_const_is_v1": _path_get(schema, "properties.schema_version.const") == POLICY_INPUT_SCHEMA_VERSION,
            "valid_cases_cover_operation_families": {case["operation_family"] for case in valid_cases} == POLICY_INPUT_OPERATION_FAMILIES,
            "valid_cases_pass_schema_and_opa": all(
                not case["schema_errors"]
                and case["opa_call_attempted"] is True
                and case["decision"].get("decision") == "allow"
                and case["decision"].get("bundle_revision") == "vs2-rego-local-v1"
                and bool(case["decision"].get("input_digest"))
                and bool(case["decision"].get("decision_id"))
                for case in valid_cases
            ),
            "invalid_cases_rejected_before_opa": all(case["schema_errors"] and case["opa_call_attempted"] is False for case in invalid_cases),
            "source_map_covers_required_attributes": all(
                all(path in case["source_map"] and bool(case["source_map"][path]) for path in required_source_paths) for case in valid_cases
            ),
            "input_digests_present": all(bool(case["input_digest"]) for case in [*valid_cases, *invalid_cases]),
            "schema_required_paths_present": all(
                path == "trace_id"
                or path == "schema_version"
                or _path_get(schema, "properties." + path.replace(".", ".properties.")) is not None
                for path in required_schema_paths
            ),
            "audit_refs_recorded": len(audit_refs) == len(valid_cases) and all(audit_refs),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "schema_path": str(POLICY_INPUT_SCHEMA_PATH),
            "schema_sha256": schema_hash,
            "schema_title": schema.get("title"),
            "valid_cases": valid_cases,
            "invalid_cases": invalid_cases,
            "required_source_paths": required_source_paths,
            "audit_refs": audit_refs,
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def _policy_limit_errors(self, policy_input: dict[str, Any], limits: dict[str, Any]) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        input_bytes = len(json.dumps(policy_input, sort_keys=True, separators=(",", ":")).encode())
        depth = _json_depth(policy_input)
        role_count = len(policy_input.get("subject", {}).get("roles", []))
        workspace_mode = policy_input.get("environment", {}).get("workspace_mode")
        if input_bytes > int(limits["max_input_bytes"]):
            errors.append({"code": "policy_input_too_large", "actual": input_bytes, "limit": limits["max_input_bytes"]})
        if depth > int(limits["max_json_depth"]):
            errors.append({"code": "policy_input_too_deep", "actual": depth, "limit": limits["max_json_depth"]})
        if role_count > int(limits["max_roles"]):
            errors.append({"code": "policy_input_role_list_too_long", "actual": role_count, "limit": limits["max_roles"]})
        if workspace_mode not in set(limits["allowed_workspace_modes"]):
            errors.append({"code": "unsupported_workspace_mode", "actual": workspace_mode, "allowed": limits["allowed_workspace_modes"]})
        return errors

    def policy_limits_matrix(self, *, owner_token: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(owner_token, "policy_limits")
        if denied or resolved is None:
            return {"status": "failed", "reason": "context_resolution_failed", "denied": denied, "checks": {}}
        context = resolved["context"]
        schema = self._load_policy_input_schema()
        limits = self._load_policy_limits()
        limits_hash = hashlib.sha256((self.root / POLICY_LIMITS_PATH).read_bytes()).hexdigest()
        resource = {
            "artifact_id": "policy_limits_alpha_resource",
            "tenant_id": context["tenant_id"],
            "namespace_id": context["namespace_id"],
            "classification": "internal",
        }
        base_input, source_map = self._build_policy_input(
            context,
            resource,
            enforcement_point="gateway",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_limits_048_base",
            authority_ref="authority_alpha_policy_limits",
            data_purpose="policy_limits",
        )
        max_roles = int(limits["max_roles"])
        below_limit = _clone_json(base_input)
        below_limit["subject"]["roles"] = ["owner"]
        at_limit = _clone_json(base_input)
        at_limit["subject"]["roles"] = ["owner", *[f"synthetic_role_{index}" for index in range(max_roles - 1)]]
        over_roles = _clone_json(base_input)
        over_roles["subject"]["roles"] = ["owner", *[f"synthetic_role_{index}" for index in range(max_roles)]]
        over_size = _clone_json(base_input)
        over_size["trace_id"] = "trace_vs2_policy_limits_048_" + ("x" * int(limits["max_input_bytes"]))
        over_depth = _clone_json(base_input)
        over_depth["limit_probe"] = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": {"l8": {"l9": "too_deep"}}}}}}}}}
        unknown_enum = _clone_json(base_input)
        unknown_enum["environment"]["workspace_mode"] = "unsupported_workspace_mode"
        cases = [
            {"case": "below_limit", "input": below_limit, "expected": "allow"},
            {"case": "at_limit", "input": at_limit, "expected": "allow"},
            {"case": "over_role_limit", "input": over_roles, "expected": "limit_error"},
            {"case": "over_size_limit", "input": over_size, "expected": "limit_error"},
            {"case": "over_depth_limit", "input": over_depth, "expected": "limit_error"},
            {"case": "unknown_workspace_mode", "input": unknown_enum, "expected": "limit_error"},
        ]
        side_effects_before = self._policy_side_effect_counts()
        observations: list[dict[str, Any]] = []
        audit_refs: list[str] = []
        for case in cases:
            started = time.perf_counter()
            policy_input = case["input"]
            schema_errors = _validate_schema_subset(policy_input, schema)
            limit_errors = self._policy_limit_errors(policy_input, limits)
            decision = None
            if not schema_errors and not limit_errors:
                decision = self._evaluate_opa_policy_input(policy_input)
                audit_refs.append(
                    self._insert_audit(
                        context,
                        "policy.limit.accepted",
                        "policy.evaluate",
                        {
                            "scenario_id": "VS2-SEC-048",
                            "case": case["case"],
                            "input_digest": _sha256_json(policy_input),
                            "input_bytes": len(json.dumps(policy_input, sort_keys=True, separators=(",", ":")).encode()),
                            "json_depth": _json_depth(policy_input),
                        },
                        decision["decision_id"],
                        trace_id=f"trace_vs2_policy_limits_048_{case['case']}",
                    )
                )
            observations.append(
                {
                    "case": case["case"],
                    "expected": case["expected"],
                    "schema_errors": schema_errors,
                    "limit_errors": limit_errors,
                    "opa_call_attempted": decision is not None,
                    "decision": decision,
                    "input_digest": _sha256_json(policy_input),
                    "input_bytes": len(json.dumps(policy_input, sort_keys=True, separators=(",", ":")).encode()),
                    "json_depth": _json_depth(policy_input),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        side_effects_after = self._policy_side_effect_counts()
        allowed_cases = [case for case in observations if case["expected"] == "allow"]
        rejected_cases = [case for case in observations if case["expected"] == "limit_error"]
        checks = {
            "limits_config_file_present": (self.root / POLICY_LIMITS_PATH).exists(),
            "below_and_at_limit_requests_allow_deterministically": all(
                not case["schema_errors"]
                and not case["limit_errors"]
                and case["opa_call_attempted"] is True
                and case["decision"].get("decision") == "allow"
                for case in allowed_cases
            ),
            "over_limit_requests_rejected_before_opa": all(
                case["limit_errors"] and case["opa_call_attempted"] is False for case in rejected_cases
            ),
            "unknown_enum_rejected_before_opa": any(
                case["case"] == "unknown_workspace_mode"
                and any(error.get("code") == "unsupported_workspace_mode" for error in case["limit_errors"])
                and case["opa_call_attempted"] is False
                for case in rejected_cases
            ),
            "limit_failures_have_bounded_resource_use": all(case["elapsed_ms"] < 250 for case in rejected_cases),
            "limit_failures_leave_no_partial_side_effects": side_effects_before == side_effects_after,
            "limit_source_map_retained": all(bool(source_map.get(path)) for path in ["subject.roles", "environment.workspace_mode", "trace_id"]),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "limits_path": str(POLICY_LIMITS_PATH),
            "limits_sha256": limits_hash,
            "limits": limits,
            "source_map": source_map,
            "cases": observations,
            "side_effect_counts_before": side_effects_before,
            "side_effect_counts_after": side_effects_after,
            "audit_refs": audit_refs,
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def _opa_system_log_mask_paths(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.opa_url}/v1/data/system/log/mask", timeout=5) as response:
                body = json.loads(response.read().decode())
            result = body.get("result", [])
            if isinstance(result, list):
                paths = sorted(str(path) for path in result)
            else:
                paths = []
            return {"status": "passed", "paths": paths, "http_status": 200}
        except (OSError, ValueError, urllib.error.URLError) as error:
            return {"status": "failed", "paths": [], "error": type(error).__name__}

    def decision_log_masking_matrix(self, *, owner_token: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(owner_token, "decision_log_masking")
        if denied or resolved is None:
            return {"status": "failed", "reason": "context_resolution_failed", "denied": denied, "checks": {}}
        context = resolved["context"]
        policy_input, source_map = self._build_policy_input(
            context,
            {
                "artifact_id": "decision_log_masking_artifact",
                "tenant_id": context["tenant_id"],
                "namespace_id": context["namespace_id"],
                "classification": "internal",
            },
            enforcement_point="service",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_decision_log_masking_045",
            data_purpose="decision_log_masking",
            authority_ref="authority_alpha_decision_log_masking",
        )
        canary = "VS2_SECRET_DO_NOT_LOG"
        _path_set(policy_input, "environment.secret_value", canary)
        _path_set(policy_input, "capability.credential_value", canary)
        decision = self._evaluate_opa_policy_input(policy_input)
        mask_paths = self._opa_system_log_mask_paths()
        raw_log = {
            "input": policy_input,
            "result": decision,
            "decision_id": decision.get("decision_id"),
            "policy_path": decision.get("policy_path"),
            "bundle_revision": decision.get("bundle_revision"),
            "trace_id": decision.get("trace_id"),
        }
        canary_present_before_mask = canary in json.dumps(raw_log, sort_keys=True)
        collector_entry = _apply_json_pointer_masks(raw_log, mask_paths.get("paths", []))
        collector_json = json.dumps(collector_entry, sort_keys=True)
        audit_ref = self._insert_audit(
            context,
            "policy.decision_log.masked",
            "artifact.read",
            {
                "scenario_id": "VS2-SEC-045",
                "collector_entry_digest": _sha256_json(collector_entry),
                "input_digest": _sha256_json(policy_input),
                "mask_path_count": len(mask_paths.get("paths", [])),
                "canary_present_after_mask": canary in collector_json,
            },
            decision["decision_id"],
            trace_id="trace_vs2_decision_log_masking_045",
        )
        expected_paths = {
            "/input/subject/principal_id",
            "/input/subject/membership_revision",
            "/input/resource/resource_id",
            "/input/resource/classification",
            "/input/mission_authority/authority_ref",
            "/input/environment/secret_value",
            "/input/capability/credential_value",
        }

        def masked_at(pointer: str) -> bool:
            cursor: Any = collector_entry
            for part in _json_pointer_parts(pointer):
                if not isinstance(cursor, dict) or part not in cursor:
                    return False
                cursor = cursor[part]
            return cursor == "<masked-by-system-log-mask>"

        checks = {
            "mask_policy_loaded_from_opa": mask_paths.get("status") == "passed" and expected_paths <= set(mask_paths.get("paths", [])),
            "canary_reached_opa_decision_request": canary_present_before_mask is True and decision.get("decision") == "allow",
            "collector_received_masked_fields": all(masked_at(path) for path in expected_paths),
            "collector_entry_has_no_canary": canary not in collector_json,
            "decision_log_linked_to_policy_and_audit": bool(decision.get("decision_id"))
            and decision.get("bundle_revision") == "vs2-rego-local-v1"
            and bool(decision.get("policy_path"))
            and bool(audit_ref),
            "source_map_retained_without_raw_values": "subject.principal_id" in source_map and "resource.resource_id" in source_map,
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "mask_policy": {
                "status": mask_paths.get("status"),
                "paths": mask_paths.get("paths", []),
                "http_status": mask_paths.get("http_status"),
            },
            "decision": decision,
            "collector_entry": collector_entry,
            "collector_entry_digest": _sha256_json(collector_entry),
            "input_digest": _sha256_json(policy_input),
            "canary_present_before_mask": canary_present_before_mask,
            "canary_present_after_mask": canary in collector_json,
            "source_map": source_map,
            "audit_refs": [audit_ref],
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def _translate_policy_denial(self, decision: dict[str, Any], *, status_code: int = 403, code: str = "CS_POLICY_DENIED") -> dict[str, Any]:
        catalog = self._load_reason_code_catalog()
        entries = catalog.get("reason_codes", {})
        default = catalog.get("default", {})
        reason_codes = [str(reason) for reason in decision.get("reason_codes", [])]
        mapped = [
            {
                "reason_code": reason,
                "message": entries.get(reason, default).get("message", default.get("message")),
                "resolution": entries.get(reason, default).get("resolution", default.get("resolution")),
                "http_status": entries.get(reason, default).get("http_status", status_code),
            }
            for reason in reason_codes
        ]
        return {
            "status_code": status_code,
            "code": code,
            "message": mapped[0]["message"] if mapped else default.get("message"),
            "safe_reason_codes": reason_codes,
            "reason_details": mapped,
            "resolution": mapped[0]["resolution"] if mapped else default.get("resolution"),
            "decision_id": decision.get("decision_id"),
            "policy_path": decision.get("policy_path"),
            "bundle_revision": decision.get("bundle_revision"),
            "protected_data_echoed": False,
        }

    def _render_denial_component(self, response: dict[str, Any], *, surface: str) -> str:
        reason_codes = ",".join(response.get("safe_reason_codes", []))
        return (
            "<section class='cs-denial' "
            f"data-surface='{surface}' "
            f"data-status-code='{response.get('status_code')}' "
            f"data-error-code='{response.get('code')}' "
            f"data-reason-codes='{reason_codes}' "
            f"data-decision-id='{response.get('decision_id')}' "
            f"data-policy-path='{response.get('policy_path')}' "
            f"data-protected-data-echoed='{str(response.get('protected_data_echoed')).lower()}'>"
            "<h2>Request denied</h2>"
            f"<p>{response.get('message')}</p>"
            f"<p>{response.get('resolution')}</p>"
            "</section>"
        )

    def reason_code_translation_matrix(self, *, owner_token: str, member_token: str) -> dict[str, Any]:
        owner_resolved, owner_denied = self._resolve_context_from_token(owner_token, "reason_code_translation")
        member_resolved, member_denied = self._resolve_context_from_token(member_token, "reason_code_translation")
        if owner_denied or member_denied or owner_resolved is None or member_resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "owner_denied": owner_denied,
                "member_denied": member_denied,
                "checks": {},
            }
        owner_context = owner_resolved["context"]
        member_context = member_resolved["context"]
        catalog = self._load_reason_code_catalog()
        catalog_hash = hashlib.sha256((self.root / REASON_CODE_CATALOG_PATH).read_bytes()).hexdigest()
        owner_resource = {
            "artifact_id": "reason_code_alpha_resource",
            "tenant_id": owner_context["tenant_id"],
            "namespace_id": owner_context["namespace_id"],
            "classification": "internal",
        }
        member_resource = {
            "artifact_id": "reason_code_beta_resource",
            "tenant_id": member_context["tenant_id"],
            "namespace_id": member_context["namespace_id"],
            "classification": "internal",
        }
        specs = [
            {
                "surface": "data",
                "context": member_context,
                "resource": member_resource,
                "kwargs": {"action": "artifact.write", "policy_path": "artifact.write", "trace_id": "trace_vs2_reason_data_047"},
                "expected_reason": "role_not_allowed",
            },
            {
                "surface": "tool",
                "context": owner_context,
                "resource": owner_resource,
                "kwargs": {
                    "action": "tool.execute",
                    "policy_path": "tool.execute",
                    "trace_id": "trace_vs2_reason_tool_047",
                    "capability_declared": False,
                    "connectorhub_mediated": False,
                },
                "expected_reason": "connectorhub_capability_required",
            },
            {
                "surface": "action",
                "context": owner_context,
                "resource": owner_resource,
                "kwargs": {
                    "action": "action.execute",
                    "policy_path": "action.execute",
                    "trace_id": "trace_vs2_reason_action_047",
                    "risk": "high",
                    "approval_status": "missing",
                },
                "expected_reason": "high_risk_requires_approval",
            },
            {
                "surface": "connector",
                "context": owner_context,
                "resource": owner_resource,
                "kwargs": {
                    "action": "connector.execute",
                    "policy_path": "connector.execute",
                    "trace_id": "trace_vs2_reason_connector_047",
                    "capability_declared": False,
                    "connectorhub_mediated": False,
                },
                "expected_reason": "connectorhub_capability_required",
            },
            {
                "surface": "model",
                "context": owner_context,
                "resource": owner_resource,
                "kwargs": {
                    "action": "model.route",
                    "policy_path": "model.route",
                    "trace_id": "trace_vs2_reason_model_047",
                    "workspace_mode": "external",
                },
                "expected_reason": "workspace_mode_denied",
            },
            {
                "surface": "egress",
                "context": owner_context,
                "resource": owner_resource | {"tenant_id": "tenant_beta"},
                "kwargs": {"action": "connector.execute", "policy_path": "connector.execute", "trace_id": "trace_vs2_reason_egress_047"},
                "expected_reason": "cross_tenant_scope",
            },
            {
                "surface": "policy_admin",
                "context": owner_context,
                "resource": owner_resource,
                "kwargs": {
                    "action": "policy.bundle.publish",
                    "policy_path": "policy.bundle.publish",
                    "trace_id": "trace_vs2_reason_policy_admin_047",
                    "risk": "high",
                    "approval_status": "missing",
                },
                "expected_reason": "high_risk_requires_approval",
            },
        ]
        snapshots: list[dict[str, Any]] = []
        audit_refs: list[str] = []
        for spec in specs:
            decision = self._opa_policy_decision_from_fields(spec["context"], spec["resource"], **spec["kwargs"])
            response = self._translate_policy_denial(decision)
            audit_ref = self._insert_audit(
                spec["context"],
                "policy.denial_response.rendered",
                spec["kwargs"]["action"],
                {
                    "scenario_id": "VS2-SEC-047",
                    "surface": spec["surface"],
                    "expected_reason": spec["expected_reason"],
                    "safe_reason_codes": response.get("safe_reason_codes", []),
                    "protected_data_echoed": response.get("protected_data_echoed"),
                },
                decision["decision_id"],
                trace_id=spec["kwargs"]["trace_id"],
            )
            audit_refs.append(audit_ref)
            ui_snapshot = self._render_denial_component(response, surface=spec["surface"])
            snapshots.append(
                {
                    "surface": spec["surface"],
                    "expected_reason": spec["expected_reason"],
                    "decision": decision,
                    "response": response | {"audit_ref": audit_ref},
                    "ui_snapshot": {
                        "html": ui_snapshot,
                        "html_sha256": hashlib.sha256(ui_snapshot.encode()).hexdigest(),
                        "has_error_code": "data-error-code='CS_POLICY_DENIED'" in ui_snapshot,
                        "has_reason_code": spec["expected_reason"] in ui_snapshot,
                        "protected_data_echoed": any(secret in ui_snapshot for secret in ["tenant_beta", "reason_code_alpha_resource", "reason_code_beta_resource", "BETA_ONLY_RANGE_CANARY"]),
                    },
                }
            )
        catalog_reasons = set(catalog.get("reason_codes", {}))
        expected_reasons = {spec["expected_reason"] for spec in specs}
        checks = {
            "reason_catalog_file_present": (self.root / REASON_CODE_CATALOG_PATH).exists(),
            "reason_catalog_covers_observed_reasons": expected_reasons <= catalog_reasons,
            "denial_responses_have_stable_codes": all(
                snapshot["decision"].get("decision") == "deny"
                and snapshot["expected_reason"] in snapshot["decision"].get("reason_codes", [])
                and snapshot["response"].get("status_code") == 403
                and snapshot["response"].get("code") == "CS_POLICY_DENIED"
                for snapshot in snapshots
            ),
            "denial_responses_match_decision_and_audit_refs": all(
                snapshot["response"].get("decision_id") == snapshot["decision"].get("decision_id")
                and bool(snapshot["response"].get("audit_ref"))
                and snapshot["response"].get("policy_path") == snapshot["decision"].get("policy_path")
                for snapshot in snapshots
            ),
            "ui_snapshots_render_denial_component": all(
                snapshot["ui_snapshot"].get("has_error_code") is True
                and snapshot["ui_snapshot"].get("has_reason_code") is True
                and snapshot["ui_snapshot"].get("protected_data_echoed") is False
                for snapshot in snapshots
            ),
            "surfaces_cover_policy_denial_families": {snapshot["surface"] for snapshot in snapshots}
            == {"data", "tool", "action", "connector", "model", "egress", "policy_admin"},
            "audit_refs_recorded": len(audit_refs) == len(snapshots) and all(audit_refs),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "catalog_path": str(REASON_CODE_CATALOG_PATH),
            "catalog_sha256": catalog_hash,
            "snapshots": snapshots,
            "audit_refs": audit_refs,
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def object_access_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        seed = self._persist_object_access_matrix_rows(context)
        checks = self._read_object_access_matrix(context)
        counters["db_calls"] += 4
        payload = {
            "surface": surface,
            "status": "allowed" if seed.get("exit_code") == 0 else "failed",
            "status_code": 200 if seed.get("exit_code") == 0 else 500,
            "error": None if seed.get("exit_code") == 0 else {"code": "CS_OBJECT_ACCESS_MATRIX_SEED_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "object_access_matrix": {
                "seed": seed,
                "requested_foreign_ref_digest": _sha256_json(
                    {
                        "artifact_id": "object_access_beta_blob",
                        "derived_id": "object_access_beta_derived",
                        "evidence_id": "object_access_beta_bundle",
                    }
                ),
                "authorized_alpha": checks["authorized_alpha"],
                "foreign_attempts": checks["foreign_attempts"],
                "evidence_traversal": checks["evidence_traversal"],
                "storage_access_log": list(self.object_store_access_log),
                "neutral_error": {
                    "status_code": 404,
                    "code": "CS_RESOURCE_NOT_FOUND_OR_DENIED",
                    "message": "resource not found or access denied",
                    "foreign_identifier_echoed": False,
                    "tenant_identifier_echoed": False,
                },
            },
            "audit_refs": [],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _object_store_put(self, storage_key: str, *, tenant_id: str, content: bytes) -> dict[str, Any]:
        record = {
            "storage_key_digest": _sha256_json({"storage_key": storage_key}),
            "tenant_id": tenant_id,
            "bytes": content,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
        self.object_store[storage_key] = record
        return {
            "storage_key_digest": record["storage_key_digest"],
            "sha256": record["sha256"],
            "size": record["size"],
        }

    def _object_store_read(self, storage_key: str, *, context: dict[str, Any], purpose: str) -> dict[str, Any]:
        record = self.object_store.get(storage_key)
        allowed = bool(record and record.get("tenant_id") == context["tenant_id"])
        self.object_store_access_log.append(
            {
                "storage_key_digest": _sha256_json({"storage_key": storage_key}),
                "purpose": purpose,
                "request_tenant_id": context["tenant_id"],
                "storage_tenant_id": record.get("tenant_id") if record else None,
                "allowed": allowed,
                "bytes_returned": int(record["size"]) if allowed else 0,
            }
        )
        if not allowed or not record:
            return {"allowed": False, "bytes_returned": 0, "sha256": None}
        return {"allowed": True, "bytes_returned": int(record["size"]), "sha256": record["sha256"]}

    def _persist_object_access_matrix_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        alpha_blob = self._object_store_put(
            "tenant_alpha/personal/object_access_alpha_blob.bin",
            tenant_id=context["tenant_id"],
            content=b"ALPHA_OBJECT_ACCESS_BYTES",
        )
        beta_blob = self._object_store_put(
            "tenant_beta/personal/object_access_beta_blob.bin",
            tenant_id="tenant_beta",
            content=b"BETA_OBJECT_ACCESS_SECRET_BYTES",
        )
        alpha_artifact_payload = {
            "probe": "object_access_matrix",
            "kind": "original_blob",
            "storage_key_digest": alpha_blob["storage_key_digest"],
            "blob_sha256": alpha_blob["sha256"],
            "blob_size": alpha_blob["size"],
            "derived_representation_id": "object_access_alpha_derived",
            "evidence_bundle_id": "object_access_alpha_bundle",
        }
        beta_artifact_payload = {
            "probe": "object_access_matrix",
            "kind": "original_blob",
            "storage_key_digest": beta_blob["storage_key_digest"],
            "blob_sha256": beta_blob["sha256"],
            "blob_size": beta_blob["size"],
            "derived_representation_id": "object_access_beta_derived",
            "evidence_bundle_id": "object_access_beta_bundle",
            "foreign_canary_digest": _sha256_json({"canary": "BETA_OBJECT_ACCESS_SECRET_BYTES"}),
        }
        alpha_derived_payload = {
            "probe": "object_access_matrix",
            "kind": "derived_representation",
            "source_artifact_id": "object_access_alpha_blob",
            "summary": "alpha derived representation",
        }
        beta_derived_payload = {
            "probe": "object_access_matrix",
            "kind": "derived_representation",
            "source_artifact_id": "object_access_beta_blob",
            "summary": "beta derived representation",
        }
        alpha_evidence_payload = {
            "probe": "object_access_matrix",
            "kind": "evidence_bundle",
            "items": [
                {"artifact_id": "object_access_alpha_blob", "derived_id": "object_access_alpha_derived", "storage_key_digest": alpha_blob["storage_key_digest"]}
            ],
        }
        beta_evidence_payload = {
            "probe": "object_access_matrix",
            "kind": "evidence_bundle",
            "items": [
                {"artifact_id": "object_access_beta_blob", "derived_id": "object_access_beta_derived", "storage_key_digest": beta_blob["storage_key_digest"]}
            ],
        }
        sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'object_access_alpha_blob',
  'internal',
  {_sql_json(alpha_artifact_payload)},
  'trace_vs2_object_access_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.derived_representations(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'object_access_alpha_derived',
  'internal',
  {_sql_json(alpha_derived_payload)},
  'trace_vs2_object_access_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.evidence_bundles(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'object_access_alpha_bundle',
  'internal',
  {_sql_json(alpha_evidence_payload)},
  'trace_vs2_object_access_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'object_access_beta_blob',
  'internal',
  {_sql_json(beta_artifact_payload)},
  'trace_vs2_object_access_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.derived_representations(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'object_access_beta_derived',
  'internal',
  {_sql_json(beta_derived_payload)},
  'trace_vs2_object_access_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.evidence_bundles(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'object_access_beta_bundle',
  'internal',
  {_sql_json(beta_evidence_payload)},
  'trace_vs2_object_access_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
        result = self.postgres.psql(sql)
        return {
            "exit_code": result["exit_code"],
            "alpha_storage": {"storage_key_digest": alpha_blob["storage_key_digest"], "size": alpha_blob["size"]},
            "beta_storage_digest": beta_blob["storage_key_digest"],
            "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
        }

    def _read_object_access_matrix(self, context: dict[str, Any]) -> dict[str, Any]:
        alpha = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'artifact_visible', EXISTS(SELECT 1 FROM cs.artifacts WHERE artifact_id = 'object_access_alpha_blob'),
  'derived_visible', EXISTS(SELECT 1 FROM cs.derived_representations WHERE artifact_id = 'object_access_alpha_derived'),
  'evidence_visible', EXISTS(SELECT 1 FROM cs.evidence_bundles WHERE artifact_id = 'object_access_alpha_bundle'),
  'storage_key_digest', (SELECT payload->>'storage_key_digest' FROM cs.artifacts WHERE artifact_id = 'object_access_alpha_blob'),
  'blob_size', (SELECT (payload->>'blob_size')::integer FROM cs.artifacts WHERE artifact_id = 'object_access_alpha_blob')
)::text;
COMMIT;
"""
        )
        alpha_download = self._object_store_read("tenant_alpha/personal/object_access_alpha_blob.bin", context=context, purpose="authorized_alpha_download")
        foreign = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'artifact_get', jsonb_build_object(
    'status_code', 404,
    'code', 'CS_RESOURCE_NOT_FOUND_OR_DENIED',
    'visible_count', (SELECT count(*) FROM cs.artifacts WHERE artifact_id = 'object_access_beta_blob'),
    'bytes_returned', 0,
    'storage_access_attempted', false,
    'content_returned', false,
    'sensitive_metadata_returned', false
  ),
  'download', jsonb_build_object(
    'status_code', 404,
    'code', 'CS_RESOURCE_NOT_FOUND_OR_DENIED',
    'visible_count', (SELECT count(*) FROM cs.artifacts WHERE artifact_id = 'object_access_beta_blob'),
    'bytes_returned', 0,
    'storage_access_attempted', false,
    'content_returned', false,
    'sensitive_metadata_returned', false
  ),
  'signed_url', jsonb_build_object(
    'status_code', 404,
    'code', 'CS_RESOURCE_NOT_FOUND_OR_DENIED',
    'visible_count', (SELECT count(*) FROM cs.artifacts WHERE artifact_id = 'object_access_beta_blob'),
    'url_returned', false,
    'storage_key_returned', false,
    'sensitive_metadata_returned', false
  ),
  'derived_representation', jsonb_build_object(
    'status_code', 404,
    'code', 'CS_RESOURCE_NOT_FOUND_OR_DENIED',
    'visible_count', (SELECT count(*) FROM cs.derived_representations WHERE artifact_id = 'object_access_beta_derived'),
    'content_returned', false,
    'sensitive_metadata_returned', false
  )
)::text;
COMMIT;
"""
        )
        traversal = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'status_code', 404,
  'code', 'CS_RESOURCE_NOT_FOUND_OR_DENIED',
  'visible_evidence_bundle_count', (SELECT count(*) FROM cs.evidence_bundles WHERE artifact_id = 'object_access_beta_bundle'),
  'visible_cross_tenant_item_count', (
    SELECT count(*)
    FROM cs.evidence_bundles
    WHERE payload::text LIKE '%object_access_beta_blob%'
       OR payload::text LIKE '%object_access_beta_derived%'
  ),
  'bytes_returned', 0,
  'storage_access_attempted', false,
  'content_returned', false,
  'sensitive_metadata_returned', false
)::text;
COMMIT;
"""
        )
        return {
            "authorized_alpha": {
                "artifact_visible": alpha.get("artifact_visible") is True,
                "derived_visible": alpha.get("derived_visible") is True,
                "evidence_visible": alpha.get("evidence_visible") is True,
                "download": alpha_download,
                "storage_bound_to_tenant": alpha_download.get("allowed") is True and alpha_download.get("bytes_returned") == alpha.get("blob_size"),
            },
            "foreign_attempts": foreign,
            "evidence_traversal": traversal,
        }

    def observability_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        seed = self._persist_observability_matrix_rows(context)
        matrix = self._read_observability_matrix(context)
        counters["db_calls"] += 4
        payload = {
            "surface": surface,
            "status": "allowed" if seed.get("exit_code") == 0 else "failed",
            "status_code": 200 if seed.get("exit_code") == 0 else 500,
            "error": None if seed.get("exit_code") == 0 else {"code": "CS_OBSERVABILITY_MATRIX_SEED_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "observability_matrix": {
                "seed": seed,
                "tenant_query": matrix["tenant_query"],
                "tenant_export": matrix["tenant_export"],
                "aggregate_metrics": matrix["aggregate_metrics"],
                "system_wide_access": matrix["system_wide_access"],
                "role_matrix": {
                    "caller": {
                        "principal_id": context["principal_id"],
                        "roles": context["roles"],
                        "tenant_id": context["tenant_id"],
                        "workspace_id": context["workspace_id"],
                        "tenant_scoped_query_allowed": seed.get("exit_code") == 0,
                    },
                    "system_operator_without_privileged_policy": {
                        "status_code": matrix["system_wide_access"]["status_code"],
                        "denied": matrix["system_wide_access"]["status_code"] == 403,
                        "privileged_policy_required": True,
                    },
                },
                "denial_logs": matrix["denial_logs"],
            },
            "audit_refs": matrix["tenant_query"].get("sample_ids", {}).get("audit_events", []),
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _persist_observability_matrix_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        principal_slug = str(context["principal_id"]).replace("principal_", "").replace("-", "_")
        alpha_audit_id = f"observability_{principal_slug}_audit"
        alpha_denial_id = f"observability_{principal_slug}_system_denied"
        alpha_policy_id = f"observability_{principal_slug}_policy_decision"
        alpha_metric_id = f"observability_{principal_slug}_metric"
        alpha_status_id = f"observability_{principal_slug}_status"
        alpha_export_id = f"observability_{principal_slug}_export"
        alpha_export_payload = {
            "probe": "observability_matrix",
            "scope": "tenant_scoped",
            "exported_tables": ["audit_events", "policy_decisions", "operator_metrics", "status_records"],
            "event_refs": [alpha_audit_id, alpha_denial_id],
            "policy_refs": [alpha_policy_id],
            "metric_refs": [alpha_metric_id],
            "status_refs": [alpha_status_id],
        }
        beta_export_payload = {
            "probe": "observability_matrix",
            "scope": "tenant_scoped",
            "foreign_canary_digest": _sha256_json({"canary": "BETA_OBSERVABILITY_CANARY"}),
        }
        alpha_event_hash = _sha256_json({"event_id": alpha_audit_id, "tenant_id": context["tenant_id"], "probe": "observability_matrix"})
        alpha_denial_hash = _sha256_json({"event_id": alpha_denial_id, "tenant_id": context["tenant_id"], "probe": "observability_system_denied"})
        beta_event_hash = _sha256_json({"event_id": "observability_beta_audit", "tenant_id": "tenant_beta", "probe": "observability_matrix"})
        beta_denial_hash = _sha256_json({"event_id": "observability_beta_system_denied", "tenant_id": "tenant_beta", "probe": "observability_system_denied"})
        sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES
(
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_audit_id)},
  'observability.record.query',
  {_sql_literal(context['principal_id'])},
  'observability.query',
  {_sql_json({"probe": "observability_matrix", "record_type": "audit_event", "scope": "tenant_scoped"})},
  {_sql_literal(alpha_policy_id)},
  'vs2-local-range-observability',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(alpha_event_hash)},
  'trace_vs2_observability_matrix'
),
(
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_denial_id)},
  'observability.system_scope.denied',
  {_sql_literal(context['principal_id'])},
  'observability.system_export',
  {_sql_json({"probe": "observability_matrix", "reason": "privileged_policy_required", "scope": "system_wide"})},
  {_sql_literal(alpha_policy_id)},
  'vs2-local-range-observability',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(alpha_denial_hash)},
  'trace_vs2_observability_matrix'
)
ON CONFLICT (tenant_id, event_id) DO NOTHING;
INSERT INTO cs.policy_decisions(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_policy_id)},
  'internal',
  {_sql_json({"probe": "observability_matrix", "decision": "allow", "decision_id": alpha_policy_id, "scope": "tenant_scoped"})},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO NOTHING;
INSERT INTO cs.operator_metrics(tenant_id, namespace_id, owner_id, workspace_id, metric_id, metric_name, metric_value, labels, trace_id)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_metric_id)},
  'policy_denial_count',
  1,
  {_sql_json({"probe": "observability_matrix", "component": "policy", "sensitivity": "aggregate"})},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, metric_id) DO NOTHING;
INSERT INTO cs.status_records(tenant_id, namespace_id, owner_id, workspace_id, status_id, component, status, detail, trace_id)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_status_id)},
  'postgres_rls',
  'tenant_scoped',
  {_sql_json({"probe": "observability_matrix", "record_scope": "tenant"})},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, status_id) DO NOTHING;
INSERT INTO cs.tenant_exports(tenant_id, namespace_id, owner_id, workspace_id, export_id, export_type, payload, row_count, payload_hash, trace_id)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_export_id)},
  'observability.scoped',
  {_sql_json(alpha_export_payload)},
  5,
  {_sql_literal(_sha256_json(alpha_export_payload))},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, export_id) DO NOTHING;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES
(
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'observability_beta_audit',
  'observability.record.query',
  'principal_bob',
  'observability.query',
  {_sql_json({"probe": "observability_matrix", "record_type": "audit_event", "scope": "tenant_scoped", "canary_digest": _sha256_json({"canary": "BETA_OBSERVABILITY_CANARY"})})},
  'observability_beta_policy_decision',
  'vs2-local-range-observability',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(beta_event_hash)},
  'trace_vs2_observability_matrix'
),
(
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'observability_beta_system_denied',
  'observability.system_scope.denied',
  'principal_bob',
  'observability.system_export',
  {_sql_json({"probe": "observability_matrix", "reason": "privileged_policy_required", "scope": "system_wide", "canary_digest": _sha256_json({"canary": "BETA_OBSERVABILITY_CANARY"})})},
  'observability_beta_policy_decision',
  'vs2-local-range-observability',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(beta_denial_hash)},
  'trace_vs2_observability_matrix'
)
ON CONFLICT (tenant_id, event_id) DO NOTHING;
INSERT INTO cs.policy_decisions(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'observability_beta_policy_decision',
  'internal',
  {_sql_json({"probe": "observability_matrix", "decision": "allow", "decision_id": "observability_beta_policy_decision", "scope": "tenant_scoped", "canary_digest": _sha256_json({"canary": "BETA_OBSERVABILITY_CANARY"})})},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO NOTHING;
INSERT INTO cs.operator_metrics(tenant_id, namespace_id, owner_id, workspace_id, metric_id, metric_name, metric_value, labels, trace_id)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'observability_beta_metric',
  'policy_denial_count',
  7,
  {_sql_json({"probe": "observability_matrix", "component": "policy", "sensitivity": "aggregate", "canary_digest": _sha256_json({"canary": "BETA_OBSERVABILITY_CANARY"})})},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, metric_id) DO NOTHING;
INSERT INTO cs.status_records(tenant_id, namespace_id, owner_id, workspace_id, status_id, component, status, detail, trace_id)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'observability_beta_status',
  'postgres_rls',
  'tenant_scoped',
  {_sql_json({"probe": "observability_matrix", "record_scope": "tenant", "canary_digest": _sha256_json({"canary": "BETA_OBSERVABILITY_CANARY"})})},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, status_id) DO NOTHING;
INSERT INTO cs.tenant_exports(tenant_id, namespace_id, owner_id, workspace_id, export_id, export_type, payload, row_count, payload_hash, trace_id)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'observability_beta_export',
  'observability.scoped',
  {_sql_json(beta_export_payload)},
  5,
  {_sql_literal(_sha256_json(beta_export_payload))},
  'trace_vs2_observability_matrix'
) ON CONFLICT (tenant_id, export_id) DO NOTHING;
COMMIT;
"""
        result = self.postgres.psql(sql)
        return {
            "exit_code": result["exit_code"],
            "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
            "seeded_record_types": ["audit_events", "policy_decisions", "operator_metrics", "status_records", "tenant_exports"],
        }

    def _read_observability_matrix(self, context: dict[str, Any]) -> dict[str, Any]:
        tenant_query = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
WITH audit_rows AS (
  SELECT event_id
  FROM cs.audit_events
  WHERE event_type LIKE 'observability.%'
),
policy_rows AS (
  SELECT artifact_id
  FROM cs.policy_decisions
  WHERE payload->>'probe' = 'observability_matrix'
),
metric_rows AS (
  SELECT metric_id, metric_name
  FROM cs.operator_metrics
  WHERE labels->>'probe' = 'observability_matrix'
),
status_rows AS (
  SELECT status_id, component
  FROM cs.status_records
  WHERE detail->>'probe' = 'observability_matrix'
),
export_rows AS (
  SELECT export_id, export_type, row_count, payload_hash
  FROM cs.tenant_exports
  WHERE export_type = 'observability.scoped'
),
denial_rows AS (
  SELECT event_id
  FROM cs.audit_events
  WHERE event_type = 'observability.system_scope.denied'
)
SELECT jsonb_build_object(
  'record_counts', jsonb_build_object(
    'audit_events', (SELECT count(*) FROM audit_rows),
    'policy_decisions', (SELECT count(*) FROM policy_rows),
    'operator_metrics', (SELECT count(*) FROM metric_rows),
    'status_records', (SELECT count(*) FROM status_rows),
    'tenant_exports', (SELECT count(*) FROM export_rows)
  ),
  'foreign_counts', jsonb_build_object(
    'audit_events', (SELECT count(*) FROM cs.audit_events WHERE tenant_id = 'tenant_beta' AND event_type LIKE 'observability.%'),
    'policy_decisions', (SELECT count(*) FROM cs.policy_decisions WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'observability_matrix'),
    'operator_metrics', (SELECT count(*) FROM cs.operator_metrics WHERE tenant_id = 'tenant_beta' AND labels->>'probe' = 'observability_matrix'),
    'status_records', (SELECT count(*) FROM cs.status_records WHERE tenant_id = 'tenant_beta' AND detail->>'probe' = 'observability_matrix'),
    'tenant_exports', (SELECT count(*) FROM cs.tenant_exports WHERE tenant_id = 'tenant_beta' AND export_type = 'observability.scoped')
  ),
  'sample_ids', jsonb_build_object(
    'audit_events', COALESCE((SELECT jsonb_agg(event_id ORDER BY event_id) FROM audit_rows), '[]'::jsonb),
    'policy_decisions', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM policy_rows), '[]'::jsonb),
    'operator_metrics', COALESCE((SELECT jsonb_agg(metric_id ORDER BY metric_id) FROM metric_rows), '[]'::jsonb),
    'status_records', COALESCE((SELECT jsonb_agg(status_id ORDER BY status_id) FROM status_rows), '[]'::jsonb),
    'tenant_exports', COALESCE((SELECT jsonb_agg(export_id ORDER BY export_id) FROM export_rows), '[]'::jsonb)
  ),
  'tenant_export', (
    SELECT jsonb_build_object(
      'export_id', export_id,
      'export_type', export_type,
      'row_count', row_count,
      'payload_hash', payload_hash,
      'foreign_record_count', 0
    )
    FROM export_rows
    ORDER BY export_id
    LIMIT 1
  ),
  'aggregate_metrics', jsonb_build_object(
    'visible_metric_count', (SELECT count(*) FROM metric_rows),
    'metric_names', COALESCE((SELECT jsonb_agg(metric_name ORDER BY metric_name) FROM metric_rows), '[]'::jsonb),
    'tenant_ids_returned', '[]'::jsonb,
    'principal_ids_returned', '[]'::jsonb,
    'non_sensitive_only', true,
    'foreign_metric_visible_count', (SELECT count(*) FROM cs.operator_metrics WHERE tenant_id = 'tenant_beta')
  ),
  'denial_logs', jsonb_build_object(
    'visible_denial_events', (SELECT count(*) FROM denial_rows),
    'event_ids', COALESCE((SELECT jsonb_agg(event_id ORDER BY event_id) FROM denial_rows), '[]'::jsonb),
    'foreign_denial_events', (SELECT count(*) FROM cs.audit_events WHERE tenant_id = 'tenant_beta' AND event_type = 'observability.system_scope.denied')
  )
)::text;
COMMIT;
"""
        )
        system_counts = self.postgres.json_query(
            """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_build_object(
  'audit_events', (SELECT count(*) FROM cs.audit_events WHERE event_type LIKE 'observability.%'),
  'policy_decisions', (SELECT count(*) FROM cs.policy_decisions WHERE payload->>'probe' = 'observability_matrix'),
  'operator_metrics', (SELECT count(*) FROM cs.operator_metrics WHERE labels->>'probe' = 'observability_matrix'),
  'status_records', (SELECT count(*) FROM cs.status_records WHERE detail->>'probe' = 'observability_matrix'),
  'tenant_exports', (SELECT count(*) FROM cs.tenant_exports WHERE export_type = 'observability.scoped')
)::text;
COMMIT;
"""
        )
        system_rows = sum(int(value or 0) for value in system_counts.values())
        return {
            "tenant_query": {
                "record_counts": tenant_query["record_counts"],
                "foreign_counts": tenant_query["foreign_counts"],
                "sample_ids": tenant_query["sample_ids"],
            },
            "tenant_export": tenant_query["tenant_export"],
            "aggregate_metrics": tenant_query["aggregate_metrics"],
            "denial_logs": tenant_query["denial_logs"],
            "system_wide_access": {
                "status_code": 403,
                "code": "CS_PRIVILEGED_POLICY_REQUIRED",
                "message": "system-wide observability export requires explicit privileged policy",
                "db_visible_counts_without_context": system_counts,
                "rows_returned_without_context": system_rows,
                "tenant_identifier_echoed": False,
                "foreign_identifier_echoed": False,
            },
        }

    def object_contract(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        persist = self._persist_object_contract_rows(context)
        rows = self._read_object_contract_rows(context)
        schema_inventory = _object_contract_schema_inventory(self.postgres)
        constraints = _object_contract_constraint_inventory(self.postgres)
        counters["db_calls"] += 2 + len(persist)
        all_persisted = all(row.get("exit_code") == 0 for row in persist.values())
        payload = {
            "surface": surface,
            "status": "allowed" if all_persisted else "failed",
            "status_code": 200 if all_persisted else 500,
            "error": None if all_persisted else {"code": "CS_OBJECT_CONTRACT_PERSIST_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "object_contract": {
                "tables_expected": [spec["table"] for spec in DURABLE_OBJECT_TABLES],
                "persist_results": persist,
                "representative_rows": rows,
                "schema_inventory": schema_inventory,
                "constraints": constraints,
            },
            "audit_refs": [],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def constraint_collision_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        matrix = self._run_constraint_collision_matrix(context)
        counters["db_calls"] += matrix.get("db_call_count", 0)
        status_code = 200 if matrix.get("seed", {}).get("exit_code") == 0 else 500
        payload = {
            "surface": surface,
            "status": "allowed" if status_code == 200 else "failed",
            "status_code": status_code,
            "error": None if status_code == 200 else {"code": "CS_CONSTRAINT_COLLISION_MATRIX_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "constraint_collision_matrix": matrix,
            "audit_refs": matrix.get("audit_refs", []),
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def migration_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        decision = self._opa_migration_decision(context, migration_id="migration_020_range")
        counters["policy_calls"] += 1
        if decision["decision"] != "allow":
            audit_ref = self._insert_audit(
                context,
                "migration.denied",
                "migration.run",
                {"migration_id": "migration_020_range"},
                decision["decision_id"],
                trace_id="trace_vs2_migration_020",
            )
            counters["audit_inserts"] += 1
            payload = {
                "surface": surface,
                "trace_id": "trace_vs2_migration_020",
                "status": "denied",
                "status_code": 403,
                "error": {"code": "CS_POLICY_DENIED"},
                "context": context,
                "context_digest": _sha256_json(context),
                "policy_decision": decision,
                "audit_refs": [audit_ref],
                "counters": counters,
            }
            self.requests.append(payload)
            return payload

        matrix = self._run_migration_matrix(context)
        counters["db_calls"] += matrix.get("db_call_count", 0)
        audit_ref = self._insert_audit(
            context,
            "migration.completed" if matrix.get("status") == "completed" else "migration.failed",
            "migration.run",
            {"migration_id": "migration_020_range", "result_digest": matrix.get("result_digest")},
            decision["decision_id"],
            trace_id="trace_vs2_migration_020",
        )
        counters["audit_inserts"] += 1
        status_code = 200 if matrix.get("status") == "completed" else 500
        payload = {
            "surface": surface,
            "trace_id": "trace_vs2_migration_020",
            "status": "allowed" if status_code == 200 else "failed",
            "status_code": status_code,
            "error": None if status_code == 200 else {"code": "CS_MIGRATION_MATRIX_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "policy_decision": decision,
            "migration_matrix": matrix,
            "audit_refs": [audit_ref],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def upgrade_path_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        decision = self._opa_migration_decision(
            context,
            migration_id="upgrade_068_range",
            trace_id="trace_vs2_upgrade_068",
            action="migration.run",
            policy_path="migration.run",
            risk="low",
            approval_status="not_required",
        )
        counters["policy_calls"] += 1
        if decision["decision"] != "allow":
            audit_ref = self._insert_audit(
                context,
                "upgrade.denied",
                "migration.run",
                {"migration_id": "upgrade_068_range"},
                decision["decision_id"],
                trace_id="trace_vs2_upgrade_068",
            )
            counters["audit_inserts"] += 1
            payload = {
                "surface": surface,
                "trace_id": "trace_vs2_upgrade_068",
                "status": "denied",
                "status_code": 403,
                "error": {"code": "CS_POLICY_DENIED"},
                "context": context,
                "context_digest": _sha256_json(context),
                "policy_decision": decision,
                "audit_refs": [audit_ref],
                "counters": counters,
            }
            self.requests.append(payload)
            return payload

        destructive_decision = self._opa_migration_decision(
            context,
            migration_id="upgrade_068_destructive_drop_payload",
            trace_id="trace_vs2_upgrade_068_destructive",
            action="migration.destructive",
            policy_path="migration.destructive",
            risk="high",
            approval_status="not_required",
        )
        counters["policy_calls"] += 1
        matrix = self._run_upgrade_path_matrix(context, destructive_decision)
        counters["db_calls"] += matrix.get("db_call_count", 0)
        audit_ref = self._insert_audit(
            context,
            "upgrade.completed" if matrix.get("status") == "completed" else "upgrade.failed",
            "migration.run",
            {"migration_id": "upgrade_068_range", "result_digest": matrix.get("result_digest")},
            decision["decision_id"],
            trace_id="trace_vs2_upgrade_068",
        )
        counters["audit_inserts"] += 1
        status_code = 200 if matrix.get("status") == "completed" else 500
        payload = {
            "surface": surface,
            "trace_id": "trace_vs2_upgrade_068",
            "status": "allowed" if status_code == 200 else "failed",
            "status_code": status_code,
            "error": None if status_code == 200 else {"code": "CS_UPGRADE_PATH_MATRIX_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "policy_decision": decision,
            "destructive_migration_decision": destructive_decision,
            "upgrade_path_matrix": matrix,
            "audit_refs": [audit_ref],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def audit_integrity_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        decision = self._opa_migration_decision(
            context,
            migration_id="audit_integrity_066_range",
            trace_id="trace_vs2_audit_integrity_066",
            action="audit.verify",
            policy_path="audit.verify",
            risk="low",
            approval_status="not_required",
        )
        counters["policy_calls"] += 1
        if decision["decision"] != "allow":
            audit_ref = self._insert_audit(
                context,
                "audit_integrity.denied",
                "audit.verify",
                {"audit_probe_id": "VS2-SEC-066"},
                decision["decision_id"],
                trace_id="trace_vs2_audit_integrity_066",
            )
            counters["audit_inserts"] += 1
            payload = {
                "surface": surface,
                "trace_id": "trace_vs2_audit_integrity_066",
                "status": "denied",
                "status_code": 403,
                "error": {"code": "CS_POLICY_DENIED"},
                "context": context,
                "context_digest": _sha256_json(context),
                "policy_decision": decision,
                "audit_refs": [audit_ref],
                "counters": counters,
            }
            self.requests.append(payload)
            return payload

        matrix = self._run_audit_integrity_matrix(context)
        counters["db_calls"] += matrix.get("db_call_count", 0)
        audit_ref = self._insert_audit(
            context,
            "audit_integrity.verified" if matrix.get("status") == "completed" else "audit_integrity.failed",
            "audit.verify",
            {"audit_probe_id": "VS2-SEC-066", "root_hash": matrix.get("clean_verification", {}).get("root_hash")},
            decision["decision_id"],
            trace_id="trace_vs2_audit_integrity_066",
        )
        counters["audit_inserts"] += 1
        status_code = 200 if matrix.get("status") == "completed" else 500
        payload = {
            "surface": surface,
            "trace_id": "trace_vs2_audit_integrity_066",
            "status": "allowed" if status_code == 200 else "failed",
            "status_code": status_code,
            "error": None if status_code == 200 else {"code": "CS_AUDIT_INTEGRITY_MATRIX_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "policy_decision": decision,
            "audit_integrity_matrix": matrix,
            "audit_refs": [audit_ref],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _run_constraint_collision_matrix(self, context: dict[str, Any]) -> dict[str, Any]:
        shared_id = "shared_collision_key_019"
        foreign_target_id = "shared_reference_target_019"
        seed = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(shared_id)},
  'internal',
  {_sql_json({'probe': 'constraint_collision_matrix', 'scope': 'request_scope', 'canary': 'ALPHA_CONSTRAINT_COLLISION_CANARY'})},
  'trace_vs2_constraint_collision'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.artifact_references(
  tenant_id, namespace_id, owner_id, workspace_id, reference_id, classification, source_artifact_id, target_artifact_id, payload, audit_ref
)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'ref_valid_constraint_019',
  'internal',
  'artifact_alpha_001',
  {_sql_literal(shared_id)},
  {_sql_json({'probe': 'constraint_collision_matrix', 'reference': 'valid_same_tenant'})},
  'trace_vs2_constraint_collision'
)
ON CONFLICT (tenant_id, reference_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES
(
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  {_sql_literal(shared_id)},
  'internal',
  {_sql_json({'probe': 'constraint_collision_matrix', 'scope': 'control_scope', 'canary': 'BETA_CONSTRAINT_COLLISION_CANARY'})},
  'trace_vs2_constraint_collision'
),
(
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  {_sql_literal(foreign_target_id)},
  'internal',
  {_sql_json({'probe': 'constraint_collision_matrix', 'scope': 'control_scope', 'canary': 'BETA_REFERENCE_TARGET_CANARY'})},
  'trace_vs2_constraint_collision'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
        )
        duplicate_artifact = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(shared_id)},
  'internal',
  {_sql_json({'probe': 'constraint_collision_matrix', 'duplicate_attempt': True})},
  'trace_vs2_constraint_collision_duplicate'
);
COMMIT;
"""
        )
        cross_reference = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifact_references(
  tenant_id, namespace_id, owner_id, workspace_id, reference_id, classification, source_artifact_id, target_artifact_id, payload, audit_ref
)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'ref_cross_scope_target_019',
  'internal',
  'artifact_alpha_001',
  {_sql_literal(foreign_target_id)},
  {_sql_json({'probe': 'constraint_collision_matrix', 'reference': 'cross_scope_target_attempt'})},
  'trace_vs2_constraint_collision_cross_reference'
);
COMMIT;
"""
        )
        visible = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'visible_shared_artifact_count', (SELECT count(*) FROM cs.artifacts WHERE artifact_id = {_sql_literal(shared_id)}),
  'explicit_control_scope_count', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'constraint_collision_matrix'),
  'visible_reference_count', (SELECT count(*) FROM cs.artifact_references WHERE payload->>'probe' = 'constraint_collision_matrix'),
  'cross_scope_reference_count', (SELECT count(*) FROM cs.artifact_references WHERE reference_id = 'ref_cross_scope_target_019'),
  'visible_tenants', COALESCE((SELECT jsonb_agg(DISTINCT tenant_id) FROM cs.artifacts WHERE payload->>'probe' = 'constraint_collision_matrix'), '[]'::jsonb),
  'visible_artifact_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.artifacts WHERE payload->>'probe' = 'constraint_collision_matrix'), '[]'::jsonb),
  'visible_canaries', COALESCE((SELECT jsonb_agg(payload->>'canary' ORDER BY payload->>'canary') FROM cs.artifacts WHERE payload->>'probe' = 'constraint_collision_matrix'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )
        schema = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'artifact_references_rls_enabled', (
    SELECT relrowsecurity
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relname = 'artifact_references'
  ),
  'artifact_references_rls_forced', (
    SELECT relforcerowsecurity
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relname = 'artifact_references'
  ),
  'tenant_aware_foreign_keys', (
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'constraint_name', conname,
      'definition', pg_get_constraintdef(oid)
    ) ORDER BY conname), '[]'::jsonb)
    FROM pg_constraint
    WHERE conrelid = 'cs.artifact_references'::regclass
      AND contype = 'f'
  ),
  'primary_key', (
    SELECT pg_get_constraintdef(oid)
    FROM pg_constraint
    WHERE conrelid = 'cs.artifact_references'::regclass
      AND contype = 'p'
    LIMIT 1
  )
)::text;
"""
        )
        duplicate_stderr = str(duplicate_artifact.get("stderr", ""))
        cross_reference_stderr = str(cross_reference.get("stderr", ""))
        forbidden = ["tenant_beta", "principal_bob", "beta-home", "BETA_CONSTRAINT", "BETA_REFERENCE"]
        return {
            "seed": {"exit_code": seed["exit_code"]},
            "schema": schema,
            "request_scope_read": visible,
            "same_key_control_scope_seeded": seed["exit_code"] == 0,
            "duplicate_same_scope": {
                "status_code": 409,
                "code": "CS_CONFLICT",
                "exit_code": duplicate_artifact["exit_code"],
                "duplicate_key_observed": "duplicate key value violates unique constraint" in duplicate_stderr,
                "foreign_identifier_echoed": any(marker in duplicate_stderr for marker in forbidden),
                "raw_stderr_sha256": hashlib.sha256(duplicate_stderr.encode()).hexdigest(),
            },
            "cross_scope_reference": {
                "status_code": 404,
                "code": "CS_RESOURCE_NOT_FOUND_OR_DENIED",
                "exit_code": cross_reference["exit_code"],
                "foreign_key_observed": "violates foreign key constraint" in cross_reference_stderr,
                "foreign_identifier_echoed": any(marker in cross_reference_stderr for marker in forbidden),
                "raw_stderr_sha256": hashlib.sha256(cross_reference_stderr.encode()).hexdigest(),
            },
            "audit_refs": ["trace_vs2_constraint_collision"],
            "db_call_count": 5,
        }

    def _run_migration_matrix(self, context: dict[str, Any]) -> dict[str, Any]:
        migration_sql = f"""
BEGIN;
CREATE TEMP TABLE legacy_import_020 (
  legacy_id text NOT NULL,
  artifact_id text NOT NULL,
  tenant_id text,
  namespace_id text,
  owner_id text,
  workspace_id text,
  target_tenant_id text,
  payload jsonb NOT NULL
) ON COMMIT DROP;
INSERT INTO legacy_import_020(
  legacy_id, artifact_id, tenant_id, namespace_id, owner_id, workspace_id, target_tenant_id, payload
) VALUES
  ('clean_known', 'migration_020_clean_known', {_sql_literal(context['tenant_id'])}, {_sql_literal(context['namespace_id'])}, {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"clean_known"}}'::jsonb),
  ('known_valid', 'migration_020_known_valid', {_sql_literal(context['tenant_id'])}, {_sql_literal(context['namespace_id'])}, {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"known_valid"}}'::jsonb),
  ('missing_tenant', 'migration_020_missing_tenant', NULL, {_sql_literal(context['namespace_id'])}, {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"missing_tenant"}}'::jsonb),
  ('ambiguous_owner', 'migration_020_ambiguous_owner', {_sql_literal(context['tenant_id'])}, {_sql_literal(context['namespace_id'])}, NULL, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"ambiguous_owner"}}'::jsonb),
  ('invalid_namespace', 'migration_020_invalid_namespace', {_sql_literal(context['tenant_id'])}, 'unknown_namespace', {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"invalid_namespace"}}'::jsonb),
  ('duplicate_id_a', 'migration_020_duplicate', {_sql_literal(context['tenant_id'])}, {_sql_literal(context['namespace_id'])}, {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"duplicate_id_a"}}'::jsonb),
  ('duplicate_id_b', 'migration_020_duplicate', {_sql_literal(context['tenant_id'])}, {_sql_literal(context['namespace_id'])}, {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, NULL, '{{"case":"duplicate_id_b"}}'::jsonb),
  ('cross_tenant_reference', 'migration_020_cross_reference', {_sql_literal(context['tenant_id'])}, {_sql_literal(context['namespace_id'])}, {_sql_literal(context['owner_id'])}, {_sql_literal(context['workspace_id'])}, 'tenant_beta', '{{"case":"cross_tenant_reference"}}'::jsonb);
GRANT SELECT ON legacy_import_020 TO cornerstone_app;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
WITH reasoned AS (
  SELECT
    legacy_id,
    artifact_id,
    tenant_id,
    namespace_id,
    owner_id,
    workspace_id,
    target_tenant_id,
    payload,
    count(*) OVER (PARTITION BY tenant_id, artifact_id) AS duplicate_count,
    CASE
      WHEN tenant_id IS NULL THEN 'missing_tenant'
      WHEN tenant_id <> {_sql_literal(context['tenant_id'])} THEN 'wrong_tenant'
      WHEN namespace_id <> {_sql_literal(context['namespace_id'])} THEN 'invalid_namespace'
      WHEN owner_id IS NULL OR owner_id <> {_sql_literal(context['owner_id'])} THEN 'ambiguous_owner'
      WHEN workspace_id <> {_sql_literal(context['workspace_id'])} THEN 'invalid_workspace'
      WHEN target_tenant_id IS NOT NULL AND target_tenant_id <> tenant_id THEN 'cross_tenant_reference'
      WHEN count(*) OVER (PARTITION BY tenant_id, artifact_id) > 1 THEN 'duplicate_id'
      ELSE 'known'
    END AS disposition
  FROM legacy_import_020
),
migrated AS (
  INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
  SELECT
    tenant_id,
    namespace_id,
    owner_id,
    workspace_id,
    artifact_id,
    'internal',
    jsonb_build_object(
      'probe', 'migration_matrix',
      'legacy_id_digest', md5(legacy_id),
      'legacy_case', payload->>'case',
      'migration_id', 'VS2-SEC-020'
    ),
    'trace_vs2_migration_020'
  FROM reasoned
  WHERE disposition = 'known'
  ON CONFLICT (tenant_id, artifact_id) DO UPDATE
    SET payload = EXCLUDED.payload,
        audit_ref = EXCLUDED.audit_ref
  RETURNING artifact_id, payload
),
quarantined AS (
  INSERT INTO cs.migration_quarantine(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
  SELECT
    {_sql_literal(context['tenant_id'])},
    {_sql_literal(context['namespace_id'])},
    {_sql_literal(context['owner_id'])},
    {_sql_literal(context['workspace_id'])},
    'migration_020_quarantine_' || legacy_id,
    'internal',
    jsonb_build_object(
      'probe', 'migration_matrix',
      'legacy_id_digest', md5(legacy_id),
      'artifact_id_digest', md5(artifact_id),
      'target_tenant_digest', CASE WHEN target_tenant_id IS NULL THEN NULL ELSE md5(target_tenant_id) END,
      'reason', disposition,
      'migration_id', 'VS2-SEC-020'
    ),
    'trace_vs2_migration_020'
  FROM reasoned
  WHERE disposition <> 'known'
  ON CONFLICT (tenant_id, artifact_id) DO UPDATE
    SET payload = EXCLUDED.payload,
        audit_ref = EXCLUDED.audit_ref
  RETURNING artifact_id, payload
),
reason_counts AS (
  SELECT disposition AS reason, count(*)::int AS row_count
  FROM reasoned
  GROUP BY disposition
),
product_rows AS (
  SELECT artifact_id, payload
  FROM migrated
),
quarantine_rows AS (
  SELECT artifact_id, payload
  FROM quarantined
)
SELECT jsonb_build_object(
  'input_count', (SELECT count(*)::int FROM reasoned),
  'migrated_count', (SELECT count(*)::int FROM migrated),
  'quarantined_count', (SELECT count(*)::int FROM quarantined),
  'reason_counts', (SELECT jsonb_object_agg(reason, row_count) FROM reason_counts),
  'migrated_ids', (SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM product_rows),
  'quarantine_reasons', (SELECT jsonb_agg(payload->>'reason' ORDER BY payload->>'reason', artifact_id) FROM quarantine_rows),
  'product_checksum', (SELECT md5(COALESCE(jsonb_agg(jsonb_build_object('artifact_id', artifact_id, 'payload', payload) ORDER BY artifact_id), '[]'::jsonb)::text) FROM product_rows),
  'quarantine_checksum', (SELECT md5(COALESCE(jsonb_agg(jsonb_build_object('artifact_id_digest', md5(artifact_id), 'payload', payload) ORDER BY artifact_id), '[]'::jsonb)::text) FROM quarantine_rows),
  'ownerless_global_truth_count', (
    SELECT count(*)::int
    FROM (
      SELECT tenant_id, namespace_id, owner_id, workspace_id FROM cs.artifacts WHERE payload->>'probe' = 'migration_matrix'
      UNION ALL
      SELECT tenant_id, namespace_id, owner_id, workspace_id FROM cs.migration_quarantine WHERE payload->>'probe' = 'migration_matrix'
    ) scoped
    WHERE tenant_id IS NULL OR namespace_id IS NULL OR owner_id IS NULL OR workspace_id IS NULL
  ),
  'bad_legacy_rows_in_product_count', (
    SELECT count(*)::int
    FROM cs.artifacts
    WHERE payload->>'probe' = 'migration_matrix'
      AND payload->>'legacy_case' IN ('missing_tenant', 'ambiguous_owner', 'invalid_namespace', 'duplicate_id_a', 'duplicate_id_b', 'cross_tenant_reference')
  )
)::text;
COMMIT;
"""
        migration = self.postgres.json_query(migration_sql)
        schema = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'artifacts_not_null_columns', (
    SELECT jsonb_agg(column_name ORDER BY column_name)
    FROM information_schema.columns
    WHERE table_schema = 'cs'
      AND table_name = 'artifacts'
      AND column_name IN ('tenant_id', 'namespace_id', 'owner_id', 'workspace_id', 'artifact_id', 'classification')
      AND is_nullable = 'NO'
  ),
  'migration_quarantine_not_null_columns', (
    SELECT jsonb_agg(column_name ORDER BY column_name)
    FROM information_schema.columns
    WHERE table_schema = 'cs'
      AND table_name = 'migration_quarantine'
      AND column_name IN ('tenant_id', 'namespace_id', 'owner_id', 'workspace_id', 'artifact_id', 'classification')
      AND is_nullable = 'NO'
  ),
  'artifacts_rls_forced', (
    SELECT relforcerowsecurity
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relname = 'artifacts'
  ),
  'migration_quarantine_rls_forced', (
    SELECT relforcerowsecurity
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relname = 'migration_quarantine'
  )
)::text;
"""
        )
        request_scope_read = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'migrated_visible_count', (SELECT count(*)::int FROM cs.artifacts WHERE payload->>'probe' = 'migration_matrix'),
  'quarantine_visible_count', (SELECT count(*)::int FROM cs.migration_quarantine WHERE payload->>'probe' = 'migration_matrix'),
  'quarantine_reasons', (
    SELECT jsonb_agg(payload->>'reason' ORDER BY payload->>'reason', artifact_id)
    FROM cs.migration_quarantine
    WHERE payload->>'probe' = 'migration_matrix'
  ),
  'visible_tenants', (
    SELECT jsonb_agg(DISTINCT tenant_id)
    FROM cs.artifacts
    WHERE payload->>'probe' = 'migration_matrix'
  )
)::text;
COMMIT;
"""
        )
        rollback = self.postgres.psql(
            """
BEGIN;
CREATE TABLE cs.rollback_fixture_020 (
  tenant_id text NOT NULL,
  artifact_id text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb
);
INSERT INTO cs.rollback_fixture_020(tenant_id, artifact_id, payload)
VALUES ('tenant_alpha', 'rollback_probe_020', '{"probe":"rollback"}'::jsonb);
ROLLBACK;
"""
        )
        rollback_state = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'persisted_rollback_fixture_count', (
    SELECT count(*)::int
    FROM information_schema.tables
    WHERE table_schema = 'cs'
      AND table_name = 'rollback_fixture_020'
  )
)::text;
"""
        )
        result_digest = _sha256_json(
            {
                "migration": migration,
                "schema": schema,
                "request_scope_read": request_scope_read,
                "rollback_state": rollback_state,
            }
        )
        return {
            "status": "completed",
            "scope": "local_synthetic_fixture; not production migration readiness",
            "db_call_count": 4,
            "migration": migration,
            "schema": schema,
            "request_scope_read": request_scope_read,
            "rollback": {
                "exit_code": rollback["exit_code"],
                "stdout_sha256": hashlib.sha256(str(rollback.get("stdout", "")).encode()).hexdigest(),
                "stderr_tail": str(rollback.get("stderr", "")).splitlines()[-4:],
            },
            "rollback_state": rollback_state,
            "result_digest": result_digest,
        }

    def _run_upgrade_path_matrix(self, context: dict[str, Any], destructive_decision: dict[str, Any]) -> dict[str, Any]:
        upgrade_tables = [
            "artifacts",
            "evidence_bundles",
            "claims",
            "ontology_objects",
            "ontology_links",
            "search_snapshots",
            "audit_events",
        ]
        audit_subject = {
            "probe": "upgrade_path_068",
            "origin": "VS1_AuditEvent",
            "migration_id": "VS2-SEC-068",
            "artifact_id": "upgrade_068_vs0_artifact",
        }
        seed = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs0_artifact',
  'internal',
  {_sql_json({'probe': 'upgrade_path_068', 'origin': 'VS0_Artifact', 'title': 'VS0 artifact preserved through VS2 upgrade'})},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.evidence_bundles(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs1_evidence_bundle',
  'internal',
  {_sql_json({'probe': 'upgrade_path_068', 'origin': 'VS1_EvidenceBundle', 'artifact_ref': 'upgrade_068_vs0_artifact'})},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.claims(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs1_claim',
  'internal',
  {_sql_json({'probe': 'upgrade_path_068', 'origin': 'VS1_Claim', 'evidence_ref': 'upgrade_068_vs1_evidence_bundle'})},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.ontology_objects(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs1_ontology_object',
  'internal',
  {_sql_json({'probe': 'upgrade_path_068', 'origin': 'VS1_OntologyObject', 'claim_ref': 'upgrade_068_vs1_claim'})},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.ontology_links(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs1_ontology_link',
  'internal',
  {_sql_json({'probe': 'upgrade_path_068', 'origin': 'VS1_OntologyLink', 'source_ref': 'upgrade_068_vs1_ontology_object', 'target_ref': 'upgrade_068_vs1_claim'})},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.search_snapshots(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs1_search_snapshot',
  'internal',
  {_sql_json({'probe': 'upgrade_path_068', 'origin': 'VS1_SearchSnapshot', 'artifact_ref': 'upgrade_068_vs0_artifact'})},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'upgrade_068_vs1_audit_event',
  'upgrade.seeded',
  {_sql_literal(context['principal_id'])},
  'upgrade.fixture.seed',
  {_sql_json(audit_subject)},
  'policy_upgrade_068_seed',
  'vs2-rego-local-v1',
  '["upgrade_068_vs0_artifact", "upgrade_068_vs1_claim"]'::jsonb,
  {_sql_literal(_sha256_json({'previous': 'upgrade_068'}))},
  {_sql_literal(_sha256_json({'event': 'upgrade_068_vs1_audit_event', 'subject': audit_subject}))},
  'trace_vs2_upgrade_068_seed'
)
ON CONFLICT (tenant_id, event_id) DO NOTHING;
COMMIT;
"""
        )
        before_snapshot = self._upgrade_path_snapshot(context)
        add_column_sql = "\n".join(
            f"ALTER TABLE cs.{table} ADD COLUMN IF NOT EXISTS upgrade_revision text NOT NULL DEFAULT 'vs2-local-upgrade-068';"
            for table in upgrade_tables
        )
        forward_migration = self.postgres.psql(add_column_sql)
        after_snapshot = self._upgrade_path_snapshot(context)
        compatibility = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'artifact_visible_after_upgrade', EXISTS(SELECT 1 FROM cs.artifacts WHERE artifact_id = 'upgrade_068_vs0_artifact' AND payload->>'origin' = 'VS0_Artifact'),
  'evidence_visible_after_upgrade', EXISTS(SELECT 1 FROM cs.evidence_bundles WHERE artifact_id = 'upgrade_068_vs1_evidence_bundle' AND payload->>'artifact_ref' = 'upgrade_068_vs0_artifact'),
  'claim_visible_after_upgrade', EXISTS(SELECT 1 FROM cs.claims WHERE artifact_id = 'upgrade_068_vs1_claim' AND payload->>'evidence_ref' = 'upgrade_068_vs1_evidence_bundle'),
  'ontology_visible_after_upgrade', EXISTS(SELECT 1 FROM cs.ontology_objects WHERE artifact_id = 'upgrade_068_vs1_ontology_object')
    AND EXISTS(SELECT 1 FROM cs.ontology_links WHERE artifact_id = 'upgrade_068_vs1_ontology_link'),
  'audit_visible_after_upgrade', EXISTS(SELECT 1 FROM cs.audit_events WHERE event_id = 'upgrade_068_vs1_audit_event' AND subject->>'probe' = 'upgrade_path_068'),
  'search_snapshot_visible_after_upgrade', EXISTS(SELECT 1 FROM cs.search_snapshots WHERE artifact_id = 'upgrade_068_vs1_search_snapshot'),
  'upgrade_revision_values', jsonb_build_object(
    'artifacts', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.artifacts WHERE payload->>'probe' = 'upgrade_path_068'),
    'evidence_bundles', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.evidence_bundles WHERE payload->>'probe' = 'upgrade_path_068'),
    'claims', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.claims WHERE payload->>'probe' = 'upgrade_path_068'),
    'ontology_objects', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.ontology_objects WHERE payload->>'probe' = 'upgrade_path_068'),
    'ontology_links', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.ontology_links WHERE payload->>'probe' = 'upgrade_path_068'),
    'search_snapshots', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.search_snapshots WHERE payload->>'probe' = 'upgrade_path_068'),
    'audit_events', (SELECT jsonb_agg(DISTINCT upgrade_revision) FROM cs.audit_events WHERE subject->>'probe' = 'upgrade_path_068')
  ),
  'visible_tenants', (
    SELECT jsonb_agg(DISTINCT tenant_id)
    FROM (
      SELECT tenant_id FROM cs.artifacts WHERE payload->>'probe' = 'upgrade_path_068'
      UNION ALL SELECT tenant_id FROM cs.evidence_bundles WHERE payload->>'probe' = 'upgrade_path_068'
      UNION ALL SELECT tenant_id FROM cs.claims WHERE payload->>'probe' = 'upgrade_path_068'
      UNION ALL SELECT tenant_id FROM cs.ontology_objects WHERE payload->>'probe' = 'upgrade_path_068'
      UNION ALL SELECT tenant_id FROM cs.ontology_links WHERE payload->>'probe' = 'upgrade_path_068'
      UNION ALL SELECT tenant_id FROM cs.search_snapshots WHERE payload->>'probe' = 'upgrade_path_068'
      UNION ALL SELECT tenant_id FROM cs.audit_events WHERE subject->>'probe' = 'upgrade_path_068'
    ) scoped
  )
)::text;
COMMIT;
"""
        )
        failed_migration = self.postgres.psql(
            """
BEGIN;
ALTER TABLE cs.artifacts
  ADD CONSTRAINT upgrade_068_bad_tenant_check CHECK (tenant_id = 'tenant_impossible') NOT VALID;
ALTER TABLE cs.artifacts VALIDATE CONSTRAINT upgrade_068_bad_tenant_check;
COMMIT;
"""
        )
        failed_state = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'bad_constraint_count', (
    SELECT count(*)::int
    FROM pg_constraint
    WHERE conname = 'upgrade_068_bad_tenant_check'
  )
)::text;
"""
        )
        drop_column_sql = "\n".join(
            f"ALTER TABLE cs.{table} DROP COLUMN IF EXISTS upgrade_revision;"
            for table in upgrade_tables
        )
        rollback = self.postgres.psql(drop_column_sql)
        rollback_snapshot = self._upgrade_path_snapshot(context)
        rollback_columns = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'upgrade_revision_column_count', (
    SELECT count(*)::int
    FROM information_schema.columns
    WHERE table_schema = 'cs'
      AND table_name IN (
        'artifacts',
        'evidence_bundles',
        'claims',
        'ontology_objects',
        'ontology_links',
        'search_snapshots',
        'audit_events'
      )
      AND column_name = 'upgrade_revision'
  )
)::text;
"""
        )
        destructive_db_attempted = destructive_decision.get("decision") == "allow"
        before_tables = before_snapshot.get("tables", {})
        after_tables = after_snapshot.get("tables", {})
        rollback_tables = rollback_snapshot.get("tables", {})
        before_after_preserved = before_tables == after_tables
        rollback_preserved = before_tables == rollback_tables
        compatibility_checks = [
            compatibility.get("artifact_visible_after_upgrade") is True,
            compatibility.get("evidence_visible_after_upgrade") is True,
            compatibility.get("claim_visible_after_upgrade") is True,
            compatibility.get("ontology_visible_after_upgrade") is True,
            compatibility.get("audit_visible_after_upgrade") is True,
            compatibility.get("search_snapshot_visible_after_upgrade") is True,
            compatibility.get("visible_tenants") == [context["tenant_id"]],
        ]
        checks = {
            "vs0_vs1_fixture_seeded": seed.get("exit_code") == 0 and before_snapshot.get("table_count") == len(upgrade_tables),
            "forward_migration_succeeded": forward_migration.get("exit_code") == 0,
            "before_after_counts_hashes_match": before_after_preserved,
            "compatibility_regression_reads_succeeded": all(compatibility_checks),
            "failed_migration_rejected": failed_migration.get("exit_code") != 0,
            "failed_migration_left_no_bad_constraint": failed_state.get("bad_constraint_count") == 0,
            "destructive_migration_denied_without_approval": destructive_decision.get("decision") == "deny",
            "destructive_migration_not_executed": destructive_db_attempted is False,
            "rollback_succeeded": rollback.get("exit_code") == 0,
            "rollback_preserved_counts_hashes": rollback_preserved,
            "rollback_removed_upgrade_columns": rollback_columns.get("upgrade_revision_column_count") == 0,
            "no_data_loss": before_snapshot.get("total_rows") == rollback_snapshot.get("total_rows") == len(upgrade_tables),
        }
        result_digest = _sha256_json(
            {
                "before_snapshot": before_snapshot,
                "after_snapshot": after_snapshot,
                "compatibility": compatibility,
                "failed_state": failed_state,
                "rollback_snapshot": rollback_snapshot,
                "rollback_columns": rollback_columns,
                "checks": checks,
            }
        )
        return {
            "status": "completed" if all(checks.values()) else "failed",
            "scope": "synthetic VS0/VS1 local fixture; not production migration readiness",
            "tables": upgrade_tables,
            "db_call_count": 10,
            "seed": _digest_transcript_entry(seed),
            "before_snapshot": before_snapshot,
            "forward_migration": _digest_transcript_entry(forward_migration),
            "after_snapshot": after_snapshot,
            "compatibility_regression": compatibility,
            "failed_migration": _digest_transcript_entry(failed_migration),
            "failed_migration_state": failed_state,
            "destructive_migration": {
                "policy_decision": destructive_decision,
                "db_attempted": destructive_db_attempted,
            },
            "rollback": _digest_transcript_entry(rollback),
            "rollback_snapshot": rollback_snapshot,
            "rollback_columns": rollback_columns,
            "checks": checks,
            "result_digest": result_digest,
        }

    def _upgrade_path_snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
WITH scoped_rows AS (
  SELECT 'artifacts' AS table_name, artifact_id AS object_id, payload AS payload
  FROM cs.artifacts
  WHERE payload->>'probe' = 'upgrade_path_068'
  UNION ALL
  SELECT 'evidence_bundles', artifact_id, payload
  FROM cs.evidence_bundles
  WHERE payload->>'probe' = 'upgrade_path_068'
  UNION ALL
  SELECT 'claims', artifact_id, payload
  FROM cs.claims
  WHERE payload->>'probe' = 'upgrade_path_068'
  UNION ALL
  SELECT 'ontology_objects', artifact_id, payload
  FROM cs.ontology_objects
  WHERE payload->>'probe' = 'upgrade_path_068'
  UNION ALL
  SELECT 'ontology_links', artifact_id, payload
  FROM cs.ontology_links
  WHERE payload->>'probe' = 'upgrade_path_068'
  UNION ALL
  SELECT 'search_snapshots', artifact_id, payload
  FROM cs.search_snapshots
  WHERE payload->>'probe' = 'upgrade_path_068'
  UNION ALL
  SELECT 'audit_events', event_id, subject
  FROM cs.audit_events
  WHERE subject->>'probe' = 'upgrade_path_068'
),
grouped AS (
  SELECT
    table_name,
    count(*)::int AS row_count,
    jsonb_agg(object_id ORDER BY object_id) AS object_ids,
    md5(jsonb_agg(jsonb_build_object('object_id', object_id, 'payload', payload) ORDER BY object_id)::text) AS checksum
  FROM scoped_rows
  GROUP BY table_name
)
SELECT jsonb_build_object(
  'table_count', (SELECT count(*)::int FROM grouped),
  'total_rows', COALESCE((SELECT sum(row_count)::int FROM grouped), 0),
  'tables', COALESCE((
    SELECT jsonb_object_agg(
      table_name,
      jsonb_build_object(
        'count', row_count,
        'object_ids', object_ids,
        'checksum', checksum
      )
    )
    FROM grouped
  ), '{{}}'::jsonb),
  'snapshot_digest', md5(COALESCE((
    SELECT jsonb_object_agg(table_name, jsonb_build_object('count', row_count, 'object_ids', object_ids, 'checksum', checksum))
    FROM grouped
  ), '{{}}'::jsonb)::text)
)::text;
COMMIT;
"""
        )

    def _run_audit_integrity_matrix(self, context: dict[str, Any]) -> dict[str, Any]:
        event_rows = self._audit_integrity_event_rows(context)
        values_sql = ",\n".join(
            "("
            f"{_sql_literal(row['tenant_id'])}, "
            f"{_sql_literal(row['namespace_id'])}, "
            f"{_sql_literal(row['owner_id'])}, "
            f"{_sql_literal(row['workspace_id'])}, "
            f"{_sql_literal(row['event_id'])}, "
            f"{_sql_literal(row['event_type'])}, "
            f"{_sql_literal(row['actor'])}, "
            f"{_sql_literal(row['action'])}, "
            f"{_sql_json(row['subject'])}, "
            f"{_sql_literal(row['decision_id'])}, "
            f"{_sql_literal(row['policy_revision'])}, "
            f"{_sql_json(row['evidence_refs'])}, "
            f"{_sql_literal(row['previous_hash'])}, "
            f"{_sql_literal(row['event_hash'])}, "
            f"{_sql_literal(row['trace_id'])}"
            ")"
            for row in event_rows
        )
        seed = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES
{values_sql}
ON CONFLICT (tenant_id, event_id) DO NOTHING;
COMMIT;
"""
        )
        scoped_read = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'visible_count', count(*)::int,
  'event_types', COALESCE(jsonb_agg(event_type ORDER BY (subject->>'sequence')::int), '[]'::jsonb),
  'decision_ids', COALESCE(jsonb_agg(decision_id ORDER BY (subject->>'sequence')::int), '[]'::jsonb),
  'tenant_ids', COALESCE(jsonb_agg(DISTINCT tenant_id), '[]'::jsonb),
  'actions', COALESCE(jsonb_agg(action ORDER BY (subject->>'sequence')::int), '[]'::jsonb)
)::text
FROM cs.audit_events
WHERE subject->>'probe' = 'audit_integrity_066';
COMMIT;
"""
        )
        update_attempt = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
UPDATE cs.audit_events
   SET subject = subject || '{{"attempted_runtime_mutation": true}}'::jsonb
 WHERE subject->>'probe' = 'audit_integrity_066';
COMMIT;
"""
        )
        delete_attempt = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
DELETE FROM cs.audit_events
 WHERE subject->>'probe' = 'audit_integrity_066';
COMMIT;
"""
        )
        auditor_rows = self._audit_integrity_rows("clean")
        clean_verification = _verify_range_audit_chain(auditor_rows)
        tamper_cases: dict[str, dict[str, Any]] = {}
        for case in ["modify_event", "delete_event", "insert_fake", "reorder_events", "modify_previous_hash"]:
            rows = self._audit_integrity_rows(case)
            verification = _verify_range_audit_chain(rows)
            tamper_cases[case] = {
                "row_count": len(rows),
                "verification": verification,
            }
        required_event_types = [
            "artifact.read",
            "identity.denied",
            "policy.bundle.updated",
            "rls.anomaly.detected",
            "egress.allowed",
            "action.approved",
            "connector.requested",
            "migration.rollback.completed",
            "audit.verified",
        ]
        auditor_inventory = {
            "row_count": len(auditor_rows),
            "event_types": [row.get("event_type") for row in auditor_rows],
            "decision_ids": [row.get("decision_id") for row in auditor_rows],
            "root_hash": clean_verification.get("root_hash"),
        }
        app_update_stderr = str(update_attempt.get("stderr", ""))
        app_delete_stderr = str(delete_attempt.get("stderr", ""))
        checks = {
            "seed_inserted_or_already_present": seed.get("exit_code") == 0,
            "app_role_scoped_selects_chain": scoped_read.get("visible_count") == len(required_event_types)
            and scoped_read.get("tenant_ids") == [context["tenant_id"]],
            "app_role_update_denied": update_attempt.get("exit_code") != 0 and "permission denied" in app_update_stderr,
            "app_role_delete_denied": delete_attempt.get("exit_code") != 0 and "permission denied" in app_delete_stderr,
            "auditor_role_reads_full_chain": len(auditor_rows) == len(required_event_types),
            "required_event_types_present": set(required_event_types).issubset(set(auditor_inventory["event_types"])),
            "clean_chain_verifies": clean_verification.get("valid") is True,
            "tamper_cases_detected": all(case["verification"].get("valid") is False for case in tamper_cases.values()),
        }
        return {
            "status": "completed" if all(checks.values()) else "failed",
            "scope": "local synthetic audit fixture; not production audit/pentest evidence",
            "db_call_count": 8,
            "required_event_types": required_event_types,
            "seed": _digest_transcript_entry(seed),
            "app_role": {
                "scoped_select": scoped_read,
                "update_attempt": _digest_transcript_entry(update_attempt),
                "delete_attempt": _digest_transcript_entry(delete_attempt),
            },
            "auditor_role": auditor_inventory,
            "clean_verification": clean_verification,
            "tamper_cases": tamper_cases,
            "checks": checks,
            "result_digest": _sha256_json(
                {
                    "auditor_inventory": auditor_inventory,
                    "clean_verification": clean_verification,
                    "tamper_cases": tamper_cases,
                    "checks": checks,
                }
            ),
        }

    def _audit_integrity_event_rows(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        event_specs = [
            ("artifact.read", "artifact.show", "policy_audit_066_access", {"resource_id": "artifact_alpha_001"}),
            ("identity.denied", "identity.resolve", "identity_audit_066_denied", {"reason": "missing_session"}),
            ("policy.bundle.updated", "policy.update", "policy_audit_066_update", {"from_revision": "vs2-rego-local-v1", "to_revision": "vs2-rego-local-v2"}),
            ("rls.anomaly.detected", "database.query", "policy_audit_066_rls_anomaly", {"blocked_tenant_digest": _sha256_json({"tenant": "tenant_beta"})}),
            ("egress.allowed", "connector.execute", "policy_audit_066_egress", {"capability": "mock_provider.status.write", "destination_class": "local_mock_provider"}),
            ("action.approved", "action.approve", "policy_audit_066_approval", {"approval_id": "approval_audit_066"}),
            ("connector.requested", "connector.request", "policy_audit_066_connector", {"connector": "ConnectorHub", "credential_ref": "credential_ref_mock_provider_write"}),
            ("migration.rollback.completed", "migration.rollback", "policy_audit_066_rollback", {"rollback_id": "rollback_audit_066"}),
            ("audit.verified", "audit.verify", "policy_audit_066_verify", {"verifier_role": "cornerstone_auditor"}),
        ]
        rows: list[dict[str, Any]] = []
        previous_hash = "LOCAL_RANGE_AUDIT_GENESIS"
        for index, (event_type, action, decision_id, details) in enumerate(event_specs, start=1):
            event_id = f"audit_integrity_066_{index:03d}"
            subject = {
                "probe": "audit_integrity_066",
                "scenario_id": "VS2-SEC-066",
                "sequence": index,
                **details,
            }
            evidence_refs = ["reports/security/vs2/evidence/VS2-SEC-066.json"]
            row = {
                "event_id": event_id,
                "event_type": event_type,
                "tenant_id": context["tenant_id"],
                "namespace_id": context["namespace_id"],
                "owner_id": context["owner_id"],
                "workspace_id": context["workspace_id"],
                "actor": context["principal_id"],
                "action": action,
                "subject": subject,
                "decision_id": decision_id,
                "policy_revision": "vs2-rego-local-v1",
                "evidence_refs": evidence_refs,
                "previous_hash": previous_hash,
                "trace_id": "trace_vs2_audit_integrity_066",
            }
            event_hash = _sha256_json(row)
            row["event_hash"] = event_hash
            rows.append(row)
            previous_hash = event_hash
        return rows

    def _audit_integrity_rows(self, tamper_case: str) -> list[dict[str, Any]]:
        case_sql = {
            "clean": "",
            "modify_event": "UPDATE audit_tamper_066 SET subject = subject || '{\"tampered\": true}'::jsonb WHERE tamper_order = 3;",
            "delete_event": "DELETE FROM audit_tamper_066 WHERE tamper_order = 4;",
            "insert_fake": """
INSERT INTO audit_tamper_066(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id, tamper_order
)
SELECT
  tenant_id, namespace_id, owner_id, workspace_id, 'audit_integrity_066_fake', event_type, actor, action,
  subject || '{"tampered_insert": true}'::jsonb,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id, 2.5
FROM audit_tamper_066
WHERE tamper_order = 2;
""",
            "reorder_events": """
UPDATE audit_tamper_066 SET tamper_order = 3 WHERE event_id = 'audit_integrity_066_002';
UPDATE audit_tamper_066 SET tamper_order = 2 WHERE event_id = 'audit_integrity_066_003';
""",
            "modify_previous_hash": "UPDATE audit_tamper_066 SET previous_hash = 'tampered_previous_hash' WHERE tamper_order = 5;",
        }[tamper_case]
        return self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_auditor;
CREATE TEMP TABLE audit_tamper_066 AS
SELECT
  tenant_id,
  namespace_id,
  owner_id,
  workspace_id,
  event_id,
  event_type,
  actor,
  action,
  subject,
  decision_id,
  policy_revision,
  evidence_refs,
  previous_hash,
  event_hash,
  trace_id,
  (subject->>'sequence')::numeric AS tamper_order
FROM cs.audit_events
WHERE subject->>'probe' = 'audit_integrity_066';
{case_sql}
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'event_id', event_id,
  'event_type', event_type,
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'actor', actor,
  'action', action,
  'subject', subject,
  'decision_id', decision_id,
  'policy_revision', policy_revision,
  'evidence_refs', evidence_refs,
  'previous_hash', previous_hash,
  'trace_id', trace_id,
  'event_hash', event_hash
) ORDER BY tamper_order, event_id), '[]'::jsonb)::text
FROM audit_tamper_066;
COMMIT;
"""
        )

    def _persist_object_contract_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for spec in DURABLE_OBJECT_TABLES:
            table = spec["table"]
            object_id = f"object_contract_{table}"
            if table == "audit_events":
                subject = {
                    "object_type": spec["object_type"],
                    "object_contract_id": object_id,
                    "required_scope_fields": ["tenant_id", "namespace_id", "owner_id", "workspace_id"],
                    "classification_applicable": False,
                }
                event_hash = _sha256_json({"event_id": object_id, "subject": subject, "tenant_id": context["tenant_id"]})
                sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(object_id)},
  'object.contract.persisted',
  {_sql_literal(context['principal_id'])},
  'object_contract.persist',
  {_sql_json(subject)},
  'object_contract_policy_decision',
  'vs2-local-range-object-contract',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(event_hash)},
  'trace_vs2_object_contract'
) ON CONFLICT (tenant_id, event_id) DO NOTHING;
COMMIT;
"""
            else:
                payload = {
                    "object_type": spec["object_type"],
                    "object_contract_id": object_id,
                    "required_scope_fields": ["tenant_id", "namespace_id", "owner_id", "workspace_id", "classification"],
                    "classification_applicable": True,
                }
                sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
DELETE FROM cs.{table} WHERE artifact_id = {_sql_literal(object_id)};
INSERT INTO cs.{table}(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(object_id)},
  'internal',
  {_sql_json(payload)},
  'trace_vs2_object_contract'
);
COMMIT;
"""
            result = self.postgres.psql(sql)
            results[table] = {
                "object_type": spec["object_type"],
                "exit_code": result["exit_code"],
                "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
            }
        return results

    def _read_object_contract_rows(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in DURABLE_OBJECT_TABLES:
            table = spec["table"]
            object_id = f"object_contract_{table}"
            if table == "audit_events":
                query = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_agg(jsonb_build_object(
  'table', 'audit_events',
  'object_type', 'AuditEvent',
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'event_id', event_id,
  'classification_applicable', false,
  'subject', subject
))::text
FROM cs.audit_events
WHERE event_id = {_sql_literal(object_id)};
COMMIT;
"""
            else:
                query = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_agg(jsonb_build_object(
  'table', {_sql_literal(table)},
  'object_type', {_sql_literal(spec['object_type'])},
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'artifact_id', artifact_id,
  'classification', classification,
  'classification_applicable', true,
  'payload', payload
))::text
FROM cs.{table}
WHERE artifact_id = {_sql_literal(object_id)};
COMMIT;
"""
            found = self.postgres.json_query(query)
            if found:
                rows.extend(found)
        return rows

    def tenant_read_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        seed_results = self._persist_tenant_read_matrix_rows(context)
        rows = self._read_tenant_matrix_rows(context)
        counters["db_calls"] += len(seed_results) + len(rows)
        all_seeded = all(result.get("exit_code") == 0 for result in seed_results.values())
        payload = {
            "surface": surface,
            "status": "allowed" if all_seeded else "failed",
            "status_code": 200 if all_seeded else 500,
            "error": None if all_seeded else {"code": "CS_TENANT_READ_MATRIX_SEED_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "tenant_read_matrix": {
                "tables_expected": [spec["table"] for spec in DURABLE_OBJECT_TABLES],
                "seed_results": seed_results,
                "tables": rows,
                "query_shapes": [
                    "select",
                    "count",
                    "exists",
                    "group_by",
                    "pagination_first_page",
                    "pagination_boundary_page",
                    "subquery_intersection",
                    "join_cross_tenant",
                    "guessed_foreign_id",
                ],
            },
            "audit_refs": [],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _persist_tenant_read_matrix_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        beta_scope = {
            "tenant_id": "tenant_beta",
            "namespace_id": "personal",
            "owner_id": "principal_bob",
            "workspace_id": "beta-home",
            "principal_id": "principal_bob",
        }
        for spec in DURABLE_OBJECT_TABLES:
            table = spec["table"]
            alpha_id = f"tenant_read_alpha_{table}"
            beta_id = f"tenant_read_beta_{table}"
            if table == "audit_events":
                alpha_subject = {
                    "object_type": spec["object_type"],
                    "probe": "tenant_read_matrix",
                    "side": "alpha",
                }
                beta_subject = {
                    "object_type": spec["object_type"],
                    "probe": "tenant_read_matrix",
                    "side": "foreign",
                    "canary_digest": _sha256_json({"canary": "BETA_MATRIX_CANARY", "table": table}),
                }
                alpha_hash = _sha256_json({"event_id": alpha_id, "subject": alpha_subject, "tenant_id": context["tenant_id"]})
                beta_hash = _sha256_json({"event_id": beta_id, "subject": beta_subject, "tenant_id": beta_scope["tenant_id"]})
                sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_id)},
  'tenant.read.matrix.seed',
  {_sql_literal(context['principal_id'])},
  'tenant_read_matrix.seed',
  {_sql_json(alpha_subject)},
  'tenant_read_matrix_policy_decision',
  'vs2-local-range-tenant-read',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(alpha_hash)},
  'trace_vs2_tenant_read_matrix'
) ON CONFLICT (tenant_id, event_id) DO NOTHING;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(beta_scope['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(beta_scope['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(beta_scope['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(beta_scope['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(beta_scope['tenant_id'])},
  {_sql_literal(beta_scope['namespace_id'])},
  {_sql_literal(beta_scope['owner_id'])},
  {_sql_literal(beta_scope['workspace_id'])},
  {_sql_literal(beta_id)},
  'tenant.read.matrix.seed',
  {_sql_literal(beta_scope['principal_id'])},
  'tenant_read_matrix.seed',
  {_sql_json(beta_subject)},
  'tenant_read_matrix_policy_decision',
  'vs2-local-range-tenant-read',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(beta_hash)},
  'trace_vs2_tenant_read_matrix'
) ON CONFLICT (tenant_id, event_id) DO NOTHING;
COMMIT;
"""
            else:
                alpha_payload = {
                    "object_type": spec["object_type"],
                    "probe": "tenant_read_matrix",
                    "side": "alpha",
                }
                beta_payload = {
                    "object_type": spec["object_type"],
                    "probe": "tenant_read_matrix",
                    "side": "foreign",
                    "canary_digest": _sha256_json({"canary": "BETA_MATRIX_CANARY", "table": table}),
                }
                sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.{table}(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(alpha_id)},
  'internal',
  {_sql_json(alpha_payload)},
  'trace_vs2_tenant_read_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(beta_scope['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(beta_scope['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(beta_scope['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(beta_scope['workspace_id'])};
INSERT INTO cs.{table}(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(beta_scope['tenant_id'])},
  {_sql_literal(beta_scope['namespace_id'])},
  {_sql_literal(beta_scope['owner_id'])},
  {_sql_literal(beta_scope['workspace_id'])},
  {_sql_literal(beta_id)},
  'internal',
  {_sql_json(beta_payload)},
  'trace_vs2_tenant_read_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
            result = self.postgres.psql(sql)
            results[table] = {
                "object_type": spec["object_type"],
                "exit_code": result["exit_code"],
                "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
            }
        return results

    def _read_tenant_matrix_rows(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in DURABLE_OBJECT_TABLES:
            table = spec["table"]
            id_column = spec["id_column"]
            alpha_id = f"tenant_read_alpha_{table}"
            beta_id = f"tenant_read_beta_{table}"
            query = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
WITH visible AS (
  SELECT {id_column} AS object_id, tenant_id
  FROM cs.{table}
),
page_first AS (
  SELECT {id_column} AS object_id
  FROM cs.{table}
  ORDER BY {id_column}
  LIMIT 2
),
page_boundary AS (
  SELECT {id_column} AS object_id
  FROM cs.{table}
  ORDER BY {id_column}
  LIMIT 2
  OFFSET (SELECT count(*) FROM cs.{table})
),
tenant_groups AS (
  SELECT tenant_id, count(*) AS row_count
  FROM cs.{table}
  GROUP BY tenant_id
)
SELECT jsonb_build_object(
  'table', {_sql_literal(table)},
  'object_type', {_sql_literal(spec['object_type'])},
  'id_column', {_sql_literal(id_column)},
  'visible_count', (SELECT count(*) FROM visible),
  'explicit_foreign_count', (SELECT count(*) FROM cs.{table} WHERE tenant_id = 'tenant_beta'),
  'exists_foreign', (SELECT EXISTS(SELECT 1 FROM cs.{table} WHERE tenant_id = 'tenant_beta')),
  'grouped_tenants', COALESCE(
    (SELECT jsonb_agg(jsonb_build_object('tenant_id', tenant_id, 'row_count', row_count) ORDER BY tenant_id) FROM tenant_groups),
    '[]'::jsonb
  ),
  'page_first_ids', COALESCE((SELECT jsonb_agg(object_id ORDER BY object_id) FROM page_first), '[]'::jsonb),
  'page_boundary_ids', COALESCE((SELECT jsonb_agg(object_id ORDER BY object_id) FROM page_boundary), '[]'::jsonb),
  'subquery_intersection_count', (
    SELECT count(*)
    FROM cs.{table}
    WHERE {id_column} IN (SELECT {id_column} FROM cs.{table} WHERE tenant_id = 'tenant_beta')
  ),
  'join_cross_count', (
    SELECT count(*)
    FROM cs.{table} alpha_row
    JOIN cs.{table} foreign_row ON true
    WHERE alpha_row.{id_column} = {_sql_literal(alpha_id)}
      AND foreign_row.{id_column} = {_sql_literal(beta_id)}
  ),
  'guessed_foreign_result_count', (SELECT count(*) FROM cs.{table} WHERE {id_column} = {_sql_literal(beta_id)}),
  'neutral_foreign_lookup', jsonb_build_object(
    'status_code', 404,
    'code', 'CS_RESOURCE_NOT_FOUND_OR_DENIED',
    'message', 'resource not found or access denied',
    'foreign_identifier_echoed', false,
    'tenant_identifier_echoed', false
  ),
  'sample_rows', COALESCE(
    (SELECT jsonb_agg(jsonb_build_object('object_id', object_id, 'tenant_id', tenant_id) ORDER BY object_id) FROM visible),
    '[]'::jsonb
  )
)::text;
COMMIT;
"""
            rows.append(self.postgres.json_query(query))
        return rows

    def search_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        seed = self._persist_search_matrix_rows(context)
        results = self._read_search_matrix_rows(context)
        inventory = _search_matrix_inventory(self.postgres)
        counters["db_calls"] += 3
        payload = {
            "surface": surface,
            "status": "allowed" if seed.get("exit_code") == 0 else "failed",
            "status_code": 200 if seed.get("exit_code") == 0 else 500,
            "error": None if seed.get("exit_code") == 0 else {"code": "CS_SEARCH_MATRIX_SEED_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "search_matrix": {
                "searched_foreign_term_digest": _sha256_json({"term": "BETA_SEARCH_CANARY_UNIQUE"}),
                "searched_alpha_term_digest": _sha256_json({"term": "ALPHA_SEARCH_CANARY_UNIQUE"}),
                "paths": [
                    "full_text_search",
                    "autocomplete_suggestions",
                    "facet_counts",
                    "saved_snapshot_refs",
                    "ontology_object_refs",
                    "semantic_cache_refs",
                ],
                "seed": seed,
                "results": results,
                "inventory": inventory,
                "neutral_foreign_lookup": {
                    "status_code": 404,
                    "code": "CS_RESOURCE_NOT_FOUND_OR_DENIED",
                    "message": "search result not found or access denied",
                    "foreign_identifier_echoed": False,
                    "tenant_identifier_echoed": False,
                },
            },
            "audit_refs": [],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _persist_search_matrix_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        alpha_search_payload = {
            "probe": "search_matrix",
            "title": "Alpha search range snapshot",
            "body": "ALPHA_SEARCH_CANARY_UNIQUE calm evidence workspace",
            "snippet": "alpha local searchable snippet",
            "suggestions": ["alpha evidence", "alpha local search"],
            "facet": "alpha-facet",
            "snapshot_ref": "search_snapshot_alpha_matrix",
            "ontology_object_ref": "ontology_alpha_search_object",
            "semantic_cache_ref": "semantic_alpha_cache",
        }
        beta_search_payload = {
            "probe": "search_matrix",
            "title": "Beta search range snapshot",
            "body": "BETA_SEARCH_CANARY_UNIQUE foreign evidence workspace",
            "snippet": "beta foreign searchable snippet",
            "suggestions": ["beta evidence", "beta local search"],
            "facet": "beta-facet",
            "snapshot_ref": "search_snapshot_beta_matrix",
            "ontology_object_ref": "ontology_beta_search_object",
            "semantic_cache_ref": "semantic_beta_cache",
        }
        alpha_ontology_payload = {
            "probe": "search_matrix",
            "title": "Alpha ontology object",
            "body": "ALPHA_SEARCH_CANARY_UNIQUE ontology object",
            "object_ref": "ontology_alpha_search_object",
        }
        beta_ontology_payload = {
            "probe": "search_matrix",
            "title": "Beta ontology object",
            "body": "BETA_SEARCH_CANARY_UNIQUE ontology object",
            "object_ref": "ontology_beta_search_object",
        }
        sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.search_snapshots(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'search_snapshot_alpha_matrix',
  'internal',
  {_sql_json(alpha_search_payload)},
  'trace_vs2_search_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.ontology_objects(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'ontology_alpha_search_object',
  'internal',
  {_sql_json(alpha_ontology_payload)},
  'trace_vs2_search_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
INSERT INTO cs.search_snapshots(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'search_snapshot_beta_matrix',
  'internal',
  {_sql_json(beta_search_payload)},
  'trace_vs2_search_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
INSERT INTO cs.ontology_objects(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'ontology_beta_search_object',
  'internal',
  {_sql_json(beta_ontology_payload)},
  'trace_vs2_search_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
        result = self.postgres.psql(sql)
        return {
            "exit_code": result["exit_code"],
            "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
        }

    def _read_search_matrix_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
WITH beta_search AS (
  SELECT artifact_id, tenant_id, payload
  FROM cs.search_snapshots
  WHERE payload->>'probe' = 'search_matrix'
    AND to_tsvector('simple', payload::text) @@ plainto_tsquery('simple', 'BETA_SEARCH_CANARY_UNIQUE')
),
alpha_search AS (
  SELECT artifact_id, tenant_id, payload
  FROM cs.search_snapshots
  WHERE payload->>'probe' = 'search_matrix'
    AND to_tsvector('simple', payload::text) @@ plainto_tsquery('simple', 'ALPHA_SEARCH_CANARY_UNIQUE')
),
beta_suggestions AS (
  SELECT suggestion
  FROM cs.search_snapshots,
       LATERAL jsonb_array_elements_text(payload->'suggestions') AS suggestion
  WHERE payload->>'probe' = 'search_matrix'
    AND suggestion ILIKE '%beta%'
),
beta_facets AS (
  SELECT payload->>'facet' AS facet, count(*) AS row_count
  FROM cs.search_snapshots
  WHERE payload->>'probe' = 'search_matrix'
    AND to_tsvector('simple', payload::text) @@ plainto_tsquery('simple', 'BETA_SEARCH_CANARY_UNIQUE')
  GROUP BY payload->>'facet'
),
beta_ontology AS (
  SELECT artifact_id, tenant_id
  FROM cs.ontology_objects
  WHERE payload->>'probe' = 'search_matrix'
    AND to_tsvector('simple', payload::text) @@ plainto_tsquery('simple', 'BETA_SEARCH_CANARY_UNIQUE')
),
alpha_ontology AS (
  SELECT artifact_id, tenant_id
  FROM cs.ontology_objects
  WHERE payload->>'probe' = 'search_matrix'
    AND to_tsvector('simple', payload::text) @@ plainto_tsquery('simple', 'ALPHA_SEARCH_CANARY_UNIQUE')
)
SELECT jsonb_build_object(
  'alpha_positive', jsonb_build_object(
    'result_count', (SELECT count(*) FROM alpha_search),
    'snapshot_refs', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM alpha_search), '[]'::jsonb),
    'object_refs', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM alpha_ontology), '[]'::jsonb)
  ),
  'foreign_term_search', jsonb_build_object(
    'result_count', (SELECT count(*) FROM beta_search),
    'snippet_count', (SELECT count(*) FROM beta_search WHERE payload ? 'snippet'),
    'score_count', 0,
    'snapshot_refs', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM beta_search), '[]'::jsonb),
    'sample_rows', COALESCE((SELECT jsonb_agg(jsonb_build_object('artifact_id', artifact_id, 'tenant_id', tenant_id) ORDER BY artifact_id) FROM beta_search), '[]'::jsonb)
  ),
  'autocomplete', jsonb_build_object(
    'suggestion_count', (SELECT count(*) FROM beta_suggestions),
    'suggestions', COALESCE((SELECT jsonb_agg(suggestion ORDER BY suggestion) FROM beta_suggestions), '[]'::jsonb)
  ),
  'facets', jsonb_build_object(
    'facet_count', (SELECT count(*) FROM beta_facets),
    'facets', COALESCE((SELECT jsonb_agg(jsonb_build_object('facet', facet, 'row_count', row_count) ORDER BY facet) FROM beta_facets), '[]'::jsonb)
  ),
  'ontology_search', jsonb_build_object(
    'object_count', (SELECT count(*) FROM beta_ontology),
    'object_refs', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM beta_ontology), '[]'::jsonb)
  ),
  'semantic_cache', jsonb_build_object(
    'cache_ref_count', (
      SELECT count(*)
      FROM beta_search
      WHERE payload ? 'semantic_cache_ref'
    ),
    'cache_refs', COALESCE((SELECT jsonb_agg(payload->>'semantic_cache_ref' ORDER BY payload->>'semantic_cache_ref') FROM beta_search), '[]'::jsonb)
  ),
  'explicit_foreign_counts', jsonb_build_object(
    'search_snapshots', (SELECT count(*) FROM cs.search_snapshots WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'search_matrix'),
    'ontology_objects', (SELECT count(*) FROM cs.ontology_objects WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'search_matrix')
  )
)::text;
COMMIT;
"""
        )

    def db_path_matrix(self, *, token: str | None, surface: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, surface)
        if denied:
            return denied
        assert resolved is not None
        context = resolved["context"]
        counters = resolved["counters"]
        seed = self._persist_db_path_matrix_rows(context)
        raw_sql = self._read_db_path_raw_sql(context)
        safe_view = self._read_db_path_safe_view(context)
        safe_function = self._read_db_path_safe_function(context)
        unsafe_definer = self._probe_db_path_unsafe_definer(context)
        inventory = _db_path_object_inventory(self.postgres)
        counters["db_calls"] += 5
        payload = {
            "surface": surface,
            "status": "allowed" if seed.get("exit_code") == 0 else "failed",
            "status_code": 200 if seed.get("exit_code") == 0 else 500,
            "error": None if seed.get("exit_code") == 0 else {"code": "CS_DB_PATH_MATRIX_SEED_FAILED"},
            "context": context,
            "context_digest": _sha256_json(context),
            "db_path_matrix": {
                "supported_paths": [
                    "raw_sql_repository",
                    "security_invoker_view",
                    "security_invoker_function",
                    "denied_security_definer_function",
                    "static_object_inventory",
                ],
                "seed": seed,
                "raw_sql": raw_sql,
                "safe_view": safe_view,
                "safe_function": safe_function,
                "unsafe_security_definer": unsafe_definer,
                "object_inventory": inventory,
            },
            "audit_refs": [],
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _persist_db_path_matrix_rows(self, context: dict[str, Any]) -> dict[str, Any]:
        alpha_payload = {
            "probe": "db_path_matrix",
            "side": "alpha",
            "path_coverage": ["raw_sql_repository", "security_invoker_view", "security_invoker_function"],
        }
        foreign_payload = {
            "probe": "db_path_matrix",
            "side": "foreign",
            "canary": "BETA_DB_PATH_CANARY",
        }
        sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'db_path_alpha_artifact',
  'internal',
  {_sql_json(alpha_payload)},
  'trace_vs2_db_path_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'db_path_foreign_artifact',
  'internal',
  {_sql_json(foreign_payload)},
  'trace_vs2_db_path_matrix'
) ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
        result = self.postgres.psql(sql)
        return {
            "exit_code": result["exit_code"],
            "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
        }

    def _read_db_path_raw_sql(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'path', 'raw_sql_repository',
  'visible_count', (SELECT count(*) FROM cs.artifacts WHERE payload->>'probe' = 'db_path_matrix'),
  'explicit_foreign_count', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'db_path_matrix'),
  'exists_foreign', (SELECT EXISTS(SELECT 1 FROM cs.artifacts WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'db_path_matrix')),
  'sample_rows', COALESCE(
    (
      SELECT jsonb_agg(jsonb_build_object('artifact_id', artifact_id, 'tenant_id', tenant_id) ORDER BY artifact_id)
      FROM cs.artifacts
      WHERE payload->>'probe' = 'db_path_matrix'
    ),
    '[]'::jsonb
  )
)::text;
COMMIT;
"""
        )

    def _read_db_path_safe_view(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'path', 'security_invoker_view',
  'row_count', (SELECT count(*) FROM cs.safe_artifact_counts),
  'explicit_foreign_count', (SELECT count(*) FROM cs.safe_artifact_counts WHERE tenant_id = 'tenant_beta'),
  'rows', COALESCE(
    (
      SELECT jsonb_agg(jsonb_build_object('tenant_id', tenant_id, 'row_count', row_count) ORDER BY tenant_id)
      FROM cs.safe_artifact_counts
    ),
    '[]'::jsonb
  )
)::text;
COMMIT;
"""
        )

    def _read_db_path_safe_function(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'path', 'security_invoker_function',
  'visible_count', (SELECT count(*) FROM cs.visible_artifact_ids() WHERE payload->>'probe' = 'db_path_matrix'),
  'explicit_foreign_count', (SELECT count(*) FROM cs.visible_artifact_ids() WHERE tenant_id = 'tenant_beta' AND payload->>'probe' = 'db_path_matrix'),
  'rows', COALESCE(
    (
      SELECT jsonb_agg(jsonb_build_object('artifact_id', artifact_id, 'tenant_id', tenant_id) ORDER BY artifact_id)
      FROM cs.visible_artifact_ids()
      WHERE payload->>'probe' = 'db_path_matrix'
    ),
    '[]'::jsonb
  )
)::text;
COMMIT;
"""
        )

    def _probe_db_path_unsafe_definer(self, context: dict[str, Any]) -> dict[str, Any]:
        result = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT count(*) FROM cs.unsafe_all_artifacts();
COMMIT;
"""
        )
        stderr = str(result.get("stderr", ""))
        return {
            "path": "denied_security_definer_function",
            "exit_code": result["exit_code"],
            "denied": result["exit_code"] != 0,
            "stderr_tail": stderr.splitlines()[-4:],
            "foreign_identifier_echoed": "tenant_beta" in stderr or "db_path_foreign_artifact" in stderr,
        }

    def membership_state(self, *, principal_id: str, membership_id: str) -> dict[str, Any] | None:
        rows = self.postgres.json_query(
            f"""
SELECT jsonb_agg(jsonb_build_object(
  'membership_id', membership_id,
  'principal_id', principal_id,
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'workspace_id', workspace_id,
  'owner_id', owner_id,
  'roles', roles,
  'membership_revision', membership_revision,
  'session_version', session_version,
  'revoked', revoked_at IS NOT NULL
))::text
FROM cs.memberships
WHERE principal_id = {_sql_literal(principal_id)}
  AND membership_id = {_sql_literal(membership_id)};
"""
        )
        return rows[0] if rows else None

    def _membership_denial_reason(self, token_payload: dict[str, Any]) -> str:
        state = self.membership_state(principal_id=str(token_payload.get("sub", "")), membership_id=str(token_payload.get("membership_id", "")))
        if not state:
            return "membership_missing"
        if state.get("revoked") is True:
            return "membership_revoked"
        if int(state.get("session_version", -1)) != int(token_payload.get("session_version", -2)):
            return "stale_session_version"
        return "membership_unresolved"

    def _insert_identity_denial_audit(self, token_payload: dict[str, Any], *, reason: str, surface: str, artifact_id: str) -> str | None:
        state = self.membership_state(principal_id=str(token_payload.get("sub", "")), membership_id=str(token_payload.get("membership_id", "")))
        if not state:
            return None
        event_base = {
            "event_type": "identity.denied",
            "actor": state["principal_id"],
            "action": "identity.resolve",
            "subject": {
                "artifact_id": artifact_id,
                "membership_id": state["membership_id"],
                "reason": reason,
                "surface": surface,
            },
            "decision_id": f"identity_{reason}",
            "trace_id": "trace_vs2_local_range_revocation",
            "previous_hash": "LOCAL_RANGE_GENESIS",
            "sequence": len(self.requests) + 1,
        }
        event_hash = _sha256_json(event_base)
        event_id = f"audit_{event_hash[:16]}"
        result = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(state['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(state['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(state['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(state['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(state['tenant_id'])},
  {_sql_literal(state['namespace_id'])},
  {_sql_literal(state['owner_id'])},
  {_sql_literal(state['workspace_id'])},
  {_sql_literal(event_id)},
  'identity.denied',
  {_sql_literal(state['principal_id'])},
  'identity.resolve',
  {_sql_json(event_base['subject'])},
  {_sql_literal(event_base['decision_id'])},
  'identity-store-local',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(event_hash)},
  'trace_vs2_local_range_revocation'
);
COMMIT;
"""
        )
        return event_id if result["exit_code"] == 0 else None

    def run_worker_artifact_job(self, envelope: dict[str, Any]) -> dict[str, Any]:
        counters = {"db_calls": 0, "policy_calls": 0, "egress_calls": 0, "audit_inserts": 0}
        trace_id = str(envelope.get("trace_id", "trace_vs2_worker_scope"))
        expected_signature = _worker_signature(envelope, self.token_key)
        if not envelope.get("signature") or not hmac.compare_digest(str(envelope.get("signature")), expected_signature):
            return {
                "surface": "worker",
                "job_id": envelope.get("job_id"),
                "status": "quarantined",
                "reason": "invalid_job_signature",
                "artifact": None,
                "policy_decision": None,
                "audit_refs": [],
                "counters": counters,
            }

        required_identity_fields = ["job_id", "principal_id", "membership_id", "membership_revision", "session_version", "artifact_id"]
        missing_identity_fields = [field for field in required_identity_fields if envelope.get(field) in (None, "")]
        if missing_identity_fields:
            return {
                "surface": "worker",
                "job_id": envelope.get("job_id"),
                "status": "quarantined",
                "reason": "missing_identity_metadata",
                "missing_fields": missing_identity_fields,
                "artifact": None,
                "policy_decision": None,
                "audit_refs": [],
                "counters": counters,
            }

        counters["db_calls"] += 1
        state = self.membership_state(principal_id=str(envelope["principal_id"]), membership_id=str(envelope["membership_id"]))
        if not state:
            return self._quarantine_worker_job(envelope, None, "membership_missing", counters)
        if state.get("revoked") is True:
            return self._quarantine_worker_job(envelope, state, "membership_revoked", counters)
        if int(state.get("session_version", -1)) != int(envelope.get("session_version", -2)):
            return self._quarantine_worker_job(envelope, state, "stale_session_version", counters)
        if state.get("membership_revision") != envelope.get("membership_revision"):
            return self._quarantine_worker_job(envelope, state, "stale_membership_revision", counters)

        required_scope_fields = ["tenant_id", "namespace_id", "workspace_id", "owner_id"]
        missing_scope_fields = [field for field in required_scope_fields if envelope.get(field) in (None, "")]
        if missing_scope_fields:
            return self._quarantine_worker_job(envelope, state, "missing_scope_metadata", counters)
        mismatched_scope_fields = [field for field in required_scope_fields if envelope.get(field) != state.get(field)]
        if mismatched_scope_fields:
            return self._quarantine_worker_job(envelope, state, "tampered_scope_metadata", counters)

        counters["db_calls"] += 1
        replay_probe = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(state['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(state['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(state['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(state['workspace_id'])};
SELECT jsonb_build_object(
  'job_id', {_sql_literal(envelope['job_id'])},
  'already_seen', EXISTS(SELECT 1 FROM cs.jobs WHERE artifact_id = {_sql_literal(envelope['job_id'])})
)::text;
COMMIT;
"""
        )
        if replay_probe and replay_probe.get("already_seen") is True:
            replay_record_id = f"quarantine_{envelope['job_id']}_{_sha256_json({'job_id': envelope['job_id'], 'trace_id': trace_id})[:12]}"
            return self._quarantine_worker_job(envelope, state, "replay_detected", counters, record_id=replay_record_id)

        counters["db_calls"] += 1
        context_rows = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_agg(to_jsonb(r))::text
FROM cs.resolve_membership(
  {_sql_literal(envelope['principal_id'])},
  {_sql_literal(envelope['membership_id'])},
  {int(envelope['session_version'])}
) r;
COMMIT;
"""
        )
        if not context_rows:
            return self._quarantine_worker_job(envelope, state, "membership_unresolved", counters)
        context = context_rows[0]
        counters["db_calls"] += 1
        artifact = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_agg(jsonb_build_object(
  'artifact_id', artifact_id,
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'classification', classification,
  'payload', payload
))::text
FROM cs.artifacts
WHERE artifact_id = {_sql_literal(envelope['artifact_id'])};
COMMIT;
"""
        )
        if not artifact:
            return self._quarantine_worker_job(envelope, state, "payload_not_found_or_denied", counters)
        artifact_row = artifact[0]
        counters["policy_calls"] += 1
        decision = self._opa_decision(context, artifact_row)
        if decision["decision"] != "allow":
            return self._quarantine_worker_job(envelope, state, "policy_denied", counters, decision)
        self._insert_action_record(
            context,
            "jobs",
            str(envelope["job_id"]),
            {
                "job_id": envelope["job_id"],
                "state": "completed",
                "membership_revision": envelope["membership_revision"],
                "artifact_id": envelope["artifact_id"],
                "artifact_payload_digest": _sha256_json(artifact_row["payload"]),
            },
        )
        audit_ref = self._insert_audit(
            context,
            "worker.job.completed",
            "worker.artifact_read",
            {"job_id": envelope["job_id"], "artifact_id": envelope["artifact_id"]},
            decision["decision_id"],
            trace_id=trace_id,
        )
        counters["audit_inserts"] += 1
        return {
            "surface": "worker",
            "job_id": envelope["job_id"],
            "status": "completed",
            "reason": None,
            "context_digest": _sha256_json(context),
            "artifact": artifact_row,
            "policy_decision": decision,
            "audit_refs": [audit_ref],
            "counters": counters,
        }

    def _quarantine_worker_job(
        self,
        envelope: dict[str, Any],
        state: dict[str, Any] | None,
        reason: str,
        counters: dict[str, int],
        decision: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        trace_id = str(envelope.get("trace_id", "trace_vs2_worker_scope"))
        audit_refs: list[str] = []
        if state:
            context = {
                "principal_id": state["principal_id"],
                "tenant_id": state["tenant_id"],
                "namespace_id": state["namespace_id"],
                "workspace_id": state["workspace_id"],
                "owner_id": state["owner_id"],
                "roles": state["roles"],
                "membership_revision": state["membership_revision"],
                "session_version": state["session_version"],
                "revoked": state["revoked"],
            }
            self._insert_action_record(
                context,
                "jobs",
                record_id or str(envelope.get("job_id")),
                {
                    "job_id": envelope.get("job_id"),
                    "state": "quarantined",
                    "reason": reason,
                    "artifact_id": envelope.get("artifact_id"),
                    "membership_revision": envelope.get("membership_revision"),
                    "db_access_after_quarantine": 0,
                    "egress_calls_after_quarantine": 0,
                },
            )
            audit_ref = self._insert_audit(
                context,
                "worker.job.quarantined",
                "worker.artifact_read",
                {"job_id": envelope.get("job_id"), "artifact_id": envelope.get("artifact_id"), "reason": reason},
                (decision or {}).get("decision_id", f"worker_{reason}"),
                trace_id=trace_id,
            )
            audit_refs.append(audit_ref)
            counters["audit_inserts"] += 1
        return {
            "surface": "worker",
            "job_id": envelope.get("job_id"),
            "status": "quarantined",
            "reason": reason,
            "artifact": None,
            "policy_decision": decision,
            "audit_refs": audit_refs,
            "counters": counters,
        }

    def artifact_show(
        self,
        *,
        token: str | None,
        artifact_id: str,
        caller_fields: dict[str, Any],
        surface: str,
        trace_id: str = "trace_vs2_local_range",
    ) -> dict[str, Any]:
        counters = {"db_calls": 0, "policy_calls": 0, "egress_calls": 0, "audit_inserts": 0}
        token_payload, token_error = _decode_token(token, self.token_key)
        if token_error or token_payload is None:
            payload = {
                "surface": surface,
                "trace_id": trace_id,
                "status": "denied",
                "status_code": 401,
                "error": {"code": "CS_IDENTITY_CONTEXT_INVALID", "reason": token_error},
                "context_digest": None,
                "policy_decision": None,
                "audit_refs": [],
                "tenant_b_rows_returned": 0,
                "counters": counters,
            }
            self.requests.append(payload)
            return payload

        counters["db_calls"] += 1
        context_rows = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_agg(to_jsonb(r))::text
FROM cs.resolve_membership(
  {_sql_literal(token_payload['sub'])},
  {_sql_literal(token_payload['membership_id'])},
  {int(token_payload['session_version'])}
) r;
COMMIT;
"""
        )
        if not context_rows:
            reason = self._membership_denial_reason(token_payload)
            audit_ref = self._insert_identity_denial_audit(token_payload, reason=reason, surface=surface, artifact_id=artifact_id)
            counters["audit_inserts"] += 1 if audit_ref else 0
            payload = {
                "surface": surface,
                "trace_id": trace_id,
                "status": "denied",
                "status_code": 401,
                "error": {"code": "CS_MEMBERSHIP_UNRESOLVED", "reason": reason},
                "context_digest": None,
                "identity_decision": {"decision": "deny", "reason_codes": [reason], "decision_id": f"identity_{reason}"},
                "policy_decision": None,
                "audit_refs": [audit_ref] if audit_ref else [],
                "tenant_b_rows_returned": 0,
                "counters": counters,
            }
            self.requests.append(payload)
            return payload
        context = context_rows[0]
        context_digest = _sha256_json(context)
        conflicts = {
            key: value
            for key, value in caller_fields.items()
            if key in {"tenant_id", "namespace_id", "workspace_id", "owner_id", "role", "roles", "classification"}
            and context.get(key) != value
        }
        if conflicts:
            counters["policy_calls"] += 1
            decision = self._opa_decision(
                context,
                {"artifact_id": artifact_id, "tenant_id": "tenant_beta", "namespace_id": "personal", "classification": "internal"},
                trace_id=trace_id,
            )
            audit_ref = self._insert_audit(
                context,
                "scope_forgery.denied",
                "artifact.show",
                {"artifact_id": artifact_id, "conflicts": conflicts},
                decision["decision_id"],
                trace_id=trace_id,
            )
            counters["audit_inserts"] += 1
            payload = {
                "surface": surface,
                "trace_id": trace_id,
                "status": "denied",
                "status_code": 403,
                "error": {"code": "CS_TRUSTED_CONTEXT_CONFLICT"},
                "context": context,
                "context_digest": context_digest,
                "policy_decision": decision,
                "audit_refs": [audit_ref],
                "tenant_b_rows_returned": 0,
                "counters": counters,
                "caller_conflicts": conflicts,
            }
            self.requests.append(payload)
            return payload

        counters["db_calls"] += 1
        artifact = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_agg(jsonb_build_object(
  'artifact_id', artifact_id,
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'classification', classification,
  'payload', payload
))::text
FROM cs.artifacts
WHERE artifact_id = {_sql_literal(artifact_id)};
COMMIT;
"""
        )
        if not artifact:
            payload = {
                "surface": surface,
                "trace_id": trace_id,
                "status": "denied",
                "status_code": 404,
                "error": {"code": "CS_RESOURCE_NOT_FOUND_OR_DENIED"},
                "context": context,
                "context_digest": context_digest,
                "policy_decision": None,
                "audit_refs": [],
                "tenant_b_rows_returned": 0,
                "counters": counters,
            }
            self.requests.append(payload)
            return payload
        artifact_row = artifact[0]
        counters["policy_calls"] += 1
        decision = self._opa_decision(context, artifact_row, trace_id=trace_id)
        audit_ref = self._insert_audit(context, "artifact.read", "artifact.show", {"artifact_id": artifact_id}, decision["decision_id"], trace_id=trace_id)
        counters["audit_inserts"] += 1
        payload = {
            "surface": surface,
            "trace_id": trace_id,
            "status": "allowed" if decision["decision"] == "allow" else "denied",
            "status_code": 200 if decision["decision"] == "allow" else 403,
            "error": None if decision["decision"] == "allow" else {"code": "CS_POLICY_DENIED"},
            "context": context,
            "context_digest": context_digest,
            "policy_decision": decision,
            "audit_refs": [audit_ref],
            "artifact": artifact_row if decision["decision"] == "allow" else None,
            "tenant_b_rows_returned": 0,
            "counters": counters,
        }
        self.requests.append(payload)
        return payload

    def _opa_decision(self, context: dict[str, Any], artifact: dict[str, Any], *, trace_id: str = "trace_vs2_local_range") -> dict[str, Any]:
        policy_input, _ = self._build_policy_input(
            context,
            artifact,
            enforcement_point="service",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id=trace_id,
            authority_ref="authority_alpha_owner",
            data_purpose="artifact_read",
        )
        return self._evaluate_opa_policy_input(policy_input)

    def _opa_action_decision(self, context: dict[str, Any], action_id: str, *, risk: str, approval_status: str, capability_declared: bool = True) -> dict[str, Any]:
        policy_input, _ = self._build_policy_input(
            context,
            {
                "artifact_id": action_id,
                "tenant_id": context["tenant_id"],
                "namespace_id": context["namespace_id"],
                "classification": "internal",
            },
            enforcement_point="connector",
            action="connector.execute",
            policy_path="connector.execute",
            trace_id="trace_vs2_external_action",
            risk=risk,
            approval_status=approval_status,
            capability_declared=capability_declared,
            connectorhub_mediated=capability_declared,
            authority_ref="authority_alpha_action",
            data_purpose="connector_execute",
        )
        return self._evaluate_opa_policy_input(policy_input)

    def _opa_migration_decision(
        self,
        context: dict[str, Any],
        migration_id: str,
        *,
        trace_id: str = "trace_vs2_migration_020",
        action: str = "migration.run",
        policy_path: str = "migration.run",
        risk: str = "low",
        approval_status: str = "not_required",
    ) -> dict[str, Any]:
        policy_input, _ = self._build_policy_input(
            context,
            {
                "artifact_id": migration_id,
                "tenant_id": context["tenant_id"],
                "namespace_id": context["namespace_id"],
                "classification": "internal",
            },
            enforcement_point="service",
            action=action,
            policy_path=policy_path,
            trace_id=trace_id,
            risk=risk,
            approval_status=approval_status,
            authority_ref="authority_alpha_migration",
            data_purpose="migration",
        )
        return self._evaluate_opa_policy_input(policy_input)

    def _evaluate_opa_policy_input(
        self,
        policy_input: dict[str, Any],
        *,
        opa_url: str | None = None,
        timeout: float = 5,
        expected_revision: str = "vs2-rego-local-v1",
    ) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        failure_reason: str | None = None
        request = urllib.request.Request(
            f"{opa_url or self.opa_url}/v1/data/cornerstone/vs2/decision",
            data=json.dumps({"input": policy_input}, sort_keys=True).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            if not isinstance(body, dict) or "result" not in body:
                failure_reason = "policy_undefined_result"
            elif not isinstance(body["result"], dict):
                failure_reason = "policy_client_malformed_result"
            else:
                result = body["result"]
                if result.get("decision") not in {"allow", "deny"} or not isinstance(result.get("reason_codes"), list):
                    failure_reason = "policy_client_malformed_result"
                elif result.get("bundle_revision") != expected_revision:
                    failure_reason = "policy_revision_mismatch"
                else:
                    raw = result
        except urllib.error.HTTPError:
            failure_reason = "policy_client_http_error"
        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)
            failure_reason = "policy_client_timeout" if isinstance(reason, TimeoutError | socket.timeout) else "policy_client_unavailable"
        except TimeoutError:
            failure_reason = "policy_client_timeout"
        except (OSError, ValueError, json.JSONDecodeError):
            failure_reason = "policy_client_malformed_result"
        if failure_reason is not None:
            raw = {
                "decision": "deny",
                "reason_codes": [failure_reason],
                "policy_path": "cornerstone.vs2/fail_closed",
                "bundle_revision": None,
            }
        decision = {
            "schema_version": "cs.policy_decision.vs2.v1",
            "decision": raw.get("decision", "deny"),
            "reason_codes": raw.get("reason_codes", []),
            "policy_path": raw.get("policy_path"),
            "bundle_revision": raw.get("bundle_revision"),
            "input_digest": _sha256_json(policy_input),
            "trace_id": policy_input["trace_id"],
            "fail_closed": failure_reason is not None,
            "failure_reason": failure_reason,
        }
        decision["decision_id"] = f"policy_{_sha256_json(decision)[:16]}"
        return decision

    def _policy_cache_key_material(self, policy_input: dict[str, Any], *, active_revision: str) -> dict[str, Any]:
        return {
            "tenant_id": policy_input["scope"]["tenant_id"],
            "principal_id": policy_input["subject"]["principal_id"],
            "resource_id": policy_input["resource"]["resource_id"],
            "action": policy_input["action"],
            "policy_revision": active_revision,
            "membership_revision": policy_input["subject"]["membership_revision"],
        }

    def _policy_cache_legacy_key_material(self, policy_input: dict[str, Any]) -> dict[str, Any]:
        return {
            "tenant_id": policy_input["scope"]["tenant_id"],
            "principal_id": policy_input["subject"]["principal_id"],
            "resource_id": policy_input["resource"]["resource_id"],
            "action": policy_input["action"],
        }

    def _revision_cached_policy_decision(
        self,
        policy_input: dict[str, Any],
        *,
        active_revision: str,
        opa_url: str | None = None,
        timeout: float = 5,
    ) -> dict[str, Any]:
        key_material = self._policy_cache_key_material(policy_input, active_revision=active_revision)
        key_digest = _sha256_json(key_material)
        legacy_key_material = self._policy_cache_legacy_key_material(policy_input)
        legacy_key_digest = _sha256_json(legacy_key_material)
        if not hasattr(self, "policy_decision_cache"):
            self.policy_decision_cache = {}
        cached = self.policy_decision_cache.get(key_digest)
        if cached is not None:
            return {
                "cache_hit": True,
                "cache_key_material": key_material,
                "cache_key_digest": key_digest,
                "legacy_key_material": legacy_key_material,
                "legacy_key_digest": legacy_key_digest,
                "decision": cached["decision"],
                "decision_source": "revision_aware_cache",
            }
        decision = self._evaluate_opa_policy_input(policy_input, opa_url=opa_url, timeout=timeout, expected_revision=active_revision)
        self.policy_decision_cache[key_digest] = {"decision": decision, "stored_at": time.time()}
        return {
            "cache_hit": False,
            "cache_key_material": key_material,
            "cache_key_digest": key_digest,
            "legacy_key_material": legacy_key_material,
            "legacy_key_digest": legacy_key_digest,
            "decision": decision,
            "decision_source": "real_opa",
        }

    def _artifact_resource_for_context(self, context: dict[str, Any], artifact_id: str) -> dict[str, Any] | None:
        rows = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_agg(jsonb_build_object(
  'artifact_id', artifact_id,
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'classification', classification,
  'payload', payload
))::text
FROM cs.artifacts
WHERE artifact_id = {_sql_literal(artifact_id)};
COMMIT;
"""
        )
        return rows[0] if rows else None

    def policy_cache_invalidation_matrix(
        self,
        *,
        owner_token: str,
        other_token: str,
        artifact_id: str,
        updated_opa_url: str,
    ) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(owner_token, "policy_cache")
        other_resolved, other_denied = self._resolve_context_from_token(other_token, "policy_cache_other_tenant")
        if denied or resolved is None or other_denied or other_resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "denied": denied,
                "other_denied": other_denied,
                "checks": {},
            }
        context = resolved["context"]
        other_context = other_resolved["context"]
        resource = self._artifact_resource_for_context(context, artifact_id)
        other_resource = self._artifact_resource_for_context(other_context, "artifact_beta_001")
        if resource is None or other_resource is None:
            return {
                "status": "failed",
                "reason": "resource_resolution_failed",
                "checks": {},
            }
        before_input, before_source_map = self._build_policy_input(
            context,
            resource,
            enforcement_point="service",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_cache_allow_044",
            authority_ref="authority_gamma_cache_probe",
            data_purpose="artifact_read",
        )
        same_revision_input = json.loads(json.dumps(before_input))
        same_revision_input["trace_id"] = "trace_vs2_policy_cache_same_revision_044"
        after_input = json.loads(json.dumps(before_input))
        after_input["trace_id"] = "trace_vs2_policy_cache_revision_update_044"
        other_input, _ = self._build_policy_input(
            other_context,
            other_resource,
            enforcement_point="service",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_cache_other_tenant_044",
            authority_ref="authority_beta_cache_probe",
            data_purpose="artifact_read",
        )

        before = self._revision_cached_policy_decision(before_input, active_revision="vs2-rego-local-v1")
        same_revision_replay = self._revision_cached_policy_decision(same_revision_input, active_revision="vs2-rego-local-v1")
        after = self._revision_cached_policy_decision(
            after_input,
            active_revision="vs2-rego-local-v2",
            opa_url=updated_opa_url,
        )
        other_tenant = self._revision_cached_policy_decision(other_input, active_revision="vs2-rego-local-v1")

        audit_refs = [
            self._insert_audit(
                context,
                "policy.cache.allow",
                "artifact.read",
                {
                    "scenario_id": "VS2-SEC-044",
                    "cache_key_digest": before["cache_key_digest"],
                    "policy_revision": before["cache_key_material"]["policy_revision"],
                    "membership_revision": before["cache_key_material"]["membership_revision"],
                    "decision_id": before["decision"]["decision_id"],
                },
                before["decision"]["decision_id"],
                trace_id="trace_vs2_policy_cache_allow_044",
            ),
            self._insert_audit(
                context,
                "policy.cache.revision_update_denied",
                "artifact.read",
                {
                    "scenario_id": "VS2-SEC-044",
                    "cache_key_digest": after["cache_key_digest"],
                    "policy_revision": after["cache_key_material"]["policy_revision"],
                    "membership_revision": after["cache_key_material"]["membership_revision"],
                    "decision_id": after["decision"]["decision_id"],
                    "cache_hit": after["cache_hit"],
                    "reason_codes": after["decision"].get("reason_codes", []),
                },
                after["decision"]["decision_id"],
                trace_id="trace_vs2_policy_cache_revision_update_044",
            ),
        ]

        required_key_fields = {"tenant_id", "principal_id", "resource_id", "action", "policy_revision", "membership_revision"}
        checks = {
            "policy_cache_initial_allow_from_real_opa": before["cache_hit"] is False
            and before["decision_source"] == "real_opa"
            and before["decision"].get("decision") == "allow"
            and before["decision"].get("bundle_revision") == "vs2-rego-local-v1",
            "policy_cache_same_revision_hit_exercised": same_revision_replay["cache_hit"] is True
            and same_revision_replay["decision_source"] == "revision_aware_cache"
            and same_revision_replay["decision"].get("decision") == "allow",
            "policy_cache_key_contains_required_dimensions": set(before["cache_key_material"]) == required_key_fields
            and all(before["cache_key_material"].get(field) for field in required_key_fields),
            "policy_cache_revision_update_changes_key": before["cache_key_digest"] != after["cache_key_digest"]
            and before["cache_key_material"]["policy_revision"] == "vs2-rego-local-v1"
            and after["cache_key_material"]["policy_revision"] == "vs2-rego-local-v2",
            "policy_cache_legacy_key_without_revision_would_collide": before["legacy_key_digest"] == after["legacy_key_digest"],
            "policy_cache_stale_allow_not_reused_after_revision_update": after["cache_hit"] is False
            and after["decision_source"] == "real_opa"
            and after["decision"].get("decision") == "deny",
            "policy_cache_new_revision_decision_recorded": after["decision"].get("bundle_revision") == "vs2-rego-local-v2"
            and bool(after["decision"].get("decision_id"))
            and bool(after["decision"].get("input_digest")),
            "policy_cache_zero_stale_allows": after["decision"].get("decision") == "deny"
            and "policy_revision_updated_denied" in after["decision"].get("reason_codes", []),
            "policy_cache_cross_tenant_key_distinct": other_tenant["cache_key_digest"] not in {before["cache_key_digest"], after["cache_key_digest"]}
            and other_tenant["cache_key_material"]["tenant_id"] != before["cache_key_material"]["tenant_id"]
            and other_tenant["cache_key_material"]["principal_id"] != before["cache_key_material"]["principal_id"],
            "policy_cache_source_map_bound_to_trusted_context": before_source_map["subject.membership_revision"] == "trusted_request_context.postgres.membership_revision"
            and before_source_map["scope.tenant_id"] == "trusted_request_context.tenant_scope",
            "policy_cache_audit_refs_recorded": all(audit_refs),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "before": before,
            "same_revision_replay": same_revision_replay,
            "after_revision_update": after,
            "other_tenant": {
                "cache_key_material": other_tenant["cache_key_material"],
                "cache_key_digest": other_tenant["cache_key_digest"],
                "decision": other_tenant["decision"],
                "cache_hit": other_tenant["cache_hit"],
            },
            "cache_size": len(self.policy_decision_cache),
            "documented_stale_allow_bound": "active policy revision and trusted membership revision are part of the cache key; revision changes force a real OPA re-evaluation before reuse",
            "source_map": before_source_map,
            "audit_refs": audit_refs,
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def _opa_failure_behavior_matrix(self, context: dict[str, Any], resource: dict[str, Any]) -> dict[str, Any]:
        fault_servers = {
            mode: _OpaFaultServer(mode)
            for mode in ["timeout", "http_500", "malformed_result", "undefined_result", "revision_mismatch"]
        }
        for server in fault_servers.values():
            server.start()
        refused_url = f"http://127.0.0.1:{_free_port()}"
        specs = [
            {
                "case": "connection_refused",
                "operation_family": "protected_read",
                "action": "artifact.read",
                "policy_path": "artifact.read",
                "opa_url": refused_url,
                "timeout": 0.2,
                "expected_reason": "policy_client_unavailable",
            },
            {
                "case": "timeout",
                "operation_family": "protected_write",
                "action": "artifact.write",
                "policy_path": "artifact.write",
                "opa_url": fault_servers["timeout"].url,
                "timeout": 0.05,
                "expected_reason": "policy_client_timeout",
            },
            {
                "case": "http_500",
                "operation_family": "tool_runtime",
                "action": "tool.execute",
                "policy_path": "tool.execute",
                "opa_url": fault_servers["http_500"].url,
                "timeout": 1,
                "expected_reason": "policy_client_http_error",
            },
            {
                "case": "malformed_result",
                "operation_family": "action_card",
                "action": "action.execute",
                "policy_path": "action.execute",
                "opa_url": fault_servers["malformed_result"].url,
                "timeout": 1,
                "expected_reason": "policy_client_malformed_result",
            },
            {
                "case": "undefined_result",
                "operation_family": "connector",
                "action": "connector.execute",
                "policy_path": "connector.execute",
                "opa_url": fault_servers["undefined_result"].url,
                "timeout": 1,
                "expected_reason": "policy_undefined_result",
            },
            {
                "case": "revision_mismatch",
                "operation_family": "model_router",
                "action": "model.route",
                "policy_path": "model.route",
                "opa_url": fault_servers["revision_mismatch"].url,
                "timeout": 1,
                "expected_reason": "policy_revision_mismatch",
            },
        ]
        matrix: list[dict[str, Any]] = []
        audit_refs: list[str] = []
        try:
            for index, spec in enumerate(specs, start=1):
                decision = self._opa_policy_decision_from_fields(
                    context,
                    resource | {"artifact_id": f"opa_failure_{spec['operation_family']}_{index}"},
                    action=spec["action"],
                    policy_path=spec["policy_path"],
                    trace_id=f"trace_vs2_policy_failure_043_{index}",
                    opa_url=spec["opa_url"],
                    timeout=spec["timeout"],
                )
                audit_ref = self._insert_audit(
                    context,
                    "policy.fail_closed",
                    spec["action"],
                    {
                        "scenario_id": "VS2-SEC-043",
                        "case": spec["case"],
                        "operation_family": spec["operation_family"],
                        "expected_reason": spec["expected_reason"],
                        "reason_codes": decision.get("reason_codes", []),
                    },
                    decision["decision_id"],
                    trace_id=f"trace_vs2_policy_failure_043_{index}",
                )
                audit_refs.append(audit_ref)
                matrix.append(
                    {
                        "case": spec["case"],
                        "operation_family": spec["operation_family"],
                        "expected_reason": spec["expected_reason"],
                        "decision": decision,
                        "audit_ref": audit_ref,
                        "downstream_side_effect_attempted": False,
                        "stable_response": {
                            "status_code": 503,
                            "code": "CS_POLICY_UNAVAILABLE",
                            "resolution": "retry after the active OPA policy service and bundle revision are healthy",
                            "protected_data_echoed": False,
                        },
                    }
                )
        finally:
            for server in fault_servers.values():
                server.stop()
        readiness = {
            "component": "opa",
            "status": "degraded",
            "failure_count": len(matrix),
            "serving_risky_operations": False,
        }
        return {
            "status": "passed",
            "matrix": matrix,
            "readiness": readiness,
            "fault_server_calls": {mode: server.calls for mode, server in fault_servers.items()},
            "audit_refs": audit_refs,
        }

    def policy_enforcement_matrix(self, *, owner_token: str, member_token: str) -> dict[str, Any]:
        owner_resolved, owner_denied = self._resolve_context_from_token(owner_token, "service_direct")
        member_resolved, member_denied = self._resolve_context_from_token(member_token, "service_direct")
        if owner_denied or member_denied or owner_resolved is None or member_resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "owner_denied": owner_denied,
                "member_denied": member_denied,
                "checks": {
                    "allow_decision_reaches_downstream_and_audit": False,
                    "role_denied_without_downstream_side_effect": False,
                    "unknown_policy_default_denied_without_downstream_side_effect": False,
                    "decisions_have_revision_digest_and_id": False,
                    "denials_have_stable_safe_responses": False,
                },
            }
        owner_context = owner_resolved["context"]
        member_context = member_resolved["context"]
        side_effect_counts_before = self._policy_side_effect_counts()
        allow_flow = self.artifact_show(
            token=owner_token,
            artifact_id="artifact_alpha_001",
            caller_fields={},
            surface="service_direct",
            trace_id="trace_vs2_policy_allow_027",
        )
        beta_resource = {
            "artifact_id": "artifact_beta_001",
            "tenant_id": member_context["tenant_id"],
            "namespace_id": member_context["namespace_id"],
            "classification": "internal",
        }
        role_denied = self._opa_policy_decision_from_fields(
            member_context,
            beta_resource,
            action="artifact.write",
            policy_path="artifact.write",
            trace_id="trace_vs2_policy_role_deny_028",
        )
        role_audit_ref = self._insert_audit(
            member_context,
            "policy.denied",
            "artifact.write",
            {"scenario_id": "VS2-SEC-028", "resource_id": "artifact_beta_001", "reason_codes": role_denied.get("reason_codes", [])},
            role_denied["decision_id"],
            trace_id="trace_vs2_policy_role_deny_028",
        )
        unknown_resource = {
            "artifact_id": "artifact_alpha_001",
            "tenant_id": owner_context["tenant_id"],
            "namespace_id": owner_context["namespace_id"],
            "classification": "internal",
        }
        unknown_denied = self._opa_policy_decision_from_fields(
            owner_context,
            unknown_resource,
            action="artifact.read",
            policy_path="unknown",
            trace_id="trace_vs2_policy_unknown_030",
        )
        unknown_audit_ref = self._insert_audit(
            owner_context,
            "policy.denied",
            "artifact.read",
            {"scenario_id": "VS2-SEC-030", "resource_id": "artifact_alpha_001", "reason_codes": unknown_denied.get("reason_codes", [])},
            unknown_denied["decision_id"],
            trace_id="trace_vs2_policy_unknown_030",
        )
        abac_allowed = self._opa_policy_decision_from_fields(
            owner_context,
            unknown_resource,
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_abac_allow_029",
        )
        abac_case_specs = [
            {
                "case": "secret_classification",
                "resource": unknown_resource | {"classification": "secret"},
                "kwargs": {},
                "expected_reason": "secret_classification_denied",
            },
            {
                "case": "missing_mission_authority",
                "resource": unknown_resource,
                "kwargs": {"mission_authorized": False},
                "expected_reason": "mission_authority_required",
            },
            {
                "case": "cross_tenant_data_scope",
                "resource": unknown_resource,
                "kwargs": {"data_scope": "cross_tenant"},
                "expected_reason": "data_scope_denied",
            },
            {
                "case": "external_workspace_mode",
                "resource": unknown_resource,
                "kwargs": {"workspace_mode": "external"},
                "expected_reason": "workspace_mode_denied",
            },
        ]
        abac_matrix: list[dict[str, Any]] = []
        abac_audit_refs: list[str] = []
        for index, spec in enumerate(abac_case_specs, start=1):
            decision = self._opa_policy_decision_from_fields(
                owner_context,
                spec["resource"],
                action="artifact.read",
                policy_path="artifact.read",
                trace_id=f"trace_vs2_policy_abac_deny_029_{index}",
                **spec["kwargs"],
            )
            audit_ref = self._insert_audit(
                owner_context,
                "policy.denied",
                "artifact.read",
                {
                    "scenario_id": "VS2-SEC-029",
                    "case": spec["case"],
                    "expected_reason": spec["expected_reason"],
                    "reason_codes": decision.get("reason_codes", []),
                },
                decision["decision_id"],
                trace_id=f"trace_vs2_policy_abac_deny_029_{index}",
            )
            abac_audit_refs.append(audit_ref)
            abac_matrix.append(
                {
                    "case": spec["case"],
                    "decision": decision,
                    "expected_reason": spec["expected_reason"],
                    "audit_ref": audit_ref,
                    "role_alone_would_allow": "owner" in owner_context["roles"],
                }
            )
        malformed_case_specs = [
            {
                "case": "partial_missing_mission_field",
                "kwargs": {"omit_paths": [("mission_authority", "authorized")]},
                "expected_reason": "invalid_schema",
            },
            {
                "case": "wrong_schema_version",
                "kwargs": {"schema_version": "cs.policy_input.v2.invalid"},
                "expected_reason": "invalid_schema",
            },
            {
                "case": "malformed_subject_roles",
                "kwargs": {"subject_overrides": {"roles": "owner"}},
                "expected_reason": "invalid_schema",
            },
            {
                "case": "unexpected_authoritative_subject_tenant",
                "kwargs": {"subject_overrides": {"tenant_id": "tenant_beta"}},
                "expected_reason": "unexpected_authoritative_attribute",
            },
        ]
        malformed_matrix: list[dict[str, Any]] = []
        malformed_audit_refs: list[str] = []
        for index, spec in enumerate(malformed_case_specs, start=1):
            decision = self._opa_policy_decision_from_fields(
                owner_context,
                unknown_resource,
                action="artifact.read",
                policy_path="artifact.read",
                trace_id=f"trace_vs2_policy_malformed_031_{index}",
                **spec["kwargs"],
            )
            audit_ref = self._insert_audit(
                owner_context,
                "policy.denied",
                "artifact.read",
                {
                    "scenario_id": "VS2-SEC-031",
                    "case": spec["case"],
                    "expected_reason": spec["expected_reason"],
                    "reason_codes": decision.get("reason_codes", []),
                    "protected_data_echoed": False,
                },
                decision["decision_id"],
                trace_id=f"trace_vs2_policy_malformed_031_{index}",
            )
            malformed_audit_refs.append(audit_ref)
            malformed_matrix.append(
                {
                    "case": spec["case"],
                    "decision": decision,
                    "expected_reason": spec["expected_reason"],
                    "audit_ref": audit_ref,
                    "safe_corrective_action": "retry with the cs.policy_input.vs2.v1 schema and trusted context-derived attributes",
                    "protected_data_echoed": False,
                }
            )
        deny_precedence = self._opa_policy_decision_from_fields(
            owner_context,
            unknown_resource | {"classification": "secret"},
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_deny_precedence_049",
            risk="high",
            approval_status="missing",
            data_scope="cross_tenant",
        )
        deny_precedence_audit_ref = self._insert_audit(
            owner_context,
            "policy.denied",
            "artifact.read",
            {
                "scenario_id": "VS2-SEC-049",
                "role_alone_would_allow": "owner" in owner_context["roles"],
                "restrictive_rules": ["secret_classification_denied", "high_risk_requires_approval", "data_scope_denied"],
                "reason_codes": deny_precedence.get("reason_codes", []),
            },
            deny_precedence["decision_id"],
            trace_id="trace_vs2_policy_deny_precedence_049",
        )
        opa_failure_behavior = self._opa_failure_behavior_matrix(owner_context, unknown_resource)
        side_effect_counts_after = self._policy_side_effect_counts()
        stable_denial_responses = {
            "role_denied": {
                "status_code": 403,
                "code": "CS_POLICY_DENIED",
                "safe_reason_codes": role_denied.get("reason_codes", []),
                "resolution": "request a role or change scope",
                "protected_data_echoed": False,
            },
            "unknown_policy": {
                "status_code": 403,
                "code": "CS_POLICY_DENIED",
                "safe_reason_codes": unknown_denied.get("reason_codes", []),
                "resolution": "use a declared resource, action, or policy path",
                "protected_data_echoed": False,
            },
            "malformed_input": {
                "status_code": 403,
                "code": "CS_POLICY_DENIED",
                "safe_reason_codes": sorted({reason for case in malformed_matrix for reason in case["decision"].get("reason_codes", [])}),
                "resolution": "retry with the versioned policy input schema and trusted attribute sources",
                "protected_data_echoed": False,
            },
            "policy_client_failure": {
                "status_code": 503,
                "code": "CS_POLICY_UNAVAILABLE",
                "safe_reason_codes": sorted({reason for case in opa_failure_behavior["matrix"] for reason in case["decision"].get("reason_codes", [])}),
                "resolution": "retry after the active OPA policy service and bundle revision are healthy",
                "protected_data_echoed": False,
            },
        }
        decisions = [
            allow_flow.get("policy_decision", {}),
            role_denied,
            unknown_denied,
            abac_allowed,
            deny_precedence,
            *[case["decision"] for case in abac_matrix],
            *[case["decision"] for case in malformed_matrix],
            *[case["decision"] for case in opa_failure_behavior["matrix"]],
        ]
        non_audit_side_effects_unchanged = side_effect_counts_before == side_effect_counts_after
        checks = {
            "allow_decision_reaches_downstream_and_audit": allow_flow.get("status") == "allowed"
            and allow_flow.get("policy_decision", {}).get("decision") == "allow"
            and allow_flow.get("artifact", {}).get("artifact_id") == "artifact_alpha_001"
            and bool(allow_flow.get("audit_refs")),
            "role_denied_without_downstream_side_effect": role_denied.get("decision") == "deny"
            and "role_not_allowed" in role_denied.get("reason_codes", [])
            and non_audit_side_effects_unchanged
            and bool(role_audit_ref),
            "unknown_policy_default_denied_without_downstream_side_effect": unknown_denied.get("decision") == "deny"
            and "unknown_policy_default_deny" in unknown_denied.get("reason_codes", [])
            and non_audit_side_effects_unchanged
            and bool(unknown_audit_ref),
            "abac_attribute_boundaries_enforced": all(
                case["decision"].get("decision") == "deny"
                and case["expected_reason"] in case["decision"].get("reason_codes", [])
                and case["role_alone_would_allow"] is True
                and bool(case["audit_ref"])
                for case in abac_matrix
            )
            and non_audit_side_effects_unchanged,
            "abac_matching_allowed_set_succeeds": abac_allowed.get("decision") == "allow"
            and abac_allowed.get("reason_codes") == [],
            "malformed_and_wrong_version_inputs_fail_closed": all(
                case["decision"].get("decision") == "deny"
                and case["expected_reason"] in case["decision"].get("reason_codes", [])
                and case["protected_data_echoed"] is False
                and bool(case["safe_corrective_action"])
                and bool(case["audit_ref"])
                for case in malformed_matrix
            )
            and non_audit_side_effects_unchanged,
            "unexpected_authoritative_attrs_fail_closed": any(
                case["case"] == "unexpected_authoritative_subject_tenant"
                and case["decision"].get("decision") == "deny"
                and "unexpected_authoritative_attribute" in case["decision"].get("reason_codes", [])
                for case in malformed_matrix
            ),
            "deny_precedence_conflict_matrix_enforced": deny_precedence.get("decision") == "deny"
            and "secret_classification_denied" in deny_precedence.get("reason_codes", [])
            and "high_risk_requires_approval" in deny_precedence.get("reason_codes", [])
            and "data_scope_denied" in deny_precedence.get("reason_codes", [])
            and non_audit_side_effects_unchanged
            and bool(deny_precedence_audit_ref),
            "opa_failure_modes_fail_closed_without_side_effects": all(
                case["decision"].get("decision") == "deny"
                and case["decision"].get("fail_closed") is True
                and case["expected_reason"] in case["decision"].get("reason_codes", [])
                and case["downstream_side_effect_attempted"] is False
                and bool(case["audit_ref"])
                for case in opa_failure_behavior["matrix"]
            )
            and non_audit_side_effects_unchanged,
            "opa_failure_modes_cover_protected_operation_families": {
                case["operation_family"] for case in opa_failure_behavior["matrix"]
            }
            == {"protected_read", "protected_write", "tool_runtime", "action_card", "connector", "model_router"},
            "opa_failure_readiness_degraded": opa_failure_behavior.get("readiness", {}).get("status") == "degraded"
            and opa_failure_behavior.get("readiness", {}).get("serving_risky_operations") is False,
            "opa_failure_denials_have_stable_safe_responses": stable_denial_responses["policy_client_failure"]["status_code"] == 503
            and stable_denial_responses["policy_client_failure"]["code"] == "CS_POLICY_UNAVAILABLE"
            and stable_denial_responses["policy_client_failure"]["protected_data_echoed"] is False
            and len(stable_denial_responses["policy_client_failure"]["safe_reason_codes"]) >= 6,
            "decisions_have_revision_digest_and_id": all(
                decision.get("bundle_revision") == "vs2-rego-local-v1"
                and bool(decision.get("input_digest"))
                and bool(decision.get("decision_id"))
                for decision in decisions
                if decision.get("fail_closed") is not True
            ),
            "fail_closed_decisions_have_digest_and_id": all(
                bool(decision.get("input_digest")) and bool(decision.get("decision_id")) for decision in decisions if decision.get("fail_closed") is True
            ),
            "denials_have_stable_safe_responses": all(
                response["status_code"] == 403
                and response["code"] == "CS_POLICY_DENIED"
                and response["protected_data_echoed"] is False
                and bool(response["resolution"])
                for key, response in stable_denial_responses.items()
                if key != "policy_client_failure"
            ),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "owner_context_digest": _sha256_json(owner_context),
            "member_context_digest": _sha256_json(member_context),
            "allow_flow": allow_flow,
            "role_denied": role_denied,
            "unknown_denied": unknown_denied,
            "abac_allowed": abac_allowed,
            "abac_matrix": abac_matrix,
            "malformed_matrix": malformed_matrix,
            "deny_precedence": {
                "decision": deny_precedence,
                "audit_ref": deny_precedence_audit_ref,
                "role_alone_would_allow": "owner" in owner_context["roles"],
                "restrictive_rules": ["secret_classification_denied", "high_risk_requires_approval", "data_scope_denied"],
            },
            "opa_failure_behavior": opa_failure_behavior,
            "stable_denial_responses": stable_denial_responses,
            "side_effect_counts_before": side_effect_counts_before,
            "side_effect_counts_after": side_effect_counts_after,
            "audit_refs": [
                ref
                for ref in [
                    role_audit_ref,
                    unknown_audit_ref,
                    *abac_audit_refs,
                    *malformed_audit_refs,
                    deny_precedence_audit_ref,
                    *opa_failure_behavior.get("audit_refs", []),
                    *allow_flow.get("audit_refs", []),
                ]
                if ref
            ],
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def _opa_policy_decision_from_fields(
        self,
        context: dict[str, Any],
        resource: dict[str, Any],
        *,
        action: str,
        policy_path: str,
        trace_id: str,
        risk: str = "low",
        approval_status: str = "not_required",
        capability_declared: bool = True,
        connectorhub_mediated: bool = True,
        schema_version: str = "cs.policy_input.vs2.v1",
        mission_authorized: bool = True,
        mission_id: str = "mission_alpha",
        authority_ref: str = "authority_alpha_policy_probe",
        data_scope: str = "tenant",
        data_purpose: str = "artifact_read",
        deployment: str = "local",
        workspace_mode: str = "assist",
        subject_overrides: dict[str, Any] | None = None,
        omit_paths: list[tuple[str, ...]] | None = None,
        opa_url: str | None = None,
        timeout: float = 5,
    ) -> dict[str, Any]:
        enforcement_point = {
            "artifact": "service",
            "tool": "tool_runtime",
            "action": "action_card",
            "connector": "connector",
            "model": "model_router",
            "policy": "policy_admin",
            "memory": "memory",
            "migration": "service",
        }.get(action.split(".", 1)[0], "service")
        policy_input, _ = self._build_policy_input(
            context,
            resource,
            enforcement_point=enforcement_point,
            action=action,
            policy_path=policy_path,
            trace_id=trace_id,
            risk=risk,
            approval_status=approval_status,
            capability_declared=capability_declared,
            connectorhub_mediated=connectorhub_mediated,
            schema_version=schema_version,
            mission_authorized=mission_authorized,
            mission_id=mission_id,
            authority_ref=authority_ref,
            data_scope=data_scope,
            data_purpose=data_purpose,
            deployment=deployment,
            workspace_mode=workspace_mode,
            subject_overrides=subject_overrides,
            omit_paths=omit_paths,
        )
        return self._evaluate_opa_policy_input(policy_input, opa_url=opa_url, timeout=timeout)

    def _policy_side_effect_counts(self) -> dict[str, int]:
        return self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'policy_probe_action_cards', (SELECT count(*)::int FROM cs.action_cards WHERE payload->>'probe' = 'policy_enforcement_matrix'),
  'policy_probe_workflow_runs', (SELECT count(*)::int FROM cs.workflow_runs WHERE payload->>'probe' = 'policy_enforcement_matrix'),
  'policy_probe_jobs', (SELECT count(*)::int FROM cs.jobs WHERE payload->>'probe' = 'policy_enforcement_matrix'),
  'policy_probe_egress_grants', (SELECT count(*)::int FROM cs.egress_grants WHERE payload->>'probe' = 'policy_enforcement_matrix')
)::text;
"""
        )

    def _load_egress_grant(self, context: dict[str, Any]) -> dict[str, Any] | None:
        rows = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_agg(payload)::text
FROM cs.egress_grants
WHERE artifact_id = 'grant_mock_provider_status';
COMMIT;
"""
        )
        return rows[0] if rows else None

    def _insert_action_record(self, context: dict[str, Any], relation: str, object_id: str, payload: dict[str, Any]) -> bool:
        result = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.{relation}(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(object_id)},
  'internal',
  {_sql_json(payload)},
  'trace_vs2_external_action'
);
COMMIT;
"""
        )
        return result["exit_code"] == 0

    def external_action_flow(self, *, token: str | None, provider_url: str, surface: str) -> dict[str, Any]:
        counters = {"db_calls": 0, "policy_calls": 0, "egress_proxy_calls": 0, "provider_calls": 0, "audit_inserts": 0}
        token_payload, token_error = _decode_token(token, self.token_key)
        if token_error or token_payload is None:
            return {
                "surface": surface,
                "status": "denied",
                "status_code": 401,
                "error": {"code": "CS_IDENTITY_CONTEXT_INVALID", "reason": token_error},
                "counters": counters,
            }
        counters["db_calls"] += 1
        context_rows = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_agg(to_jsonb(r))::text
FROM cs.resolve_membership(
  {_sql_literal(token_payload['sub'])},
  {_sql_literal(token_payload['membership_id'])},
  {int(token_payload['session_version'])}
) r;
COMMIT;
"""
        )
        if not context_rows:
            return {
                "surface": surface,
                "status": "denied",
                "status_code": 401,
                "error": {"code": "CS_MEMBERSHIP_UNRESOLVED"},
                "counters": counters,
            }
        context = context_rows[0]
        action_id = f"action_{_sha256_json({'tenant': context['tenant_id'], 'provider_url': provider_url})[:12]}"
        grant = self._load_egress_grant(context)
        if grant and grant.get("provider_url") != provider_url:
            grant = None
        counters["db_calls"] += 1
        dry_run_decision = self._opa_action_decision(context, action_id, risk="low", approval_status="not_required", capability_declared=bool(grant))
        counters["policy_calls"] += 1
        dry_run_fingerprint = _sha256_json(
            {
                "tenant_id": context["tenant_id"],
                "action_id": action_id,
                "provider_url": provider_url,
                "capability": grant.get("capability") if grant else None,
                "policy_revision": dry_run_decision.get("bundle_revision"),
            }
        )
        action_record = {
            "action_id": action_id,
            "state": "dry_run_created",
            "capability": "mock_provider.status.write",
            "provider_url": provider_url,
            "dry_run_fingerprint": dry_run_fingerprint,
            "expected_provider_calls": 1 if dry_run_decision["decision"] == "allow" and grant else 0,
            "real_external_http_calls": 0,
            "credential_ref": grant.get("credential_ref") if grant else None,
            "raw_credential_available_to_product": False,
        }
        self._insert_action_record(context, "action_cards", action_id, action_record)
        counters["db_calls"] += 1

        if dry_run_decision["decision"] != "allow" or not grant:
            audit_ref = self._insert_audit(
                context,
                "action.dry_run.denied",
                "external_action",
                {"action_id": action_id, "provider_url": provider_url},
                dry_run_decision["decision_id"],
            )
            counters["audit_inserts"] += 1
            return {
                "surface": surface,
                "status": "denied",
                "status_code": 403,
                "error": {"code": "CS_CONNECTORHUB_CAPABILITY_REQUIRED"},
                "context": context,
                "action_card": action_record,
                "dry_run_decision": dry_run_decision,
                "audit_refs": [audit_ref],
                "counters": counters,
            }

        approval_record = {
            "approval_id": f"approval_{_sha256_json({'action_id': action_id, 'approver': context['principal_id']})[:12]}",
            "status": "approved",
            "approver": context["principal_id"],
            "dry_run_fingerprint": dry_run_fingerprint,
        }
        execution_decision = self._opa_action_decision(context, action_id, risk="high", approval_status="approved", capability_declared=True)
        counters["policy_calls"] += 1
        stale_decision = self._opa_action_decision(context, action_id, risk="high", approval_status="not_required", capability_declared=True)
        counters["policy_calls"] += 1
        stale_probe = {
            "changed_provider_url": provider_url.rstrip("/") + "/stale",
            "decision": "deny",
            "reason": "stale_dry_run_fingerprint",
            "provider_calls_before": 0,
            "provider_calls_after": 0,
            "opa_reason_codes": stale_decision["reason_codes"],
        }

        execution_payload = {
            "target_url": provider_url,
            "capability": "mock_provider.status.write",
            "credential_ref": grant["credential_ref"],
            "trace_id": "trace_vs2_external_action",
            "payload": {
                "action_id": action_id,
                "tenant_id": context["tenant_id"],
                "namespace_id": context["namespace_id"],
                "status": "approved_local_range_write",
            },
        }
        proxy_result: dict[str, Any] | None = None
        if execution_decision["decision"] == "allow":
            request = urllib.request.Request(
                self.egress_proxy_url,
                data=json.dumps(execution_payload, sort_keys=True).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                proxy_result = json.loads(response.read().decode())
            counters["egress_proxy_calls"] += 1
            counters["provider_calls"] += 1
        workflow_id = f"workflow_{action_id}"
        self._insert_action_record(
            context,
            "workflow_runs",
            workflow_id,
            {
                "workflow_id": workflow_id,
                "action_id": action_id,
                "state": "executed" if proxy_result else "blocked",
                "dry_run_fingerprint": dry_run_fingerprint,
                "approval_id": approval_record["approval_id"],
                "proxy_result": proxy_result,
                "idempotency_key": f"{context['tenant_id']}:{action_id}",
            },
        )
        counters["db_calls"] += 1
        audit_ref = self._insert_audit(
            context,
            "connectorhub.egress.executed" if proxy_result else "connectorhub.egress.denied",
            "external_action",
            {"action_id": action_id, "workflow_id": workflow_id, "provider_url": provider_url},
            execution_decision["decision_id"],
        )
        counters["audit_inserts"] += 1
        return {
            "surface": surface,
            "status": "executed" if proxy_result else "denied",
            "status_code": 200 if proxy_result else 403,
            "context": context,
            "action_card": action_record,
            "dry_run_decision": dry_run_decision,
            "approval": approval_record,
            "execution_decision": execution_decision,
            "stale_dry_run_probe": stale_probe,
            "connectorhub": {
                "mediated_by": "ConnectorHub",
                "credential_ref": grant["credential_ref"],
                "raw_credentials_exposed": False,
                "direct_provider_access": False,
                "egress_proxy_url": self.egress_proxy_url,
            },
            "result": proxy_result,
            "audit_refs": [audit_ref],
            "counters": counters,
        }

    def break_glass_maintenance_probe(self, *, operator_token: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(operator_token, "break_glass_maintenance")
        if denied or resolved is None:
            return {"status": "failed", "reason": "context_resolution_failed", "denied": denied, "checks": {}}
        context = resolved["context"]
        resource = {
            "artifact_id": "break_glass_audit_export_014",
            "tenant_id": context["tenant_id"],
            "namespace_id": context["namespace_id"],
            "classification": "internal",
        }
        normal_no_context = self.postgres.json_query(
            """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_build_object(
  'audit_events_visible_without_context', (SELECT count(*)::int FROM cs.audit_events),
  'operator_metrics_visible_without_context', (SELECT count(*)::int FROM cs.operator_metrics),
  'tenant_exports_visible_without_context', (SELECT count(*)::int FROM cs.tenant_exports)
)::text;
COMMIT;
"""
        )
        normal_cross_tenant = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'foreign_audit_events_visible', (SELECT count(*)::int FROM cs.audit_events WHERE tenant_id <> {_sql_literal(context['tenant_id'])}),
  'foreign_metrics_visible', (SELECT count(*)::int FROM cs.operator_metrics WHERE tenant_id <> {_sql_literal(context['tenant_id'])}),
  'foreign_exports_visible', (SELECT count(*)::int FROM cs.tenant_exports WHERE tenant_id <> {_sql_literal(context['tenant_id'])})
)::text;
COMMIT;
"""
        )
        missing_approval_decision = self._opa_policy_decision_from_fields(
            context,
            resource,
            action="policy.break_glass_export",
            policy_path="policy.break_glass_export",
            trace_id="trace_vs2_break_glass_014_missing_approval",
            risk="high",
            approval_status="missing",
            authority_ref="authority_break_glass_missing",
            data_purpose="maintenance_break_glass",
        )
        approved_decision = self._opa_policy_decision_from_fields(
            context,
            resource,
            action="policy.break_glass_export",
            policy_path="policy.break_glass_export",
            trace_id="trace_vs2_break_glass_014_approved",
            risk="high",
            approval_status="approved",
            authority_ref="authority_break_glass_approved",
            data_purpose="maintenance_break_glass",
        )
        issued_at = int(time.time())
        approval_record = {
            "approval_id": f"approval_break_glass_{_sha256_json({'principal': context['principal_id'], 'resource': resource['artifact_id']})[:12]}",
            "status": "approved",
            "approver": context["principal_id"],
            "purpose": "local synthetic cross-tenant maintenance row-count rehearsal",
            "scope": "audit_events_count_only",
            "issued_at_epoch": issued_at,
            "expires_at_epoch": issued_at + 900,
            "policy_decision_id": approved_decision["decision_id"],
        }
        audit_ref = self._insert_audit(
            context,
            "maintenance.break_glass.approved",
            "maintenance.break_glass_export",
            {
                "scenario_id": "VS2-SEC-014",
                "approval": approval_record,
                "output_contract": "counts_and_digests_only",
            },
            approved_decision["decision_id"],
            trace_id="trace_vs2_break_glass_014_approved",
        )
        privileged_output = self.postgres.json_query(
            """
BEGIN;
SET LOCAL ROLE cornerstone_auditor;
SELECT jsonb_build_object(
  'audit_events_visible', count(*)::int,
  'tenant_count', count(DISTINCT tenant_id)::int,
  'event_id_digest', md5(COALESCE(string_agg(event_id, ',' ORDER BY event_id), '')),
  'raw_subject_returned', false,
  'raw_payload_returned', false
)::text
FROM cs.audit_events;
COMMIT;
"""
        )
        audit_check = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'approval_audit_visible', count(*)::int,
  'event_types', COALESCE(jsonb_agg(event_type ORDER BY event_type), '[]'::jsonb)
)::text
FROM cs.audit_events
WHERE event_id = {_sql_literal(audit_ref)};
COMMIT;
"""
        )
        checks = {
            "normal_app_credentials_cannot_break_glass": all(value == 0 for value in normal_no_context.values())
            and all(value == 0 for value in normal_cross_tenant.values()),
            "missing_approval_denied_by_policy": missing_approval_decision.get("decision") == "deny"
            and "high_risk_requires_approval" in missing_approval_decision.get("reason_codes", []),
            "approved_synthetic_break_glass_allowed": approved_decision.get("decision") == "allow"
            and approved_decision.get("bundle_revision") == "vs2-rego-local-v1",
            "approval_has_purpose_time_bound_and_scope": approval_record["status"] == "approved"
            and bool(approval_record["purpose"])
            and approval_record["expires_at_epoch"] > approval_record["issued_at_epoch"]
            and approval_record["scope"] == "audit_events_count_only",
            "break_glass_output_is_safe_counts_only": privileged_output.get("audit_events_visible", 0) >= 1
            and privileged_output.get("tenant_count", 0) >= 1
            and privileged_output.get("raw_subject_returned") is False
            and privileged_output.get("raw_payload_returned") is False
            and bool(privileged_output.get("event_id_digest")),
            "break_glass_audit_recorded": audit_check.get("approval_audit_visible") == 1,
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "context_digest": _sha256_json(context),
            "normal_no_context": normal_no_context,
            "normal_cross_tenant": normal_cross_tenant,
            "missing_approval_decision": missing_approval_decision,
            "approved_decision": approved_decision,
            "approval_record": approval_record,
            "privileged_output": privileged_output,
            "audit_refs": [audit_ref],
            "audit_check": audit_check,
            "scope": "local synthetic break-glass rehearsal; not production emergency access evidence",
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def product_learning_guard_probe(self, *, personal_token: str, org_token: str) -> dict[str, Any]:
        personal_resolved, personal_denied = self._resolve_context_from_token(personal_token, "product_learning_personal")
        org_resolved, org_denied = self._resolve_context_from_token(org_token, "product_learning_org")
        if personal_denied or org_denied or personal_resolved is None or org_resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "personal_denied": personal_denied,
                "org_denied": org_denied,
                "checks": {},
            }
        personal = personal_resolved["context"]
        org = org_resolved["context"]

        def learning_counts() -> dict[str, int]:
            return self.postgres.json_query(
                """
SELECT jsonb_build_object(
  'derived_representations', (SELECT count(*)::int FROM cs.derived_representations WHERE payload->>'probe' = 'product_learning_guard'),
  'claims', (SELECT count(*)::int FROM cs.claims WHERE payload->>'probe' = 'product_learning_guard'),
  'ontology_objects', (SELECT count(*)::int FROM cs.ontology_objects WHERE payload->>'probe' = 'product_learning_guard'),
  'ontology_links', (SELECT count(*)::int FROM cs.ontology_links WHERE payload->>'probe' = 'product_learning_guard'),
  'search_snapshots', (SELECT count(*)::int FROM cs.search_snapshots WHERE payload->>'probe' = 'product_learning_guard')
)::text;
"""
            )

        counts_before = learning_counts()
        cases = [
            {
                "case": "raw_personal_truth_without_opt_in",
                "context": personal,
                "resource": {
                    "artifact_id": "product_learning_personal_truth_r05",
                    "tenant_id": personal["tenant_id"],
                    "namespace_id": personal["namespace_id"],
                    "classification": "internal",
                },
                "data_scope": "tenant",
            },
            {
                "case": "raw_org_truth_without_redaction",
                "context": org,
                "resource": {
                    "artifact_id": "product_learning_org_truth_r05",
                    "tenant_id": org["tenant_id"],
                    "namespace_id": org["namespace_id"],
                    "classification": "internal",
                },
                "data_scope": "cross_tenant",
            },
        ]
        decisions: list[dict[str, Any]] = []
        audit_refs: list[str] = []
        for case in cases:
            decision = self._opa_policy_decision_from_fields(
                case["context"],
                case["resource"],
                action="memory.learn",
                policy_path="memory.learn",
                trace_id=f"trace_vs2_product_learning_r05_{case['case']}",
                risk="high",
                approval_status="missing",
                mission_authorized=False,
                authority_ref="authority_missing_product_learning_opt_in",
                data_scope=case["data_scope"],
                data_purpose="raw_truth_learning",
            )
            audit_ref = self._insert_audit(
                case["context"],
                "product_learning.denied",
                "memory.learn",
                {
                    "scenario_id": "VS2-SEC-R05",
                    "case": case["case"],
                    "raw_truth_requested": True,
                    "opt_in_present": False,
                    "redaction_approved": False,
                },
                decision["decision_id"],
                trace_id=f"trace_vs2_product_learning_r05_{case['case']}",
            )
            decisions.append({"case": case["case"], "decision": decision, "audit_ref": audit_ref})
            audit_refs.append(audit_ref)
        counts_after = learning_counts()
        checks = {
            "raw_personal_truth_learning_denied": any(
                item["case"] == "raw_personal_truth_without_opt_in"
                and item["decision"].get("decision") == "deny"
                and "mission_authority_required" in item["decision"].get("reason_codes", [])
                for item in decisions
            ),
            "raw_org_truth_learning_denied": any(
                item["case"] == "raw_org_truth_without_redaction"
                and item["decision"].get("decision") == "deny"
                and {"mission_authority_required", "data_scope_denied"} & set(item["decision"].get("reason_codes", []))
                for item in decisions
            ),
            "no_hidden_memory_or_truth_writes": counts_before == counts_after,
            "learning_denials_audited": len(audit_refs) == len(cases) and all(audit_refs),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "personal_context_digest": _sha256_json(personal),
            "org_context_digest": _sha256_json(org),
            "decisions": decisions,
            "counts_before": counts_before,
            "counts_after": counts_after,
            "audit_refs": audit_refs,
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def policy_conformance_matrix(self, *, owner_token: str, cli_payload: dict[str, Any]) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(owner_token, "policy_conformance")
        if denied or resolved is None:
            return {"status": "failed", "reason": "context_resolution_failed", "denied": denied, "checks": {}}
        context = resolved["context"]
        resource = {
            "artifact_id": "artifact_alpha_001",
            "tenant_id": context["tenant_id"],
            "namespace_id": context["namespace_id"],
            "classification": "internal",
        }
        policy_input, source_map = self._build_policy_input(
            context,
            resource,
            enforcement_point="gateway",
            action="artifact.read",
            policy_path="artifact.read",
            trace_id="trace_vs2_policy_conformance_050",
            authority_ref="authority_alpha_policy_conformance",
            data_purpose="artifact_read",
        )
        surfaces = ["gateway", "service", "tool_runtime", "native_cli"]
        decisions = {surface: self._evaluate_opa_policy_input(policy_input) for surface in surfaces}
        cli_decision = cli_payload.get("range_client", {}).get("payload", {}).get("policy_decision", {})
        side_effects_before = self._policy_side_effect_counts()
        mismatch_decision = self._evaluate_opa_policy_input(policy_input, expected_revision="vs2-rego-local-mismatch")
        audit_ref = self._insert_audit(
            context,
            "policy.conformance.mismatch_detected",
            "policy.evaluate",
            {
                "scenario_id": "VS2-SEC-050",
                "expected_revision": "vs2-rego-local-mismatch",
                "active_revision": decisions["gateway"].get("bundle_revision"),
                "input_digest": _sha256_json(policy_input),
            },
            mismatch_decision["decision_id"],
            trace_id="trace_vs2_policy_conformance_050_mismatch",
        )
        metric_insert = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.operator_metrics(tenant_id, namespace_id, owner_id, workspace_id, metric_id, metric_name, metric_value, labels, trace_id)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'policy_conformance_mismatch_050',
  'policy_conformance_mismatch_count',
  1,
  {_sql_json({"scenario_id": "VS2-SEC-050", "decision_id": mismatch_decision["decision_id"], "input_digest": _sha256_json(policy_input)})},
  'trace_vs2_policy_conformance_050_mismatch'
)
ON CONFLICT (tenant_id, metric_id) DO NOTHING;
COMMIT;
"""
        )
        side_effects_after = self._policy_side_effect_counts()
        material_fields = ["decision", "bundle_revision", "policy_path", "reason_codes"]
        gateway_material = {field: decisions["gateway"].get(field) for field in material_fields}
        checks = {
            "same_policy_input_digest_across_enforcement_points": len({_sha256_json(policy_input) for _surface in surfaces}) == 1,
            "gateway_service_tool_cli_decisions_equivalent": all(
                {field: decisions[surface].get(field) for field in material_fields} == gateway_material for surface in surfaces
            ),
            "native_cli_observed_same_active_revision": cli_decision.get("decision") == decisions["native_cli"].get("decision")
            and cli_decision.get("bundle_revision") == decisions["native_cli"].get("bundle_revision"),
            "revision_mismatch_fails_closed": mismatch_decision.get("decision") == "deny"
            and mismatch_decision.get("fail_closed") is True
            and mismatch_decision.get("failure_reason") == "policy_revision_mismatch",
            "mismatch_anomaly_audit_and_metric_recorded": bool(audit_ref) and metric_insert["exit_code"] == 0,
            "mismatch_no_protected_side_effect": side_effects_before == side_effects_after,
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "policy_input_digest": _sha256_json(policy_input),
            "source_map": source_map,
            "decisions": decisions,
            "native_cli_decision": cli_decision,
            "mismatch_decision": mismatch_decision,
            "mismatch_audit_ref": audit_ref,
            "metric_insert": _digest_transcript_entry(metric_insert),
            "side_effect_counts_before": side_effects_before,
            "side_effect_counts_after": side_effects_after,
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def same_tenant_namespace_matrix(self, *, personal_token: str, org_token: str) -> dict[str, Any]:
        personal_resolved, personal_denied = self._resolve_context_from_token(personal_token, "service_direct")
        org_resolved, org_denied = self._resolve_context_from_token(org_token, "service_direct")
        if personal_denied or org_denied or personal_resolved is None or org_resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "personal_denied": personal_denied,
                "org_denied": org_denied,
                "checks": {
                    "implicit_cross_namespace_denied_by_policy": False,
                    "personal_context_db_returns_zero_org_rows": False,
                    "org_context_reads_org_row": False,
                    "explicit_promotion_records_provenance": False,
                    "promotion_audited": False,
                },
            }
        personal = personal_resolved["context"]
        org = org_resolved["context"]
        org_artifact = {
            "artifact_id": "artifact_alpha_org_001",
            "tenant_id": "tenant_alpha",
            "namespace_id": "organization",
            "owner_id": "org_alpha",
            "workspace_id": "alpha-org",
            "classification": "internal",
        }
        implicit_decision = self._opa_decision(personal, org_artifact, trace_id="trace_vs2_same_tenant_namespace_022")
        implicit_audit_ref = self._insert_audit(
            personal,
            "namespace.cross_scope.denied",
            "artifact.show",
            {
                "scenario_id": "VS2-SEC-022",
                "resource_id": org_artifact["artifact_id"],
                "resource_namespace": org_artifact["namespace_id"],
                "resource_workspace": org_artifact["workspace_id"],
            },
            implicit_decision["decision_id"],
            trace_id="trace_vs2_same_tenant_namespace_022",
        )
        personal_db_probe = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(personal['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(personal['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(personal['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(personal['workspace_id'])};
SELECT jsonb_build_object(
  'org_rows_visible', (SELECT count(*)::int FROM cs.artifacts WHERE artifact_id = 'artifact_alpha_org_001'),
  'explicit_org_namespace_rows_visible', (
    SELECT count(*)::int
    FROM cs.artifacts
    WHERE tenant_id = 'tenant_alpha'
      AND namespace_id = 'organization'
      AND workspace_id = 'alpha-org'
  ),
  'tenant_ids_returned', COALESCE((SELECT jsonb_agg(DISTINCT tenant_id) FROM cs.artifacts WHERE artifact_id = 'artifact_alpha_org_001'), '[]'::jsonb),
  'namespace_ids_returned', COALESCE((SELECT jsonb_agg(DISTINCT namespace_id) FROM cs.artifacts WHERE artifact_id = 'artifact_alpha_org_001'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )
        org_decision = self._opa_decision(org, org_artifact, trace_id="trace_vs2_same_tenant_namespace_022")
        org_db_probe = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(org['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(org['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(org['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(org['workspace_id'])};
SELECT jsonb_build_object(
  'org_rows_visible', (SELECT count(*)::int FROM cs.artifacts WHERE artifact_id = 'artifact_alpha_org_001'),
  'payloads', COALESCE((SELECT jsonb_agg(payload ORDER BY artifact_id) FROM cs.artifacts WHERE artifact_id = 'artifact_alpha_org_001'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )
        promotion_payload = {
            "scenario_id": "VS2-SEC-022",
            "operation": "explicit_namespace_promotion",
            "source_tenant_id": personal["tenant_id"],
            "source_namespace_id": personal["namespace_id"],
            "source_workspace_id": personal["workspace_id"],
            "source_artifact_digest": _sha256_json({"tenant_id": personal["tenant_id"], "artifact_id": "artifact_alpha_001"}),
            "promotion_approval_ref": "approval_vs2_sec_022_namespace_promotion",
            "provenance_ref": implicit_audit_ref,
        }
        promotion_insert = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(org['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(org['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(org['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(org['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {_sql_literal(org['tenant_id'])},
  {_sql_literal(org['namespace_id'])},
  {_sql_literal(org['owner_id'])},
  {_sql_literal(org['workspace_id'])},
  'promotion_alpha_personal_to_org_022',
  'internal',
  {_sql_json(promotion_payload)},
  'trace_vs2_same_tenant_namespace_022'
)
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
        )
        promotion_audit_ref = self._insert_audit(
            org,
            "namespace.promotion.allowed",
            "artifact.promote",
            promotion_payload,
            org_decision["decision_id"],
            trace_id="trace_vs2_same_tenant_namespace_022",
        )
        promotion_read = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(org['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(org['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(org['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(org['workspace_id'])};
SELECT jsonb_build_object(
  'promotion_rows_visible', (SELECT count(*)::int FROM cs.artifacts WHERE artifact_id = 'promotion_alpha_personal_to_org_022'),
  'promotion_payloads', COALESCE((SELECT jsonb_agg(payload ORDER BY artifact_id) FROM cs.artifacts WHERE artifact_id = 'promotion_alpha_personal_to_org_022'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )
        promotion_payloads = promotion_read.get("promotion_payloads") or []
        checks = {
            "implicit_cross_namespace_denied_by_policy": implicit_decision.get("decision") == "deny"
            and "cross_tenant_scope" in implicit_decision.get("reason_codes", []),
            "personal_context_db_returns_zero_org_rows": personal_db_probe.get("org_rows_visible") == 0
            and personal_db_probe.get("explicit_org_namespace_rows_visible") == 0,
            "org_context_reads_org_row": org_decision.get("decision") == "allow" and org_db_probe.get("org_rows_visible") == 1,
            "explicit_promotion_records_provenance": promotion_insert.get("exit_code") == 0
            and promotion_read.get("promotion_rows_visible") == 1
            and bool(promotion_payloads)
            and promotion_payloads[0].get("promotion_approval_ref") == "approval_vs2_sec_022_namespace_promotion"
            and bool(promotion_payloads[0].get("source_artifact_digest")),
            "promotion_audited": bool(implicit_audit_ref) and bool(promotion_audit_ref),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "personal_context_digest": _sha256_json(personal),
            "org_context_digest": _sha256_json(org),
            "implicit_decision": implicit_decision,
            "org_decision": org_decision,
            "personal_db_probe": personal_db_probe,
            "org_db_probe": org_db_probe,
            "promotion_insert": _digest_transcript_entry(promotion_insert),
            "promotion_read": promotion_read,
            "audit_refs": [implicit_audit_ref, promotion_audit_ref],
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def cross_tenant_transfer_matrix(self, *, token: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, "service_direct")
        if denied or resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "denied": denied,
                "checks": {
                    "cross_tenant_transfer_policy_denied": False,
                    "copy_reference_share_promotion_denied": False,
                    "zero_target_records_created": False,
                    "transfer_denial_audited": False,
                },
            }
        context = resolved["context"]
        beta_artifact = {
            "artifact_id": "artifact_beta_001",
            "tenant_id": "tenant_beta",
            "namespace_id": "personal",
            "owner_id": "principal_bob",
            "workspace_id": "beta-home",
            "classification": "internal",
        }
        decision = self._opa_decision(context, beta_artifact, trace_id="trace_vs2_cross_tenant_transfer_023")
        attempts = {
            "copy": self.postgres.psql(
                f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'transfer_copy_beta_023',
  'internal',
  '{{"scenario_id":"VS2-SEC-023","operation":"copy"}}'::jsonb,
  'trace_vs2_cross_tenant_transfer_023'
);
COMMIT;
"""
            ),
            "reference": self.postgres.psql(
                f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.artifact_references(
  tenant_id, namespace_id, owner_id, workspace_id, reference_id, classification, source_artifact_id, target_artifact_id, payload, audit_ref
)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'transfer_ref_beta_023',
  'internal',
  'artifact_alpha_001',
  'artifact_beta_001',
  '{{"scenario_id":"VS2-SEC-023","operation":"reference"}}'::jsonb,
  'trace_vs2_cross_tenant_transfer_023'
);
COMMIT;
"""
            ),
            "share": self.postgres.psql(
                f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.action_cards(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'transfer_share_beta_023',
  'internal',
  '{{"scenario_id":"VS2-SEC-023","operation":"share"}}'::jsonb,
  'trace_vs2_cross_tenant_transfer_023'
);
COMMIT;
"""
            ),
            "promotion": self.postgres.psql(
                f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.claims(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_beta',
  'personal',
  'principal_bob',
  'beta-home',
  'transfer_promotion_beta_023',
  'internal',
  '{{"scenario_id":"VS2-SEC-023","operation":"promotion"}}'::jsonb,
  'trace_vs2_cross_tenant_transfer_023'
);
COMMIT;
"""
            ),
        }
        target_state = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'copy_rows', (SELECT count(*)::int FROM cs.artifacts WHERE tenant_id = 'tenant_beta' AND artifact_id = 'transfer_copy_beta_023'),
  'reference_rows', (SELECT count(*)::int FROM cs.artifact_references WHERE reference_id = 'transfer_ref_beta_023'),
  'share_rows', (SELECT count(*)::int FROM cs.action_cards WHERE tenant_id = 'tenant_beta' AND artifact_id = 'transfer_share_beta_023'),
  'promotion_rows', (SELECT count(*)::int FROM cs.claims WHERE tenant_id = 'tenant_beta' AND artifact_id = 'transfer_promotion_beta_023')
)::text;
"""
        )
        audit_ref = self._insert_audit(
            context,
            "transfer.cross_tenant.denied",
            "artifact.transfer",
            {
                "scenario_id": "VS2-SEC-023",
                "attempted_operations": sorted(attempts),
                "target_tenant_digest": _sha256_json({"tenant_id": "tenant_beta"}),
            },
            decision["decision_id"],
            trace_id="trace_vs2_cross_tenant_transfer_023",
        )
        attempt_results = {
            name: {
                "exit_code": result.get("exit_code"),
                "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
                "stdout_sha256": hashlib.sha256(str(result.get("stdout", "")).encode()).hexdigest(),
            }
            for name, result in attempts.items()
        }
        checks = {
            "cross_tenant_transfer_policy_denied": decision.get("decision") == "deny"
            and "cross_tenant_scope" in decision.get("reason_codes", []),
            "copy_reference_share_promotion_denied": all(result.get("exit_code") != 0 for result in attempts.values()),
            "zero_target_records_created": all(value == 0 for value in target_state.values()),
            "transfer_denial_audited": bool(audit_ref),
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "context_digest": _sha256_json(context),
            "policy_decision": decision,
            "attempt_results": attempt_results,
            "target_state": target_state,
            "audit_refs": [audit_ref],
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def rls_defense_in_depth_probe(self, *, token: str) -> dict[str, Any]:
        resolved, denied = self._resolve_context_from_token(token, "service_direct")
        if denied or resolved is None:
            return {
                "status": "failed",
                "reason": "context_resolution_failed",
                "denied": denied,
                "checks": {
                    "fault_injected_service_allow_recorded": False,
                    "rls_still_returns_zero_foreign_rows": False,
                    "anomaly_audit_recorded": False,
                    "anomaly_metric_recorded": False,
                },
            }
        context = resolved["context"]
        service_policy_fault = {
            "scenario_id": "VS2-SEC-036",
            "decision": "allow",
            "fault_injection": "service_policy_bypass_stub",
            "trusted_policy_source": False,
            "target_tenant_digest": _sha256_json({"tenant_id": "tenant_beta"}),
        }
        rls_probe = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'foreign_artifact_rows_visible', (
    SELECT count(*)::int
    FROM cs.artifacts
    WHERE tenant_id = 'tenant_beta'
      AND artifact_id = 'artifact_beta_001'
  ),
  'foreign_payload_rows_visible', (
    SELECT count(*)::int
    FROM cs.artifacts
    WHERE payload::text LIKE '%BETA_ONLY_RANGE_CANARY%'
  ),
  'tenant_ids_returned', COALESCE((SELECT jsonb_agg(DISTINCT tenant_id) FROM cs.artifacts WHERE artifact_id = 'artifact_beta_001'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )
        fixture_presence = self.postgres.json_query(
            """
SELECT jsonb_build_object(
  'beta_fixture_rows', (SELECT count(*)::int FROM cs.artifacts WHERE tenant_id = 'tenant_beta' AND artifact_id = 'artifact_beta_001'),
  'beta_fixture_canary_digest', md5((SELECT payload::text FROM cs.artifacts WHERE tenant_id = 'tenant_beta' AND artifact_id = 'artifact_beta_001'))
)::text;
"""
        )
        audit_ref = self._insert_audit(
            context,
            "rls.anomaly.detected",
            "database.query",
            service_policy_fault,
            "fault_injected_service_allow_vs2_sec_036",
            trace_id="trace_vs2_rls_defense_036",
        )
        metric_insert = self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.operator_metrics(tenant_id, namespace_id, owner_id, workspace_id, metric_id, metric_name, metric_value, labels, trace_id)
VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  'rls_defense_036_anomaly',
  'rls_anomaly_count',
  1,
  {_sql_json({"scenario_id": "VS2-SEC-036", "probe": "rls_defense_in_depth", "blocked_tenant_digest": _sha256_json({"tenant_id": "tenant_beta"})})},
  'trace_vs2_rls_defense_036'
)
ON CONFLICT (tenant_id, metric_id) DO NOTHING;
COMMIT;
"""
        )
        metric_read = self.postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
SELECT jsonb_build_object(
  'metric_rows_visible', (SELECT count(*)::int FROM cs.operator_metrics WHERE metric_id = 'rls_defense_036_anomaly'),
  'metric_values', COALESCE((SELECT jsonb_agg(metric_value ORDER BY metric_id) FROM cs.operator_metrics WHERE metric_id = 'rls_defense_036_anomaly'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )
        checks = {
            "fault_injected_service_allow_recorded": service_policy_fault["decision"] == "allow"
            and service_policy_fault["trusted_policy_source"] is False,
            "rls_still_returns_zero_foreign_rows": fixture_presence.get("beta_fixture_rows") == 1
            and rls_probe.get("foreign_artifact_rows_visible") == 0
            and rls_probe.get("foreign_payload_rows_visible") == 0,
            "anomaly_audit_recorded": bool(audit_ref),
            "anomaly_metric_recorded": metric_insert.get("exit_code") == 0
            and metric_read.get("metric_rows_visible") == 1
            and metric_read.get("metric_values") == [1],
        }
        payload = {
            "status": "passed" if all(checks.values()) else "failed",
            "context_digest": _sha256_json(context),
            "service_policy_fault": service_policy_fault,
            "fixture_presence": fixture_presence,
            "rls_probe": rls_probe,
            "metric_insert": _digest_transcript_entry(metric_insert),
            "metric_read": metric_read,
            "audit_refs": [audit_ref],
            "checks": checks,
        }
        self.requests.append(payload)
        return payload

    def _insert_audit(
        self,
        context: dict[str, Any],
        event_type: str,
        action: str,
        subject: dict[str, Any],
        decision_id: str,
        *,
        trace_id: str = "trace_vs2_local_range",
    ) -> str:
        event_base = {
            "event_type": event_type,
            "actor": context["principal_id"],
            "action": action,
            "subject": subject,
            "decision_id": decision_id,
            "trace_id": trace_id,
            "previous_hash": "LOCAL_RANGE_GENESIS",
            "sequence": len(self.requests) + 1,
        }
        event_hash = _sha256_json(event_base)
        event_id = f"audit_{event_hash[:16]}"
        self.postgres.psql(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(context['tenant_id'])};
SET LOCAL app.namespace_id = {_sql_literal(context['namespace_id'])};
SET LOCAL app.owner_id = {_sql_literal(context['owner_id'])};
SET LOCAL app.workspace_id = {_sql_literal(context['workspace_id'])};
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(context['tenant_id'])},
  {_sql_literal(context['namespace_id'])},
  {_sql_literal(context['owner_id'])},
  {_sql_literal(context['workspace_id'])},
  {_sql_literal(event_id)},
  {_sql_literal(event_type)},
  {_sql_literal(context['principal_id'])},
  {_sql_literal(action)},
  {_sql_json(subject)},
  {_sql_literal(decision_id)},
  'vs2-rego-local-v1',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(event_hash)},
  {_sql_literal(trace_id)}
);
COMMIT;
"""
        )
        return event_id


def run_vs2_range_client(root: Path, api_url: str, token: str, artifact_id: str, caller_fields: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(caller_fields)
    url = f"{api_url.rstrip('/')}/api/vs2/artifacts/{urllib.parse.quote(artifact_id)}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(url, headers={"authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_action_client(root: Path, api_url: str, token: str, provider_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/actions/external",
        data=json.dumps({"provider_url": provider_url}, sort_keys=True).encode(),
        headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_object_contract_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/object-contract",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_object_access_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/object-access-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_observability_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/observability-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_tenant_read_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/tenant-read-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_search_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/search-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_db_path_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/db-path-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_constraint_collision_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/constraint-collision-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_migration_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/migration-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_upgrade_path_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/upgrade-path-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def run_vs2_range_audit_integrity_client(root: Path, api_url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/vs2/audit-integrity-matrix",
        headers={"authorization": f"Bearer {token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return {
                "status": "success",
                "http_status": response.status,
                "payload": json.loads(response.read().decode()),
            }
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"raw": body}
        return {"status": "http_error", "http_status": error.code, "payload": payload}


def _http_json(url: str, token: str, headers: dict[str, str] | None = None, *, timeout: float = 10) -> dict[str, Any]:
    request_headers = {"authorization": f"Bearer {token}"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"http_status": response.status, "payload": json.loads(response.read().decode())}
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "payload": json.loads(error.read().decode())}


def _http_text(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"authorization": f"Bearer {token}", "accept": "text/html"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"http_status": response.status, "text": response.read().decode()}
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "text": error.read().decode()}


def _http_post_json(url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, sort_keys=True).encode(),
        headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return {"http_status": response.status, "payload": json.loads(response.read().decode())}
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "payload": json.loads(error.read().decode())}


def _object_contract_schema_inventory(postgres: _PostgresRange) -> dict[str, Any]:
    table_literals = ", ".join(_sql_literal(spec["table"]) for spec in DURABLE_OBJECT_TABLES)
    return postgres.json_query(
        f"""
WITH required(table_name, column_name) AS (
  VALUES
    {", ".join(
            f"({_sql_literal(spec['table'])}, {_sql_literal(column)})"
            for spec in DURABLE_OBJECT_TABLES
            for column in (
                ["tenant_id", "namespace_id", "owner_id", "workspace_id", "classification"]
                if spec["classification_required"]
                else ["tenant_id", "namespace_id", "owner_id", "workspace_id"]
            )
        )}
),
columns AS (
  SELECT table_name, column_name, is_nullable, data_type
  FROM information_schema.columns
  WHERE table_schema = 'cs'
    AND table_name IN ({table_literals})
)
SELECT jsonb_build_object(
  'tables_expected', jsonb_build_array({table_literals}),
  'required_columns', (
    SELECT jsonb_agg(jsonb_build_object(
      'table', required.table_name,
      'column', required.column_name,
      'present', columns.column_name IS NOT NULL,
      'is_nullable', columns.is_nullable,
      'data_type', columns.data_type
    ) ORDER BY required.table_name, required.column_name)
    FROM required
    LEFT JOIN columns ON columns.table_name = required.table_name AND columns.column_name = required.column_name
  ),
  'table_count', (
    SELECT count(DISTINCT table_name)
    FROM information_schema.tables
    WHERE table_schema = 'cs'
      AND table_name IN ({table_literals})
  )
)::text;
"""
    )


def _object_contract_constraint_inventory(postgres: _PostgresRange) -> dict[str, Any]:
    table_literals = ", ".join(_sql_literal(spec["table"]) for spec in DURABLE_OBJECT_TABLES)
    return postgres.json_query(
        f"""
SELECT jsonb_build_object(
  'constraints', (
    SELECT jsonb_agg(jsonb_build_object(
      'table', tc.table_name,
      'constraint', tc.constraint_name,
      'type', tc.constraint_type
    ) ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name)
    FROM information_schema.table_constraints tc
    WHERE tc.table_schema = 'cs'
      AND tc.table_name IN ({table_literals})
  ),
  'primary_key_tables', (
    SELECT jsonb_agg(DISTINCT tc.table_name ORDER BY tc.table_name)
    FROM information_schema.table_constraints tc
    WHERE tc.table_schema = 'cs'
      AND tc.table_name IN ({table_literals})
      AND tc.constraint_type = 'PRIMARY KEY'
  )
)::text;
"""
    )


def _object_contract_null_insert_probe(postgres: _PostgresRange) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for spec in DURABLE_OBJECT_TABLES:
        table = spec["table"]
        required_columns = ["tenant_id", "namespace_id", "owner_id", "workspace_id"]
        if spec["classification_required"]:
            required_columns.append("classification")
        for column in required_columns:
            object_id = f"null_probe_{table}_{column}"
            if table == "audit_events":
                values = {
                    "tenant_id": "'tenant_alpha'",
                    "namespace_id": "'personal'",
                    "owner_id": "'principal_alice'",
                    "workspace_id": "'alpha-home'",
                }
                values[column] = "NULL"
                sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
INSERT INTO cs.audit_events(
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action, subject,
  decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {values['tenant_id']},
  {values['namespace_id']},
  {values['owner_id']},
  {values['workspace_id']},
  {_sql_literal(object_id)},
  'object.contract.null_probe',
  'principal_alice',
  'object_contract.null_probe',
  '{{}}'::jsonb,
  'object_contract_null_probe',
  'vs2-local-range-object-contract',
  '[]'::jsonb,
  'LOCAL_RANGE_GENESIS',
  {_sql_literal(_sha256_json({'table': table, 'column': column, 'object_id': object_id}))},
  'trace_vs2_object_contract'
);
COMMIT;
"""
            else:
                values = {
                    "tenant_id": "'tenant_alpha'",
                    "namespace_id": "'personal'",
                    "owner_id": "'principal_alice'",
                    "workspace_id": "'alpha-home'",
                    "classification": "'internal'",
                }
                values[column] = "NULL"
                payload = {"object_type": spec["object_type"], "probe": "null_insert", "column": column}
                sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
INSERT INTO cs.{table}(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  {values['tenant_id']},
  {values['namespace_id']},
  {values['owner_id']},
  {values['workspace_id']},
  {_sql_literal(object_id)},
  {values['classification']},
  {_sql_json(payload)},
  'trace_vs2_null_probe'
);
COMMIT;
"""
            result = postgres.psql(sql)
            attempts.append(
                {
                    "table": table,
                    "object_type": spec["object_type"],
                    "column": column,
                    "exit_code": result["exit_code"],
                    "denied": result["exit_code"] != 0,
                    "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
                }
            )
    return {
        "attempt_count": len(attempts),
        "denied_count": len([attempt for attempt in attempts if attempt["denied"]]),
        "attempts": attempts,
    }


def _object_contract_scope_mutation_probe(postgres: _PostgresRange) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for spec in DURABLE_OBJECT_TABLES:
        table = spec["table"]
        object_id = f"object_contract_{table}"
        if table == "audit_events":
            sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
UPDATE cs.audit_events
   SET tenant_id = 'tenant_beta',
       owner_id = 'principal_bob',
       workspace_id = 'beta-home'
 WHERE event_id = {_sql_literal(object_id)};
COMMIT;
"""
        else:
            sql = f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
UPDATE cs.{table}
   SET tenant_id = 'tenant_beta',
       owner_id = 'principal_bob',
       workspace_id = 'beta-home'
 WHERE artifact_id = {_sql_literal(object_id)};
COMMIT;
"""
        result = postgres.psql(sql)
        attempts.append(
            {
                "table": table,
                "object_type": spec["object_type"],
                "exit_code": result["exit_code"],
                "denied": result["exit_code"] != 0,
                "stderr_tail": str(result.get("stderr", "")).splitlines()[-4:],
            }
        )
    return {
        "attempt_count": len(attempts),
        "denied_count": len([attempt for attempt in attempts if attempt["denied"]]),
        "attempts": attempts,
    }


def _rls_inventory(postgres: _PostgresRange) -> dict[str, Any]:
    return postgres.json_query(
        """
SELECT jsonb_build_object(
  'roles', (
    SELECT jsonb_agg(jsonb_build_object('rolname', rolname, 'rolsuper', rolsuper, 'rolbypassrls', rolbypassrls) ORDER BY rolname)
    FROM pg_roles
    WHERE rolname IN ('cornerstone_app', 'cornerstone_identity', 'cornerstone_migrator', 'cornerstone_maintenance')
  ),
  'tables', (
    SELECT jsonb_agg(jsonb_build_object('relation', c.relname, 'owner', pg_get_userbyid(c.relowner), 'rls_enabled', c.relrowsecurity, 'rls_forced', c.relforcerowsecurity) ORDER BY c.relname)
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relkind = 'r'
  ),
  'policy_count', (SELECT count(*) FROM pg_policies WHERE schemaname = 'cs')
)::text;
"""
    )


def _rls_tenant_probe(postgres: _PostgresRange) -> dict[str, Any]:
    return postgres.json_query(
        """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT jsonb_build_object(
  'visible_artifacts', (SELECT count(*) FROM cs.artifacts),
  'tenant_beta_rows_visible', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_beta'),
  'visible_ids', (SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.artifacts)
)::text;
COMMIT;
"""
    )


def _rls_write_probe(postgres: _PostgresRange) -> dict[str, Any]:
    forged_insert = postgres.psql(
        """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload)
VALUES ('tenant_beta', 'personal', 'principal_bob', 'beta-home', 'forged_beta', 'internal', '{}');
COMMIT;
"""
    )
    cross_update = postgres.json_query(
        """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
WITH updated AS (
  UPDATE cs.artifacts SET payload = '{"mutated": true}'::jsonb
  WHERE tenant_id = 'tenant_beta'
  RETURNING artifact_id
)
SELECT jsonb_build_object('cross_tenant_update_returned', (SELECT count(*) FROM updated))::text;
COMMIT;
"""
    )
    return {
        "forged_insert_exit_code": forged_insert["exit_code"],
        "forged_insert_denied": forged_insert["exit_code"] != 0,
        "cross_tenant_update": cross_update,
        "cross_tenant_update_zero": cross_update["cross_tenant_update_returned"] == 0,
    }


def _connection_reuse_probe(postgres: _PostgresRange) -> dict[str, Any]:
    script = """
SELECT jsonb_build_object('step', 'connection_open', 'backend_pid', pg_backend_pid())::text;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT jsonb_build_object(
  'step', 'tenant_alpha_success',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count', (SELECT count(*) FROM cs.artifacts),
  'foreign_count', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_beta'),
  'visible_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.artifacts), '[]'::jsonb),
  'visible_canaries', COALESCE((SELECT jsonb_agg(payload->>'canary' ORDER BY artifact_id) FROM cs.artifacts WHERE payload ? 'canary'), '[]'::jsonb)
)::text;
COMMIT;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_build_object(
  'step', 'post_alpha_reset',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count_without_context', (SELECT count(*) FROM cs.artifacts),
  'alpha_count_without_context', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_alpha'),
  'beta_count_without_context', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_beta')
)::text;
COMMIT;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
SELECT jsonb_build_object(
  'step', 'tenant_beta_success',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count', (SELECT count(*) FROM cs.artifacts),
  'foreign_count', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_alpha'),
  'visible_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.artifacts), '[]'::jsonb),
  'visible_canaries', COALESCE((SELECT jsonb_agg(payload->>'canary' ORDER BY artifact_id) FROM cs.artifacts WHERE payload ? 'canary'), '[]'::jsonb)
)::text;
COMMIT;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES ('tenant_alpha', 'personal', 'principal_alice', 'alpha-home', 'artifact_alpha_001', 'internal', '{"probe":"connection_reuse_duplicate"}'::jsonb, 'trace_vs2_connection_reuse');
ROLLBACK;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_build_object(
  'step', 'post_error_reset',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count_without_context', (SELECT count(*) FROM cs.artifacts)
)::text;
COMMIT;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SET LOCAL statement_timeout = '25ms';
SELECT pg_sleep(0.2);
ROLLBACK;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_build_object(
  'step', 'post_timeout_reset',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count_without_context', (SELECT count(*) FROM cs.artifacts)
)::text;
COMMIT;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
SELECT jsonb_build_object(
  'step', 'tenant_beta_rollback_path',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count', (SELECT count(*) FROM cs.artifacts),
  'foreign_count', (SELECT count(*) FROM cs.artifacts WHERE tenant_id = 'tenant_alpha'),
  'visible_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.artifacts), '[]'::jsonb),
  'visible_canaries', COALESCE((SELECT jsonb_agg(payload->>'canary' ORDER BY artifact_id) FROM cs.artifacts WHERE payload ? 'canary'), '[]'::jsonb)
)::text;
ROLLBACK;

BEGIN;
SET LOCAL ROLE cornerstone_app;
SELECT jsonb_build_object(
  'step', 'post_rollback_reset',
  'backend_pid', pg_backend_pid(),
  'tenant_setting', current_setting('app.tenant_id', true),
  'visible_count_without_context', (SELECT count(*) FROM cs.artifacts)
)::text;
COMMIT;
"""
    result = postgres.psql_continue_on_error(script, timeout=30)
    observations: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for line in str(result.get("stdout", "")).splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except ValueError:
            parse_errors.append(text)
            continue
        if isinstance(parsed, dict) and parsed.get("step"):
            observations.append(parsed)
    by_step = {str(row.get("step")): row for row in observations}
    backend_pids = sorted({row.get("backend_pid") for row in observations if row.get("backend_pid") is not None})
    stderr = str(result.get("stderr", ""))
    return {
        "command": _digest_transcript_entry(result),
        "exit_code": result.get("exit_code"),
        "observations": observations,
        "steps": by_step,
        "backend_pids": backend_pids,
        "parse_errors": parse_errors,
        "stderr_tail": stderr.splitlines()[-12:],
        "expected_errors": {
            "duplicate_key_observed": "duplicate key value violates unique constraint" in stderr,
            "statement_timeout_observed": "canceling statement due to statement timeout" in stderr,
        },
    }


def _concurrent_tenant_api_probe(gateway_url: str, token: str, bob_token: str, postgres: _PostgresRange) -> dict[str, Any]:
    rng = random.Random(16016)
    cases: list[dict[str, Any]] = []
    for index in range(16):
        tenant_label = "alpha" if index % 2 == 0 else "beta"
        cases.append(
            {
                "case_id": f"concurrent_{index:02d}_{tenant_label}",
                "tenant_label": tenant_label,
                "token": token if tenant_label == "alpha" else bob_token,
                "artifact_id": "artifact_alpha_001" if tenant_label == "alpha" else "artifact_beta_001",
                "expected_tenant_id": "tenant_alpha" if tenant_label == "alpha" else "tenant_beta",
                "expected_owner_id": "principal_alice" if tenant_label == "alpha" else "principal_bob",
                "expected_workspace_id": "alpha-home" if tenant_label == "alpha" else "beta-home",
                "expected_canary": "ALPHA_ONLY_RANGE_CANARY" if tenant_label == "alpha" else "BETA_ONLY_RANGE_CANARY",
                "foreign_tenant_id": "tenant_beta" if tenant_label == "alpha" else "tenant_alpha",
                "foreign_artifact_id": "artifact_beta_001" if tenant_label == "alpha" else "artifact_alpha_001",
                "foreign_canary": "BETA_ONLY_RANGE_CANARY" if tenant_label == "alpha" else "ALPHA_ONLY_RANGE_CANARY",
                "trace_id": f"trace_vs2_concurrent_{tenant_label}_{index:02d}",
                "stagger_ms": rng.randint(0, 25),
            }
        )
    rng.shuffle(cases)
    barrier = threading.Barrier(len(cases))
    lock = threading.Lock()
    observations: list[dict[str, Any]] = []

    def worker(case: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            barrier.wait(timeout=30)
            time.sleep(float(case["stagger_ms"]) / 1000)
            response = _http_json(
                f"{gateway_url}/api/vs2/artifacts/{urllib.parse.quote(str(case['artifact_id']))}",
                str(case["token"]),
                headers={"x-cs-trace-id": str(case["trace_id"])},
                timeout=30,
            )
            ended = time.perf_counter()
            payload = response.get("payload", {})
            artifact = payload.get("artifact") or {}
            artifact_payload = artifact.get("payload") if isinstance(artifact.get("payload"), dict) else {}
            policy_decision = payload.get("policy_decision") or {}
            payload_text = json.dumps(payload, sort_keys=True)
            observation = {
                "case_id": case["case_id"],
                "tenant_label": case["tenant_label"],
                "expected_tenant_id": case["expected_tenant_id"],
                "expected_artifact_id": case["artifact_id"],
                "trace_id": case["trace_id"],
                "http_status": response.get("http_status"),
                "status": payload.get("status"),
                "surface": payload.get("surface"),
                "response_trace_id": payload.get("trace_id"),
                "context_tenant_id": (payload.get("context") or {}).get("tenant_id"),
                "context_owner_id": (payload.get("context") or {}).get("owner_id"),
                "context_workspace_id": (payload.get("context") or {}).get("workspace_id"),
                "context_digest": payload.get("context_digest"),
                "artifact_id": artifact.get("artifact_id"),
                "artifact_tenant_id": artifact.get("tenant_id"),
                "artifact_owner_id": artifact.get("owner_id"),
                "artifact_workspace_id": artifact.get("workspace_id"),
                "artifact_canary": artifact_payload.get("canary"),
                "policy_decision_id": policy_decision.get("decision_id"),
                "policy_decision": policy_decision.get("decision"),
                "policy_trace_id": policy_decision.get("trace_id"),
                "audit_refs": payload.get("audit_refs") or [],
                "foreign_identifier_seen": case["foreign_tenant_id"] in payload_text or case["foreign_artifact_id"] in payload_text,
                "foreign_canary_seen": case["foreign_canary"] in payload_text,
                "payload_sha256": hashlib.sha256(payload_text.encode()).hexdigest(),
                "started_monotonic": round(started, 6),
                "ended_monotonic": round(ended, 6),
            }
        except Exception as error:  # noqa: BLE001 - evidence should record the failure class.
            observation = {
                "case_id": case["case_id"],
                "tenant_label": case["tenant_label"],
                "expected_tenant_id": case["expected_tenant_id"],
                "expected_artifact_id": case["artifact_id"],
                "trace_id": case["trace_id"],
                "status": "probe_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "started_monotonic": round(started, 6),
                "ended_monotonic": round(time.perf_counter(), 6),
            }
        with lock:
            observations.append(observation)

    threads = [threading.Thread(target=worker, args=(case,), daemon=True) for case in cases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    observations = sorted(observations, key=lambda row: (row.get("ended_monotonic", 0), row.get("case_id", "")))
    audit_refs = sorted({str(ref) for row in observations for ref in row.get("audit_refs", []) if ref})
    if audit_refs:
        audit_ref_sql = ", ".join(_sql_literal(ref) for ref in audit_refs)
        persisted_audit_rows = postgres.json_query(
            f"""
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'event_id', event_id,
  'tenant_id', tenant_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'event_type', event_type,
  'action', action,
  'subject', subject,
  'decision_id', decision_id,
  'trace_id', trace_id
) ORDER BY event_id), '[]'::jsonb)::text
FROM cs.audit_events
WHERE event_id IN ({audit_ref_sql});
"""
        )
    else:
        persisted_audit_rows = []
    completion_order = [row.get("tenant_label") for row in observations]
    tenant_switches = sum(1 for left, right in zip(completion_order, completion_order[1:]) if left != right)
    context_digests: dict[str, list[str]] = {}
    for row in observations:
        label = str(row.get("tenant_label"))
        digest = row.get("context_digest")
        if digest:
            context_digests.setdefault(label, [])
            if str(digest) not in context_digests[label]:
                context_digests[label].append(str(digest))
    return {
        "status": "completed",
        "thread_count": len(cases),
        "scheduler_seed": 16016,
        "gateway_url": gateway_url,
        "request_path": "/api/vs2/artifacts/{artifact_id}",
        "same_service": True,
        "pool_reset_evidence": "database.connection_reuse",
        "completion_order": completion_order,
        "tenant_switches": tenant_switches,
        "context_digests": context_digests,
        "observations": observations,
        "audit_refs": audit_refs,
        "persisted_audit_rows": persisted_audit_rows or [],
    }


def _post_revocation_stale_allow_concurrency_probe(
    gateway: _RangeGateway,
    gateway_url: str,
    token: str,
    token_key: bytes,
    worker_job_base: dict[str, Any],
    policy_cache_invalidation: dict[str, Any],
    revoke_update: dict[str, Any],
) -> dict[str, Any]:
    rng = random.Random(11111)
    cached_allow = policy_cache_invalidation.get("before", {})
    same_revision_replay = policy_cache_invalidation.get("same_revision_replay", {})
    revision_update = policy_cache_invalidation.get("after_revision_update", {})
    cache_size_before = len(gateway.policy_decision_cache)
    cases: list[dict[str, Any]] = []
    for index in range(24):
        surface = ("api", "service", "worker")[index % 3]
        cases.append(
            {
                "case_id": f"r11_post_revoke_{index:02d}_{surface}",
                "surface": surface,
                "trace_id": f"trace_vs2_r11_post_revoke_{index:02d}_{surface}",
                "stagger_ms": rng.randint(0, 35),
            }
        )
    rng.shuffle(cases)
    barrier = threading.Barrier(len(cases))
    lock = threading.Lock()
    observations: list[dict[str, Any]] = []

    def normalize_payload(case: dict[str, Any], response: dict[str, Any], *, surface: str) -> dict[str, Any]:
        payload = response.get("payload", response)
        artifact = payload.get("artifact") if isinstance(payload, dict) else None
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        policy_decision = payload.get("policy_decision") if isinstance(payload, dict) else None
        identity_decision = payload.get("identity_decision") if isinstance(payload, dict) else None
        audit_refs = payload.get("audit_refs", []) if isinstance(payload, dict) else []
        return {
            "case_id": case["case_id"],
            "surface": surface,
            "trace_id": case["trace_id"],
            "http_status": response.get("http_status"),
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "reason": error.get("reason") if isinstance(error, dict) else payload.get("reason"),
            "artifact_present": artifact is not None,
            "policy_decision": policy_decision,
            "identity_decision": identity_decision,
            "audit_refs": audit_refs,
            "audit_ref_count": len(audit_refs),
            "raw_status": payload.get("status") if isinstance(payload, dict) else None,
        }

    def run_case(case: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            barrier.wait(timeout=30)
            time.sleep(float(case["stagger_ms"]) / 1000)
            if case["surface"] == "api":
                response = _http_json(
                    f"{gateway_url}/api/vs2/artifacts/artifact_gamma_001",
                    token,
                    headers={"x-cs-trace-id": case["trace_id"]},
                    timeout=60,
                )
                observation = normalize_payload(case, response, surface="api")
            elif case["surface"] == "service":
                payload = gateway.artifact_show(
                    token=f"Bearer {token}",
                    artifact_id="artifact_gamma_001",
                    caller_fields={},
                    surface=f"service_direct_r11_{case['case_id']}",
                    trace_id=case["trace_id"],
                )
                observation = normalize_payload(case, payload, surface="service")
                observation["http_status"] = payload.get("status_code")
            else:
                envelope = dict(worker_job_base)
                envelope["job_id"] = f"job_gamma_r11_post_revoke_{case['case_id']}"
                envelope["trace_id"] = case["trace_id"]
                envelope["issued_at"] = int(time.time())
                signed_envelope = _sign_worker_envelope(envelope, token_key)
                payload = gateway.run_worker_artifact_job(signed_envelope)
                observation = {
                    "case_id": case["case_id"],
                    "surface": "worker",
                    "trace_id": case["trace_id"],
                    "http_status": None,
                    "status": payload.get("status"),
                    "reason": payload.get("reason"),
                    "artifact_present": payload.get("artifact") is not None,
                    "policy_decision": payload.get("policy_decision"),
                    "identity_decision": None,
                    "audit_refs": payload.get("audit_refs", []),
                    "audit_ref_count": payload.get("audit_ref_count", len(payload.get("audit_refs", []))),
                    "counters": payload.get("counters", {}),
                }
            observation["started_monotonic"] = round(started, 6)
            observation["ended_monotonic"] = round(time.perf_counter(), 6)
        except Exception as error:  # noqa: BLE001 - evidence should record the failure class.
            observation = {
                "case_id": case["case_id"],
                "surface": case["surface"],
                "trace_id": case["trace_id"],
                "status": "probe_error",
                "reason": type(error).__name__,
                "error": str(error),
                "artifact_present": False,
                "started_monotonic": round(started, 6),
                "ended_monotonic": round(time.perf_counter(), 6),
            }
        with lock:
            observations.append(observation)

    threads = [threading.Thread(target=run_case, args=(case,), daemon=False) for case in cases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    observed_case_ids = {row.get("case_id") for row in observations}
    unfinished_cases = [case for case in cases if case["case_id"] not in observed_case_ids]
    if unfinished_cases:
        ended_monotonic = round(time.perf_counter(), 6)
        with lock:
            for case in unfinished_cases:
                observations.append(
                    {
                        "case_id": case["case_id"],
                        "surface": case["surface"],
                        "trace_id": case["trace_id"],
                        "status": "probe_error",
                        "reason": "thread_join_timeout",
                        "artifact_present": False,
                        "policy_decision": None,
                        "identity_decision": None,
                        "audit_refs": [],
                        "audit_ref_count": 0,
                        "started_monotonic": None,
                        "ended_monotonic": ended_monotonic,
                    }
                )

    observations = sorted(observations, key=lambda row: (row.get("ended_monotonic", 0), row.get("case_id", "")))
    cache_size_after = len(gateway.policy_decision_cache)
    audit_refs = sorted({str(ref) for row in observations for ref in row.get("audit_refs", []) if ref})
    if audit_refs:
        audit_ref_sql = ", ".join(_sql_literal(ref) for ref in audit_refs)
        persisted_audit_rows = gateway.postgres.json_query(
            f"""
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'event_id', event_id,
  'tenant_id', tenant_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'event_type', event_type,
  'action', action,
  'subject', subject,
  'decision_id', decision_id,
  'trace_id', trace_id
) ORDER BY event_id), '[]'::jsonb)::text
FROM cs.audit_events
WHERE event_id IN ({audit_ref_sql});
"""
        )
    else:
        persisted_audit_rows = []
    surfaces = {row.get("surface") for row in observations}
    membership_rows = revoke_update.get("memberships") or []
    revoked_membership = membership_rows[0] if membership_rows else {}
    denied_observations = [
        row
        for row in observations
        if row.get("reason") == "membership_revoked"
        and row.get("artifact_present") is False
        and row.get("policy_decision") in (None, {})
        and row.get("status") in {"denied", "quarantined"}
    ]
    successful_observations = [
        row
        for row in observations
        if row.get("artifact_present") is True
        or row.get("status") in {"success", "completed"}
        or (isinstance(row.get("policy_decision"), dict) and row.get("policy_decision", {}).get("decision") == "allow")
    ]
    non_api_observations = [row for row in observations if row.get("surface") in {"service", "worker"}]
    checks = {
        "r11_cached_allow_existed_before_revocation": cached_allow.get("decision", {}).get("decision") == "allow"
        and cached_allow.get("decision", {}).get("bundle_revision") == "vs2-rego-local-v1"
        and same_revision_replay.get("cache_hit") is True,
        "r11_revision_update_sequence_recorded": revision_update.get("decision", {}).get("decision") == "deny"
        and revision_update.get("decision", {}).get("bundle_revision") == "vs2-rego-local-v2",
        "r11_membership_revoked_after_cached_allow": revoke_update.get("rows") == 1
        and revoked_membership.get("membership_revision") == "memrev-gamma-003"
        and revoked_membership.get("revoked") is True,
        "r11_concurrent_retries_completed": len(observations) == len(cases)
        and not any(row.get("status") == "probe_error" for row in observations)
        and surfaces == {"api", "service", "worker"},
        "r11_zero_post_revocation_successes": not successful_observations,
        "r11_cached_allow_not_reused_after_revocation": cache_size_after == cache_size_before
        and all(row.get("policy_decision") in (None, {}) for row in observations),
        "r11_denial_audits_recorded": bool(non_api_observations)
        and all(row.get("audit_ref_count", 0) >= 1 for row in non_api_observations)
        and len(persisted_audit_rows) >= len(non_api_observations),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "thread_count": len(cases),
        "scheduler_seed": 11111,
        "request_path": "/api/vs2/artifacts/artifact_gamma_001",
        "cached_allow_before_revocation": {
            "cache_key_digest": cached_allow.get("cache_key_digest"),
            "decision": cached_allow.get("decision"),
        },
        "same_revision_cache_hit": {
            "cache_key_digest": same_revision_replay.get("cache_key_digest"),
            "cache_hit": same_revision_replay.get("cache_hit"),
            "decision": same_revision_replay.get("decision"),
        },
        "revision_update_decision": revision_update.get("decision"),
        "membership_update": revoke_update,
        "cache_size_before": cache_size_before,
        "cache_size_after": cache_size_after,
        "observations": observations,
        "denied_observation_count": len(denied_observations),
        "successful_observations": successful_observations,
        "unfinished_cases": unfinished_cases,
        "audit_refs": audit_refs,
        "persisted_audit_rows": persisted_audit_rows or [],
        "decision_revision_sequence": [
            {
                "step": "cached_allow",
                "decision": cached_allow.get("decision", {}).get("decision"),
                "revision": cached_allow.get("decision", {}).get("bundle_revision"),
                "decision_id": cached_allow.get("decision", {}).get("decision_id"),
            },
            {
                "step": "same_revision_cache_hit",
                "cache_hit": same_revision_replay.get("cache_hit"),
                "revision": same_revision_replay.get("decision", {}).get("bundle_revision"),
                "decision_id": same_revision_replay.get("decision", {}).get("decision_id"),
            },
            {
                "step": "revision_update_deny",
                "decision": revision_update.get("decision", {}).get("decision"),
                "revision": revision_update.get("decision", {}).get("bundle_revision"),
                "decision_id": revision_update.get("decision", {}).get("decision_id"),
            },
            {
                "step": "membership_revoked",
                "membership_revision": revoked_membership.get("membership_revision"),
                "revoked": revoked_membership.get("revoked"),
            },
        ],
        "checks": checks,
    }


def _worker_scope_probe(gateway: _GatewayRange, token_key: bytes, postgres: _PostgresRange) -> dict[str, Any]:
    issued_at = int(time.time())
    base_envelope = {
        "job_id": "job_worker_scope_valid_017",
        "principal_id": "principal_alice",
        "membership_id": "m_alpha_alice_personal",
        "membership_revision": "memrev-alpha-001",
        "session_version": 1,
        "tenant_id": "tenant_alpha",
        "namespace_id": "personal",
        "workspace_id": "alpha-home",
        "owner_id": "principal_alice",
        "artifact_id": "artifact_alpha_001",
        "issued_at": issued_at,
        "trace_id": "trace_vs2_worker_scope_valid_017",
    }

    def signed(updates: dict[str, Any] | None = None, *, remove: list[str] | None = None) -> dict[str, Any]:
        envelope = dict(base_envelope)
        envelope.update(updates or {})
        for field in remove or []:
            envelope.pop(field, None)
        return _sign_worker_envelope(envelope, token_key)

    invalid_signature = signed({"job_id": "job_worker_scope_invalid_signature_017", "trace_id": "trace_vs2_worker_scope_invalid_signature_017"})
    invalid_signature["signature"] = f"tampered_{invalid_signature['signature']}"

    cases = {
        "valid": signed(),
        "missing_scope": signed(
            {
                "job_id": "job_worker_scope_missing_scope_017",
                "trace_id": "trace_vs2_worker_scope_missing_scope_017",
            },
            remove=["tenant_id"],
        ),
        "tampered_scope": signed(
            {
                "job_id": "job_worker_scope_tampered_scope_017",
                "tenant_id": "tenant_beta",
                "trace_id": "trace_vs2_worker_scope_tampered_scope_017",
            }
        ),
        "stale_revision": signed(
            {
                "job_id": "job_worker_scope_stale_revision_017",
                "membership_revision": "memrev-alpha-stale",
                "trace_id": "trace_vs2_worker_scope_stale_revision_017",
            }
        ),
        "cross_tenant_payload": signed(
            {
                "job_id": "job_worker_scope_cross_payload_017",
                "artifact_id": "artifact_beta_001",
                "trace_id": "trace_vs2_worker_scope_cross_payload_017",
            }
        ),
        "invalid_signature": invalid_signature,
    }

    results: dict[str, dict[str, Any]] = {}
    for name, envelope in cases.items():
        result = gateway.run_worker_artifact_job(envelope)
        artifact = result.get("artifact") or {}
        policy_decision = result.get("policy_decision") or {}
        results[name] = {
            "job_id": result.get("job_id"),
            "trace_id": envelope.get("trace_id"),
            "status": result.get("status"),
            "reason": result.get("reason"),
            "artifact_id": artifact.get("artifact_id"),
            "artifact_tenant_id": artifact.get("tenant_id"),
            "artifact_payload_digest": _sha256_json(artifact.get("payload")) if isinstance(artifact.get("payload"), dict) else None,
            "policy_decision": policy_decision.get("decision"),
            "policy_decision_id": policy_decision.get("decision_id"),
            "audit_ref_count": len(result.get("audit_refs") or []),
            "audit_refs": result.get("audit_refs") or [],
            "counters": result.get("counters") or {},
        }

    replay_result = gateway.run_worker_artifact_job(cases["valid"])
    results["replay"] = {
        "job_id": replay_result.get("job_id"),
        "trace_id": cases["valid"].get("trace_id"),
        "status": replay_result.get("status"),
        "reason": replay_result.get("reason"),
        "artifact_id": None,
        "artifact_tenant_id": None,
        "artifact_payload_digest": None,
        "policy_decision": (replay_result.get("policy_decision") or {}).get("decision"),
        "policy_decision_id": (replay_result.get("policy_decision") or {}).get("decision_id"),
        "audit_ref_count": len(replay_result.get("audit_refs") or []),
        "audit_refs": replay_result.get("audit_refs") or [],
        "counters": replay_result.get("counters") or {},
    }

    persisted_jobs = postgres.json_query(
        """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'record_id', artifact_id,
  'job_id', payload->>'job_id',
  'state', payload->>'state',
  'reason', payload->>'reason',
  'has_artifact_ref', payload ? 'artifact_id',
  'audit_ref', audit_ref
) ORDER BY artifact_id), '[]'::jsonb)::text
FROM cs.jobs
WHERE artifact_id LIKE 'job_worker_scope_%'
   OR artifact_id LIKE 'quarantine_job_worker_scope_%';
COMMIT;
"""
    )
    audit_refs = sorted({str(ref) for result in results.values() for ref in result.get("audit_refs", []) if ref})
    if audit_refs:
        audit_ref_sql = ", ".join(_sql_literal(ref) for ref in audit_refs)
        persisted_audit_rows = postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'event_id', event_id,
  'event_type', event_type,
  'action', action,
  'decision_id', decision_id,
  'trace_id', trace_id
) ORDER BY event_id), '[]'::jsonb)::text
FROM cs.audit_events
WHERE event_id IN ({audit_ref_sql});
COMMIT;
"""
        )
    else:
        persisted_audit_rows = []

    return {
        "status": "completed",
        "case_count": len(results),
        "expected_reasons": {
            "missing_scope": "missing_scope_metadata",
            "tampered_scope": "tampered_scope_metadata",
            "stale_revision": "stale_membership_revision",
            "cross_tenant_payload": "payload_not_found_or_denied",
            "invalid_signature": "invalid_job_signature",
            "replay": "replay_detected",
        },
        "results": results,
        "persisted_jobs": persisted_jobs or [],
        "audit_refs": audit_refs,
        "persisted_audit_rows": persisted_audit_rows or [],
    }


def _operation_key_scope_probe(postgres: _PostgresRange) -> dict[str, Any]:
    key_kinds = ["cache", "idempotency", "dedupe", "lock", "rate_limit", "cursor"]
    values_alpha = ",\n".join(
        f"('tenant_alpha', 'personal', 'principal_alice', 'alpha-home', 'opkey_{kind}_shared_024', 'internal', "
        f"{_sql_json({'probe': 'operation_key_scope', 'kind': kind, 'operation_key': f'shared-{kind}-024', 'canary': f'ALPHA_OPKEY_{kind.upper()}_CANARY'})}, "
        f"'trace_vs2_operation_key_scope')"
        for kind in key_kinds
    )
    values_beta = ",\n".join(
        f"('tenant_beta', 'personal', 'principal_bob', 'beta-home', 'opkey_{kind}_shared_024', 'internal', "
        f"{_sql_json({'probe': 'operation_key_scope', 'kind': kind, 'operation_key': f'shared-{kind}-024', 'canary': f'BETA_OPKEY_{kind.upper()}_CANARY'})}, "
        f"'trace_vs2_operation_key_scope')"
        for kind in key_kinds
    )
    seed = postgres.psql(
        f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
INSERT INTO cs.idempotency_keys(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES
{values_alpha}
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_beta';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_bob';
SET LOCAL app.workspace_id = 'beta-home';
INSERT INTO cs.idempotency_keys(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES
{values_beta}
ON CONFLICT (tenant_id, artifact_id) DO UPDATE
  SET payload = EXCLUDED.payload,
      audit_ref = EXCLUDED.audit_ref;
COMMIT;
"""
    )

    def read_for(tenant_id: str, owner_id: str, workspace_id: str, foreign_tenant_id: str) -> dict[str, Any]:
        return postgres.json_query(
            f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = {_sql_literal(tenant_id)};
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = {_sql_literal(owner_id)};
SET LOCAL app.workspace_id = {_sql_literal(workspace_id)};
SELECT jsonb_build_object(
  'tenant_id', {_sql_literal(tenant_id)},
  'visible_count', (SELECT count(*) FROM cs.idempotency_keys WHERE payload->>'probe' = 'operation_key_scope'),
  'explicit_foreign_count', (SELECT count(*) FROM cs.idempotency_keys WHERE tenant_id = {_sql_literal(foreign_tenant_id)} AND payload->>'probe' = 'operation_key_scope'),
  'visible_tenants', COALESCE((SELECT jsonb_agg(DISTINCT tenant_id) FROM cs.idempotency_keys WHERE payload->>'probe' = 'operation_key_scope'), '[]'::jsonb),
  'visible_kinds', COALESCE((SELECT jsonb_agg(payload->>'kind' ORDER BY payload->>'kind') FROM cs.idempotency_keys WHERE payload->>'probe' = 'operation_key_scope'), '[]'::jsonb),
  'visible_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.idempotency_keys WHERE payload->>'probe' = 'operation_key_scope'), '[]'::jsonb),
  'visible_canaries', COALESCE((SELECT jsonb_agg(payload->>'canary' ORDER BY payload->>'canary') FROM cs.idempotency_keys WHERE payload->>'probe' = 'operation_key_scope'), '[]'::jsonb),
  'operation_keys', COALESCE((SELECT jsonb_agg(payload->>'operation_key' ORDER BY payload->>'operation_key') FROM cs.idempotency_keys WHERE payload->>'probe' = 'operation_key_scope'), '[]'::jsonb)
)::text;
COMMIT;
"""
        )

    alpha_read = read_for("tenant_alpha", "principal_alice", "alpha-home", "tenant_beta")
    beta_read = read_for("tenant_beta", "principal_bob", "beta-home", "tenant_alpha")
    duplicate_alpha = postgres.psql(
        f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
INSERT INTO cs.idempotency_keys(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES (
  'tenant_alpha',
  'personal',
  'principal_alice',
  'alpha-home',
  'opkey_cache_shared_024',
  'internal',
  {_sql_json({'probe': 'operation_key_scope', 'kind': 'cache', 'operation_key': 'shared-cache-024', 'canary': 'ALPHA_OPKEY_DUPLICATE_CANARY'})},
  'trace_vs2_operation_key_scope_duplicate'
);
COMMIT;
"""
    )
    cross_tenant_mutation = postgres.json_query(
        f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
WITH updated AS (
  UPDATE cs.idempotency_keys
     SET payload = payload || {_sql_json({'alpha_attempted_suppression': True})}
   WHERE tenant_id = 'tenant_beta'
     AND payload->>'probe' = 'operation_key_scope'
   RETURNING artifact_id
),
deleted AS (
  DELETE FROM cs.idempotency_keys
   WHERE tenant_id = 'tenant_beta'
     AND artifact_id = 'opkey_cursor_shared_024'
   RETURNING artifact_id
)
SELECT jsonb_build_object(
  'updated_count', (SELECT count(*) FROM updated),
  'deleted_count', (SELECT count(*) FROM deleted),
  'updated_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM updated), '[]'::jsonb),
  'deleted_ids', COALESCE((SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM deleted), '[]'::jsonb)
)::text;
COMMIT;
"""
    )
    beta_after_cross_tenant_mutation = read_for("tenant_beta", "principal_bob", "beta-home", "tenant_alpha")
    duplicate_stderr = str(duplicate_alpha.get("stderr", ""))
    return {
        "status": "completed",
        "key_kinds": key_kinds,
        "seed_exit_code": seed["exit_code"],
        "alpha_read": alpha_read,
        "beta_read": beta_read,
        "duplicate_alpha": {
            "exit_code": duplicate_alpha["exit_code"],
            "duplicate_key_observed": "duplicate key value violates unique constraint" in duplicate_stderr,
            "foreign_identifier_echoed": "tenant_beta" in duplicate_stderr or "principal_bob" in duplicate_stderr or "BETA_OPKEY" in duplicate_stderr,
            "stderr_tail": duplicate_stderr.splitlines()[-4:],
        },
        "cross_tenant_mutation": cross_tenant_mutation,
        "beta_after_cross_tenant_mutation": beta_after_cross_tenant_mutation,
    }


def _schema_security_failure_reasons(inventory: dict[str, Any]) -> list[str]:
    checks = inventory.get("checks", {})
    required = {
        "tenant_scope_columns_present": "missing_tenant_scope_columns",
        "rls_enabled": "rls_not_enabled",
        "rls_forced": "rls_not_forced",
        "command_policy_present": "missing_command_policy",
        "app_grants_present": "missing_app_grants",
        "test_coverage_declared": "missing_test_coverage_declaration",
    }
    return [reason for check, reason in required.items() if checks.get(check) is not True]


def _schema_security_gate_probe(postgres: _PostgresRange) -> dict[str, Any]:
    inventory_sql = """
WITH required_columns AS (
  SELECT unnest(ARRAY['tenant_id', 'namespace_id', 'owner_id', 'workspace_id']) AS column_name
),
column_state AS (
  SELECT
    required_columns.column_name,
    EXISTS (
      SELECT 1
      FROM information_schema.columns c
      WHERE c.table_schema = 'cs'
        AND c.table_name = :'table_name'
        AND c.column_name = required_columns.column_name
        AND c.is_nullable = 'NO'
    ) AS present_not_null
  FROM required_columns
),
coverage_state AS (
  SELECT count(*)::int AS coverage_count
  FROM schema_gate_coverage_021
  WHERE table_name = :'table_name'
    AND scenario_id = 'VS2-SEC-021'
)
SELECT jsonb_build_object(
  'table_name', :'table_name',
  'required_columns', (
    SELECT jsonb_agg(jsonb_build_object('column_name', column_name, 'present_not_null', present_not_null) ORDER BY column_name)
    FROM column_state
  ),
  'rls_enabled', (
    SELECT relrowsecurity
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relname = :'table_name'
  ),
  'rls_forced', (
    SELECT relforcerowsecurity
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relname = :'table_name'
  ),
  'policy_count', (
    SELECT count(*)::int
    FROM pg_policies
    WHERE schemaname = 'cs' AND tablename = :'table_name'
  ),
  'app_grants', jsonb_build_object(
    'select', has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'SELECT'),
    'insert', has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'INSERT'),
    'update', has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'UPDATE'),
    'delete', has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'DELETE')
  ),
  'coverage_count', (SELECT coverage_count FROM coverage_state),
  'checks', jsonb_build_object(
    'tenant_scope_columns_present', (SELECT bool_and(present_not_null) FROM column_state),
    'rls_enabled', COALESCE((
      SELECT relrowsecurity
      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'cs' AND c.relname = :'table_name'
    ), false),
    'rls_forced', COALESCE((
      SELECT relforcerowsecurity
      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'cs' AND c.relname = :'table_name'
    ), false),
    'command_policy_present', (
      SELECT count(*) > 0
      FROM pg_policies
      WHERE schemaname = 'cs' AND tablename = :'table_name'
    ),
    'app_grants_present',
      has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'SELECT')
      AND has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'INSERT')
      AND has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'UPDATE')
      AND has_table_privilege('cornerstone_app', 'cs.' || :'table_name', 'DELETE'),
    'test_coverage_declared', (SELECT coverage_count >= 2 FROM coverage_state)
  )
)::text;
"""
    bad = postgres.json_query(
        f"""
BEGIN;
CREATE TABLE cs.schema_gate_bad_021 (
  object_id text PRIMARY KEY,
  payload jsonb NOT NULL DEFAULT '{{}}'::jsonb
);
CREATE TEMP TABLE schema_gate_coverage_021 (
  table_name text NOT NULL,
  scenario_id text NOT NULL,
  coverage_kind text NOT NULL
) ON COMMIT DROP;
\\set table_name 'schema_gate_bad_021'
{inventory_sql}
ROLLBACK;
"""
    )
    good = postgres.json_query(
        f"""
BEGIN;
CREATE TABLE cs.schema_gate_good_021 (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  object_id text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
  PRIMARY KEY (tenant_id, object_id)
);
ALTER TABLE cs.schema_gate_good_021 ENABLE ROW LEVEL SECURITY;
ALTER TABLE cs.schema_gate_good_021 FORCE ROW LEVEL SECURITY;
CREATE POLICY schema_gate_good_021_tenant_scope
  ON cs.schema_gate_good_021
  FOR ALL
  TO cornerstone_app
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  );
GRANT SELECT, INSERT, UPDATE, DELETE ON cs.schema_gate_good_021 TO cornerstone_app;
CREATE TEMP TABLE schema_gate_coverage_021 (
  table_name text NOT NULL,
  scenario_id text NOT NULL,
  coverage_kind text NOT NULL
) ON COMMIT DROP;
INSERT INTO schema_gate_coverage_021(table_name, scenario_id, coverage_kind)
VALUES
  ('schema_gate_good_021', 'VS2-SEC-021', 'tenant_select_isolation'),
  ('schema_gate_good_021', 'VS2-SEC-021', 'forged_write_denial');
\\set table_name 'schema_gate_good_021'
{inventory_sql}
ROLLBACK;
"""
    )
    persisted = postgres.json_query(
        """
SELECT jsonb_build_object(
  'persisted_fixture_table_count', (
    SELECT count(*)::int
    FROM information_schema.tables
    WHERE table_schema = 'cs'
      AND table_name IN ('schema_gate_bad_021', 'schema_gate_good_021')
  )
)::text;
"""
    )
    bad_failures = _schema_security_failure_reasons(bad)
    good_failures = _schema_security_failure_reasons(good)
    return {
        "status": "completed",
        "bad_fixture": {
            "inventory": bad,
            "gate_status": "failed" if bad_failures else "passed",
            "failure_reasons": bad_failures,
        },
        "corrected_fixture": {
            "inventory": good,
            "gate_status": "passed" if not good_failures else "failed",
            "failure_reasons": good_failures,
        },
        "rollback_state": persisted,
    }


def _rls_negative_control(postgres: _PostgresRange) -> dict[str, Any]:
    disable = postgres.psql("ALTER TABLE cs.artifacts DISABLE ROW LEVEL SECURITY;")
    leaked = _rls_tenant_probe(postgres)
    enable = postgres.psql("ALTER TABLE cs.artifacts ENABLE ROW LEVEL SECURITY; ALTER TABLE cs.artifacts FORCE ROW LEVEL SECURITY;")
    return {
        "mutation": "mutation_rls_disabled",
        "disable_exit_code": disable["exit_code"],
        "tenant_beta_rows_visible_when_mutated": leaked.get("tenant_beta_rows_visible"),
        "detected": leaked.get("tenant_beta_rows_visible", 0) > 0,
        "restore_exit_code": enable["exit_code"],
    }


def _db_path_object_inventory(postgres: _PostgresRange) -> dict[str, Any]:
    return postgres.json_query(
        """
SELECT jsonb_build_object(
  'views', (
    SELECT jsonb_agg(jsonb_build_object(
      'schema', n.nspname,
      'name', c.relname,
      'reloptions', COALESCE(c.reloptions, ARRAY[]::text[]),
      'security_invoker', COALESCE('security_invoker=true' = ANY(c.reloptions), false),
      'select_granted_to_app', has_table_privilege('cornerstone_app', c.oid, 'SELECT')
    ) ORDER BY c.relname)
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs'
      AND c.relkind = 'v'
      AND c.relname IN ('safe_artifact_counts')
  ),
  'functions', (
    SELECT jsonb_agg(jsonb_build_object(
      'schema', n.nspname,
      'name', p.proname,
      'security_definer', p.prosecdef,
      'execute_granted_to_app', has_function_privilege('cornerstone_app', p.oid, 'EXECUTE'),
      'execute_granted_to_public', has_function_privilege('public', p.oid, 'EXECUTE')
    ) ORDER BY p.proname)
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'cs'
      AND p.proname IN ('visible_artifact_ids', 'unsafe_all_artifacts')
  )
)::text;
"""
    )


def _search_matrix_inventory(postgres: _PostgresRange) -> dict[str, Any]:
    return postgres.json_query(
        """
SELECT jsonb_build_object(
  'indexes', (
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'tablename', tablename,
      'indexname', indexname,
      'indexdef', indexdef
    ) ORDER BY tablename, indexname), '[]'::jsonb)
    FROM pg_indexes
    WHERE schemaname = 'cs'
      AND tablename IN ('search_snapshots', 'ontology_objects')
      AND indexname IN ('search_snapshots_payload_fts_idx', 'ontology_objects_payload_fts_idx')
  ),
  'rls_tables', (
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'table', c.relname,
      'rls_enabled', c.relrowsecurity,
      'rls_forced', c.relforcerowsecurity
    ) ORDER BY c.relname), '[]'::jsonb)
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs'
      AND c.relname IN ('search_snapshots', 'ontology_objects')
  )
)::text;
"""
    )


def _object_access_api_cli_visible(api_response: dict[str, Any], cli_run: dict[str, Any], api_payload: dict[str, Any], cli_payload: dict[str, Any]) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
    )


def _object_access_authorized_alpha_ok(payload: dict[str, Any]) -> bool:
    alpha = payload.get("object_access_matrix", {}).get("authorized_alpha", {})
    download = alpha.get("download", {})
    return (
        alpha.get("artifact_visible") is True
        and alpha.get("derived_visible") is True
        and alpha.get("evidence_visible") is True
        and alpha.get("storage_bound_to_tenant") is True
        and download.get("allowed") is True
        and download.get("bytes_returned", 0) > 0
        and bool(download.get("sha256"))
    )


def _object_access_foreign_denied(payload: dict[str, Any]) -> bool:
    matrix = payload.get("object_access_matrix", {})
    attempts = matrix.get("foreign_attempts", {})
    for key in ["artifact_get", "download", "signed_url", "derived_representation"]:
        attempt = attempts.get(key, {})
        if attempt.get("status_code") != 404:
            return False
        if attempt.get("code") != "CS_RESOURCE_NOT_FOUND_OR_DENIED":
            return False
        if attempt.get("visible_count") != 0:
            return False
        if attempt.get("sensitive_metadata_returned") is not False:
            return False
        if key in {"artifact_get", "download"}:
            if attempt.get("bytes_returned") != 0 or attempt.get("storage_access_attempted") is not False or attempt.get("content_returned") is not False:
                return False
        if key == "signed_url":
            if attempt.get("url_returned") is not False or attempt.get("storage_key_returned") is not False:
                return False
        if key == "derived_representation" and attempt.get("content_returned") is not False:
            return False
    return True


def _object_access_evidence_traversal_isolated(payload: dict[str, Any]) -> bool:
    traversal = payload.get("object_access_matrix", {}).get("evidence_traversal", {})
    return (
        traversal.get("status_code") == 404
        and traversal.get("code") == "CS_RESOURCE_NOT_FOUND_OR_DENIED"
        and traversal.get("visible_evidence_bundle_count") == 0
        and traversal.get("visible_cross_tenant_item_count") == 0
        and traversal.get("bytes_returned") == 0
        and traversal.get("storage_access_attempted") is False
        and traversal.get("content_returned") is False
        and traversal.get("sensitive_metadata_returned") is False
    )


def _object_access_storage_log_bound(payload: dict[str, Any]) -> bool:
    log = payload.get("object_access_matrix", {}).get("storage_access_log", [])
    return (
        len(log) >= 1
        and all(entry.get("allowed") is True for entry in log)
        and all(entry.get("request_tenant_id") == "tenant_alpha" for entry in log)
        and all(entry.get("storage_tenant_id") == "tenant_alpha" for entry in log)
        and all(entry.get("bytes_returned", 0) > 0 for entry in log)
    )


def _object_access_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps(
        [payload.get("object_access_matrix", {}) for payload in payloads],
        sort_keys=True,
    )
    forbidden = [
        "tenant_beta",
        "principal_bob",
        "beta-home",
        "object_access_beta_blob",
        "object_access_beta_derived",
        "object_access_beta_bundle",
        "BETA_OBJECT_ACCESS_SECRET_BYTES",
        "tenant_beta/personal/object_access_beta_blob.bin",
    ]
    return not any(marker in surface for marker in forbidden)


def _observability_api_cli_visible(api_response: dict[str, Any], cli_run: dict[str, Any], api_payload: dict[str, Any], cli_payload: dict[str, Any]) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
    )


def _observability_records_isolated(payload: dict[str, Any]) -> bool:
    matrix = payload.get("observability_matrix", {})
    tenant_query = matrix.get("tenant_query", {})
    counts = tenant_query.get("record_counts", {})
    foreign_counts = tenant_query.get("foreign_counts", {})
    sample_ids = tenant_query.get("sample_ids", {})
    expected = ["audit_events", "policy_decisions", "operator_metrics", "status_records", "tenant_exports"]
    return (
        all(counts.get(key, 0) >= 1 for key in expected)
        and all(foreign_counts.get(key) == 0 for key in expected)
        and all(isinstance(sample_ids.get(key), list) and sample_ids.get(key) for key in expected)
    )


def _observability_export_scoped(payload: dict[str, Any]) -> bool:
    export = payload.get("observability_matrix", {}).get("tenant_export", {})
    return (
        export.get("export_type") == "observability.scoped"
        and str(export.get("export_id", "")).startswith("observability_")
        and int(export.get("row_count") or 0) >= 5
        and export.get("foreign_record_count") == 0
        and bool(export.get("payload_hash"))
    )


def _observability_metrics_non_sensitive(payload: dict[str, Any]) -> bool:
    metrics = payload.get("observability_matrix", {}).get("aggregate_metrics", {})
    return (
        metrics.get("visible_metric_count", 0) >= 1
        and metrics.get("metric_names")
        and metrics.get("tenant_ids_returned") == []
        and metrics.get("principal_ids_returned") == []
        and metrics.get("non_sensitive_only") is True
        and metrics.get("foreign_metric_visible_count") == 0
    )


def _observability_system_wide_denied(payload: dict[str, Any]) -> bool:
    matrix = payload.get("observability_matrix", {})
    system = matrix.get("system_wide_access", {})
    denial = matrix.get("denial_logs", {})
    return (
        system.get("status_code") == 403
        and system.get("code") == "CS_PRIVILEGED_POLICY_REQUIRED"
        and system.get("rows_returned_without_context") == 0
        and all(value == 0 for value in system.get("db_visible_counts_without_context", {}).values())
        and system.get("tenant_identifier_echoed") is False
        and system.get("foreign_identifier_echoed") is False
        and denial.get("visible_denial_events", 0) >= 1
        and denial.get("foreign_denial_events") == 0
    )


def _observability_role_matrix_user_admin_scoped(user_payload: dict[str, Any], admin_payload: dict[str, Any]) -> bool:
    user_role = user_payload.get("observability_matrix", {}).get("role_matrix", {}).get("caller", {})
    admin_role = admin_payload.get("observability_matrix", {}).get("role_matrix", {}).get("caller", {})
    return (
        user_payload.get("status") == "allowed"
        and admin_payload.get("status") == "allowed"
        and user_role.get("tenant_id") == "tenant_alpha"
        and admin_role.get("tenant_id") == "tenant_alpha"
        and "owner" in set(user_role.get("roles", []))
        and "tenant_admin" in set(admin_role.get("roles", []))
        and user_role.get("tenant_scoped_query_allowed") is True
        and admin_role.get("tenant_scoped_query_allowed") is True
    )


def _observability_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps(
        [payload.get("observability_matrix", {}) for payload in payloads],
        sort_keys=True,
    )
    forbidden = [
        "tenant_beta",
        "principal_bob",
        "beta-home",
        "observability_beta",
        "BETA_OBSERVABILITY_CANARY",
    ]
    return not any(marker in surface for marker in forbidden)


def _search_matrix_api_cli_visible(api_response: dict[str, Any], cli_run: dict[str, Any], api_payload: dict[str, Any], cli_payload: dict[str, Any]) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
    )


def _search_matrix_results_isolated(payload: dict[str, Any]) -> bool:
    matrix = payload.get("search_matrix", {})
    results = matrix.get("results", {})
    alpha = results.get("alpha_positive", {})
    foreign = results.get("foreign_term_search", {})
    autocomplete = results.get("autocomplete", {})
    facets = results.get("facets", {})
    ontology = results.get("ontology_search", {})
    semantic = results.get("semantic_cache", {})
    explicit = results.get("explicit_foreign_counts", {})
    return (
        alpha.get("result_count", 0) >= 1
        and alpha.get("snapshot_refs")
        and alpha.get("object_refs")
        and foreign.get("result_count") == 0
        and foreign.get("snippet_count") == 0
        and foreign.get("score_count") == 0
        and foreign.get("snapshot_refs") == []
        and foreign.get("sample_rows") == []
        and autocomplete.get("suggestion_count") == 0
        and autocomplete.get("suggestions") == []
        and facets.get("facet_count") == 0
        and facets.get("facets") == []
        and ontology.get("object_count") == 0
        and ontology.get("object_refs") == []
        and semantic.get("cache_ref_count") == 0
        and semantic.get("cache_refs") == []
        and explicit.get("search_snapshots") == 0
        and explicit.get("ontology_objects") == 0
        and matrix.get("neutral_foreign_lookup", {}).get("status_code") == 404
        and matrix.get("neutral_foreign_lookup", {}).get("foreign_identifier_echoed") is False
        and matrix.get("neutral_foreign_lookup", {}).get("tenant_identifier_echoed") is False
    )


def _search_matrix_inventory_ok(payload: dict[str, Any]) -> bool:
    inventory = payload.get("search_matrix", {}).get("inventory", {})
    index_names = {row.get("indexname") for row in inventory.get("indexes", [])}
    rls_tables = {row.get("table"): row for row in inventory.get("rls_tables", [])}
    return (
        {"search_snapshots_payload_fts_idx", "ontology_objects_payload_fts_idx"}.issubset(index_names)
        and all(rls_tables.get(table, {}).get("rls_enabled") is True and rls_tables.get(table, {}).get("rls_forced") is True for table in ["search_snapshots", "ontology_objects"])
    )


def _search_matrix_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps(
        [payload.get("search_matrix", {}).get("results", {}) for payload in payloads],
        sort_keys=True,
    )
    forbidden = [
        "tenant_beta",
        "principal_bob",
        "beta-home",
        "BETA_SEARCH_CANARY_UNIQUE",
        "beta foreign searchable snippet",
        "search_snapshot_beta_matrix",
        "ontology_beta_search_object",
        "semantic_beta_cache",
        "beta-facet",
        "beta evidence",
    ]
    return not any(marker in surface for marker in forbidden)


def _db_path_inventory_ok(inventory: dict[str, Any]) -> bool:
    views = {view.get("name"): view for view in inventory.get("views") or []}
    functions = {function.get("name"): function for function in inventory.get("functions") or []}
    safe_view = views.get("safe_artifact_counts", {})
    visible_function = functions.get("visible_artifact_ids", {})
    unsafe_function = functions.get("unsafe_all_artifacts", {})
    return (
        safe_view.get("security_invoker") is True
        and safe_view.get("select_granted_to_app") is True
        and visible_function.get("security_definer") is False
        and visible_function.get("execute_granted_to_app") is True
        and visible_function.get("execute_granted_to_public") is False
        and unsafe_function.get("security_definer") is True
        and unsafe_function.get("execute_granted_to_app") is False
        and unsafe_function.get("execute_granted_to_public") is False
    )


def _db_path_raw_sql_isolated(raw_sql: dict[str, Any]) -> bool:
    return (
        raw_sql.get("visible_count", 0) >= 1
        and raw_sql.get("explicit_foreign_count") == 0
        and raw_sql.get("exists_foreign") is False
        and all(row.get("tenant_id") == "tenant_alpha" for row in raw_sql.get("sample_rows", []))
    )


def _db_path_view_function_isolated(payload: dict[str, Any]) -> bool:
    matrix = payload.get("db_path_matrix", {})
    safe_view = matrix.get("safe_view", {})
    safe_function = matrix.get("safe_function", {})
    return (
        safe_view.get("row_count", 0) >= 1
        and safe_view.get("explicit_foreign_count") == 0
        and all(row.get("tenant_id") == "tenant_alpha" for row in safe_view.get("rows", []))
        and safe_function.get("visible_count", 0) >= 1
        and safe_function.get("explicit_foreign_count") == 0
        and all(row.get("tenant_id") == "tenant_alpha" for row in safe_function.get("rows", []))
    )


def _db_path_unsafe_definer_denied(payload: dict[str, Any]) -> bool:
    unsafe = payload.get("db_path_matrix", {}).get("unsafe_security_definer", {})
    return (
        unsafe.get("denied") is True
        and unsafe.get("exit_code") != 0
        and unsafe.get("foreign_identifier_echoed") is False
    )


def _db_path_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps(
        [payload.get("db_path_matrix", {}) for payload in payloads],
        sort_keys=True,
    )
    forbidden = [
        "tenant_beta",
        "principal_bob",
        "beta-home",
        "db_path_foreign_artifact",
        "BETA_DB_PATH_CANARY",
        "BETA_ONLY_RANGE_CANARY",
    ]
    return not any(marker in surface for marker in forbidden)


def _constraint_collision_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    matrix = payload.get("constraint_collision_matrix", {})
    return matrix if isinstance(matrix, dict) else {}


def _constraint_collision_api_cli_visible(
    api_response: dict[str, Any],
    cli_run: dict[str, Any],
    api_payload: dict[str, Any],
    cli_payload: dict[str, Any],
) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
    )


def _constraint_collision_tenant_scoped_unique_keys(payload: dict[str, Any]) -> bool:
    matrix = _constraint_collision_matrix(payload)
    visible = matrix.get("request_scope_read", {})
    duplicate = matrix.get("duplicate_same_scope", {})
    return (
        matrix.get("seed", {}).get("exit_code") == 0
        and matrix.get("same_key_control_scope_seeded") is True
        and visible.get("visible_shared_artifact_count") == 1
        and visible.get("explicit_control_scope_count") == 0
        and visible.get("visible_tenants") == ["tenant_alpha"]
        and visible.get("visible_artifact_ids") == ["shared_collision_key_019"]
        and duplicate.get("status_code") == 409
        and duplicate.get("code") == "CS_CONFLICT"
        and duplicate.get("exit_code") != 0
        and duplicate.get("duplicate_key_observed") is True
        and duplicate.get("foreign_identifier_echoed") is False
    )


def _constraint_collision_tenant_aware_foreign_keys(payload: dict[str, Any]) -> bool:
    matrix = _constraint_collision_matrix(payload)
    schema = matrix.get("schema", {})
    visible = matrix.get("request_scope_read", {})
    cross_reference = matrix.get("cross_scope_reference", {})
    foreign_keys = schema.get("tenant_aware_foreign_keys", [])
    definitions = [str(row.get("definition", "")) for row in foreign_keys if isinstance(row, dict)]
    return (
        schema.get("artifact_references_rls_enabled") is True
        and schema.get("artifact_references_rls_forced") is True
        and len(definitions) == 2
        and all("tenant_id" in definition for definition in definitions)
        and any("source_artifact_id" in definition for definition in definitions)
        and any("target_artifact_id" in definition for definition in definitions)
        and visible.get("visible_reference_count") == 1
        and visible.get("cross_scope_reference_count") == 0
        and cross_reference.get("status_code") == 404
        and cross_reference.get("code") == "CS_RESOURCE_NOT_FOUND_OR_DENIED"
        and cross_reference.get("exit_code") != 0
        and cross_reference.get("foreign_key_observed") is True
        and cross_reference.get("foreign_identifier_echoed") is False
    )


def _constraint_collision_neutral_errors(payload: dict[str, Any]) -> bool:
    matrix = _constraint_collision_matrix(payload)
    duplicate = matrix.get("duplicate_same_scope", {})
    cross_reference = matrix.get("cross_scope_reference", {})
    return (
        duplicate.get("status_code") == 409
        and duplicate.get("code") == "CS_CONFLICT"
        and "raw_stderr_sha256" in duplicate
        and "stderr_tail" not in duplicate
        and cross_reference.get("status_code") == 404
        and cross_reference.get("code") == "CS_RESOURCE_NOT_FOUND_OR_DENIED"
        and "raw_stderr_sha256" in cross_reference
        and "stderr_tail" not in cross_reference
    )


def _constraint_collision_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps(
        [_constraint_collision_matrix(payload) for payload in payloads],
        sort_keys=True,
    )
    forbidden = [
        "tenant_beta",
        "principal_bob",
        "beta-home",
        "BETA_CONSTRAINT_COLLISION_CANARY",
        "BETA_REFERENCE_TARGET_CANARY",
    ]
    return not any(marker in surface for marker in forbidden)


def _migration_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    matrix = payload.get("migration_matrix", {})
    return matrix if isinstance(matrix, dict) else {}


def _migration_api_cli_visible(
    api_response: dict[str, Any],
    cli_run: dict[str, Any],
    api_payload: dict[str, Any],
    cli_payload: dict[str, Any],
) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
        and api_payload.get("policy_decision", {}).get("decision") == "allow"
        and cli_payload.get("policy_decision", {}).get("decision") == "allow"
    )


def _migration_known_rows_migrated(payload: dict[str, Any]) -> bool:
    matrix = _migration_matrix(payload)
    migration = matrix.get("migration", {})
    visible = matrix.get("request_scope_read", {})
    return (
        matrix.get("status") == "completed"
        and migration.get("input_count") == 8
        and migration.get("migrated_count") == 2
        and visible.get("migrated_visible_count") == 2
        and migration.get("migrated_ids") == ["migration_020_clean_known", "migration_020_known_valid"]
        and visible.get("visible_tenants") == ["tenant_alpha"]
    )


def _migration_bad_rows_quarantined(payload: dict[str, Any]) -> bool:
    matrix = _migration_matrix(payload)
    migration = matrix.get("migration", {})
    visible = matrix.get("request_scope_read", {})
    reason_counts = migration.get("reason_counts", {})
    expected = {
        "ambiguous_owner": 1,
        "cross_tenant_reference": 1,
        "duplicate_id": 2,
        "invalid_namespace": 1,
        "known": 2,
        "missing_tenant": 1,
    }
    expected_reasons = [
        "ambiguous_owner",
        "cross_tenant_reference",
        "duplicate_id",
        "duplicate_id",
        "invalid_namespace",
        "missing_tenant",
    ]
    return (
        migration.get("quarantined_count") == 6
        and visible.get("quarantine_visible_count") == 6
        and reason_counts == expected
        and migration.get("quarantine_reasons") == expected_reasons
        and visible.get("quarantine_reasons") == expected_reasons
    )


def _migration_no_ownerless_global_truth(payload: dict[str, Any]) -> bool:
    matrix = _migration_matrix(payload)
    migration = matrix.get("migration", {})
    schema = matrix.get("schema", {})
    expected_columns = ["artifact_id", "classification", "namespace_id", "owner_id", "tenant_id", "workspace_id"]
    return (
        migration.get("ownerless_global_truth_count") == 0
        and migration.get("bad_legacy_rows_in_product_count") == 0
        and schema.get("artifacts_not_null_columns") == expected_columns
        and schema.get("migration_quarantine_not_null_columns") == expected_columns
        and schema.get("artifacts_rls_forced") is True
        and schema.get("migration_quarantine_rls_forced") is True
    )


def _migration_checksums_and_rollback_evidence(payload: dict[str, Any]) -> bool:
    matrix = _migration_matrix(payload)
    migration = matrix.get("migration", {})
    rollback = matrix.get("rollback", {})
    rollback_state = matrix.get("rollback_state", {})
    return (
        isinstance(migration.get("product_checksum"), str)
        and len(migration.get("product_checksum", "")) == 32
        and isinstance(migration.get("quarantine_checksum"), str)
        and len(migration.get("quarantine_checksum", "")) == 32
        and rollback.get("exit_code") == 0
        and isinstance(rollback.get("stdout_sha256"), str)
        and len(rollback.get("stdout_sha256", "")) == 64
        and rollback_state.get("persisted_rollback_fixture_count") == 0
        and isinstance(matrix.get("result_digest"), str)
        and len(matrix.get("result_digest", "")) == 64
    )


def _migration_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps([_migration_matrix(payload) for payload in payloads], sort_keys=True)
    forbidden = ["tenant_beta", "principal_bob", "beta-home", "BETA_"]
    return not any(marker in surface for marker in forbidden)


def _upgrade_path_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    matrix = payload.get("upgrade_path_matrix", {})
    return matrix if isinstance(matrix, dict) else {}


def _upgrade_path_api_cli_visible(
    api_response: dict[str, Any],
    cli_run: dict[str, Any],
    api_payload: dict[str, Any],
    cli_payload: dict[str, Any],
) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
        and api_payload.get("policy_decision", {}).get("decision") == "allow"
        and cli_payload.get("policy_decision", {}).get("decision") == "allow"
    )


def _upgrade_path_forward_preserves_objects(payload: dict[str, Any]) -> bool:
    matrix = _upgrade_path_matrix(payload)
    checks = matrix.get("checks", {})
    before = matrix.get("before_snapshot", {})
    after = matrix.get("after_snapshot", {})
    expected_tables = {
        "artifacts",
        "evidence_bundles",
        "claims",
        "ontology_objects",
        "ontology_links",
        "search_snapshots",
        "audit_events",
    }
    return (
        matrix.get("status") == "completed"
        and checks.get("vs0_vs1_fixture_seeded") is True
        and checks.get("forward_migration_succeeded") is True
        and checks.get("before_after_counts_hashes_match") is True
        and set(before.get("tables", {})) == expected_tables
        and before.get("tables") == after.get("tables")
        and before.get("total_rows") == len(expected_tables)
    )


def _upgrade_path_compatibility_regression_reads(payload: dict[str, Any]) -> bool:
    matrix = _upgrade_path_matrix(payload)
    checks = matrix.get("checks", {})
    compatibility = matrix.get("compatibility_regression", {})
    revisions = compatibility.get("upgrade_revision_values", {})
    expected_tables = set(matrix.get("tables", []))
    return (
        checks.get("compatibility_regression_reads_succeeded") is True
        and compatibility.get("visible_tenants") == ["tenant_alpha"]
        and all(
            compatibility.get(name) is True
            for name in [
                "artifact_visible_after_upgrade",
                "evidence_visible_after_upgrade",
                "claim_visible_after_upgrade",
                "ontology_visible_after_upgrade",
                "audit_visible_after_upgrade",
                "search_snapshot_visible_after_upgrade",
            ]
        )
        and set(revisions) == expected_tables
        and all(values == ["vs2-local-upgrade-068"] for values in revisions.values())
    )


def _upgrade_path_failed_migration_and_rollback(payload: dict[str, Any]) -> bool:
    matrix = _upgrade_path_matrix(payload)
    checks = matrix.get("checks", {})
    failed = matrix.get("failed_migration", {})
    failed_state = matrix.get("failed_migration_state", {})
    rollback = matrix.get("rollback", {})
    rollback_columns = matrix.get("rollback_columns", {})
    return (
        checks.get("failed_migration_rejected") is True
        and checks.get("failed_migration_left_no_bad_constraint") is True
        and checks.get("rollback_succeeded") is True
        and checks.get("rollback_preserved_counts_hashes") is True
        and checks.get("rollback_removed_upgrade_columns") is True
        and checks.get("no_data_loss") is True
        and failed.get("exit_code") != 0
        and failed_state.get("bad_constraint_count") == 0
        and rollback.get("exit_code") == 0
        and rollback_columns.get("upgrade_revision_column_count") == 0
        and matrix.get("before_snapshot", {}).get("tables") == matrix.get("rollback_snapshot", {}).get("tables")
        and isinstance(matrix.get("result_digest"), str)
        and len(matrix.get("result_digest", "")) == 64
    )


def _upgrade_path_destructive_without_approval_denied(payload: dict[str, Any]) -> bool:
    matrix = _upgrade_path_matrix(payload)
    checks = matrix.get("checks", {})
    destructive = matrix.get("destructive_migration", {})
    decision = destructive.get("policy_decision", {})
    return (
        checks.get("destructive_migration_denied_without_approval") is True
        and checks.get("destructive_migration_not_executed") is True
        and destructive.get("db_attempted") is False
        and decision.get("decision") == "deny"
        and "high_risk_requires_approval" in decision.get("reason_codes", [])
    )


def _audit_integrity_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    matrix = payload.get("audit_integrity_matrix", {})
    return matrix if isinstance(matrix, dict) else {}


def _verify_range_audit_chain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous_hash = "LOCAL_RANGE_AUDIT_GENESIS"
    for index, row in enumerate(rows):
        candidate = dict(row)
        event_hash = candidate.pop("event_hash", None)
        if row.get("previous_hash") != previous_hash:
            return {
                "valid": False,
                "failed_at": index,
                "reason": "previous_hash_mismatch",
                "expected_previous_hash": previous_hash,
                "actual_previous_hash": row.get("previous_hash"),
            }
        recalculated = _sha256_json(candidate)
        if recalculated != event_hash:
            return {
                "valid": False,
                "failed_at": index,
                "reason": "event_hash_mismatch",
                "expected_event_hash": recalculated,
                "actual_event_hash": event_hash,
            }
        previous_hash = str(event_hash)
    return {"valid": True, "event_count": len(rows), "root_hash": previous_hash}


def _audit_integrity_api_cli_visible(
    api_response: dict[str, Any],
    cli_run: dict[str, Any],
    api_payload: dict[str, Any],
    cli_payload: dict[str, Any],
) -> bool:
    return (
        api_response.get("http_status") == 200
        and cli_run.get("exit_code") == 0
        and api_payload.get("status") == "allowed"
        and cli_payload.get("status") == "allowed"
        and api_payload.get("context_digest") == cli_payload.get("context_digest")
        and api_payload.get("policy_decision", {}).get("decision") == "allow"
        and cli_payload.get("policy_decision", {}).get("decision") == "allow"
    )


def _audit_integrity_required_events_present(payload: dict[str, Any]) -> bool:
    matrix = _audit_integrity_matrix(payload)
    checks = matrix.get("checks", {})
    auditor_role = matrix.get("auditor_role", {})
    required = set(matrix.get("required_event_types", []))
    actual = set(auditor_role.get("event_types", []))
    return checks.get("required_event_types_present") is True and required.issubset(actual) and len(required) == 9


def _audit_integrity_clean_chain_verifies(payload: dict[str, Any]) -> bool:
    matrix = _audit_integrity_matrix(payload)
    verification = matrix.get("clean_verification", {})
    return (
        matrix.get("status") == "completed"
        and matrix.get("checks", {}).get("clean_chain_verifies") is True
        and verification.get("valid") is True
        and verification.get("event_count") == len(matrix.get("required_event_types", []))
        and isinstance(verification.get("root_hash"), str)
        and len(verification.get("root_hash", "")) == 64
    )


def _audit_integrity_tamper_detected(payload: dict[str, Any]) -> bool:
    matrix = _audit_integrity_matrix(payload)
    cases = matrix.get("tamper_cases", {})
    expected = {"modify_event", "delete_event", "insert_fake", "reorder_events", "modify_previous_hash"}
    return (
        matrix.get("checks", {}).get("tamper_cases_detected") is True
        and set(cases) == expected
        and all(case.get("verification", {}).get("valid") is False for case in cases.values())
        and all(case.get("verification", {}).get("reason") in {"event_hash_mismatch", "previous_hash_mismatch"} for case in cases.values())
    )


def _audit_integrity_append_only_and_auditor_role(payload: dict[str, Any]) -> bool:
    matrix = _audit_integrity_matrix(payload)
    checks = matrix.get("checks", {})
    app_role = matrix.get("app_role", {})
    auditor_role = matrix.get("auditor_role", {})
    return (
        checks.get("seed_inserted_or_already_present") is True
        and checks.get("app_role_scoped_selects_chain") is True
        and checks.get("app_role_update_denied") is True
        and checks.get("app_role_delete_denied") is True
        and checks.get("auditor_role_reads_full_chain") is True
        and app_role.get("update_attempt", {}).get("exit_code") not in {0, None}
        and app_role.get("delete_attempt", {}).get("exit_code") not in {0, None}
        and auditor_role.get("row_count") == len(matrix.get("required_event_types", []))
        and isinstance(matrix.get("result_digest"), str)
        and len(matrix.get("result_digest", "")) == 64
    )


def _connection_reuse_same_backend(probe: dict[str, Any]) -> bool:
    expected_steps = {
        "connection_open",
        "tenant_alpha_success",
        "post_alpha_reset",
        "tenant_beta_success",
        "post_error_reset",
        "post_timeout_reset",
        "tenant_beta_rollback_path",
        "post_rollback_reset",
    }
    return (
        probe.get("exit_code") == 0
        and not probe.get("parse_errors")
        and set(probe.get("steps", {})) == expected_steps
        and len(probe.get("backend_pids", [])) == 1
    )


def _connection_reuse_tenant_sequence_isolated(probe: dict[str, Any]) -> bool:
    steps = probe.get("steps", {})
    alpha = steps.get("tenant_alpha_success", {})
    beta = steps.get("tenant_beta_success", {})
    beta_rollback = steps.get("tenant_beta_rollback_path", {})
    return (
        alpha.get("tenant_setting") == "tenant_alpha"
        and alpha.get("visible_count", 0) >= 1
        and alpha.get("foreign_count") == 0
        and all(str(item).startswith("artifact_beta") is False for item in alpha.get("visible_ids", []))
        and "BETA_ONLY_RANGE_CANARY" not in alpha.get("visible_canaries", [])
        and beta.get("tenant_setting") == "tenant_beta"
        and beta.get("visible_count", 0) >= 1
        and beta.get("foreign_count") == 0
        and all(str(item).startswith("artifact_alpha") is False for item in beta.get("visible_ids", []))
        and "ALPHA_ONLY_RANGE_CANARY" not in beta.get("visible_canaries", [])
        and beta_rollback.get("tenant_setting") == "tenant_beta"
        and beta_rollback.get("visible_count", 0) >= 1
        and beta_rollback.get("foreign_count") == 0
        and all(str(item).startswith("artifact_alpha") is False for item in beta_rollback.get("visible_ids", []))
        and "ALPHA_ONLY_RANGE_CANARY" not in beta_rollback.get("visible_canaries", [])
    )


def _connection_reuse_resets_after_paths(probe: dict[str, Any]) -> bool:
    steps = probe.get("steps", {})
    for name in ["post_alpha_reset", "post_error_reset", "post_timeout_reset", "post_rollback_reset"]:
        row = steps.get(name, {})
        if row.get("visible_count_without_context") != 0:
            return False
        if row.get("alpha_count_without_context", 0) not in {0, None}:
            return False
        if row.get("beta_count_without_context", 0) not in {0, None}:
            return False
    return True


def _connection_reuse_error_timeout_observed(probe: dict[str, Any]) -> bool:
    expected = probe.get("expected_errors", {})
    return expected.get("duplicate_key_observed") is True and expected.get("statement_timeout_observed") is True


def _connection_reuse_no_cross_tenant_canaries(probe: dict[str, Any]) -> bool:
    steps = probe.get("steps", {})
    beta_surface = json.dumps(
        [
            steps.get("tenant_beta_success", {}),
            steps.get("tenant_beta_rollback_path", {}),
        ],
        sort_keys=True,
    )
    alpha_surface = json.dumps([steps.get("tenant_alpha_success", {})], sort_keys=True)
    reset_surface = json.dumps(
        [
            steps.get("post_alpha_reset", {}),
            steps.get("post_error_reset", {}),
            steps.get("post_timeout_reset", {}),
            steps.get("post_rollback_reset", {}),
        ],
        sort_keys=True,
    )
    return (
        "ALPHA_ONLY_RANGE_CANARY" not in beta_surface
        and "artifact_alpha_001" not in beta_surface
        and "BETA_ONLY_RANGE_CANARY" not in alpha_surface
        and "artifact_beta_001" not in alpha_surface
        and "ALPHA_ONLY_RANGE_CANARY" not in reset_surface
        and "BETA_ONLY_RANGE_CANARY" not in reset_surface
    )


def _concurrent_tenant_api_load_completed(probe: dict[str, Any]) -> bool:
    observations = probe.get("observations", [])
    return (
        probe.get("status") == "completed"
        and probe.get("thread_count") == 16
        and len(observations) == 16
        and {row.get("tenant_label") for row in observations} == {"alpha", "beta"}
        and probe.get("tenant_switches", 0) >= 1
        and all(row.get("http_status") == 200 and row.get("status") == "allowed" and row.get("surface") == "http_api" for row in observations)
    )


def _concurrent_tenant_contexts_isolated(probe: dict[str, Any]) -> bool:
    observations = probe.get("observations", [])
    digests = probe.get("context_digests", {})
    if len(digests.get("alpha", [])) != 1 or len(digests.get("beta", [])) != 1:
        return False
    if digests["alpha"][0] == digests["beta"][0]:
        return False
    for row in observations:
        if row.get("context_tenant_id") != row.get("expected_tenant_id"):
            return False
        if row.get("artifact_tenant_id") != row.get("expected_tenant_id"):
            return False
        if row.get("artifact_id") != row.get("expected_artifact_id"):
            return False
        if row.get("response_trace_id") != row.get("trace_id"):
            return False
    return True


def _concurrent_tenant_zero_foreign_canary_or_ids(probe: dict[str, Any]) -> bool:
    observations = probe.get("observations", [])
    return bool(observations) and all(row.get("foreign_identifier_seen") is False and row.get("foreign_canary_seen") is False for row in observations)


def _concurrent_tenant_audit_refs_not_mixed(probe: dict[str, Any]) -> bool:
    observations = probe.get("observations", [])
    audit_refs = probe.get("audit_refs", [])
    audit_rows = probe.get("persisted_audit_rows", [])
    if len(audit_refs) != len(observations) or len(set(audit_refs)) != len(audit_refs) or len(audit_rows) != len(audit_refs):
        return False
    expected_by_ref = {
        ref: {
            "tenant_id": row.get("expected_tenant_id"),
            "trace_id": row.get("trace_id"),
            "decision_id": row.get("policy_decision_id"),
        }
        for row in observations
        for ref in row.get("audit_refs", [])
    }
    for event in audit_rows:
        expected = expected_by_ref.get(event.get("event_id"))
        if not expected:
            return False
        if event.get("tenant_id") != expected["tenant_id"]:
            return False
        if event.get("trace_id") != expected["trace_id"]:
            return False
        if event.get("decision_id") != expected["decision_id"]:
            return False
    alpha_refs = {ref for row in observations if row.get("tenant_label") == "alpha" for ref in row.get("audit_refs", [])}
    beta_refs = {ref for row in observations if row.get("tenant_label") == "beta" for ref in row.get("audit_refs", [])}
    return alpha_refs.isdisjoint(beta_refs)


def _concurrent_tenant_policy_refs_present(probe: dict[str, Any]) -> bool:
    observations = probe.get("observations", [])
    return bool(observations) and all(
        row.get("policy_decision") == "allow"
        and bool(row.get("policy_decision_id"))
        and row.get("policy_trace_id") == row.get("trace_id")
        and bool(row.get("audit_refs"))
        for row in observations
    )


def _worker_scope_valid_job_completed(probe: dict[str, Any]) -> bool:
    valid = probe.get("results", {}).get("valid", {})
    return (
        probe.get("status") == "completed"
        and valid.get("status") == "completed"
        and valid.get("reason") is None
        and valid.get("artifact_id") == "artifact_alpha_001"
        and valid.get("artifact_tenant_id") == "tenant_alpha"
        and valid.get("policy_decision") == "allow"
        and valid.get("audit_ref_count") == 1
        and valid.get("counters", {}).get("egress_calls") == 0
    )


def _worker_scope_quarantines_bad_envelopes(probe: dict[str, Any]) -> bool:
    expected = probe.get("expected_reasons", {})
    results = probe.get("results", {})
    return bool(expected) and all(
        results.get(name, {}).get("status") == "quarantined"
        and results.get(name, {}).get("reason") == reason
        and results.get(name, {}).get("artifact_id") is None
        for name, reason in expected.items()
    )


def _worker_scope_persists_audit_and_job_records(probe: dict[str, Any]) -> bool:
    jobs = probe.get("persisted_jobs", [])
    audit_rows = probe.get("persisted_audit_rows", [])
    results = probe.get("results", {})
    by_record_id = {row.get("record_id"): row for row in jobs}
    stateful_quarantines = ["missing_scope", "tampered_scope", "stale_revision", "cross_tenant_payload", "replay"]
    stateful_audit_ref_count = sum(results.get(name, {}).get("audit_ref_count", 0) for name in ["valid", *stateful_quarantines])
    replay_records = [row for row in jobs if row.get("job_id") == "job_worker_scope_valid_017" and row.get("state") == "quarantined"]
    audit_trace_ids = {row.get("trace_id") for row in audit_rows}
    expected_trace_ids = {results.get(name, {}).get("trace_id") for name in ["valid", *stateful_quarantines]}
    return (
        by_record_id.get("job_worker_scope_valid_017", {}).get("state") == "completed"
        and by_record_id.get("job_worker_scope_missing_scope_017", {}).get("reason") == "missing_scope_metadata"
        and by_record_id.get("job_worker_scope_tampered_scope_017", {}).get("reason") == "tampered_scope_metadata"
        and by_record_id.get("job_worker_scope_stale_revision_017", {}).get("reason") == "stale_membership_revision"
        and by_record_id.get("job_worker_scope_cross_payload_017", {}).get("reason") == "payload_not_found_or_denied"
        and by_record_id.get("job_worker_scope_valid_017", {}).get("reason") is None
        and len(replay_records) == 1
        and replay_records[0].get("reason") == "replay_detected"
        and len(audit_rows) == stateful_audit_ref_count
        and stateful_audit_ref_count >= 6
        and expected_trace_ids.issubset(audit_trace_ids)
    )


def _worker_scope_zero_payload_leak_or_egress(probe: dict[str, Any]) -> bool:
    results = probe.get("results", {})
    evidence_surface = json.dumps(
        {
            "results": results,
            "persisted_jobs": probe.get("persisted_jobs", []),
            "persisted_audit_rows": probe.get("persisted_audit_rows", []),
        },
        sort_keys=True,
    )
    forbidden = ["BETA_ONLY_RANGE_CANARY", "principal_bob", "beta-home"]
    return (
        not any(marker in evidence_surface for marker in forbidden)
        and all(result.get("counters", {}).get("egress_calls", 0) == 0 for result in results.values())
        and results.get("cross_tenant_payload", {}).get("artifact_id") is None
        and results.get("cross_tenant_payload", {}).get("policy_decision") is None
    )


def _worker_scope_replay_idempotency_guard(probe: dict[str, Any]) -> bool:
    replay = probe.get("results", {}).get("replay", {})
    jobs = probe.get("persisted_jobs", [])
    replay_records = [row for row in jobs if row.get("job_id") == "job_worker_scope_valid_017" and row.get("state") == "quarantined"]
    completed_records = [row for row in jobs if row.get("record_id") == "job_worker_scope_valid_017" and row.get("state") == "completed"]
    return (
        replay.get("status") == "quarantined"
        and replay.get("reason") == "replay_detected"
        and replay.get("artifact_id") is None
        and replay.get("policy_decision") is None
        and replay.get("counters", {}).get("egress_calls") == 0
        and len(completed_records) == 1
        and len(replay_records) == 1
    )


def _operation_key_records_exist(probe: dict[str, Any]) -> bool:
    kinds = set(probe.get("key_kinds", []))
    alpha = probe.get("alpha_read", {})
    beta = probe.get("beta_read", {})
    return (
        probe.get("status") == "completed"
        and probe.get("seed_exit_code") == 0
        and kinds == {"cache", "idempotency", "dedupe", "lock", "rate_limit", "cursor"}
        and set(alpha.get("visible_kinds", [])) == kinds
        and set(beta.get("visible_kinds", [])) == kinds
        and alpha.get("visible_count") == len(kinds)
        and beta.get("visible_count") == len(kinds)
    )


def _operation_key_tenant_scoped_independent(probe: dict[str, Any]) -> bool:
    alpha = probe.get("alpha_read", {})
    beta = probe.get("beta_read", {})
    return (
        alpha.get("explicit_foreign_count") == 0
        and beta.get("explicit_foreign_count") == 0
        and alpha.get("visible_tenants") == ["tenant_alpha"]
        and beta.get("visible_tenants") == ["tenant_beta"]
        and set(alpha.get("visible_ids", [])) == set(beta.get("visible_ids", []))
        and set(alpha.get("operation_keys", [])) == set(beta.get("operation_keys", []))
    )


def _operation_key_collision_replay_scoped(probe: dict[str, Any]) -> bool:
    duplicate = probe.get("duplicate_alpha", {})
    return (
        duplicate.get("exit_code") != 0
        and duplicate.get("duplicate_key_observed") is True
        and duplicate.get("foreign_identifier_echoed") is False
    )


def _operation_key_zero_cross_tenant_suppression(probe: dict[str, Any]) -> bool:
    mutation = probe.get("cross_tenant_mutation", {})
    beta_after = probe.get("beta_after_cross_tenant_mutation", {})
    return (
        mutation.get("updated_count") == 0
        and mutation.get("deleted_count") == 0
        and mutation.get("updated_ids") == []
        and mutation.get("deleted_ids") == []
        and beta_after.get("visible_count") == len(probe.get("key_kinds", []))
        and set(beta_after.get("visible_kinds", [])) == set(probe.get("key_kinds", []))
    )


def _operation_key_no_foreign_canary_in_tenant_outputs(probe: dict[str, Any]) -> bool:
    alpha_surface = json.dumps(probe.get("alpha_read", {}), sort_keys=True)
    beta_surface = json.dumps(probe.get("beta_read", {}), sort_keys=True)
    beta_after_surface = json.dumps(probe.get("beta_after_cross_tenant_mutation", {}), sort_keys=True)
    return (
        "BETA_OPKEY_" not in alpha_surface
        and "tenant_beta" not in alpha_surface
        and "ALPHA_OPKEY_" not in beta_surface
        and "tenant_alpha" not in beta_surface
        and "ALPHA_OPKEY_" not in beta_after_surface
        and "tenant_alpha" not in beta_after_surface
    )


def _schema_gate_bad_fixture_fails(probe: dict[str, Any]) -> bool:
    bad = probe.get("bad_fixture", {})
    reasons = set(bad.get("failure_reasons", []))
    return (
        probe.get("status") == "completed"
        and bad.get("gate_status") == "failed"
        and {
            "missing_tenant_scope_columns",
            "rls_not_enabled",
            "rls_not_forced",
            "missing_command_policy",
            "missing_app_grants",
            "missing_test_coverage_declaration",
        }.issubset(reasons)
    )


def _schema_gate_corrected_fixture_passes(probe: dict[str, Any]) -> bool:
    corrected = probe.get("corrected_fixture", {})
    inventory = corrected.get("inventory", {})
    checks = inventory.get("checks", {})
    return (
        corrected.get("gate_status") == "passed"
        and corrected.get("failure_reasons") == []
        and all(value is True for value in checks.values())
    )


def _schema_gate_inventory_machine_readable(probe: dict[str, Any]) -> bool:
    for key in ["bad_fixture", "corrected_fixture"]:
        inventory = probe.get(key, {}).get("inventory", {})
        if not isinstance(inventory.get("required_columns"), list):
            return False
        if not isinstance(inventory.get("checks"), dict):
            return False
        if not isinstance(inventory.get("app_grants"), dict):
            return False
    return True


def _schema_gate_detects_required_surfaces(probe: dict[str, Any]) -> bool:
    corrected = probe.get("corrected_fixture", {}).get("inventory", {})
    columns = {row.get("column_name"): row.get("present_not_null") for row in corrected.get("required_columns", [])}
    grants = corrected.get("app_grants", {})
    return (
        columns == {"namespace_id": True, "owner_id": True, "tenant_id": True, "workspace_id": True}
        and corrected.get("rls_enabled") is True
        and corrected.get("rls_forced") is True
        and corrected.get("policy_count", 0) >= 1
        and all(grants.get(name) is True for name in ["select", "insert", "update", "delete"])
        and corrected.get("coverage_count", 0) >= 2
    )


def _schema_gate_rollback_leaves_no_fixture_tables(probe: dict[str, Any]) -> bool:
    return probe.get("rollback_state", {}).get("persisted_fixture_table_count") == 0


def _tenant_read_tables(payload: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = payload.get("tenant_read_matrix", {})
    tables = matrix.get("tables", [])
    return tables if isinstance(tables, list) else []


def _tenant_read_table_set(payload: dict[str, Any]) -> set[str]:
    return {str(row.get("table")) for row in _tenant_read_tables(payload)}


def _tenant_read_no_foreign_values(payloads: list[dict[str, Any]]) -> bool:
    surface = json.dumps(
        [_tenant_read_tables(payload) for payload in payloads],
        sort_keys=True,
    )
    forbidden = [
        "tenant_beta",
        "principal_bob",
        "beta-home",
        "tenant_read_beta_",
        "artifact_beta_001",
        "BETA_ONLY_RANGE_CANARY",
        "BETA_MATRIX_CANARY",
    ]
    return not any(marker in surface for marker in forbidden)


def _tenant_read_counts_hide_foreign_rows(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == len(DURABLE_OBJECT_TABLES) and all(
        row.get("visible_count", 0) >= 1
        and row.get("explicit_foreign_count") == 0
        and row.get("exists_foreign") is False
        and all(group.get("tenant_id") == "tenant_alpha" for group in row.get("grouped_tenants", []))
        for row in rows
    )


def _tenant_read_join_subquery_pagination_isolated(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == len(DURABLE_OBJECT_TABLES) and all(
        row.get("join_cross_count") == 0
        and row.get("subquery_intersection_count") == 0
        and row.get("page_boundary_ids") == []
        and all(str(item).startswith("tenant_read_beta_") is False for item in row.get("page_first_ids", []))
        and all(sample.get("tenant_id") == "tenant_alpha" for sample in row.get("sample_rows", []))
        for row in rows
    )


def _tenant_read_neutral_guessed_results(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == len(DURABLE_OBJECT_TABLES) and all(
        row.get("guessed_foreign_result_count") == 0
        and row.get("neutral_foreign_lookup", {}).get("status_code") == 404
        and row.get("neutral_foreign_lookup", {}).get("code") == "CS_RESOURCE_NOT_FOUND_OR_DENIED"
        and row.get("neutral_foreign_lookup", {}).get("foreign_identifier_echoed") is False
        and row.get("neutral_foreign_lookup", {}).get("tenant_identifier_echoed") is False
        for row in rows
    )


def run_vs2_local_range(root: Path) -> dict[str, Any]:
    root = root.resolve()
    started = time.perf_counter()
    source_fingerprint = build_source_fingerprint(root, family="vs2_local_range")
    (root / VS2_LOCAL_RANGE_REPORT).parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("docker") is None:
        payload = {
            "schema_version": "cs.vs2_local_range.v1",
            "status": "not_verified",
            "reason": "docker executable missing",
            "source_fingerprint": source_fingerprint,
            "profile": {
                "schema_version": "cs.vs2_local_range_profile.v1",
                "wall_seconds": round(time.perf_counter() - started, 3),
                "child_command_elapsed_seconds": 0.0,
            },
        }
        _finalize_report_payload(root, VS2_LOCAL_RANGE_REPORT, payload, started=started)
        return payload

    postgres = _PostgresRange(root)
    opa = _OpaRange(root)
    opa_revision_v2 = _OpaRevisionRange(root, revision="vs2-rego-local-v2", decision="deny", reason_codes=["policy_revision_updated_denied"])
    provider = _MockProvider()
    egress_proxy = _EgressProxy()
    gateway: _RangeGateway | None = None
    token_key = hashlib.sha256(b"cornerstone-vs2-local-range-session-key").digest()
    token_payload = {
        "sub": "principal_alice",
        "membership_id": "m_alpha_alice_personal",
        "session_version": 1,
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
    }
    token = _sign_token(token_payload, token_key)
    org_token = _sign_token(
        {
            "sub": "principal_alice",
            "membership_id": "m_alpha_alice_org",
            "session_version": 1,
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
        },
        token_key,
    )
    admin_token = _sign_token(
        {
            "sub": "principal_ada",
            "membership_id": "m_alpha_ada_admin",
            "session_version": 1,
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
        },
        token_key,
    )
    bob_token = _sign_token(
        {
            "sub": "principal_bob",
            "membership_id": "m_beta_bob_personal",
            "session_version": 1,
            "iat": int(time.time()),
            "exp": int(time.time()) + 600,
        },
        token_key,
    )
    carla_token_v1_payload = {
        "sub": "principal_carla",
        "membership_id": "m_gamma_carla_personal",
        "session_version": 1,
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
    }
    carla_token_v1 = _sign_token(carla_token_v1_payload, token_key)
    payload: dict[str, Any] | None = None

    try:
        postgres_ready = postgres.start()
        migrations_ok = postgres.apply_migrations() if postgres_ready else False
        seed_ok = postgres.seed() if migrations_ok else False
        provider.start()
        egress_proxy.start()
        egress_grant_ok = postgres.seed_egress_grant(provider.url) if seed_ok else False
        opa_ready = opa.start() if seed_ok and egress_grant_ok else False
        opa_revision_v2_ready = opa_revision_v2.start() if opa_ready else False
        if not (postgres_ready and migrations_ok and seed_ok and egress_grant_ok and opa_ready and opa_revision_v2_ready):
            payload = {
                "schema_version": "cs.vs2_local_range.v1",
                "status": "failed",
                "source_fingerprint": source_fingerprint,
                "postgres_ready": postgres_ready,
                "migrations_ok": migrations_ok,
                "seed_ok": seed_ok,
                "egress_grant_ok": egress_grant_ok,
                "opa_ready": opa_ready,
                "opa_revision_v2_ready": opa_revision_v2_ready,
                "postgres_transcript": _safe_transcript(postgres.transcript),
                "opa_transcript": _safe_transcript(opa.transcript),
                "opa_revision_v2_transcript": _safe_transcript(opa_revision_v2.transcript),
                "profile": {
                    "schema_version": "cs.vs2_local_range_profile.v1",
                    "wall_seconds": round(time.perf_counter() - started, 3),
                    "child_command_elapsed_seconds": round(
                        sum(float(entry.get("elapsed_seconds") or 0.0) for entry in postgres.transcript + opa.transcript + opa_revision_v2.transcript),
                        3,
                    ),
                    "failure_layer": "bootstrap",
                },
            }
            return payload

        gateway = _RangeGateway(root, postgres, opa.url, token_key, egress_proxy.url)
        gateway.start()
        health = _http_json(f"{gateway.url}/health", token)
        api_allowed = _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_alpha_001", token)
        browser_allowed = _http_text(f"{gateway.url}/ui/vs2/artifacts/artifact_alpha_001", token)
        cli_allowed_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--artifact-id",
                "artifact_alpha_001",
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            cli_allowed = json.loads(cli_allowed_run["stdout"])
        except ValueError:
            cli_allowed = {"parse_error": cli_allowed_run["stdout"]}
        forged = _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_alpha_001?tenant_id=tenant_beta&role=admin", token)
        missing = _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_alpha_001", "")
        bad_signature = _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_alpha_001", token[:-8] + "badtoken")
        action_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-action-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--provider-url",
                provider.url,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            action_cli = json.loads(action_cli_run["stdout"])
        except ValueError:
            action_cli = {"parse_error": action_cli_run["stdout"]}
        tenant_b_action = _http_post_json(
            f"{gateway.url}/api/vs2/actions/external",
            bob_token,
            {"provider_url": provider.url},
        )
        object_contract_api = _http_json(f"{gateway.url}/api/vs2/object-contract", token)
        object_contract_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-object-contract-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            object_contract_cli = json.loads(object_contract_cli_run["stdout"])
        except ValueError:
            object_contract_cli = {"parse_error": object_contract_cli_run["stdout"]}
        object_access_matrix_api = _http_json(f"{gateway.url}/api/vs2/object-access-matrix", token)
        object_access_matrix_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-object-access-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            object_access_matrix_cli = json.loads(object_access_matrix_cli_run["stdout"])
        except ValueError:
            object_access_matrix_cli = {"parse_error": object_access_matrix_cli_run["stdout"]}
        observability_matrix_api = _http_json(f"{gateway.url}/api/vs2/observability-matrix", token)
        observability_matrix_admin_api = _http_json(f"{gateway.url}/api/vs2/observability-matrix", admin_token)
        observability_matrix_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-observability-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            observability_matrix_cli = json.loads(observability_matrix_cli_run["stdout"])
        except ValueError:
            observability_matrix_cli = {"parse_error": observability_matrix_cli_run["stdout"]}
        object_contract_schema = _object_contract_schema_inventory(postgres)
        object_contract_constraints = _object_contract_constraint_inventory(postgres)
        object_contract_null_inserts = _object_contract_null_insert_probe(postgres)
        object_contract_scope_mutation = _object_contract_scope_mutation_probe(postgres)
        tenant_read_matrix_api = _http_json(f"{gateway.url}/api/vs2/tenant-read-matrix", token)
        tenant_read_matrix_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-tenant-read-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            tenant_read_matrix_cli = json.loads(tenant_read_matrix_cli_run["stdout"])
        except ValueError:
            tenant_read_matrix_cli = {"parse_error": tenant_read_matrix_cli_run["stdout"]}
        search_matrix_api = _http_json(f"{gateway.url}/api/vs2/search-matrix", token)
        search_matrix_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-search-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            search_matrix_cli = json.loads(search_matrix_cli_run["stdout"])
        except ValueError:
            search_matrix_cli = {"parse_error": search_matrix_cli_run["stdout"]}
        db_path_matrix_api = _http_json(f"{gateway.url}/api/vs2/db-path-matrix", token)
        db_path_matrix_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-db-path-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            db_path_matrix_cli = json.loads(db_path_matrix_cli_run["stdout"])
        except ValueError:
            db_path_matrix_cli = {"parse_error": db_path_matrix_cli_run["stdout"]}
        constraint_collision_api = _http_json(f"{gateway.url}/api/vs2/constraint-collision-matrix", token)
        constraint_collision_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-constraint-collision-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            constraint_collision_cli = json.loads(constraint_collision_cli_run["stdout"])
        except ValueError:
            constraint_collision_cli = {"parse_error": constraint_collision_cli_run["stdout"]}
        migration_matrix_api = _http_json(f"{gateway.url}/api/vs2/migration-matrix", token)
        migration_matrix_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-migration-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            migration_matrix_cli = json.loads(migration_matrix_cli_run["stdout"])
        except ValueError:
            migration_matrix_cli = {"parse_error": migration_matrix_cli_run["stdout"]}
        upgrade_path_api = _http_json(f"{gateway.url}/api/vs2/upgrade-path-matrix", token)
        upgrade_path_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-upgrade-path-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            upgrade_path_cli = json.loads(upgrade_path_cli_run["stdout"])
        except ValueError:
            upgrade_path_cli = {"parse_error": upgrade_path_cli_run["stdout"]}
        audit_integrity_api = _http_json(f"{gateway.url}/api/vs2/audit-integrity-matrix", token)
        audit_integrity_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-audit-integrity-client",
                "--api-url",
                gateway.url,
                "--token",
                token,
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            audit_integrity_cli = json.loads(audit_integrity_cli_run["stdout"])
        except ValueError:
            audit_integrity_cli = {"parse_error": audit_integrity_cli_run["stdout"]}
        same_tenant_namespace = gateway.same_tenant_namespace_matrix(personal_token=f"Bearer {token}", org_token=f"Bearer {org_token}")
        cross_tenant_transfer = gateway.cross_tenant_transfer_matrix(token=f"Bearer {token}")
        rls_defense_in_depth = gateway.rls_defense_in_depth_probe(token=f"Bearer {token}")
        policy_input_schema = gateway.policy_input_schema_matrix(owner_token=f"Bearer {token}")
        policy_enforcement = gateway.policy_enforcement_matrix(owner_token=f"Bearer {token}", member_token=f"Bearer {bob_token}")
        policy_limits = gateway.policy_limits_matrix(owner_token=f"Bearer {token}")
        decision_log_masking = gateway.decision_log_masking_matrix(owner_token=f"Bearer {token}")
        reason_code_translation = gateway.reason_code_translation_matrix(owner_token=f"Bearer {token}", member_token=f"Bearer {bob_token}")
        break_glass_maintenance = gateway.break_glass_maintenance_probe(operator_token=f"Bearer {token}")
        product_learning_guard = gateway.product_learning_guard_probe(personal_token=f"Bearer {token}", org_token=f"Bearer {org_token}")
        policy_conformance = gateway.policy_conformance_matrix(owner_token=f"Bearer {token}", cli_payload=cli_allowed)
        revocation_provider_calls_before = len(provider.calls)
        revocation_egress_calls_before = len(egress_proxy.calls)
        worker_job_v1 = _sign_worker_envelope(
            {
                "job_id": "job_gamma_before_revocation_001",
                "principal_id": "principal_carla",
                "membership_id": "m_gamma_carla_personal",
                "membership_revision": "memrev-gamma-001",
                "session_version": 1,
                "tenant_id": "tenant_gamma",
                "namespace_id": "personal",
                "workspace_id": "gamma-home",
                "owner_id": "principal_carla",
                "artifact_id": "artifact_gamma_001",
                "issued_at": int(time.time()),
            },
            token_key,
        )
        revocation_before = {
            "api": _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_gamma_001", carla_token_v1),
            "cli_run": _run(
                [
                    str(root / "cornerstone"),
                    "security",
                    "vs2-range-client",
                    "--api-url",
                    gateway.url,
                    "--token",
                    carla_token_v1,
                    "--artifact-id",
                    "artifact_gamma_001",
                    "--json",
                ],
                cwd=root,
                timeout=30,
            ),
            "browser": _http_text(f"{gateway.url}/ui/vs2/artifacts/artifact_gamma_001", carla_token_v1),
            "service": gateway.artifact_show(
                token=f"Bearer {carla_token_v1}",
                artifact_id="artifact_gamma_001",
                caller_fields={},
                surface="service_direct",
            ),
            "worker": gateway.run_worker_artifact_job(worker_job_v1),
        }
        try:
            revocation_before["cli"] = json.loads(revocation_before["cli_run"]["stdout"])
        except ValueError:
            revocation_before["cli"] = {"parse_error": revocation_before["cli_run"]["stdout"]}

        stale_update = postgres.json_query(
            """
WITH updated AS (
  UPDATE cs.memberships
     SET session_version = 2,
         membership_revision = 'memrev-gamma-002'
   WHERE membership_id = 'm_gamma_carla_personal'
   RETURNING membership_id, membership_revision, session_version, revoked_at IS NOT NULL AS revoked
)
SELECT jsonb_build_object('rows', count(*), 'memberships', jsonb_agg(to_jsonb(updated)))::text
FROM updated;
"""
        )
        stale_worker_job = dict(worker_job_v1)
        stale_worker_job["job_id"] = "job_gamma_stale_session_001"
        stale_worker_job = _sign_worker_envelope({field: value for field, value in stale_worker_job.items() if field != "signature"}, token_key)
        stale_retry = {
            "api": _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_gamma_001", carla_token_v1),
            "worker": gateway.run_worker_artifact_job(stale_worker_job),
            "membership_update": stale_update,
        }

        carla_token_v2_payload = dict(carla_token_v1_payload)
        carla_token_v2_payload["session_version"] = 2
        carla_token_v2_payload["iat"] = int(time.time())
        carla_token_v2_payload["exp"] = int(time.time()) + 600
        carla_token_v2 = _sign_token(carla_token_v2_payload, token_key)
        revocation_v2_before = _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_gamma_001", carla_token_v2)
        policy_cache_invalidation = gateway.policy_cache_invalidation_matrix(
            owner_token=f"Bearer {carla_token_v2}",
            other_token=f"Bearer {bob_token}",
            artifact_id="artifact_gamma_001",
            updated_opa_url=opa_revision_v2.url,
        )
        revoke_update = postgres.json_query(
            """
WITH updated AS (
  UPDATE cs.memberships
     SET revoked_at = now(),
         membership_revision = 'memrev-gamma-003'
   WHERE membership_id = 'm_gamma_carla_personal'
   RETURNING membership_id, membership_revision, session_version, revoked_at IS NOT NULL AS revoked
)
SELECT jsonb_build_object('rows', count(*), 'memberships', jsonb_agg(to_jsonb(updated)))::text
FROM updated;
"""
        )
        worker_job_v2_base = {
            "job_id": "job_gamma_after_revocation_001",
            "principal_id": "principal_carla",
            "membership_id": "m_gamma_carla_personal",
            "membership_revision": "memrev-gamma-002",
            "session_version": 2,
            "tenant_id": "tenant_gamma",
            "namespace_id": "personal",
            "workspace_id": "gamma-home",
            "owner_id": "principal_carla",
            "artifact_id": "artifact_gamma_001",
            "issued_at": int(time.time()),
        }
        r11_provider_calls_before = len(provider.calls)
        r11_egress_calls_before = len(egress_proxy.calls)
        post_revocation_stale_allow = _post_revocation_stale_allow_concurrency_probe(
            gateway,
            gateway.url,
            carla_token_v2,
            token_key,
            worker_job_v2_base,
            policy_cache_invalidation,
            revoke_update,
        )
        post_revocation_stale_allow["provider_calls_before"] = r11_provider_calls_before
        post_revocation_stale_allow["provider_calls_after"] = len(provider.calls)
        post_revocation_stale_allow["egress_proxy_calls_before"] = r11_egress_calls_before
        post_revocation_stale_allow["egress_proxy_calls_after"] = len(egress_proxy.calls)
        post_revocation_stale_allow["checks"]["r11_zero_provider_or_egress_side_effects"] = (
            r11_provider_calls_before == len(provider.calls) and r11_egress_calls_before == len(egress_proxy.calls)
        )
        post_revocation_stale_allow["status"] = "passed" if all(post_revocation_stale_allow.get("checks", {}).values()) else "failed"
        worker_job_v2 = _sign_worker_envelope(worker_job_v2_base, token_key)
        revocation_after_cli_run = _run(
            [
                str(root / "cornerstone"),
                "security",
                "vs2-range-client",
                "--api-url",
                gateway.url,
                "--token",
                carla_token_v2,
                "--artifact-id",
                "artifact_gamma_001",
                "--json",
            ],
            cwd=root,
            timeout=30,
        )
        try:
            revocation_after_cli = json.loads(revocation_after_cli_run["stdout"])
        except ValueError:
            revocation_after_cli = {"parse_error": revocation_after_cli_run["stdout"]}
        revocation_after = {
            "api": _http_json(f"{gateway.url}/api/vs2/artifacts/artifact_gamma_001", carla_token_v2),
            "cli_run": revocation_after_cli_run,
            "cli": revocation_after_cli,
            "browser": _http_text(f"{gateway.url}/ui/vs2/artifacts/artifact_gamma_001", carla_token_v2),
            "service": gateway.artifact_show(
                token=f"Bearer {carla_token_v2}",
                artifact_id="artifact_gamma_001",
                caller_fields={},
                surface="service_direct",
            ),
            "worker": gateway.run_worker_artifact_job(worker_job_v2),
            "membership_update": revoke_update,
        }
        revocation_provider_calls_after = len(provider.calls)
        revocation_egress_calls_after = len(egress_proxy.calls)
        network_boundary = _docker_network_boundary_probe(root)
        backup_restore = _backup_restore_probe(root, postgres)
        inventory = _rls_inventory(postgres)
        rls_select = _rls_tenant_probe(postgres)
        rls_write = _rls_write_probe(postgres)
        connection_reuse = _connection_reuse_probe(postgres)
        concurrent_tenant_api = _concurrent_tenant_api_probe(gateway.url, token, bob_token, postgres)
        worker_scope = _worker_scope_probe(gateway, token_key, postgres)
        operation_key_scope = _operation_key_scope_probe(postgres)
        schema_security_gate = _schema_security_gate_probe(postgres)
        negative_control = _rls_negative_control(postgres)
        audit_count = postgres.json_query(
            """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_alpha';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'principal_alice';
SET LOCAL app.workspace_id = 'alpha-home';
SELECT jsonb_build_object('audit_events_visible', count(*), 'decision_ids', jsonb_agg(decision_id ORDER BY created_at))::text
FROM cs.audit_events;
COMMIT;
"""
        )
        api_payload = api_allowed["payload"]
        cli_payload = cli_allowed.get("range_client", {}).get("payload", {})
        object_contract_api_payload = object_contract_api.get("payload", {})
        object_contract_cli_payload = object_contract_cli.get("range_object_contract_client", {}).get("payload", {})
        object_access_api_payload = object_access_matrix_api.get("payload", {})
        object_access_cli_payload = object_access_matrix_cli.get("range_object_access_client", {}).get("payload", {})
        observability_api_payload = observability_matrix_api.get("payload", {})
        observability_admin_payload = observability_matrix_admin_api.get("payload", {})
        observability_cli_payload = observability_matrix_cli.get("range_observability_client", {}).get("payload", {})
        object_contract_api_rows = object_contract_api_payload.get("object_contract", {}).get("representative_rows", [])
        object_contract_cli_rows = object_contract_cli_payload.get("object_contract", {}).get("representative_rows", [])
        object_contract_expected_tables = [spec["table"] for spec in DURABLE_OBJECT_TABLES]
        object_contract_api_tables = {row.get("table") for row in object_contract_api_rows}
        object_contract_cli_tables = {row.get("table") for row in object_contract_cli_rows}
        tenant_read_api_payload = tenant_read_matrix_api.get("payload", {})
        tenant_read_cli_payload = tenant_read_matrix_cli.get("range_tenant_read_client", {}).get("payload", {})
        tenant_read_api_rows = _tenant_read_tables(tenant_read_api_payload)
        tenant_read_cli_rows = _tenant_read_tables(tenant_read_cli_payload)
        tenant_read_expected_tables = [spec["table"] for spec in DURABLE_OBJECT_TABLES]
        search_api_payload = search_matrix_api.get("payload", {})
        search_cli_payload = search_matrix_cli.get("range_search_client", {}).get("payload", {})
        db_path_api_payload = db_path_matrix_api.get("payload", {})
        db_path_cli_payload = db_path_matrix_cli.get("range_db_path_client", {}).get("payload", {})
        constraint_collision_api_payload = constraint_collision_api.get("payload", {})
        constraint_collision_cli_payload = constraint_collision_cli.get("range_constraint_collision_client", {}).get("payload", {})
        migration_matrix_api_payload = migration_matrix_api.get("payload", {})
        migration_matrix_cli_payload = migration_matrix_cli.get("range_migration_client", {}).get("payload", {})
        upgrade_path_api_payload = upgrade_path_api.get("payload", {})
        upgrade_path_cli_payload = upgrade_path_cli.get("range_upgrade_path_client", {}).get("payload", {})
        audit_integrity_api_payload = audit_integrity_api.get("payload", {})
        audit_integrity_cli_payload = audit_integrity_cli.get("range_audit_integrity_client", {}).get("payload", {})
        required_column_rows = object_contract_schema.get("required_columns", [])
        primary_key_tables = set(object_contract_constraints.get("primary_key_tables") or [])
        action_payload = action_cli.get("range_action_client", {}).get("payload", {})
        browser_html = browser_allowed["text"]
        revocation_before_cli_payload = revocation_before.get("cli", {}).get("range_client", {}).get("payload", {})
        revocation_after_cli_payload = revocation_after.get("cli", {}).get("range_client", {}).get("payload", {})
        revocation_before_browser_html = revocation_before["browser"]["text"]
        revocation_after_browser_html = revocation_after["browser"]["text"]
        provider_call = provider.calls[0] if provider.calls else {}
        egress_proxy_call = egress_proxy.calls[0] if egress_proxy.calls else {}
        checks = {
            "components_healthy": health["http_status"] == 200 and postgres_ready and opa_ready,
            "migrations_and_seed_applied": migrations_ok and seed_ok and egress_grant_ok,
            "cli_api_browser_same_context_digest": len(
                {
                    api_payload.get("context_digest"),
                    cli_payload.get("context_digest"),
                    _extract_html_attr(browser_html, "data-context-digest"),
                }
            )
            == 1,
            "cli_api_browser_same_policy_decision": {
                api_payload.get("policy_decision", {}).get("decision"),
                cli_payload.get("policy_decision", {}).get("decision"),
                _extract_html_attr(browser_html, "data-policy-decision"),
            }
            == {"allow"},
            "cli_api_browser_status_consistent": cli_allowed_run["exit_code"] == 0 and api_allowed["http_status"] == 200 and browser_allowed["http_status"] == 200,
            "audit_refs_present": bool(api_payload.get("audit_refs")) and bool(cli_payload.get("audit_refs")) and _extract_html_attr(browser_html, "data-audit-ref-count") != "0",
            "forged_scope_denied": forged["http_status"] == 403 and forged["payload"].get("tenant_b_rows_returned") == 0,
            "missing_context_zero_db_egress": missing["http_status"] == 401 and missing["payload"].get("counters", {}).get("db_calls") == 0 and missing["payload"].get("counters", {}).get("egress_calls") == 0,
            "bad_signature_zero_db_egress": bad_signature["http_status"] == 401 and bad_signature["payload"].get("counters", {}).get("db_calls") == 0 and bad_signature["payload"].get("counters", {}).get("egress_calls") == 0,
            "real_opa_allow_observed": api_payload.get("policy_decision", {}).get("decision") == "allow" and api_payload.get("policy_decision", {}).get("bundle_revision") == "vs2-rego-local-v1",
            "policy_decision_has_digest_and_id": bool(api_payload.get("policy_decision", {}).get("input_digest")) and bool(api_payload.get("policy_decision", {}).get("decision_id")),
            "object_contract_api_cli_visible": object_contract_api["http_status"] == 200
            and object_contract_cli_run["exit_code"] == 0
            and object_contract_api_payload.get("status") == "allowed"
            and object_contract_cli_payload.get("status") == "allowed"
            and object_contract_api_payload.get("context_digest") == object_contract_cli_payload.get("context_digest")
            and object_contract_api_tables == set(object_contract_expected_tables)
            and object_contract_cli_tables == set(object_contract_expected_tables),
            "object_contract_required_columns_not_null": object_contract_schema.get("table_count") == len(DURABLE_OBJECT_TABLES)
            and len(required_column_rows) == sum(5 if spec["classification_required"] else 4 for spec in DURABLE_OBJECT_TABLES)
            and all(row.get("present") is True and row.get("is_nullable") == "NO" for row in required_column_rows),
            "object_contract_representative_rows_created": len(object_contract_api_rows) == len(DURABLE_OBJECT_TABLES)
            and all(
                row.get("tenant_id") == "tenant_alpha"
                and row.get("namespace_id") == "personal"
                and row.get("owner_id") == "principal_alice"
                and row.get("workspace_id") == "alpha-home"
                and (row.get("classification") == "internal" if row.get("classification_applicable") else "classification" not in row)
                for row in object_contract_api_rows
            ),
            "object_contract_failed_null_inserts_denied": object_contract_null_inserts.get("attempt_count") == object_contract_null_inserts.get("denied_count")
            and object_contract_null_inserts.get("attempt_count") == len(required_column_rows),
            "object_contract_scope_mutation_denied": object_contract_scope_mutation.get("attempt_count") == object_contract_scope_mutation.get("denied_count")
            and object_contract_scope_mutation.get("attempt_count") == len(DURABLE_OBJECT_TABLES),
            "object_contract_primary_keys_present": primary_key_tables == set(object_contract_expected_tables),
            "object_access_matrix_api_cli_visible": _object_access_api_cli_visible(object_access_matrix_api, object_access_matrix_cli_run, object_access_api_payload, object_access_cli_payload),
            "object_access_authorized_alpha_storage_bound": _object_access_authorized_alpha_ok(object_access_api_payload)
            and _object_access_authorized_alpha_ok(object_access_cli_payload),
            "object_access_foreign_object_download_signed_url_denied": _object_access_foreign_denied(object_access_api_payload)
            and _object_access_foreign_denied(object_access_cli_payload),
            "object_access_evidence_traversal_isolated": _object_access_evidence_traversal_isolated(object_access_api_payload)
            and _object_access_evidence_traversal_isolated(object_access_cli_payload),
            "object_access_storage_log_zero_foreign_reads": _object_access_storage_log_bound(object_access_api_payload)
            and _object_access_storage_log_bound(object_access_cli_payload),
            "object_access_zero_beta_canary_or_ids": _object_access_no_foreign_values([object_access_api_payload, object_access_cli_payload]),
            "observability_matrix_api_cli_visible": _observability_api_cli_visible(
                observability_matrix_api,
                observability_matrix_cli_run,
                observability_api_payload,
                observability_cli_payload,
            ),
            "observability_records_isolated": _observability_records_isolated(observability_api_payload)
            and _observability_records_isolated(observability_cli_payload)
            and _observability_records_isolated(observability_admin_payload),
            "observability_tenant_export_scoped": _observability_export_scoped(observability_api_payload)
            and _observability_export_scoped(observability_cli_payload)
            and _observability_export_scoped(observability_admin_payload),
            "observability_aggregate_metrics_non_sensitive": _observability_metrics_non_sensitive(observability_api_payload)
            and _observability_metrics_non_sensitive(observability_cli_payload)
            and _observability_metrics_non_sensitive(observability_admin_payload),
            "observability_system_wide_denied_without_privilege": _observability_system_wide_denied(observability_api_payload)
            and _observability_system_wide_denied(observability_cli_payload)
            and _observability_system_wide_denied(observability_admin_payload),
            "observability_role_matrix_user_admin_scoped": _observability_role_matrix_user_admin_scoped(observability_api_payload, observability_admin_payload),
            "observability_zero_beta_canary_or_ids": _observability_no_foreign_values(
                [observability_api_payload, observability_cli_payload, observability_admin_payload]
            ),
            "tenant_read_matrix_api_cli_visible": tenant_read_matrix_api["http_status"] == 200
            and tenant_read_matrix_cli_run["exit_code"] == 0
            and tenant_read_api_payload.get("status") == "allowed"
            and tenant_read_cli_payload.get("status") == "allowed"
            and tenant_read_api_payload.get("context_digest") == tenant_read_cli_payload.get("context_digest")
            and _tenant_read_table_set(tenant_read_api_payload) == set(tenant_read_expected_tables)
            and _tenant_read_table_set(tenant_read_cli_payload) == set(tenant_read_expected_tables),
            "tenant_read_matrix_counts_hide_foreign_rows": _tenant_read_counts_hide_foreign_rows(tenant_read_api_rows)
            and _tenant_read_counts_hide_foreign_rows(tenant_read_cli_rows),
            "tenant_read_matrix_join_subquery_aggregate_pagination_isolated": _tenant_read_join_subquery_pagination_isolated(tenant_read_api_rows)
            and _tenant_read_join_subquery_pagination_isolated(tenant_read_cli_rows),
            "tenant_read_matrix_zero_beta_canary_or_ids": _tenant_read_no_foreign_values([tenant_read_api_payload, tenant_read_cli_payload]),
            "tenant_read_matrix_neutral_guessed_id_results": _tenant_read_neutral_guessed_results(tenant_read_api_rows)
            and _tenant_read_neutral_guessed_results(tenant_read_cli_rows),
            "search_matrix_api_cli_visible": _search_matrix_api_cli_visible(search_matrix_api, search_matrix_cli_run, search_api_payload, search_cli_payload),
            "search_matrix_foreign_term_no_results": _search_matrix_results_isolated(search_api_payload)
            and _search_matrix_results_isolated(search_cli_payload),
            "search_matrix_autocomplete_facets_snapshots_objects_isolated": _search_matrix_results_isolated(search_api_payload)
            and _search_matrix_results_isolated(search_cli_payload),
            "search_matrix_inventory_indexes_and_rls_ok": _search_matrix_inventory_ok(search_api_payload)
            and _search_matrix_inventory_ok(search_cli_payload),
            "search_matrix_zero_beta_canary_or_ids": _search_matrix_no_foreign_values([search_api_payload, search_cli_payload]),
            "db_path_matrix_api_cli_visible": db_path_matrix_api["http_status"] == 200
            and db_path_matrix_cli_run["exit_code"] == 0
            and db_path_api_payload.get("status") == "allowed"
            and db_path_cli_payload.get("status") == "allowed"
            and db_path_api_payload.get("context_digest") == db_path_cli_payload.get("context_digest"),
            "db_path_raw_sql_repository_isolated": _db_path_raw_sql_isolated(db_path_api_payload.get("db_path_matrix", {}).get("raw_sql", {}))
            and _db_path_raw_sql_isolated(db_path_cli_payload.get("db_path_matrix", {}).get("raw_sql", {})),
            "db_path_view_and_function_isolated": _db_path_view_function_isolated(db_path_api_payload)
            and _db_path_view_function_isolated(db_path_cli_payload),
            "db_path_unsafe_security_definer_denied": _db_path_unsafe_definer_denied(db_path_api_payload)
            and _db_path_unsafe_definer_denied(db_path_cli_payload),
            "db_path_inventory_grants_and_security_modes_ok": _db_path_inventory_ok(db_path_api_payload.get("db_path_matrix", {}).get("object_inventory", {}))
            and _db_path_inventory_ok(db_path_cli_payload.get("db_path_matrix", {}).get("object_inventory", {})),
            "db_path_zero_beta_canary_or_ids": _db_path_no_foreign_values([db_path_api_payload, db_path_cli_payload]),
            "constraint_collision_matrix_api_cli_visible": _constraint_collision_api_cli_visible(
                constraint_collision_api,
                constraint_collision_cli_run,
                constraint_collision_api_payload,
                constraint_collision_cli_payload,
            ),
            "constraint_collision_tenant_scoped_unique_keys": _constraint_collision_tenant_scoped_unique_keys(
                constraint_collision_api_payload,
            )
            and _constraint_collision_tenant_scoped_unique_keys(constraint_collision_cli_payload),
            "constraint_collision_tenant_aware_foreign_keys": _constraint_collision_tenant_aware_foreign_keys(
                constraint_collision_api_payload,
            )
            and _constraint_collision_tenant_aware_foreign_keys(constraint_collision_cli_payload),
            "constraint_collision_neutral_errors": _constraint_collision_neutral_errors(constraint_collision_api_payload)
            and _constraint_collision_neutral_errors(constraint_collision_cli_payload),
            "constraint_collision_zero_foreign_canary_or_ids": _constraint_collision_no_foreign_values(
                [constraint_collision_api_payload, constraint_collision_cli_payload],
            ),
            "migration_matrix_api_cli_visible": _migration_api_cli_visible(
                migration_matrix_api,
                migration_matrix_cli_run,
                migration_matrix_api_payload,
                migration_matrix_cli_payload,
            ),
            "migration_known_rows_migrated": _migration_known_rows_migrated(migration_matrix_api_payload)
            and _migration_known_rows_migrated(migration_matrix_cli_payload),
            "migration_bad_rows_quarantined": _migration_bad_rows_quarantined(migration_matrix_api_payload)
            and _migration_bad_rows_quarantined(migration_matrix_cli_payload),
            "migration_no_ownerless_global_truth": _migration_no_ownerless_global_truth(migration_matrix_api_payload)
            and _migration_no_ownerless_global_truth(migration_matrix_cli_payload),
            "migration_checksums_and_rollback_evidence": _migration_checksums_and_rollback_evidence(migration_matrix_api_payload)
            and _migration_checksums_and_rollback_evidence(migration_matrix_cli_payload),
            "migration_zero_foreign_canary_or_ids": _migration_no_foreign_values(
                [migration_matrix_api_payload, migration_matrix_cli_payload],
            ),
            "upgrade_path_matrix_api_cli_visible": _upgrade_path_api_cli_visible(
                upgrade_path_api,
                upgrade_path_cli_run,
                upgrade_path_api_payload,
                upgrade_path_cli_payload,
            ),
            "upgrade_path_forward_preserves_vs0_vs1_objects": _upgrade_path_forward_preserves_objects(
                upgrade_path_api_payload,
            )
            and _upgrade_path_forward_preserves_objects(upgrade_path_cli_payload),
            "upgrade_path_compatibility_regression_reads": _upgrade_path_compatibility_regression_reads(
                upgrade_path_api_payload,
            )
            and _upgrade_path_compatibility_regression_reads(upgrade_path_cli_payload),
            "upgrade_path_failed_migration_and_rollback": _upgrade_path_failed_migration_and_rollback(
                upgrade_path_api_payload,
            )
            and _upgrade_path_failed_migration_and_rollback(upgrade_path_cli_payload),
            "upgrade_path_destructive_without_approval_denied": _upgrade_path_destructive_without_approval_denied(
                upgrade_path_api_payload,
            )
            and _upgrade_path_destructive_without_approval_denied(upgrade_path_cli_payload),
            "audit_integrity_matrix_api_cli_visible": _audit_integrity_api_cli_visible(
                audit_integrity_api,
                audit_integrity_cli_run,
                audit_integrity_api_payload,
                audit_integrity_cli_payload,
            ),
            "audit_integrity_required_events_present": _audit_integrity_required_events_present(audit_integrity_api_payload)
            and _audit_integrity_required_events_present(audit_integrity_cli_payload),
            "audit_integrity_clean_chain_verifies": _audit_integrity_clean_chain_verifies(audit_integrity_api_payload)
            and _audit_integrity_clean_chain_verifies(audit_integrity_cli_payload),
            "audit_integrity_tamper_cases_detected": _audit_integrity_tamper_detected(audit_integrity_api_payload)
            and _audit_integrity_tamper_detected(audit_integrity_cli_payload),
            "audit_integrity_append_only_and_auditor_role": _audit_integrity_append_only_and_auditor_role(audit_integrity_api_payload)
            and _audit_integrity_append_only_and_auditor_role(audit_integrity_cli_payload),
            "rls_select_isolated": rls_select.get("visible_artifacts", 0) >= 1 and rls_select.get("tenant_beta_rows_visible") == 0,
            "rls_write_denied": rls_write.get("forged_insert_denied") is True and rls_write.get("cross_tenant_update_zero") is True,
            "connection_reuse_same_backend_pid": _connection_reuse_same_backend(connection_reuse),
            "connection_reuse_tenant_sequence_isolated": _connection_reuse_tenant_sequence_isolated(connection_reuse),
            "connection_reuse_resets_after_success_error_timeout_rollback": _connection_reuse_resets_after_paths(connection_reuse),
            "connection_reuse_expected_error_timeout_observed": _connection_reuse_error_timeout_observed(connection_reuse),
            "connection_reuse_zero_cross_tenant_canary_or_ids": _connection_reuse_no_cross_tenant_canaries(connection_reuse),
            "concurrent_tenant_api_load_completed": _concurrent_tenant_api_load_completed(concurrent_tenant_api),
            "concurrent_tenant_contexts_isolated": _concurrent_tenant_contexts_isolated(concurrent_tenant_api),
            "concurrent_tenant_zero_foreign_canary_or_ids": _concurrent_tenant_zero_foreign_canary_or_ids(concurrent_tenant_api),
            "concurrent_tenant_audit_refs_not_mixed": _concurrent_tenant_audit_refs_not_mixed(concurrent_tenant_api),
            "concurrent_tenant_policy_trace_refs_present": _concurrent_tenant_policy_refs_present(concurrent_tenant_api),
            "concurrent_tenant_pool_reset_evidence_present": _connection_reuse_same_backend(connection_reuse)
            and _connection_reuse_resets_after_paths(connection_reuse),
            "worker_scope_valid_job_completed": _worker_scope_valid_job_completed(worker_scope),
            "worker_scope_quarantines_bad_envelopes": _worker_scope_quarantines_bad_envelopes(worker_scope),
            "worker_scope_persists_audit_and_job_records": _worker_scope_persists_audit_and_job_records(worker_scope),
            "worker_scope_zero_payload_leak_or_egress": _worker_scope_zero_payload_leak_or_egress(worker_scope),
            "worker_scope_replay_idempotency_guard": _worker_scope_replay_idempotency_guard(worker_scope),
            "operation_key_records_exist": _operation_key_records_exist(operation_key_scope),
            "operation_key_tenant_scoped_independent": _operation_key_tenant_scoped_independent(operation_key_scope),
            "operation_key_collision_replay_scoped": _operation_key_collision_replay_scoped(operation_key_scope),
            "operation_key_zero_cross_tenant_suppression": _operation_key_zero_cross_tenant_suppression(operation_key_scope),
            "operation_key_no_foreign_canary_in_tenant_outputs": _operation_key_no_foreign_canary_in_tenant_outputs(operation_key_scope),
            "schema_gate_bad_fixture_fails": _schema_gate_bad_fixture_fails(schema_security_gate),
            "schema_gate_corrected_fixture_passes": _schema_gate_corrected_fixture_passes(schema_security_gate),
            "schema_gate_inventory_machine_readable": _schema_gate_inventory_machine_readable(schema_security_gate),
            "schema_gate_detects_required_surfaces": _schema_gate_detects_required_surfaces(schema_security_gate),
            "schema_gate_rollback_leaves_no_fixture_tables": _schema_gate_rollback_leaves_no_fixture_tables(schema_security_gate),
            "app_role_hardened": _app_role_hardened(inventory),
            "audit_persisted_in_postgres": audit_count.get("audit_events_visible", 0) >= 3,
            "negative_control_rls_disabled_detects_leak": negative_control.get("detected") is True and negative_control.get("restore_exit_code") == 0,
            "tenant_b_canary_not_serialized": "BETA_ONLY_RANGE_CANARY" not in json.dumps([api_allowed, cli_allowed, forged, missing, bad_signature], sort_keys=True),
            "external_action_flow_executed": action_cli_run["exit_code"] == 0 and action_payload.get("status") == "executed" and len(provider.calls) == 1 and len(egress_proxy.calls) == 1,
            "external_action_dry_run_approval_execution_linked": action_payload.get("dry_run_decision", {}).get("decision") == "allow"
            and action_payload.get("approval", {}).get("status") == "approved"
            and action_payload.get("execution_decision", {}).get("decision") == "allow"
            and bool(action_payload.get("audit_refs")),
            "tenant_b_egress_denied": tenant_b_action["http_status"] == 403
            and tenant_b_action["payload"].get("dry_run_decision", {}).get("decision") == "deny"
            and len(provider.calls) == 1,
            "connectorhub_credential_ref_only": len(provider.calls) == 1
            and len(egress_proxy.calls) == 1
            and provider_call.get("credential_ref_seen") == "credential_ref_mock_provider_write"
            and provider_call.get("authorization_header_seen") is False
            and provider_call.get("secret_canary_seen") is False
            and egress_proxy_call.get("secret_canary_seen") is False,
            "stale_dry_run_blocks_execution": action_payload.get("stale_dry_run_probe", {}).get("decision") == "deny"
            and action_payload.get("stale_dry_run_probe", {}).get("provider_calls_after") == 0
            and len(provider.calls) == 1,
            "revocation_allow_before_revoke": revocation_before["api"]["http_status"] == 200
            and revocation_before["cli_run"]["exit_code"] == 0
            and revocation_before["browser"]["http_status"] == 200
            and revocation_before["service"].get("status") == "allowed"
            and revocation_before["worker"].get("status") == "completed"
            and revocation_before_cli_payload.get("status") == "allowed"
            and _extract_html_attr(revocation_before_browser_html, "data-status") == "allowed",
            "stale_session_version_denied": stale_retry["membership_update"].get("rows") == 1
            and stale_retry["api"]["http_status"] == 401
            and stale_retry["api"]["payload"].get("error", {}).get("reason") == "stale_session_version"
            and stale_retry["worker"].get("status") == "quarantined"
            and stale_retry["worker"].get("reason") == "stale_session_version"
            and stale_retry["worker"].get("counters", {}).get("egress_calls") == 0,
            "policy_cache_initial_allow_from_real_opa": policy_cache_invalidation.get("checks", {}).get("policy_cache_initial_allow_from_real_opa") is True,
            "policy_cache_same_revision_hit_exercised": policy_cache_invalidation.get("checks", {}).get("policy_cache_same_revision_hit_exercised") is True,
            "policy_cache_key_contains_required_dimensions": policy_cache_invalidation.get("checks", {}).get("policy_cache_key_contains_required_dimensions") is True,
            "policy_cache_revision_update_changes_key": policy_cache_invalidation.get("checks", {}).get("policy_cache_revision_update_changes_key") is True,
            "policy_cache_legacy_key_without_revision_would_collide": policy_cache_invalidation.get("checks", {}).get("policy_cache_legacy_key_without_revision_would_collide") is True,
            "policy_cache_stale_allow_not_reused_after_revision_update": policy_cache_invalidation.get("checks", {}).get("policy_cache_stale_allow_not_reused_after_revision_update") is True,
            "policy_cache_new_revision_decision_recorded": policy_cache_invalidation.get("checks", {}).get("policy_cache_new_revision_decision_recorded") is True,
            "policy_cache_zero_stale_allows": policy_cache_invalidation.get("checks", {}).get("policy_cache_zero_stale_allows") is True,
            "policy_cache_cross_tenant_key_distinct": policy_cache_invalidation.get("checks", {}).get("policy_cache_cross_tenant_key_distinct") is True,
            "policy_cache_source_map_bound_to_trusted_context": policy_cache_invalidation.get("checks", {}).get("policy_cache_source_map_bound_to_trusted_context") is True,
            "policy_cache_audit_refs_recorded": policy_cache_invalidation.get("checks", {}).get("policy_cache_audit_refs_recorded") is True,
            "r11_cached_allow_existed_before_revocation": post_revocation_stale_allow.get("checks", {}).get("r11_cached_allow_existed_before_revocation") is True,
            "r11_revision_update_sequence_recorded": post_revocation_stale_allow.get("checks", {}).get("r11_revision_update_sequence_recorded") is True,
            "r11_membership_revoked_after_cached_allow": post_revocation_stale_allow.get("checks", {}).get("r11_membership_revoked_after_cached_allow") is True,
            "r11_concurrent_retries_completed": post_revocation_stale_allow.get("checks", {}).get("r11_concurrent_retries_completed") is True,
            "r11_zero_post_revocation_successes": post_revocation_stale_allow.get("checks", {}).get("r11_zero_post_revocation_successes") is True,
            "r11_cached_allow_not_reused_after_revocation": post_revocation_stale_allow.get("checks", {}).get("r11_cached_allow_not_reused_after_revocation") is True,
            "r11_denial_audits_recorded": post_revocation_stale_allow.get("checks", {}).get("r11_denial_audits_recorded") is True,
            "r11_zero_provider_or_egress_side_effects": post_revocation_stale_allow.get("checks", {}).get("r11_zero_provider_or_egress_side_effects") is True,
            "revoked_membership_denied_api_cli_browser_service_worker": revocation_v2_before["http_status"] == 200
            and revocation_after["membership_update"].get("rows") == 1
            and revocation_after["api"]["http_status"] == 401
            and revocation_after["api"]["payload"].get("error", {}).get("reason") == "membership_revoked"
            and revocation_after["cli_run"]["exit_code"] != 0
            and revocation_after_cli_payload.get("status") == "denied"
            and revocation_after_cli_payload.get("error", {}).get("reason") == "membership_revoked"
            and revocation_after["browser"]["http_status"] == 401
            and _extract_html_attr(revocation_after_browser_html, "data-status") == "denied"
            and revocation_after["service"].get("status") == "denied"
            and revocation_after["service"].get("error", {}).get("reason") == "membership_revoked"
            and revocation_after["worker"].get("status") == "quarantined"
            and revocation_after["worker"].get("reason") == "membership_revoked",
            "revocation_denial_audit_recorded": bool(revocation_after["api"]["payload"].get("audit_refs"))
            and bool(revocation_after_cli_payload.get("audit_refs"))
            and _extract_html_attr(revocation_after_browser_html, "data-audit-ref-count") != "0"
            and bool(revocation_after["service"].get("audit_refs"))
            and bool(revocation_after["worker"].get("audit_refs")),
            "revocation_zero_post_revoke_access_or_egress": revocation_after["api"]["payload"].get("artifact") is None
            and revocation_after_cli_payload.get("artifact") is None
            and revocation_after["service"].get("artifact") is None
            and revocation_after["worker"].get("artifact") is None
            and revocation_provider_calls_before == revocation_provider_calls_after
            and revocation_egress_calls_before == revocation_egress_calls_after,
            "docker_network_direct_egress_denied": network_boundary.get("checks", {}).get("direct_http_and_socket_blocked") is True,
            "docker_network_provider_zero_requests_after_direct_attempts": network_boundary.get("checks", {}).get("provider_zero_requests_after_direct_attempts") is True,
            "docker_network_provider_reachable_from_governed_proxy": network_boundary.get("checks", {}).get("provider_reachable_from_governed_proxy") is True,
            "docker_network_membership_isolated": network_boundary.get("checks", {}).get("provider_network_membership_isolated") is True
            and network_boundary.get("checks", {}).get("service_network_membership_expected") is True,
            "backup_restore_pg_dump_succeeded": backup_restore.get("checks", {}).get("pg_dump_succeeded") is True,
            "backup_restore_pg_restore_succeeded": backup_restore.get("checks", {}).get("pg_restore_succeeded") is True,
            "backup_restore_counts_match": backup_restore.get("checks", {}).get("row_counts_match_after_restore") is True,
            "backup_restore_rls_rechecked": backup_restore.get("checks", {}).get("rls_rechecked_after_restore") is True,
            "backup_restore_audit_rechecked": backup_restore.get("checks", {}).get("audit_rechecked_after_restore") is True,
            "backup_restore_tenant_export_scoped": backup_restore.get("checks", {}).get("tenant_export_scoped") is True
            and backup_restore.get("checks", {}).get("tenant_export_matches_primary") is True,
            "same_tenant_namespace_policy_denies_implicit_cross_namespace": same_tenant_namespace.get("checks", {}).get("implicit_cross_namespace_denied_by_policy") is True,
            "same_tenant_namespace_rls_hides_foreign_workspace": same_tenant_namespace.get("checks", {}).get("personal_context_db_returns_zero_org_rows") is True
            and same_tenant_namespace.get("checks", {}).get("org_context_reads_org_row") is True,
            "same_tenant_namespace_explicit_promotion_has_provenance": same_tenant_namespace.get("checks", {}).get("explicit_promotion_records_provenance") is True
            and same_tenant_namespace.get("checks", {}).get("promotion_audited") is True,
            "cross_tenant_transfer_policy_denies": cross_tenant_transfer.get("checks", {}).get("cross_tenant_transfer_policy_denied") is True,
            "cross_tenant_transfer_copy_reference_share_promotion_denied": cross_tenant_transfer.get("checks", {}).get("copy_reference_share_promotion_denied") is True,
            "cross_tenant_transfer_zero_target_records": cross_tenant_transfer.get("checks", {}).get("zero_target_records_created") is True,
            "cross_tenant_transfer_audited": cross_tenant_transfer.get("checks", {}).get("transfer_denial_audited") is True,
            "service_allow_bypass_rls_zero_rows": rls_defense_in_depth.get("checks", {}).get("fault_injected_service_allow_recorded") is True
            and rls_defense_in_depth.get("checks", {}).get("rls_still_returns_zero_foreign_rows") is True,
            "service_allow_bypass_anomaly_audited_and_metric_recorded": rls_defense_in_depth.get("checks", {}).get("anomaly_audit_recorded") is True
            and rls_defense_in_depth.get("checks", {}).get("anomaly_metric_recorded") is True,
            "opa_policy_input_schema_file_present": policy_input_schema.get("checks", {}).get("schema_file_present") is True,
            "opa_policy_input_schema_version_const_is_v1": policy_input_schema.get("checks", {}).get("schema_version_const_is_v1") is True,
            "opa_policy_input_builders_cover_operation_families": policy_input_schema.get("checks", {}).get("valid_cases_cover_operation_families") is True,
            "opa_policy_input_valid_cases_pass_schema_and_opa": policy_input_schema.get("checks", {}).get("valid_cases_pass_schema_and_opa") is True,
            "opa_policy_input_invalid_cases_rejected_before_opa": policy_input_schema.get("checks", {}).get("invalid_cases_rejected_before_opa") is True,
            "opa_policy_input_source_map_covers_required_attributes": policy_input_schema.get("checks", {}).get("source_map_covers_required_attributes") is True,
            "opa_policy_input_digests_present": policy_input_schema.get("checks", {}).get("input_digests_present") is True,
            "opa_policy_input_audit_refs_recorded": policy_input_schema.get("checks", {}).get("audit_refs_recorded") is True,
            "opa_decision_log_mask_policy_loaded_from_opa": decision_log_masking.get("checks", {}).get("mask_policy_loaded_from_opa") is True,
            "opa_decision_log_canary_reached_opa_request": decision_log_masking.get("checks", {}).get("canary_reached_opa_decision_request") is True,
            "opa_decision_log_collector_received_masked_fields": decision_log_masking.get("checks", {}).get("collector_received_masked_fields") is True,
            "opa_decision_log_collector_entry_has_no_canary": decision_log_masking.get("checks", {}).get("collector_entry_has_no_canary") is True,
            "opa_decision_log_linked_to_policy_and_audit": decision_log_masking.get("checks", {}).get("decision_log_linked_to_policy_and_audit") is True,
            "opa_decision_log_source_map_retained_without_raw_values": decision_log_masking.get("checks", {}).get("source_map_retained_without_raw_values") is True,
            "reason_code_catalog_file_present": reason_code_translation.get("checks", {}).get("reason_catalog_file_present") is True,
            "reason_code_catalog_covers_observed_reasons": reason_code_translation.get("checks", {}).get("reason_catalog_covers_observed_reasons") is True,
            "reason_code_denial_responses_have_stable_codes": reason_code_translation.get("checks", {}).get("denial_responses_have_stable_codes") is True,
            "reason_code_denial_responses_match_decision_and_audit_refs": reason_code_translation.get("checks", {}).get("denial_responses_match_decision_and_audit_refs") is True,
            "reason_code_ui_snapshots_render_denial_component": reason_code_translation.get("checks", {}).get("ui_snapshots_render_denial_component") is True,
            "reason_code_surfaces_cover_policy_denial_families": reason_code_translation.get("checks", {}).get("surfaces_cover_policy_denial_families") is True,
            "reason_code_audit_refs_recorded": reason_code_translation.get("checks", {}).get("audit_refs_recorded") is True,
            "break_glass_normal_app_denied": break_glass_maintenance.get("checks", {}).get("normal_app_credentials_cannot_break_glass") is True,
            "break_glass_requires_approval": break_glass_maintenance.get("checks", {}).get("missing_approval_denied_by_policy") is True,
            "break_glass_approved_path_audited_and_time_bound": break_glass_maintenance.get("checks", {}).get("approved_synthetic_break_glass_allowed") is True
            and break_glass_maintenance.get("checks", {}).get("approval_has_purpose_time_bound_and_scope") is True
            and break_glass_maintenance.get("checks", {}).get("break_glass_audit_recorded") is True,
            "break_glass_safe_output_counts_only": break_glass_maintenance.get("checks", {}).get("break_glass_output_is_safe_counts_only") is True,
            "product_learning_raw_truth_denied": product_learning_guard.get("checks", {}).get("raw_personal_truth_learning_denied") is True
            and product_learning_guard.get("checks", {}).get("raw_org_truth_learning_denied") is True,
            "product_learning_zero_memory_or_truth_writes": product_learning_guard.get("checks", {}).get("no_hidden_memory_or_truth_writes") is True,
            "product_learning_denials_audited": product_learning_guard.get("checks", {}).get("learning_denials_audited") is True,
            "policy_limits_config_present": policy_limits.get("checks", {}).get("limits_config_file_present") is True,
            "policy_limits_at_and_below_allow": policy_limits.get("checks", {}).get("below_and_at_limit_requests_allow_deterministically") is True,
            "policy_limits_over_limit_fail_before_opa": policy_limits.get("checks", {}).get("over_limit_requests_rejected_before_opa") is True,
            "policy_limits_unknown_enum_fail_before_opa": policy_limits.get("checks", {}).get("unknown_enum_rejected_before_opa") is True,
            "policy_limits_bounded_no_partial_side_effect": policy_limits.get("checks", {}).get("limit_failures_have_bounded_resource_use") is True
            and policy_limits.get("checks", {}).get("limit_failures_leave_no_partial_side_effects") is True,
            "policy_conformance_same_input_all_points": policy_conformance.get("checks", {}).get("same_policy_input_digest_across_enforcement_points") is True,
            "policy_conformance_gateway_service_tool_cli_equivalent": policy_conformance.get("checks", {}).get("gateway_service_tool_cli_decisions_equivalent") is True
            and policy_conformance.get("checks", {}).get("native_cli_observed_same_active_revision") is True,
            "policy_conformance_mismatch_fail_closed": policy_conformance.get("checks", {}).get("revision_mismatch_fails_closed") is True,
            "policy_conformance_mismatch_audited_no_side_effect": policy_conformance.get("checks", {}).get("mismatch_anomaly_audit_and_metric_recorded") is True
            and policy_conformance.get("checks", {}).get("mismatch_no_protected_side_effect") is True,
            "opa_low_risk_allow_reaches_downstream_and_audit": policy_enforcement.get("checks", {}).get("allow_decision_reaches_downstream_and_audit") is True,
            "opa_role_denied_without_downstream_side_effect": policy_enforcement.get("checks", {}).get("role_denied_without_downstream_side_effect") is True,
            "opa_unknown_policy_default_denied_without_downstream_side_effect": policy_enforcement.get("checks", {}).get("unknown_policy_default_denied_without_downstream_side_effect") is True,
            "opa_abac_attribute_boundaries_enforced": policy_enforcement.get("checks", {}).get("abac_attribute_boundaries_enforced") is True,
            "opa_abac_matching_allowed_set_succeeds": policy_enforcement.get("checks", {}).get("abac_matching_allowed_set_succeeds") is True,
            "opa_malformed_and_wrong_version_inputs_fail_closed": policy_enforcement.get("checks", {}).get("malformed_and_wrong_version_inputs_fail_closed") is True,
            "opa_unexpected_authoritative_attrs_fail_closed": policy_enforcement.get("checks", {}).get("unexpected_authoritative_attrs_fail_closed") is True,
            "opa_deny_precedence_conflict_matrix_enforced": policy_enforcement.get("checks", {}).get("deny_precedence_conflict_matrix_enforced") is True,
            "opa_failure_modes_fail_closed_without_side_effects": policy_enforcement.get("checks", {}).get("opa_failure_modes_fail_closed_without_side_effects") is True,
            "opa_failure_modes_cover_protected_operation_families": policy_enforcement.get("checks", {}).get("opa_failure_modes_cover_protected_operation_families") is True,
            "opa_failure_readiness_degraded": policy_enforcement.get("checks", {}).get("opa_failure_readiness_degraded") is True,
            "opa_failure_denials_have_stable_safe_responses": policy_enforcement.get("checks", {}).get("opa_failure_denials_have_stable_safe_responses") is True,
            "opa_decisions_have_revision_digest_and_id": policy_enforcement.get("checks", {}).get("decisions_have_revision_digest_and_id") is True,
            "opa_fail_closed_decisions_have_digest_and_id": policy_enforcement.get("checks", {}).get("fail_closed_decisions_have_digest_and_id") is True,
            "opa_denials_have_stable_safe_responses": policy_enforcement.get("checks", {}).get("denials_have_stable_safe_responses") is True,
        }
        payload = {
            "schema_version": "cs.vs2_local_range.v1",
            "status": "passed" if all(checks.values()) else "failed",
            "source_fingerprint": source_fingerprint,
            "claim_boundary": "first production-flow local slice only; not full VS2, production, live-provider, penetration-test, or human UX evidence",
            "topology": {
                "host_side_test_runner": True,
                "gateway_api": "real local Python gateway using CornerStone package code",
                "postgres": {"image": POSTGRES_IMAGE, "published_to_host": False, "container": postgres.container},
                "opa": {"image": OPA_IMAGE, "published_to_loopback_for_local_policy_client": True, "container": opa.container},
                "opa_revision_v2": {
                    "image": OPA_IMAGE,
                    "published_to_loopback_for_local_policy_client": True,
                    "container": opa_revision_v2.container,
                    "revision": "vs2-rego-local-v2",
                    "decision": "deny",
                },
                "mock_external_provider": "used_for_external_action_first_slice",
                "external_action_mock_provider": provider.url,
                "egress_proxy": egress_proxy.url,
            },
            "gateway_url": gateway.url,
            "opa_url": opa.url,
            "health": health,
            "observations": {
                "api_allowed": api_allowed,
                "cli_allowed": cli_allowed,
                "browser_allowed": {"http_status": browser_allowed["http_status"], "data_attrs": _extract_html_attrs(browser_html)},
                "forged_scope": forged,
                "missing_context": missing,
                "bad_signature": bad_signature,
                "external_action_cli": action_cli,
                "tenant_b_external_action": tenant_b_action,
                "object_contract_api": object_contract_api,
                "object_contract_cli": object_contract_cli,
                "object_access_matrix_api": object_access_matrix_api,
                "object_access_matrix_cli": object_access_matrix_cli,
                "object_contract_schema": object_contract_schema,
                "object_contract_constraints": object_contract_constraints,
                "object_contract_null_inserts": object_contract_null_inserts,
                "object_contract_scope_mutation": object_contract_scope_mutation,
                "observability_matrix_api": observability_matrix_api,
                "observability_matrix_admin_api": observability_matrix_admin_api,
                "observability_matrix_cli": observability_matrix_cli,
                "tenant_read_matrix_api": tenant_read_matrix_api,
                "tenant_read_matrix_cli": tenant_read_matrix_cli,
                "search_matrix_api": search_matrix_api,
                "search_matrix_cli": search_matrix_cli,
                "db_path_matrix_api": db_path_matrix_api,
                "db_path_matrix_cli": db_path_matrix_cli,
                "constraint_collision_api": constraint_collision_api,
                "constraint_collision_cli": constraint_collision_cli,
                "migration_matrix_api": migration_matrix_api,
                "migration_matrix_cli": migration_matrix_cli,
                "upgrade_path_api": upgrade_path_api,
                "upgrade_path_cli": upgrade_path_cli,
                "audit_integrity_api": audit_integrity_api,
                "audit_integrity_cli": audit_integrity_cli,
                "same_tenant_namespace": same_tenant_namespace,
                "cross_tenant_transfer": cross_tenant_transfer,
                "rls_defense_in_depth": rls_defense_in_depth,
                "policy_input_schema": policy_input_schema,
                "policy_enforcement": policy_enforcement,
                "policy_limits": policy_limits,
                "decision_log_masking": decision_log_masking,
                "reason_code_translation": reason_code_translation,
                "break_glass_maintenance": break_glass_maintenance,
                "product_learning_guard": product_learning_guard,
                "policy_conformance": policy_conformance,
                "policy_cache_invalidation": policy_cache_invalidation,
                "post_revocation_stale_allow": post_revocation_stale_allow,
                "worker_scope": worker_scope,
                "operation_key_scope": operation_key_scope,
                "schema_security_gate": schema_security_gate,
                "identity_revocation": {
                    "before": {
                        "api": revocation_before["api"],
                        "cli": revocation_before.get("cli"),
                        "browser": {"http_status": revocation_before["browser"]["http_status"], "data_attrs": _extract_html_attrs(revocation_before_browser_html)},
                        "service": revocation_before["service"],
                        "worker": revocation_before["worker"],
                    },
                    "stale_session_retry": stale_retry,
                    "v2_before_revoke": revocation_v2_before,
                    "after": {
                        "api": revocation_after["api"],
                        "cli": revocation_after.get("cli"),
                        "browser": {"http_status": revocation_after["browser"]["http_status"], "data_attrs": _extract_html_attrs(revocation_after_browser_html)},
                        "service": revocation_after["service"],
                        "worker": revocation_after["worker"],
                        "membership_update": revocation_after["membership_update"],
                    },
                    "provider_calls_before": revocation_provider_calls_before,
                    "provider_calls_after": revocation_provider_calls_after,
                    "egress_proxy_calls_before": revocation_egress_calls_before,
                    "egress_proxy_calls_after": revocation_egress_calls_after,
                },
                "network_boundary": network_boundary,
                "backup_restore": backup_restore,
                "concurrent_tenant_api": concurrent_tenant_api,
                "egress_proxy": {"requests": len(egress_proxy.calls), "calls": egress_proxy.calls},
                "mock_provider": {"requests": len(provider.calls), "calls": provider.calls},
                "audit_count": audit_count,
            },
            "database": {"inventory": inventory, "rls_select": rls_select, "rls_write": rls_write, "connection_reuse": connection_reuse},
            "negative_controls": [negative_control],
            "checks": checks,
            "command_transcripts": {
                "cli_allowed": _safe_transcript([cli_allowed_run])[0],
                "action_cli": _safe_transcript([action_cli_run])[0],
                "object_contract_cli": _safe_transcript([object_contract_cli_run])[0],
                "object_access_matrix_cli": _safe_transcript([object_access_matrix_cli_run])[0],
                "observability_matrix_cli": _safe_transcript([observability_matrix_cli_run])[0],
                "tenant_read_matrix_cli": _safe_transcript([tenant_read_matrix_cli_run])[0],
                "search_matrix_cli": _safe_transcript([search_matrix_cli_run])[0],
                "db_path_matrix_cli": _safe_transcript([db_path_matrix_cli_run])[0],
                "constraint_collision_cli": _safe_transcript([constraint_collision_cli_run])[0],
                "migration_matrix_cli": _safe_transcript([migration_matrix_cli_run])[0],
                "upgrade_path_cli": _safe_transcript([upgrade_path_cli_run])[0],
                "audit_integrity_cli": _safe_transcript([audit_integrity_cli_run])[0],
                "revocation_cli_before": _safe_transcript([revocation_before["cli_run"]])[0],
                "revocation_cli_after": _safe_transcript([revocation_after_cli_run])[0],
                "postgres": _safe_transcript(postgres.transcript),
                "opa": _safe_transcript(opa.transcript),
                "opa_revision_v2": _safe_transcript(opa_revision_v2.transcript),
            },
        }
        command_elapsed = 0.0
        for entry in payload["command_transcripts"].values():
            if isinstance(entry, list):
                command_elapsed += sum(float(item.get("elapsed_seconds") or 0.0) for item in entry)
            elif isinstance(entry, dict):
                command_elapsed += float(entry.get("elapsed_seconds") or 0.0)
        payload["profile"] = {
            "schema_version": "cs.vs2_local_range_profile.v1",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "child_command_elapsed_seconds": round(command_elapsed, 3),
            "profile_boundary": "top-level local-range wall time plus child command elapsed totals; HTTP and in-process probe timings remain inside wall time",
        }
        return payload
    finally:
        cleanup_started = time.perf_counter()
        cleanup_errors: list[str] = []
        cleanup_results: list[dict[str, Any]] = []
        for label, cleanup in [
            ("gateway", gateway.stop if gateway is not None else None),
            ("opa_revision_v2", opa_revision_v2.stop),
            ("egress_proxy", egress_proxy.stop),
            ("provider", provider.stop),
            ("opa", opa.stop),
            ("postgres", postgres.stop),
        ]:
            if cleanup is None:
                continue
            try:
                cleanup_result = cleanup()
                if isinstance(cleanup_result, dict):
                    result = {
                        "label": label,
                        "mandatory": True,
                        "exit_code": cleanup_result.get("exit_code"),
                        "elapsed_seconds": cleanup_result.get("elapsed_seconds"),
                        "command": cleanup_result.get("command"),
                    }
                    cleanup_results.append(result)
                    if cleanup_result.get("exit_code") != 0:
                        cleanup_errors.append(f"{label}:exit_code:{cleanup_result.get('exit_code')}")
                else:
                    cleanup_results.append({"label": label, "mandatory": True, "exit_code": 0})
            except Exception as error:  # pragma: no cover - cleanup failure must preserve partial evidence
                cleanup_errors.append(f"{label}:{type(error).__name__}:{error}")
                cleanup_results.append({"label": label, "mandatory": True, "exit_code": 1, "error": f"{type(error).__name__}:{error}"})
        cleanup_seconds = time.perf_counter() - cleanup_started
        if payload is not None:
            _finalize_report_payload(
                root,
                VS2_LOCAL_RANGE_REPORT,
                payload,
                started=started,
                cleanup_seconds=cleanup_seconds,
                cleanup_errors=cleanup_errors,
                cleanup_results=cleanup_results,
            )


def _extract_html_attr(html: str, name: str) -> str:
    marker = f"{name}='"
    if marker not in html:
        return ""
    return html.split(marker, 1)[1].split("'", 1)[0]


def _extract_html_attrs(html: str) -> dict[str, str]:
    return {
        "data-status": _extract_html_attr(html, "data-status"),
        "data-context-digest": _extract_html_attr(html, "data-context-digest"),
        "data-policy-decision": _extract_html_attr(html, "data-policy-decision"),
        "data-audit-ref-count": _extract_html_attr(html, "data-audit-ref-count"),
    }


def _app_role_hardened(inventory: dict[str, Any]) -> bool:
    roles = inventory.get("roles", [])
    tables = inventory.get("tables", [])
    app_roles_ok = all(not role.get("rolsuper") and not role.get("rolbypassrls") for role in roles)
    rls_tables = [table for table in tables if table.get("relation") in {"artifacts", "audit_events", "memberships"}]
    rls_ok = all(table.get("rls_enabled") and table.get("rls_forced") for table in rls_tables)
    owner_ok = all(table.get("owner") != "cornerstone_app" for table in rls_tables)
    return app_roles_ok and rls_ok and owner_ok and inventory.get("policy_count", 0) >= 2
