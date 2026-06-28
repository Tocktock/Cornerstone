from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_PATH = Path("reports/security/vs2-production-like-integration-2026-06-27.json")

ROLE_DEFINITIONS = {
    "cornerstone_schema_owner": "CREATE ROLE cornerstone_schema_owner NOLOGIN NOSUPERUSER NOBYPASSRLS",
    "cornerstone_app": "CREATE ROLE cornerstone_app LOGIN NOSUPERUSER NOBYPASSRLS",
    "cornerstone_identity": "CREATE ROLE cornerstone_identity NOLOGIN NOSUPERUSER NOBYPASSRLS",
    "cornerstone_migrator": "CREATE ROLE cornerstone_migrator LOGIN NOSUPERUSER NOBYPASSRLS",
    "cornerstone_maintenance": "CREATE ROLE cornerstone_maintenance LOGIN NOSUPERUSER NOBYPASSRLS",
    "cornerstone_auditor": "CREATE ROLE cornerstone_auditor NOLOGIN NOSUPERUSER NOBYPASSRLS",
}

CORE_RLS_TABLES = [
    "artifacts",
    "derived_representations",
    "search_snapshots",
    "evidence_bundles",
    "claims",
    "ontology_objects",
    "ontology_links",
    "action_cards",
    "workflow_runs",
    "policy_decisions",
    "jobs",
    "idempotency_keys",
    "egress_grants",
    "migration_quarantine",
]

RELATION_ID_COLUMNS = {
    **{table: "artifact_id" for table in CORE_RLS_TABLES},
    "audit_events": "event_id",
    "operator_metrics": "metric_id",
    "status_records": "status_id",
    "tenant_exports": "export_id",
    "artifact_references": "reference_id",
}


def _run(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "command": command,
            "exit_code": 124,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
        }
    return {
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def _run_bytes(
    command: list[str],
    *,
    cwd: Path,
    input_bytes: bytes | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "exit_code": 124,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout": error.stdout or b"",
            "stderr": error.stderr or b"",
            "timed_out": True,
        }
    return {
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def _summarize(entry: dict[str, Any]) -> dict[str, Any]:
    stdout = entry.get("stdout", "")
    stderr = entry.get("stderr", "")
    if isinstance(stdout, bytes):
        stdout = f"<{len(stdout)} bytes>"
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return {
        "command": entry.get("command"),
        "exit_code": entry.get("exit_code"),
        "elapsed_seconds": entry.get("elapsed_seconds"),
        "timed_out": entry.get("timed_out", False),
        "stdout_tail": str(stdout).splitlines()[-8:],
        "stderr_tail": str(stderr).splitlines()[-8:],
    }


def _compose(compose_file: Path, args: list[str], *, root: Path, input_text: str | None = None, timeout: int = 120) -> dict[str, Any]:
    return _run(["docker", "compose", "-f", str(compose_file), *args], cwd=root, input_text=input_text, timeout=timeout)


def _compose_exec(
    compose_file: Path,
    service: str,
    command: list[str],
    *,
    root: Path,
    input_text: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    return _compose(compose_file, ["exec", "-T", service, *command], root=root, input_text=input_text, timeout=timeout)


def _compose_logs(compose_file: Path, service: str, *, root: Path, timeout: int = 30) -> dict[str, Any]:
    return _compose(compose_file, ["logs", "--no-color", service], root=root, timeout=timeout)


def _psql(compose_file: Path, database: str, sql: str, *, root: Path, timeout: int = 120) -> dict[str, Any]:
    return _compose_exec(
        compose_file,
        "postgres",
        ["psql", "-U", "postgres", "-d", database, "-v", "ON_ERROR_STOP=1", "-X", "-q", "-t", "-A"],
        root=root,
        input_text=sql,
        timeout=timeout,
    )


def _json_from_psql(result: dict[str, Any]) -> Any:
    if result["exit_code"] != 0:
        raise RuntimeError(result.get("stderr") or result.get("stdout") or "psql failed")
    for line in reversed(str(result.get("stdout", "")).splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return json.loads(candidate)
    return None


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _json_literal(value: Any) -> str:
    return _sql_literal(json.dumps(value, sort_keys=True, separators=(",", ":"))) + "::jsonb"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _audit_hash(event: dict[str, Any]) -> str:
    return _sha256_json(
        {
            "tenant_id": event["tenant_id"],
            "namespace_id": event["namespace_id"],
            "owner_id": event["owner_id"],
            "workspace_id": event["workspace_id"],
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "actor": event["actor"],
            "action": event["action"],
            "subject": event["subject"],
            "previous_hash": event["previous_hash"],
            "trace_id": event["trace_id"],
        }
    )


def _scoped_sql(tenant_id: str, body: str) -> str:
    return f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = '{tenant_id}';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'local-user';
SET LOCAL app.workspace_id = 'default';
{body}
COMMIT;
"""


def _bootstrap_roles_sql() -> str:
    blocks = []
    for role, statement in ROLE_DEFINITIONS.items():
        blocks.append(
            f"""
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
    {statement};
  END IF;
END $$;
"""
        )
    return "\n".join(blocks)


def _migration_sql(path: Path) -> str:
    text = path.read_text()
    if path.name == "0001_identity.sql":
        lines = [line for line in text.splitlines() if not line.lstrip().startswith("CREATE ROLE cornerstone_")]
        return "\n".join(lines) + "\n"
    return text


def _wait_for_stack(compose_file: Path, root: Path) -> dict[str, Any]:
    transcript = []
    postgres_ready = False
    opa_ready = False
    for _ in range(60):
        pg = _compose_exec(compose_file, "postgres", ["pg_isready", "-U", "postgres", "-d", "cornerstone"], root=root, timeout=10)
        transcript.append(pg)
        postgres_ready = pg["exit_code"] == 0
        opa = _compose_exec(compose_file, "egress-gateway", ["python", "-"], root=root, input_text=_opa_health_probe_script(), timeout=10)
        transcript.append(opa)
        opa_ready = opa["exit_code"] == 0 and _load_json_stdout(opa).get("health_status") == 200
        if postgres_ready and opa_ready:
            break
        time.sleep(0.5)
    return {
        "postgres_ready": postgres_ready,
        "opa_http_ready_inside_stack": opa_ready,
        "transcript": [_summarize(item) for item in transcript[-12:]],
    }


def _load_json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    stdout = str(result.get("stdout", "")).strip()
    if not stdout:
        return {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {}


def _opa_health_probe_script() -> str:
    return """
import json
import urllib.request

try:
    with urllib.request.urlopen("http://opa:8181/health", timeout=5) as response:
        print(json.dumps({"health_status": response.status, "body": response.read().decode()}))
except Exception as error:
    print(json.dumps({"health_status": 599, "error_class": type(error).__name__, "error": str(error)}))
    raise SystemExit(1)
"""


def _opa_decision_probe_script() -> str:
    sample_input = {
        "schema_version": "cs.policy_input.vs2.v1",
        "trace_id": "trace_vs2_production_like",
        "subject": {
            "principal_id": "principal_alice",
            "roles": ["owner"],
            "membership_revision": "memrev-alpha-001",
            "revoked": False,
        },
        "scope": {"tenant_id": "tenant_a", "namespace_id": "personal", "workspace_id": "default"},
        "resource": {
            "resource_id": "artifact_a",
            "tenant_id": "tenant_a",
            "namespace_id": "personal",
            "classification": "internal",
        },
        "action": "artifact.read",
        "risk": "low",
        "policy_path": "artifact.read",
        "mission_authority": {"mission_id": "mission_alpha", "authorized": True, "authority_ref": "authority_alpha_owner"},
        "data_scope": {"scope": "tenant", "purpose": "artifact_read"},
        "approval": {"required": False, "status": "not_required"},
        "capability": {"declared": True, "connectorhub_mediated": True},
        "environment": {"deployment": "local", "workspace_mode": "assist"},
    }
    cross_tenant = json.loads(json.dumps(sample_input))
    cross_tenant["resource"]["tenant_id"] = "tenant_b"
    invalid = {"schema_version": "bad"}
    return (
        """
import json
import urllib.request

cases = __CASES__
results = []
for name, payload in cases:
    request = urllib.request.Request(
        "http://opa:8181/v1/data/cornerstone/vs2/decision",
        data=json.dumps({"input": payload}, sort_keys=True).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = json.loads(response.read().decode())
        results.append({"case": name, "http_status": response.status, "result": body.get("result", {})})
print(json.dumps({"cases": results}, sort_keys=True))
"""
    ).replace("__CASES__", repr([("allow", sample_input), ("cross_tenant", cross_tenant), ("invalid", invalid)]))


def _controlled_sink_probe_script(run_id: str) -> str:
    sample_input = {
        "schema_version": "cs.policy_input.vs2.v1",
        "trace_id": f"trace_vs2_sink_{run_id}",
        "subject": {
            "principal_id": "principal_alice",
            "roles": ["owner"],
            "membership_revision": "memrev-alpha-001",
            "revoked": False,
        },
        "scope": {"tenant_id": "tenant_a", "namespace_id": "personal", "workspace_id": "default"},
        "resource": {
            "resource_id": "provider_sink",
            "tenant_id": "tenant_a",
            "namespace_id": "personal",
            "classification": "internal",
        },
        "action": "connector.egress",
        "risk": "low",
        "policy_path": "artifact.read",
        "mission_authority": {"mission_id": "mission_alpha", "authorized": True, "authority_ref": "authority_alpha_owner"},
        "data_scope": {"scope": "tenant", "purpose": "connector_egress_probe"},
        "approval": {"required": False, "status": "not_required"},
        "capability": {"declared": True, "connectorhub_mediated": True},
        "environment": {"deployment": "local", "workspace_mode": "assist"},
    }
    forbidden_input = json.loads(json.dumps(sample_input))
    forbidden_input["trace_id"] = f"trace_vs2_forbidden_sink_{run_id}"
    forbidden_input["resource"]["resource_id"] = "forbidden_sink"
    forbidden_input["resource"]["tenant_id"] = "tenant_b"
    provider_url = f"http://provider-sink:8080/?probe=provider_allowed_{run_id}"
    forbidden_url = f"http://forbidden-sink:8080/?probe=forbidden_denied_{run_id}"
    return (
        """
import json
import urllib.request

provider_url = __PROVIDER_URL__
forbidden_url = __FORBIDDEN_URL__
allow_input = __ALLOW_INPUT__
forbidden_input = __FORBIDDEN_INPUT__

def opa_decision(payload):
    request = urllib.request.Request(
        "http://opa:8181/v1/data/cornerstone/vs2/decision",
        data=json.dumps({"input": payload}, sort_keys=True).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = json.loads(response.read().decode())
        return {"http_status": response.status, "result": body.get("result", {})}

allow_decision = opa_decision(allow_input)
forbidden_decision = opa_decision(forbidden_input)
provider_request_sent = False
provider_status = None
provider_error = None
if allow_decision["result"].get("decision") == "allow":
    provider_request_sent = True
    try:
        with urllib.request.urlopen(provider_url, timeout=5) as response:
            provider_status = response.status
            response.read()
    except Exception as error:
        provider_error = {"class": type(error).__name__, "message": str(error)}

forbidden_request_sent = False
forbidden_status = None
forbidden_error = None
if forbidden_decision["result"].get("decision") == "allow":
    forbidden_request_sent = True
    try:
        with urllib.request.urlopen(forbidden_url, timeout=5) as response:
            forbidden_status = response.status
            response.read()
    except Exception as error:
        forbidden_error = {"class": type(error).__name__, "message": str(error)}

print(json.dumps({
    "provider_url": provider_url,
    "forbidden_url": forbidden_url,
    "allow_decision": allow_decision,
    "forbidden_decision": forbidden_decision,
    "provider_request_sent": provider_request_sent,
    "provider_status": provider_status,
    "provider_error": provider_error,
    "forbidden_request_sent": forbidden_request_sent,
    "forbidden_status": forbidden_status,
    "forbidden_error": forbidden_error,
}, sort_keys=True))
"""
    ).replace("__PROVIDER_URL__", repr(provider_url)).replace(
        "__FORBIDDEN_URL__", repr(forbidden_url)
    ).replace("__ALLOW_INPUT__", repr(sample_input)).replace("__FORBIDDEN_INPUT__", repr(forbidden_input))


def _network_probe_script(gateway_url: str) -> str:
    return f"""
import json
import urllib.request

result = {{"gateway_url": {gateway_url!r}}}
try:
    with urllib.request.urlopen({gateway_url!r}, timeout=5) as response:
        result["controlled_gateway_status"] = response.status
        result["controlled_gateway_reachable"] = True
except Exception as error:
    result["controlled_gateway_reachable"] = False
    result["controlled_gateway_error_class"] = type(error).__name__
try:
    urllib.request.urlopen("http://example.com", timeout=3)
    result["external_direct_reachable"] = True
except Exception as error:
    result["external_direct_reachable"] = False
    result["external_direct_error_class"] = type(error).__name__
print(json.dumps(result, sort_keys=True))
"""


def _apply_migrations(compose_file: Path, root: Path, database: str) -> dict[str, Any]:
    transcript = []
    role_bootstrap = _psql(compose_file, "postgres", _bootstrap_roles_sql(), root=root)
    transcript.append(role_bootstrap)
    create_database = _psql(compose_file, "postgres", f"CREATE DATABASE {database};", root=root)
    transcript.append(create_database)
    if role_bootstrap["exit_code"] != 0 or create_database["exit_code"] != 0:
        return {"status": "failed", "transcript": [_summarize(item) for item in transcript]}

    for migration in sorted((root / "migrations" / "vs2").glob("*.sql")):
        result = _psql(compose_file, database, _migration_sql(migration), root=root)
        transcript.append(result)
        if result["exit_code"] != 0:
            return {
                "status": "failed",
                "failed_migration": str(migration.relative_to(root)),
                "transcript": [_summarize(item) for item in transcript],
            }
    return {"status": "passed", "database": database, "transcript": [_summarize(item) for item in transcript]}


def _audit_seed_events(run_id: str) -> list[dict[str, Any]]:
    first = {
        "tenant_id": "tenant_a",
        "namespace_id": "personal",
        "owner_id": "local-user",
        "workspace_id": "default",
        "event_id": f"audit_{run_id}_001",
        "event_type": "artifact.ingested",
        "actor": "principal_alice",
        "action": "artifact.ingest",
        "subject": {"artifact_id": "artifacts_a", "source": "production_like_fixture"},
        "decision_id": "decision_allow_artifact_read",
        "policy_revision": "vs2-rego-local-v1",
        "evidence_refs": ["artifact:artifacts_a"],
        "previous_hash": "GENESIS",
        "trace_id": "trace_vs2_production_like",
    }
    first["event_hash"] = _audit_hash(first)
    second = {
        "tenant_id": "tenant_a",
        "namespace_id": "personal",
        "owner_id": "local-user",
        "workspace_id": "default",
        "event_id": f"audit_{run_id}_002",
        "event_type": "policy.decision",
        "actor": "cornerstone_policy",
        "action": "policy.evaluate",
        "subject": {"decision": "allow", "resource_id": "artifact_a"},
        "decision_id": "decision_allow_artifact_read",
        "policy_revision": "vs2-rego-local-v1",
        "evidence_refs": ["policy:decision_allow_artifact_read"],
        "previous_hash": first["event_hash"],
        "trace_id": "trace_vs2_production_like",
    }
    second["event_hash"] = _audit_hash(second)
    return [first, second]


def _event_insert_sql(event: dict[str, Any]) -> str:
    return f"""
INSERT INTO cs.audit_events (
  tenant_id, namespace_id, owner_id, workspace_id, event_id, event_type, actor, action,
  subject, decision_id, policy_revision, evidence_refs, previous_hash, event_hash, trace_id
) VALUES (
  {_sql_literal(event['tenant_id'])},
  {_sql_literal(event['namespace_id'])},
  {_sql_literal(event['owner_id'])},
  {_sql_literal(event['workspace_id'])},
  {_sql_literal(event['event_id'])},
  {_sql_literal(event['event_type'])},
  {_sql_literal(event['actor'])},
  {_sql_literal(event['action'])},
  {_json_literal(event['subject'])},
  {_sql_literal(event['decision_id'])},
  {_sql_literal(event['policy_revision'])},
  {_json_literal(event['evidence_refs'])},
  {_sql_literal(event['previous_hash'])},
  {_sql_literal(event['event_hash'])},
  {_sql_literal(event['trace_id'])}
);
"""


def _seed_database(compose_file: Path, root: Path, database: str, run_id: str) -> dict[str, Any]:
    tenant_sql = """
INSERT INTO cs.principals(principal_id, display_name) VALUES
  ('principal_alice', 'Alice Owner'),
  ('principal_bob', 'Bob Other')
ON CONFLICT DO NOTHING;
INSERT INTO cs.tenants(tenant_id, display_name) VALUES
  ('tenant_a', 'Tenant A'),
  ('tenant_b', 'Tenant B')
ON CONFLICT DO NOTHING;
INSERT INTO cs.memberships(membership_id, principal_id, tenant_id, namespace_id, workspace_id, owner_id, roles, membership_revision, session_version) VALUES
  ('membership_alice_a', 'principal_alice', 'tenant_a', 'personal', 'default', 'local-user', ARRAY['owner'], 'memrev-alpha-001', 1),
  ('membership_bob_b', 'principal_bob', 'tenant_b', 'personal', 'default', 'local-user', ARRAY['owner'], 'memrev-beta-001', 1)
ON CONFLICT DO NOTHING;
"""
    transcript = [_psql(compose_file, database, tenant_sql, root=root)]
    audit_events = _audit_seed_events(run_id)
    for tenant_id, suffix in [("tenant_a", "a"), ("tenant_b", "b")]:
        statements = []
        for table in CORE_RLS_TABLES:
            statements.append(
                f"""
INSERT INTO cs.{table}(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload, audit_ref)
VALUES ({_sql_literal(tenant_id)}, 'personal', 'local-user', 'default', {_sql_literal(f'{table}_{suffix}')}, 'internal', {_json_literal({'tenant': tenant_id, 'relation': table})}, {_sql_literal(f'audit:{table}_{suffix}')});
"""
            )
        statements.append(
            f"""
INSERT INTO cs.operator_metrics(tenant_id, namespace_id, owner_id, workspace_id, metric_id, metric_name, metric_value, labels, trace_id)
VALUES ({_sql_literal(tenant_id)}, 'personal', 'local-user', 'default', {_sql_literal(f'metric_{suffix}')}, 'artifact_visible_count', 1, {_json_literal({'tenant': tenant_id})}, 'trace_vs2_production_like');
INSERT INTO cs.status_records(tenant_id, namespace_id, owner_id, workspace_id, status_id, component, status, detail, trace_id)
VALUES ({_sql_literal(tenant_id)}, 'personal', 'local-user', 'default', {_sql_literal(f'status_{suffix}')}, 'postgres', 'ok', {_json_literal({'tenant': tenant_id})}, 'trace_vs2_production_like');
INSERT INTO cs.tenant_exports(tenant_id, namespace_id, owner_id, workspace_id, export_id, export_type, payload, row_count, payload_hash, trace_id)
VALUES ({_sql_literal(tenant_id)}, 'personal', 'local-user', 'default', {_sql_literal(f'export_{suffix}')}, 'tenant_scope', {_json_literal({'tenant': tenant_id})}, 1, {_sql_literal(_sha256_json({'tenant': tenant_id}))}, 'trace_vs2_production_like');
INSERT INTO cs.artifact_references(tenant_id, namespace_id, owner_id, workspace_id, reference_id, classification, source_artifact_id, target_artifact_id, payload, audit_ref)
VALUES ({_sql_literal(tenant_id)}, 'personal', 'local-user', 'default', {_sql_literal(f'ref_{suffix}')}, 'internal', {_sql_literal(f'artifacts_{suffix}')}, {_sql_literal(f'artifacts_{suffix}')}, {_json_literal({'tenant': tenant_id})}, {_sql_literal(f'audit:ref_{suffix}')});
"""
        )
        if tenant_id == "tenant_a":
            statements.extend(_event_insert_sql(event) for event in audit_events)
        transcript.append(_psql(compose_file, database, _scoped_sql(tenant_id, "\n".join(statements)), root=root))
    status = "passed" if all(item["exit_code"] == 0 for item in transcript) else "failed"
    return {"status": status, "audit_events": audit_events, "transcript": [_summarize(item) for item in transcript]}


def _relation_counts_sql() -> str:
    selects = []
    for relation, id_column in RELATION_ID_COLUMNS.items():
        selects.append(
            f"""
SELECT
  '{relation}' AS relation,
  count(*)::integer AS visible_count,
  count(*) FILTER (WHERE {id_column} LIKE '%_b' OR {id_column} IN ('ref_b', 'metric_b', 'status_b', 'export_b'))::integer AS foreign_visible_count,
  jsonb_agg({id_column} ORDER BY {id_column}) AS visible_ids
FROM cs.{relation}
""".strip()
        )
    return "\nUNION ALL\n".join(selects)


def _verify_rls(compose_file: Path, root: Path, database: str) -> dict[str, Any]:
    transcript = []
    relation_counts = _json_from_psql(
        _psql(
            compose_file,
            database,
            _scoped_sql(
                "tenant_a",
                f"""
SELECT jsonb_build_object(
  'relation_counts', (
    SELECT jsonb_agg(jsonb_build_object(
      'relation', relation,
      'visible_count', visible_count,
      'foreign_visible_count', foreign_visible_count,
      'visible_ids', visible_ids
    ) ORDER BY relation)
    FROM ({_relation_counts_sql()}) counts
  ),
  'visible_artifact_function_ids', (SELECT jsonb_agg(artifact_id ORDER BY artifact_id) FROM cs.visible_artifact_ids()),
  'safe_artifact_count_rows', (SELECT count(*) FROM cs.safe_artifact_counts),
  'resolved_membership_tenant', (SELECT tenant_id FROM cs.resolve_membership('principal_alice', 'membership_alice_a', 1))
)::text;
""",
            ),
            root=root,
        )
    )
    cross_delete = _json_from_psql(
        _psql(
            compose_file,
            database,
            _scoped_sql(
                "tenant_a",
                """
WITH deleted AS (
  DELETE FROM cs.artifacts WHERE tenant_id = 'tenant_b' RETURNING artifact_id
)
SELECT jsonb_build_object('deleted_count', (SELECT count(*) FROM deleted))::text;
""",
            ),
            root=root,
        )
    )
    forged_insert = _psql(
        compose_file,
        database,
        _scoped_sql(
            "tenant_a",
            "INSERT INTO cs.artifacts(tenant_id, namespace_id, owner_id, workspace_id, artifact_id, classification, payload) VALUES ('tenant_b','personal','local-user','default','forged','internal','{}');",
        ),
        root=root,
    )
    forged_update = _psql(
        compose_file,
        database,
        _scoped_sql("tenant_a", "UPDATE cs.artifacts SET tenant_id = 'tenant_b' WHERE artifact_id = 'artifacts_a';"),
        root=root,
    )
    unsafe_function = _psql(compose_file, database, _scoped_sql("tenant_a", "SELECT count(*) FROM cs.unsafe_all_artifacts();"), root=root)
    inventory = _json_from_psql(
        _psql(
            compose_file,
            database,
            """
SELECT jsonb_build_object(
  'roles', (
    SELECT jsonb_agg(jsonb_build_object('rolname', rolname, 'rolsuper', rolsuper, 'rolbypassrls', rolbypassrls) ORDER BY rolname)
    FROM pg_roles
    WHERE rolname IN ('cornerstone_app','cornerstone_schema_owner','cornerstone_identity','cornerstone_migrator','cornerstone_maintenance','cornerstone_auditor')
  ),
  'rls_tables', (
    SELECT jsonb_agg(jsonb_build_object('relation', c.relname, 'rls_enabled', c.relrowsecurity, 'rls_forced', c.relforcerowsecurity) ORDER BY c.relname)
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relkind = 'r' AND c.relname NOT IN ('principals','tenants')
  ),
  'policy_count', (SELECT count(*) FROM pg_policies WHERE schemaname = 'cs'),
  'unsafe_function_public_execute', has_function_privilege('public', 'cs.unsafe_all_artifacts()', 'EXECUTE')
)::text;
""",
            root=root,
        )
    )
    transcript.extend([forged_insert, forged_update, unsafe_function])
    relation_rows = relation_counts.get("relation_counts", [])
    checks = {
        "tenant_a_rows_visible": bool(relation_rows) and all(row["visible_count"] >= 1 for row in relation_rows),
        "tenant_b_rows_hidden": bool(relation_rows) and all(row["foreign_visible_count"] == 0 for row in relation_rows),
        "safe_view_is_rls_bound": relation_counts.get("safe_artifact_count_rows") == 1,
        "security_invoker_function_is_rls_bound": relation_counts.get("visible_artifact_function_ids") == ["artifacts_a"],
        "membership_resolved_by_security_definer": relation_counts.get("resolved_membership_tenant") == "tenant_a",
        "cross_tenant_delete_zero": cross_delete.get("deleted_count") == 0,
        "forged_cross_tenant_insert_denied": forged_insert["exit_code"] != 0,
        "forged_cross_tenant_update_denied": forged_update["exit_code"] != 0,
        "unsafe_security_definer_execute_denied": unsafe_function["exit_code"] != 0,
        "rls_enabled_and_forced": all(row["rls_enabled"] and row["rls_forced"] for row in inventory.get("rls_tables", [])),
        "app_role_cannot_bypass_rls": any(row["rolname"] == "cornerstone_app" and not row["rolsuper"] and not row["rolbypassrls"] for row in inventory.get("roles", [])),
        "policy_inventory_present": inventory.get("policy_count", 0) >= 17,
        "unsafe_function_not_public": inventory.get("unsafe_function_public_execute") is False,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "relation_counts": relation_counts,
        "cross_delete": cross_delete,
        "inventory": inventory,
        "checks": checks,
        "negative_transcript": [_summarize(item) for item in transcript],
    }


def _select_audit_events(compose_file: Path, root: Path, database: str) -> list[dict[str, Any]]:
    selected = _json_from_psql(
        _psql(
            compose_file,
            database,
            _scoped_sql(
                "tenant_a",
                """
SELECT jsonb_agg(jsonb_build_object(
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'event_id', event_id,
  'event_type', event_type,
  'actor', actor,
  'action', action,
  'subject', subject,
  'previous_hash', previous_hash,
  'event_hash', event_hash,
  'trace_id', trace_id
) ORDER BY event_id)::text
FROM cs.audit_events
WHERE event_id LIKE 'audit_%';
""",
            ),
            root=root,
        )
    )
    return selected or []


def _verify_audit(compose_file: Path, root: Path, database: str) -> dict[str, Any]:
    events = _select_audit_events(compose_file, root, database)
    chain_ok = all(_audit_hash(event) == event["event_hash"] for event in events)
    linked = len(events) >= 2 and events[1]["previous_hash"] == events[0]["event_hash"]
    tamper = _json_from_psql(
        _psql(
            compose_file,
            database,
            """
BEGIN;
UPDATE cs.audit_events
SET subject = '{"tampered":true}'::jsonb
WHERE event_id = (SELECT event_id FROM cs.audit_events WHERE tenant_id = 'tenant_a' ORDER BY event_id LIMIT 1);
SELECT jsonb_build_object(
  'event_id', event_id,
  'tenant_id', tenant_id,
  'namespace_id', namespace_id,
  'owner_id', owner_id,
  'workspace_id', workspace_id,
  'event_type', event_type,
  'actor', actor,
  'action', action,
  'subject', subject,
  'previous_hash', previous_hash,
  'event_hash', event_hash,
  'trace_id', trace_id
)::text
FROM cs.audit_events
WHERE tenant_id = 'tenant_a'
ORDER BY event_id
LIMIT 1;
ROLLBACK;
""",
            root=root,
        )
    )
    tamper_detected = _audit_hash(tamper) != tamper.get("event_hash")
    checks = {
        "audit_events_visible_in_scope": len(events) >= 2,
        "audit_hash_chain_valid": chain_ok,
        "audit_events_linked": linked,
        "controlled_tamper_detected_before_rollback": tamper_detected,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "event_count": len(events), "checks": checks}


def _verify_backup_restore(compose_file: Path, root: Path, database: str, restore_database: str, run_id: str) -> dict[str, Any]:
    backup_dir = root / "tmp" / "vs2-production-like" / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "backup.dump"
    dump = _run_bytes(
        ["docker", "compose", "-f", str(compose_file), "exec", "-T", "postgres", "pg_dump", "-U", "postgres", "-Fc", database],
        cwd=root,
        timeout=180,
    )
    if dump["exit_code"] == 0:
        backup_path.write_bytes(dump["stdout"])
    create_restore = _psql(compose_file, "postgres", f"CREATE DATABASE {restore_database};", root=root)
    restore = _run_bytes(
        ["docker", "compose", "-f", str(compose_file), "exec", "-T", "postgres", "pg_restore", "-U", "postgres", "-d", restore_database, "--no-owner"],
        cwd=root,
        input_bytes=dump["stdout"] if dump["exit_code"] == 0 else b"",
        timeout=180,
    )
    before_counts = _verify_rls(compose_file, root, database)
    after_counts = _verify_rls(compose_file, root, restore_database) if restore["exit_code"] == 0 else {"status": "skipped"}
    restore_audit = _verify_audit(compose_file, root, restore_database) if restore["exit_code"] == 0 else {"status": "skipped"}
    checks = {
        "pg_dump_succeeded": dump["exit_code"] == 0 and backup_path.exists() and backup_path.stat().st_size > 0,
        "pg_restore_succeeded": create_restore["exit_code"] == 0 and restore["exit_code"] == 0,
        "rls_rechecked_after_restore": after_counts.get("status") == "passed",
        "audit_rechecked_after_restore": restore_audit.get("status") == "passed",
        "tenant_counts_match_after_restore": before_counts.get("relation_counts", {}) == after_counts.get("relation_counts", {}),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "backup_path": str(backup_path.relative_to(root)),
        "backup_sha256": hashlib.sha256(backup_path.read_bytes()).hexdigest() if backup_path.exists() else None,
        "backup_bytes": backup_path.stat().st_size if backup_path.exists() else 0,
        "restore_database": restore_database,
        "checks": checks,
        "transcript": [_summarize(dump), _summarize(create_restore), _summarize(restore)],
    }


def _container_id(compose_file: Path, root: Path, service: str) -> str:
    result = _compose(compose_file, ["ps", "-q", service], root=root)
    if result["exit_code"] != 0 or not result["stdout"].strip():
        raise RuntimeError(f"cannot resolve container id for {service}")
    return result["stdout"].strip()


def _inspect_json(root: Path, object_id: str) -> dict[str, Any]:
    result = _run(["docker", "inspect", object_id], cwd=root)
    if result["exit_code"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"])
    return json.loads(result["stdout"])[0]


def _verify_network(compose_file: Path, root: Path) -> dict[str, Any]:
    services = ["postgres", "opa", "egress-gateway", "provider-sink", "forbidden-sink"]
    containers = {service: _container_id(compose_file, root, service) for service in services}
    inspected = {service: _inspect_json(root, container_id) for service, container_id in containers.items()}
    networks_by_service = {
        service: set(item["NetworkSettings"]["Networks"].keys())
        for service, item in inspected.items()
    }
    common_internal = sorted(networks_by_service["postgres"] & networks_by_service["opa"] & networks_by_service["egress-gateway"])
    internal_network = common_internal[0] if common_internal else ""
    external_networks = sorted(networks_by_service["egress-gateway"] - networks_by_service["postgres"] - networks_by_service["opa"])
    internal_inspect = _inspect_json(root, internal_network) if internal_network else {}
    egress_internal_ip = inspected["egress-gateway"]["NetworkSettings"]["Networks"][internal_network]["IPAddress"] if internal_network else ""
    port_bindings = inspected["postgres"].get("HostConfig", {}).get("PortBindings", {})
    host_socket = _host_socket_open("127.0.0.1", 55432)
    host_opa = _host_http_available("http://127.0.0.1:8181/health")
    probe = _run(
        ["docker", "run", "--rm", "-i", "--network", internal_network, "python:3.12-alpine", "python", "-"],
        cwd=root,
        input_text=_network_probe_script(f"http://{egress_internal_ip}:8080"),
        timeout=30,
    ) if internal_network else {"exit_code": 1, "stdout": "{}", "stderr": "internal network missing"}
    probe_payload = _load_json_stdout(probe)
    checks = {
        "internal_network_exists": bool(internal_network),
        "internal_network_is_docker_internal": internal_inspect.get("Internal") is True,
        "postgres_and_opa_only_internal": networks_by_service["postgres"] == {internal_network} and networks_by_service["opa"] == {internal_network},
        "egress_gateway_bridges_internal_to_external_test": bool(external_networks) and internal_network in networks_by_service["egress-gateway"],
        "provider_sink_external_only": networks_by_service["provider-sink"] == set(external_networks),
        "forbidden_sink_external_only": networks_by_service["forbidden-sink"] == set(external_networks),
        "postgres_host_bound_to_loopback": port_bindings.get("5432/tcp") == [{"HostIp": "127.0.0.1", "HostPort": "55432"}] and host_socket,
        "opa_not_published_to_host": not host_opa,
        "controlled_gateway_reachable_from_internal_probe": probe_payload.get("controlled_gateway_reachable") is True and probe_payload.get("controlled_gateway_status") == 200,
        "direct_external_from_internal_probe_denied": probe_payload.get("external_direct_reachable") is False,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "containers": containers,
        "networks_by_service": {key: sorted(value) for key, value in networks_by_service.items()},
        "internal_network": internal_network,
        "external_networks": external_networks,
        "postgres_port_bindings": port_bindings,
        "probe": probe_payload,
        "checks": checks,
        "probe_transcript": _summarize(probe),
    }


def _verify_controlled_sink_egress(compose_file: Path, root: Path, run_id: str) -> dict[str, Any]:
    probe = _compose_exec(
        compose_file,
        "egress-gateway",
        ["python", "-"],
        root=root,
        input_text=_controlled_sink_probe_script(run_id),
        timeout=30,
    )
    payload = _load_json_stdout(probe)
    provider_logs = _compose_logs(compose_file, "provider-sink", root=root)
    forbidden_logs = _compose_logs(compose_file, "forbidden-sink", root=root)
    provider_log_text = str(provider_logs.get("stdout", ""))
    forbidden_log_text = str(forbidden_logs.get("stdout", ""))
    provider_token = f"provider_allowed_{run_id}"
    forbidden_token = f"forbidden_denied_{run_id}"
    allow_decision = payload.get("allow_decision", {}).get("result", {})
    forbidden_decision = payload.get("forbidden_decision", {}).get("result", {})
    checks = {
        "provider_sink_service_reached_once": payload.get("provider_request_sent") is True
        and payload.get("provider_status") == 200
        and provider_log_text.count(provider_token) == 1,
        "forbidden_sink_denied_by_policy": forbidden_decision.get("decision") == "deny"
        and "cross_tenant_scope" in forbidden_decision.get("reason_codes", []),
        "forbidden_sink_not_contacted": payload.get("forbidden_request_sent") is False
        and forbidden_token not in forbidden_log_text,
        "opa_allow_decision_recorded": allow_decision.get("decision") == "allow",
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "probe": payload,
        "log_evidence": {
            "provider_sink_token": provider_token,
            "provider_sink_token_count": provider_log_text.count(provider_token),
            "forbidden_sink_token": forbidden_token,
            "forbidden_sink_token_count": forbidden_log_text.count(forbidden_token),
            "provider_sink_log_tail": provider_log_text.splitlines()[-8:],
            "forbidden_sink_log_tail": forbidden_log_text.splitlines()[-8:],
        },
        "transcripts": {
            "probe": _summarize(probe),
            "provider_logs": _summarize(provider_logs),
            "forbidden_logs": _summarize(forbidden_logs),
        },
    }


def _host_socket_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _host_http_available(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _verify_opa(compose_file: Path, root: Path) -> dict[str, Any]:
    decision = _compose_exec(compose_file, "egress-gateway", ["python", "-"], root=root, input_text=_opa_decision_probe_script(), timeout=30)
    payload = _load_json_stdout(decision)
    cases = {item.get("case"): item.get("result", {}) for item in payload.get("cases", [])}
    checks = {
        "opa_http_decision_api_reachable_inside_stack": decision["exit_code"] == 0,
        "allow_case_allowed": cases.get("allow", {}).get("decision") == "allow",
        "cross_tenant_case_denied": cases.get("cross_tenant", {}).get("decision") == "deny"
        and "cross_tenant_scope" in cases.get("cross_tenant", {}).get("reason_codes", []),
        "invalid_schema_denied": cases.get("invalid", {}).get("decision") == "deny"
        and "invalid_schema" in cases.get("invalid", {}).get("reason_codes", []),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "cases": payload.get("cases", []),
        "transcript": _summarize(decision),
    }


def _overclaim_checks() -> dict[str, Any]:
    checks = {
        "local_production_like_only": True,
        "production_release_ready": False,
        "live_connector_ready": False,
        "human_security_acceptance": False,
        "external_provider_mutation": False,
    }
    return {
        "status": "passed",
        "claim_boundary": "Local production-like Docker integration rehearsal only. This is not production readiness, live connector readiness, penetration-test completion, or human acceptance.",
        "checks": checks,
        "human_required": [
            {
                "id": "VS2-SEC-H04",
                "why_ai_cannot_verify": "Real production topology, network policy ownership, and independent security review require human or external evidence.",
                "required_human_action": "Run or approve the production/network security review and attach redacted evidence.",
                "expected_evidence": "Reviewed topology, policy decision logs, network-control transcript, backup/restore drill, audit proof, and sign-off.",
                "release_impact": "Blocks production PASS, but not this local production-like rehearsal.",
            }
        ],
    }


def run_vs2_production_like_integration(root: Path, *, compose_file: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    compose = (compose_file or root / "compose.vs2.yml").resolve()
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    database = f"cornerstone_plike_{run_id.lower()}".replace("-", "_")
    restore_database = f"{database}_restore"

    scenario_contract = [
        {
            "id": "PLIKE-001",
            "type": "MUST_PASS",
            "expected": "Production-like local stack starts with real Postgres, OPA, and egress gateway services.",
            "verification": "docker compose up, service readiness probes, Docker network inspection.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-002",
            "type": "MUST_PASS",
            "expected": "Real VS2 migrations apply to a fresh Postgres database using hardened roles.",
            "verification": "psql migration transcript through compose Postgres.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-003",
            "type": "MUST_PASS",
            "expected": "Postgres RLS hides tenant B and denies forged cross-tenant writes for cornerstone_app.",
            "verification": "SQL checks through SET LOCAL ROLE cornerstone_app and tenant session settings.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-004",
            "type": "MUST_PASS",
            "expected": "OPA HTTP decision API allows valid owner input and denies cross-tenant and invalid-schema input.",
            "verification": "Internal stack HTTP POSTs from egress-gateway to opa:8181.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-005",
            "type": "MUST_PASS",
            "expected": "Docker internal network blocks direct external egress while controlled egress gateway remains reachable.",
            "verification": "Ephemeral internal-network probe container attempts controlled gateway and external URL.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-006",
            "type": "MUST_PASS",
            "expected": "Controlled local sink rehearsal proves egress-gateway can reach only the policy-allowed provider sink and does not contact the forbidden sink.",
            "verification": "OPA-mediated egress-gateway probe plus provider-sink and forbidden-sink Docker logs.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-007",
            "type": "MUST_PASS",
            "expected": "pg_dump/pg_restore preserve tenant-scoped counts, RLS checks, and audit verification.",
            "verification": "Dump fresh database, restore into fresh database, rerun RLS and audit checks.",
            "owner": "AI",
        },
        {
            "id": "PLIKE-R01",
            "type": "REGRESSION_GUARD",
            "expected": "Local rehearsal does not claim production readiness, live-provider readiness, or human acceptance.",
            "verification": "Report claim-boundary fields and human-required row.",
            "owner": "AI",
        },
    ]

    compose_up = _compose(compose, ["up", "-d"], root=root, timeout=180)
    stack = _wait_for_stack(compose, root) if compose_up["exit_code"] == 0 else {"postgres_ready": False, "opa_http_ready_inside_stack": False}
    migrations = _apply_migrations(compose, root, database) if stack.get("postgres_ready") else {"status": "skipped"}
    seed = _seed_database(compose, root, database, run_id) if migrations.get("status") == "passed" else {"status": "skipped"}
    rls = _verify_rls(compose, root, database) if seed.get("status") == "passed" else {"status": "skipped"}
    audit = _verify_audit(compose, root, database) if seed.get("status") == "passed" else {"status": "skipped"}
    opa = _verify_opa(compose, root) if stack.get("opa_http_ready_inside_stack") else {"status": "skipped"}
    network = _verify_network(compose, root) if compose_up["exit_code"] == 0 else {"status": "skipped"}
    controlled_sink_egress = (
        _verify_controlled_sink_egress(compose, root, run_id)
        if compose_up["exit_code"] == 0 and stack.get("opa_http_ready_inside_stack") and network.get("status") == "passed"
        else {"status": "skipped"}
    )
    backup_restore = (
        _verify_backup_restore(compose, root, database, restore_database, run_id)
        if seed.get("status") == "passed" and rls.get("status") == "passed" and audit.get("status") == "passed"
        else {"status": "skipped"}
    )
    overclaim = _overclaim_checks()

    scenario_results = [
        {
            "id": "PLIKE-001",
            "status": "PASS" if compose_up["exit_code"] == 0 and stack.get("postgres_ready") and stack.get("opa_http_ready_inside_stack") and network.get("status") == "passed" else "FAIL",
            "evidence": ["docker compose -f compose.vs2.yml up -d", "docker inspect networks", "internal OPA health probe"],
        },
        {
            "id": "PLIKE-002",
            "status": "PASS" if migrations.get("status") == "passed" and seed.get("status") == "passed" else "FAIL",
            "evidence": [f"database:{database}", "migrations/vs2/*.sql"],
        },
        {
            "id": "PLIKE-003",
            "status": "PASS" if rls.get("status") == "passed" else "FAIL",
            "evidence": ["SET LOCAL ROLE cornerstone_app SQL transcript", "pg_policies inventory"],
        },
        {
            "id": "PLIKE-004",
            "status": "PASS" if opa.get("status") == "passed" else "FAIL",
            "evidence": ["POST /v1/data/cornerstone/vs2/decision from egress-gateway"],
        },
        {
            "id": "PLIKE-005",
            "status": "PASS" if network.get("status") == "passed" else "FAIL",
            "evidence": ["Docker internal network inspect", "ephemeral internal-network probe"],
        },
        {
            "id": "PLIKE-006",
            "status": "PASS" if controlled_sink_egress.get("status") == "passed" else "FAIL",
            "evidence": ["egress-gateway OPA-mediated sink probe", "provider-sink logs", "forbidden-sink logs"],
        },
        {
            "id": "PLIKE-007",
            "status": "PASS" if backup_restore.get("status") == "passed" else "FAIL",
            "evidence": [backup_restore.get("backup_path", ""), f"restore_database:{restore_database}"],
        },
        {
            "id": "PLIKE-R01",
            "status": "PASS" if overclaim.get("status") == "passed" else "FAIL",
            "evidence": ["claim_boundary", "human_required"],
        },
    ]
    passed = all(row["status"] == "PASS" for row in scenario_results)
    finished_at = datetime.now(UTC)
    report = {
        "schema_version": "cs.vs2.production_like_integration.v1",
        "status": "passed" if passed else "failed",
        "product": "CornerStone",
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round((finished_at - started_at).total_seconds(), 3),
        "compose_file": str(compose.relative_to(root)),
        "database": database,
        "restore_database": restore_database,
        "claim_boundary": overclaim["claim_boundary"],
        "scenario_contract": scenario_contract,
        "scenario_results": scenario_results,
        "compose_up": _summarize(compose_up),
        "stack": stack,
        "migrations": migrations,
        "seed": seed,
        "rls": rls,
        "audit": audit,
        "opa": opa,
        "network": network,
        "controlled_sink_egress": controlled_sink_egress,
        "backup_restore": backup_restore,
        "overclaim": overclaim,
        "stack_left_running": True,
    }
    output = root / REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(REPORT_PATH)
    report["report_payload_sha256_without_hash"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
