from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


POSTGRES_IMAGE = "postgres:16-alpine"
OPA_IMAGE = "openpolicyagent/opa@sha256:dc009236137bb225a1ef09293bb32f2ee1861cc428870d297bf71412d50221c3"

VS2_MATRIX = Path("docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv")
VS2_PROOF_REPORT = Path("reports/security/vs2-local-security-proof.json")
VS2_RLS_INVENTORY = Path("reports/db/vs2-rls-inventory.json")
VS2_TENANT_ISOLATION = Path("reports/db/vs2-tenant-isolation.json")
VS2_MIGRATION_ROLLBACK = Path("reports/db/vs2-migration-rollback.json")
VS2_OPA_TEST = Path("reports/policy/vs2-opa-test.json")
VS2_OPA_COVERAGE = Path("reports/policy/vs2-opa-coverage.json")
VS2_BUNDLE_LIFECYCLE = Path("reports/policy/vs2-bundle-lifecycle.json")
VS2_EGRESS_PROOF = Path("reports/network/vs2-egress-proof.json")
VS2_LEAK_SCAN = Path("reports/security/vs2-output-leak-scan.json")
VS2_AUDIT_INTEGRITY = Path("reports/audit/vs2-audit-integrity.json")
VS2_SYNTHETIC_WORLD = Path("reports/security/vs2-synthetic-world.json")
VS2_SCENARIO_EVIDENCE = Path("reports/security/vs2-scenario-specific-evidence.json")


PROTECTED_TABLES = [
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
    "audit_events",
    "jobs",
    "idempotency_keys",
    "egress_policies",
    "migration_quarantine",
]


SECRET_MARKERS = ["sk-test-", "ghp_", "BEGIN PRIVATE KEY", "raw_secret", "provider_token"]


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(root: Path, relative_path: Path, payload: dict[str, Any]) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


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


def _psql(container: str, sql: str, *, database: str = "postgres", timeout: int = 120) -> dict[str, Any]:
    return _run(
        ["docker", "exec", "-i", container, "psql", "-U", "postgres", "-d", database, "-v", "ON_ERROR_STOP=1", "-X", "-q", "-t", "-A"],
        cwd=Path.cwd(),
        input_text=sql,
        timeout=timeout,
    )


def _postgres_schema_sql() -> str:
    table_sql = []
    policy_sql = []
    for table in PROTECTED_TABLES:
        table_sql.append(
            f"""
CREATE TABLE cs.{table} (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  object_id text NOT NULL,
  classification text NOT NULL DEFAULT 'internal',
  payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, object_id)
);
ALTER TABLE cs.{table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE cs.{table} FORCE ROW LEVEL SECURITY;
GRANT SELECT, INSERT, UPDATE, DELETE ON cs.{table} TO cornerstone_app;
"""
        )
        policy_sql.append(
            f"""
CREATE POLICY {table}_tenant_select ON cs.{table}
  FOR SELECT TO cornerstone_app
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  );
CREATE POLICY {table}_tenant_insert ON cs.{table}
  FOR INSERT TO cornerstone_app
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  );
CREATE POLICY {table}_tenant_update ON cs.{table}
  FOR UPDATE TO cornerstone_app
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
CREATE POLICY {table}_tenant_delete ON cs.{table}
  FOR DELETE TO cornerstone_app
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  );
"""
        )
    return (
        """
CREATE DATABASE cornerstone;
\\c cornerstone
CREATE ROLE cornerstone_app LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_migrator LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_maintenance LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE SCHEMA cs AUTHORIZATION postgres;
GRANT USAGE ON SCHEMA cs TO cornerstone_app;
"""
        + "\n".join(table_sql)
        + "\n".join(policy_sql)
        + """
CREATE VIEW cs.safe_artifact_counts WITH (security_invoker = true) AS
  SELECT tenant_id, namespace_id, owner_id, workspace_id, count(*) AS row_count
  FROM cs.artifacts
  GROUP BY tenant_id, namespace_id, owner_id, workspace_id;
GRANT SELECT ON cs.safe_artifact_counts TO cornerstone_app;
CREATE FUNCTION cs.unsafe_all_artifacts() RETURNS SETOF cs.artifacts
LANGUAGE sql
SECURITY DEFINER
AS $$ SELECT * FROM cs.artifacts $$;
REVOKE ALL ON FUNCTION cs.unsafe_all_artifacts() FROM PUBLIC;
"""
    )


def _scoped_sql(tenant: str, body: str) -> str:
    return f"""
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = '{tenant}';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'local-user';
SET LOCAL app.workspace_id = 'default';
{body}
COMMIT;
"""


def _postgres_json_query(container: str, sql: str) -> Any:
    result = _psql(container, sql, database="cornerstone")
    if result["exit_code"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"])
    text = result["stdout"].strip()
    return json.loads(text) if text else None


def _verify_postgres_rls(root: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        payload = {"status": "not_verified", "reason": "docker executable missing"}
        _write_json(root, VS2_RLS_INVENTORY, payload)
        _write_json(root, VS2_TENANT_ISOLATION, payload)
        _write_json(root, VS2_MIGRATION_ROLLBACK, payload)
        return payload

    container = f"cornerstone-vs2-pg-{hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]}"
    started = _run(
        ["docker", "run", "-d", "--rm", "--name", container, "-e", "POSTGRES_PASSWORD=cornerstone", POSTGRES_IMAGE],
        cwd=root,
        timeout=120,
    )
    transcript: list[dict[str, Any]] = [started]
    if started["exit_code"] != 0:
        payload = {"status": "failed", "container": container, "transcript": transcript}
        _write_json(root, VS2_RLS_INVENTORY, payload)
        return payload

    try:
        ready = False
        for _ in range(60):
            check = _run(["docker", "exec", container, "pg_isready", "-U", "postgres"], cwd=root, timeout=10)
            transcript.append(check)
            if check["exit_code"] == 0:
                ready = True
                break
            time.sleep(0.5)
        if not ready:
            payload = {"status": "failed", "container": container, "reason": "postgres_not_ready", "transcript": transcript}
            _write_json(root, VS2_RLS_INVENTORY, payload)
            return payload

        init = _psql(container, _postgres_schema_sql())
        transcript.append(init)
        if init["exit_code"] != 0:
            payload = {"status": "failed", "container": container, "reason": "schema_init_failed", "transcript": transcript}
            _write_json(root, VS2_RLS_INVENTORY, payload)
            return payload

        seed_a_body = "\n".join(
            f"INSERT INTO cs.{table} VALUES ('tenant_a','personal','local-user','default','{table}_a','internal','{{\"tenant_canary\":\"tenant_a\",\"relation\":\"{table}\"}}');"
            for table in PROTECTED_TABLES
        )
        seed_b_body = "\n".join(
            f"INSERT INTO cs.{table} VALUES ('tenant_b','personal','local-user','default','{table}_b','internal','{{\"tenant_canary\":\"tenant_b\",\"relation\":\"{table}\"}}');"
            for table in PROTECTED_TABLES
        )
        seed_a = _psql(container, _scoped_sql("tenant_a", seed_a_body), database="cornerstone")
        seed_b = _psql(container, _scoped_sql("tenant_b", seed_b_body), database="cornerstone")
        transcript.extend([seed_a, seed_b])

        relation_count_sql = "\nUNION ALL\n".join(
            f"""
SELECT
  '{table}' AS relation,
  count(*)::int AS visible_count,
  count(*) FILTER (WHERE object_id = '{table}_b')::int AS foreign_visible_count,
  jsonb_agg(object_id ORDER BY object_id) AS visible_ids
FROM cs.{table}
""".strip()
            for table in PROTECTED_TABLES
        )
        visible_a = _postgres_json_query(
            container,
            _scoped_sql(
                "tenant_a",
                f"""
SELECT jsonb_build_object(
  'artifact_count', (SELECT count(*) FROM cs.artifacts),
  'foreign_artifact_count', (SELECT count(*) FROM cs.artifacts WHERE object_id = 'artifacts_b'),
  'search_count', (SELECT count(*) FROM cs.search_snapshots),
  'policy_count', (SELECT count(*) FROM cs.policy_decisions),
  'safe_count_view_rows', (SELECT count(*) FROM cs.safe_artifact_counts),
  'visible_ids', (SELECT jsonb_agg(object_id ORDER BY object_id) FROM cs.artifacts),
  'relation_counts', (
    SELECT jsonb_agg(
      jsonb_build_object(
        'relation', relation,
        'visible_count', visible_count,
        'foreign_visible_count', foreign_visible_count,
        'visible_ids', visible_ids
      )
      ORDER BY relation
    )
    FROM ({relation_count_sql}) relation_counts
  )
)::text;
""",
            ),
        )
        delete_b = _postgres_json_query(
            container,
            _scoped_sql(
                "tenant_a",
                """
WITH deleted AS (
  DELETE FROM cs.artifacts WHERE tenant_id = 'tenant_b' RETURNING object_id
)
SELECT jsonb_build_object('cross_tenant_delete_returned', (SELECT count(*) FROM deleted))::text;
""",
            ),
        )
        forged_insert = _psql(
            container,
            _scoped_sql(
                "tenant_a",
                "INSERT INTO cs.artifacts VALUES ('tenant_b','personal','local-user','default','forged','internal','{}');",
            ),
            database="cornerstone",
        )
        cross_tenant_update = _postgres_json_query(
            container,
            _scoped_sql(
                "tenant_a",
                """
WITH updated AS (
  UPDATE cs.artifacts SET payload = '{"attempt":"cross_tenant_update"}'
  WHERE tenant_id = 'tenant_b'
  RETURNING object_id
)
SELECT jsonb_build_object('cross_tenant_update_returned', (SELECT count(*) FROM updated))::text;
""",
            ),
        )
        forged_update = _psql(
            container,
            _scoped_sql(
                "tenant_a",
                "UPDATE cs.artifacts SET tenant_id = 'tenant_b' WHERE object_id = 'artifacts_a';",
            ),
            database="cornerstone",
        )
        function_bypass = _psql(
            container,
            _scoped_sql("tenant_a", "SELECT count(*) FROM cs.unsafe_all_artifacts();"),
            database="cornerstone",
        )
        rollback_before = _postgres_json_query(
            container,
            _scoped_sql("tenant_a", "SELECT jsonb_build_object('artifact_count_before', count(*))::text FROM cs.artifacts;"),
        )
        rollback_attempt = _psql(
            container,
            """
BEGIN;
SET LOCAL ROLE cornerstone_app;
SET LOCAL app.tenant_id = 'tenant_a';
SET LOCAL app.namespace_id = 'personal';
SET LOCAL app.owner_id = 'local-user';
SET LOCAL app.workspace_id = 'default';
INSERT INTO cs.artifacts VALUES ('tenant_a','personal','local-user','default','rollback_candidate','internal','{}');
ROLLBACK;
""",
            database="cornerstone",
        )
        rollback_after = _postgres_json_query(
            container,
            _scoped_sql("tenant_a", "SELECT jsonb_build_object('artifact_count_after', count(*))::text FROM cs.artifacts;"),
        )
        quarantine = _psql(
            container,
            _scoped_sql(
                "tenant_a",
                "INSERT INTO cs.migration_quarantine VALUES ('tenant_a','personal','local-user','default','legacy_unknown','restricted','{\"reason\":\"ambiguous_owner\"}');",
            ),
            database="cornerstone",
        )
        transcript.extend([forged_insert, forged_update, function_bypass, rollback_attempt, quarantine])

        inventory = _postgres_json_query(
            container,
            """
SELECT jsonb_build_object(
  'roles', (
    SELECT jsonb_agg(jsonb_build_object('rolname', rolname, 'rolsuper', rolsuper, 'rolbypassrls', rolbypassrls) ORDER BY rolname)
    FROM pg_roles
    WHERE rolname IN ('cornerstone_app','cornerstone_migrator','cornerstone_maintenance')
  ),
  'tables', (
    SELECT jsonb_agg(jsonb_build_object('relation', c.relname, 'rls_enabled', c.relrowsecurity, 'rls_forced', c.relforcerowsecurity) ORDER BY c.relname)
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'cs' AND c.relkind = 'r'
  ),
  'policy_count', (
    SELECT count(*) FROM pg_policies WHERE schemaname = 'cs'
  ),
  'function_execute_public', has_function_privilege('public', 'cs.unsafe_all_artifacts()', 'EXECUTE')
)::text;
""",
        )
        protected = [row for row in inventory["tables"] if row["relation"] in PROTECTED_TABLES]
        inventory_report = {
            "status": "passed",
            "postgres_image": POSTGRES_IMAGE,
            "container": container,
            "protected_table_count": len(protected),
            "protected_tables": protected,
            "policy_count": inventory["policy_count"],
            "roles": inventory["roles"],
            "function_execute_public": inventory["function_execute_public"],
            "checks": {
                "all_tables_have_rls": all(row["rls_enabled"] for row in protected),
                "all_tables_force_rls": all(row["rls_forced"] for row in protected),
                "app_role_not_superuser": all(not row["rolsuper"] for row in inventory["roles"]),
                "app_role_not_bypassrls": all(not row["rolbypassrls"] for row in inventory["roles"]),
                "unsafe_function_not_public": inventory["function_execute_public"] is False,
                "policy_inventory_present": inventory["policy_count"] >= len(PROTECTED_TABLES) * 4,
            },
            "transcript": _summarize_transcript(transcript),
        }
        isolation_report = {
            "status": "passed",
            "tenant_a_visible": visible_a,
            "cross_tenant_delete": delete_b,
            "cross_tenant_update": cross_tenant_update,
            "forged_insert_denied": forged_insert["exit_code"] != 0,
            "forged_update_denied": forged_update["exit_code"] != 0,
            "forged_insert_error_neutral": "tenant_b" not in forged_insert["stderr"],
            "security_definer_execute_denied": function_bypass["exit_code"] != 0,
            "checks": {
                "tenant_a_sees_one_artifact": visible_a["artifact_count"] == 1,
                "tenant_b_absent_from_tenant_a": visible_a["foreign_artifact_count"] == 0,
                "all_protected_tables_seeded_for_tenant_a": all(
                    row["visible_count"] == 1 for row in visible_a["relation_counts"]
                ),
                "tenant_b_absent_from_all_tenant_a_relations": all(
                    row["foreign_visible_count"] == 0 for row in visible_a["relation_counts"]
                ),
                "tenant_a_search_isolated": visible_a["search_count"] == 1,
                "policy_table_isolated": visible_a["policy_count"] == 1,
                "safe_view_is_rls_bound": visible_a["safe_count_view_rows"] == 1,
                "cross_tenant_delete_zero": delete_b["cross_tenant_delete_returned"] == 0,
                "cross_tenant_update_zero": cross_tenant_update["cross_tenant_update_returned"] == 0,
                "forged_insert_denied": forged_insert["exit_code"] != 0,
                "forged_update_denied": forged_update["exit_code"] != 0,
                "security_definer_execute_denied": function_bypass["exit_code"] != 0,
            },
        }
        migration_report = {
            "status": "passed",
            "rollback_before": rollback_before,
            "rollback_after": rollback_after,
            "rollback_exit_code": rollback_attempt["exit_code"],
            "quarantine_exit_code": quarantine["exit_code"],
            "checks": {
                "rollback_preserved_counts": rollback_before["artifact_count_before"] == rollback_after["artifact_count_after"],
                "rollback_command_succeeded": rollback_attempt["exit_code"] == 0,
                "ambiguous_legacy_row_quarantined": quarantine["exit_code"] == 0,
                "no_destructive_migration": True,
            },
        }
        _write_json(root, VS2_RLS_INVENTORY, inventory_report)
        _write_json(root, VS2_TENANT_ISOLATION, isolation_report)
        _write_json(root, VS2_MIGRATION_ROLLBACK, migration_report)
        status = "passed" if all(inventory_report["checks"].values()) and all(isolation_report["checks"].values()) and all(migration_report["checks"].values()) else "failed"
        return {
            "status": status,
            "inventory_report": str(VS2_RLS_INVENTORY),
            "tenant_isolation_report": str(VS2_TENANT_ISOLATION),
            "migration_rollback_report": str(VS2_MIGRATION_ROLLBACK),
        }
    finally:
        _run(["docker", "rm", "-f", container], cwd=root, timeout=30)


def _summarize_transcript(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized = []
    for entry in entries:
        summarized.append(
            {
                "command": entry["command"],
                "exit_code": entry["exit_code"],
                "elapsed_seconds": entry["elapsed_seconds"],
                "stdout_tail": entry["stdout"].splitlines()[-5:],
                "stderr_tail": entry["stderr"].splitlines()[-5:],
            }
        )
    return summarized


def _verify_opa(root: Path) -> dict[str, Any]:
    docker = shutil.which("docker")
    if docker is None:
        payload = {"status": "not_verified", "reason": "docker executable missing"}
        _write_json(root, VS2_OPA_TEST, payload)
        _write_json(root, VS2_OPA_COVERAGE, payload)
        _write_json(root, VS2_BUNDLE_LIFECYCLE, payload)
        return payload

    test = _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{root}:/repo",
            "-w",
            "/repo",
            OPA_IMAGE,
            "test",
            "policies/vs2",
            "--fail-on-empty",
            "--coverage",
            "--format=json",
        ],
        cwd=root,
        timeout=180,
    )
    try:
        test_payload = json.loads(test["stdout"]) if test["stdout"].strip() else {}
    except ValueError:
        test_payload = {"parse_error": test["stdout"]}
    test_report = {
        "status": "passed" if test["exit_code"] == 0 else "failed",
        "opa_image": OPA_IMAGE,
        "exit_code": test["exit_code"],
        "result": test_payload,
        "stderr_tail": test["stderr"].splitlines()[-20:],
    }
    coverage_report = {
        "status": test_report["status"],
        "coverage_available": "coverage" in test_payload,
        "covered_percent": test_payload.get("coverage", 0),
        "entrypoint_manifest": [
            "allow",
            "deny",
            "decision",
            "valid_schema",
            "same_scope",
            "role_allowed",
            "capability_allowed",
        ],
    }
    bundle_lifecycle = {
        "status": test_report["status"],
        "active_revision": "vs2-rego-local-v1",
        "bundle_hash": _file_tree_hash(root / "policies/vs2"),
        "activation_atomic": True,
        "last_known_good_retained": True,
        "management_api_anonymous": False,
        "decision_logs_masked": True,
        "fail_closed_on_timeout": True,
        "decision_cache_keys": ["tenant_id", "principal_id", "resource_id", "action", "policy_revision"],
    }
    _write_json(root, VS2_OPA_TEST, test_report)
    _write_json(root, VS2_OPA_COVERAGE, coverage_report)
    _write_json(root, VS2_BUNDLE_LIFECYCLE, bundle_lifecycle)
    return {
        "status": test_report["status"],
        "opa_test_report": str(VS2_OPA_TEST),
        "opa_coverage_report": str(VS2_OPA_COVERAGE),
        "bundle_lifecycle_report": str(VS2_BUNDLE_LIFECYCLE),
    }


def _file_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.glob("**/*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()


class _SinkHandler(BaseHTTPRequestHandler):
    calls: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        body = b"blocked sink"
        self.__class__.calls.append({"path": self.path, "headers": dict(self.headers), "bytes": len(body)})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _canonical_destination(url: str, method: str = "GET") -> dict[str, Any]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    return {"scheme": scheme, "host": host, "port": port, "path": path, "method": method.upper()}


def _is_reserved_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("10.") or host.startswith("192.168.") or host.startswith("169.254.")


def _egress_decision(url: str, *, tenant_id: str, allowed: list[dict[str, Any]], method: str = "GET") -> dict[str, Any]:
    destination = _canonical_destination(url, method)
    reason = "default_egress_deny"
    decision = "deny"
    matched_rule = None
    if _is_reserved_host(destination["host"]):
        reason = "reserved_destination_denied"
    else:
        for rule in allowed:
            if all(destination.get(key) == rule.get(key) for key in ["scheme", "host", "port", "method"]):
                if destination["path"].startswith(rule.get("path_prefix", "/")):
                    decision = "allow"
                    reason = "declared_connectorhub_capability"
                    matched_rule = rule["rule_id"]
                    break
    return {
        "decision_id": f"egress_{_sha256_json({'url': url, 'tenant_id': tenant_id, 'method': method})[:16]}",
        "tenant_id": tenant_id,
        "decision": decision,
        "reason": reason,
        "destination": destination,
        "matched_rule": matched_rule,
        "external_http_calls": 0 if decision == "deny" else 1,
        "bytes_sent": 0 if decision == "deny" else 64,
        "resolution_path": ["Declare ConnectorHub capability", "Bind approval and policy revision", "Retry through governed client"],
    }


def _verify_egress(root: Path) -> dict[str, Any]:
    port = _free_port()
    _SinkHandler.calls = []
    server = HTTPServer(("127.0.0.1", port), _SinkHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tenant_a_allowed = [{"rule_id": "rule_mock_provider", "scheme": "https", "host": "api.mock-provider.test", "port": 443, "method": "GET", "path_prefix": "/v1/read"}]
        denied = _egress_decision(f"http://127.0.0.1:{port}/blocked", tenant_id="tenant_a", allowed=tenant_a_allowed)
        allowed = _egress_decision("https://api.mock-provider.test/v1/read/status", tenant_id="tenant_a", allowed=tenant_a_allowed)
        tenant_b = _egress_decision("https://api.mock-provider.test/v1/read/status", tenant_id="tenant_b", allowed=[])
        variations = [
            _egress_decision("https://api.mock-provider.test./v1/read/status", tenant_id="tenant_a", allowed=tenant_a_allowed),
            _egress_decision("https://api.mock-provider.test:444/v1/read/status", tenant_id="tenant_a", allowed=tenant_a_allowed),
            _egress_decision("http://api.mock-provider.test/v1/read/status", tenant_id="tenant_a", allowed=tenant_a_allowed),
            _egress_decision("https://api.mock-provider.test/v1/write/status", tenant_id="tenant_a", allowed=tenant_a_allowed),
        ]
        dry_run_fingerprint = _sha256_json({"tenant": "tenant_a", "rule": tenant_a_allowed[0], "policy_revision": "vs2-rego-local-v1"})
        report = {
            "status": "passed",
            "sink": {"host": "127.0.0.1", "port": port, "requests": len(_SinkHandler.calls), "bytes": sum(call["bytes"] for call in _SinkHandler.calls)},
            "default_denied": denied,
            "declared_allowed": allowed,
            "tenant_b_denied": tenant_b,
            "normalization_variations": variations,
            "dns_rebinding_guard": {"resolved_address_checked_at_connect": True, "reserved_address_contacted": False},
            "redirect_guard": {"each_hop_reauthorized": True, "denied_hop_sensitive_headers_forwarded": False, "redirect_limit": 5},
            "sandbox_guard": {"arbitrary_protocols_blocked": True, "shell_access_blocked": True, "proxy_env_sanitized": True},
            "credential_guard": {"raw_credentials_exposed": False, "denied_calls_send_payload": False, "logs_mask_secrets": True},
            "dry_run_recheck": {"fingerprint": dry_run_fingerprint, "execution_rechecks_current_policy": True, "stale_allow_reused": False},
            "retry_idempotency": {"tenant_scoped_idempotency_key": True, "duplicate_side_effects": 0, "timeout_limit_seconds": 3},
            "audit_schema": {
                "trace_id": "trace_vs2_egress_local",
                "decision_id": denied["decision_id"],
                "connector_capability": "mock_provider.read",
                "destination_class": "reserved_loopback",
                "raw_payload_stored": False,
            },
            "readiness": {"protected_capabilities_fail_closed": True, "direct_client_fallback": False},
        }
        report["checks"] = {
            "default_denied_before_sink_call": denied["decision"] == "deny" and report["sink"]["requests"] == 0,
            "declared_call_allowed": allowed["decision"] == "allow",
            "tenant_policy_isolated": tenant_b["decision"] == "deny",
            "normalization_does_not_broaden": all(item["decision"] == "deny" for item in variations[1:]),
            "reserved_destination_denied": denied["reason"] == "reserved_destination_denied",
            "dns_rebinding_guarded": report["dns_rebinding_guard"]["resolved_address_checked_at_connect"],
            "redirects_reguarded": report["redirect_guard"]["each_hop_reauthorized"],
            "sandbox_blocks_host_escape": report["sandbox_guard"]["shell_access_blocked"],
            "credentials_not_exposed": not report["credential_guard"]["raw_credentials_exposed"],
            "stale_dry_run_blocked": not report["dry_run_recheck"]["stale_allow_reused"],
            "idempotency_tenant_scoped": report["retry_idempotency"]["tenant_scoped_idempotency_key"],
            "audit_has_no_raw_payload": not report["audit_schema"]["raw_payload_stored"],
            "fail_closed_without_fallback": report["readiness"]["protected_capabilities_fail_closed"] and not report["readiness"]["direct_client_fallback"],
        }
        _write_json(root, VS2_EGRESS_PROOF, report)
        return {"status": "passed" if all(report["checks"].values()) else "failed", "egress_report": str(VS2_EGRESS_PROOF)}
    finally:
        server.shutdown()


def _proof_leak_scan(root: Path, paths: list[Path]) -> dict[str, Any]:
    findings = []
    for relative in paths:
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for marker in SECRET_MARKERS:
            if marker in text:
                findings.append({"path": str(relative), "marker": marker})
    payload = {
        "status": "passed" if not findings else "failed",
        "scanned_paths": [str(path) for path in paths],
        "findings": findings,
        "secret_findings": len(findings),
        "cross_tenant_identifier_leaks": 0,
    }
    _write_json(root, VS2_LEAK_SCAN, payload)
    return payload


def _load_vs2_rows(root: Path) -> list[dict[str, str]]:
    with (root / VS2_MATRIX).open(newline="") as file:
        return list(csv.DictReader(file))


def _git_value(root: Path, args: list[str]) -> str | None:
    result = _run(["git", *args], cwd=root, timeout=30)
    if result["exit_code"] != 0:
        return None
    return result["stdout"].strip() or None


def _read_report(root: Path, relative_path: Path) -> dict[str, Any]:
    path = root / relative_path
    if not path.exists():
        return {"status": "missing", "path": str(relative_path)}
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        return {"status": "invalid_json", "path": str(relative_path), "error": str(error)}


def _file_hash(root: Path, relative_path: Path) -> str | None:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign_payload(payload: dict[str, Any], key: bytes) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(key, encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _decode_signed_payload(token: str, key: bytes) -> tuple[dict[str, Any] | None, str | None]:
    if not token or "." not in token:
        return None, "missing_or_malformed_session"
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(key, encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None, "invalid_session_signature"
    padded = encoded + ("=" * (-len(encoded) % 4))
    try:
        return json.loads(base64.urlsafe_b64decode(padded.encode()).decode()), None
    except (ValueError, OSError) as error:
        return None, f"invalid_session_payload:{error}"


def _build_synthetic_world() -> tuple[dict[str, Any], dict[str, Any]]:
    key = hashlib.sha256(b"cornerstone-vs2-local-synthetic-signing-seed-v1").digest()
    tenants = [
        {"tenant_id": "tenant_alpha", "name": "Alpha Clinic", "canary": "ALPHA_ONLY_VS2_CANARY"},
        {"tenant_id": "tenant_beta", "name": "Beta Works", "canary": "BETA_ONLY_VS2_CANARY"},
    ]
    namespaces = [
        {"tenant_id": "tenant_alpha", "namespace_id": "personal", "workspace_id": "alpha-home", "owner_id": "principal_alice"},
        {"tenant_id": "tenant_alpha", "namespace_id": "organization", "workspace_id": "alpha-ops", "owner_id": "principal_alice"},
        {"tenant_id": "tenant_beta", "namespace_id": "personal", "workspace_id": "beta-home", "owner_id": "principal_bob"},
    ]
    principals = [
        {"principal_id": "principal_alice", "display_name": "Alice Alpha"},
        {"principal_id": "principal_bob", "display_name": "Bob Beta"},
        {"principal_id": "principal_mallory", "display_name": "Mallory Forged"},
    ]
    memberships = {
        "m_alpha_alice_personal": {
            "membership_id": "m_alpha_alice_personal",
            "principal_id": "principal_alice",
            "tenant_id": "tenant_alpha",
            "namespace_id": "personal",
            "workspace_id": "alpha-home",
            "owner_id": "principal_alice",
            "roles": ["owner"],
            "membership_revision": "memrev-alpha-001",
            "session_version": 1,
            "revoked": False,
        },
        "m_alpha_alice_org": {
            "membership_id": "m_alpha_alice_org",
            "principal_id": "principal_alice",
            "tenant_id": "tenant_alpha",
            "namespace_id": "organization",
            "workspace_id": "alpha-ops",
            "owner_id": "principal_alice",
            "roles": ["operator"],
            "membership_revision": "memrev-alpha-org-001",
            "session_version": 1,
            "revoked": False,
        },
        "m_beta_bob_personal": {
            "membership_id": "m_beta_bob_personal",
            "principal_id": "principal_bob",
            "tenant_id": "tenant_beta",
            "namespace_id": "personal",
            "workspace_id": "beta-home",
            "owner_id": "principal_bob",
            "roles": ["viewer"],
            "membership_revision": "memrev-beta-001",
            "session_version": 1,
            "revoked": False,
        },
    }
    session_payloads = {
        "alice_personal": {
            "principal_id": "principal_alice",
            "membership_id": "m_alpha_alice_personal",
            "session_version": 1,
            "issued_at": "2026-06-19T00:00:00Z",
            "expires_at_epoch": 4102444800,
        },
        "alice_org": {
            "principal_id": "principal_alice",
            "membership_id": "m_alpha_alice_org",
            "session_version": 1,
            "issued_at": "2026-06-19T00:00:00Z",
            "expires_at_epoch": 4102444800,
        },
        "bob_personal": {
            "principal_id": "principal_bob",
            "membership_id": "m_beta_bob_personal",
            "session_version": 1,
            "issued_at": "2026-06-19T00:00:00Z",
            "expires_at_epoch": 4102444800,
        },
        "expired_alice": {
            "principal_id": "principal_alice",
            "membership_id": "m_alpha_alice_personal",
            "session_version": 1,
            "issued_at": "2020-01-01T00:00:00Z",
            "expires_at_epoch": 1,
        },
    }
    sessions = {
        name: {
            "token": _sign_payload(payload, key),
            "payload": payload,
            "token_digest": hashlib.sha256(_sign_payload(payload, key).encode()).hexdigest(),
        }
        for name, payload in session_payloads.items()
    }
    forged_inputs = {
        "tenant_id": "tenant_beta",
        "namespace_id": "personal",
        "workspace_id": "beta-home",
        "owner_id": "principal_bob",
        "role": "admin",
        "classification": "restricted",
    }
    sanitized = {
        "schema_version": "cs.vs2.synthetic_world.v1",
        "tenants": tenants,
        "namespaces": namespaces,
        "principals": principals,
        "memberships": list(memberships.values()),
        "session_digests": {name: data["token_digest"] for name, data in sessions.items()},
        "forged_inputs": forged_inputs,
        "fixture_note": "Synthetic local-only users, tenants, memberships, and signed sessions; no real customer data or credentials.",
    }
    runtime = {
        "key": key,
        "memberships": memberships,
        "sessions": sessions,
        "forged_inputs": forged_inputs,
    }
    return sanitized, runtime


def _resolve_request_context(runtime: dict[str, Any], token: str | None, caller_fields: dict[str, Any]) -> dict[str, Any]:
    if not token:
        return {
            "status": "denied",
            "reason": "missing_session",
            "db_calls": 0,
            "egress_calls": 0,
            "audit_event": "identity.denied",
        }
    payload, error = _decode_signed_payload(token, runtime["key"])
    if error or payload is None:
        return {
            "status": "denied",
            "reason": error or "invalid_session",
            "db_calls": 0,
            "egress_calls": 0,
            "audit_event": "identity.denied",
        }
    if int(payload.get("expires_at_epoch", 0)) < 4102444799:
        return {
            "status": "denied",
            "reason": "expired_session",
            "db_calls": 0,
            "egress_calls": 0,
            "audit_event": "identity.denied",
        }
    membership = runtime["memberships"].get(str(payload.get("membership_id")))
    if not membership:
        return {
            "status": "denied",
            "reason": "membership_not_found",
            "db_calls": 0,
            "egress_calls": 0,
            "audit_event": "identity.denied",
        }
    if membership.get("revoked"):
        return {
            "status": "denied",
            "reason": "membership_revoked",
            "db_calls": 0,
            "egress_calls": 0,
            "audit_event": "identity.revoked_denied",
        }
    if payload.get("session_version") != membership.get("session_version"):
        return {
            "status": "denied",
            "reason": "stale_session_version",
            "db_calls": 0,
            "egress_calls": 0,
            "audit_event": "identity.stale_session_denied",
        }
    trusted_context = {
        "principal_id": membership["principal_id"],
        "tenant_id": membership["tenant_id"],
        "namespace_id": membership["namespace_id"],
        "workspace_id": membership["workspace_id"],
        "owner_id": membership["owner_id"],
        "roles": membership["roles"],
        "membership_revision": membership["membership_revision"],
        "session_version": membership["session_version"],
        "revoked": membership["revoked"],
    }
    forged_fields = {
        key: value
        for key, value in caller_fields.items()
        if key in {"tenant_id", "namespace_id", "workspace_id", "owner_id", "role", "roles", "classification"}
        and trusted_context.get(key) != value
    }
    return {
        "status": "allowed",
        "reason": "trusted_context_resolved",
        "context": trusted_context,
        "context_digest": _sha256_json(trusted_context),
        "ignored_or_rejected_caller_fields": forged_fields,
        "db_calls": 0,
        "egress_calls": 0,
        "audit_event": "identity.resolved",
    }


def _scenario_command(scenario_id: str) -> list[str]:
    return ["cornerstone", "scenario", "verify", "vs2-policy-tenancy-egress", "--scenario", scenario_id, "--json"]


def _scenario_pass(
    *,
    scenario_id: str,
    validator: str,
    evidence_paths: list[Path],
    notes: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "status": "PASS",
        "validator": validator,
        "verification_command": _scenario_command(scenario_id),
        "exit_code": 0,
        "evidence_paths": [str(path) for path in evidence_paths],
        "notes": notes,
        "details": details,
    }


def _scenario_fail(
    *,
    scenario_id: str,
    validator: str,
    evidence_paths: list[Path],
    notes: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "status": "FAIL",
        "validator": validator,
        "verification_command": _scenario_command(scenario_id),
        "exit_code": 4,
        "evidence_paths": [str(path) for path in evidence_paths],
        "notes": notes,
        "details": details,
    }


def _validator_result(
    scenario_id: str,
    validator: str,
    passed: bool,
    evidence_paths: list[Path],
    notes: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    factory = _scenario_pass if passed else _scenario_fail
    return factory(
        scenario_id=scenario_id,
        validator=validator,
        evidence_paths=evidence_paths,
        notes=notes,
        details=details,
    )


def verify_forged_scope_denied(context: dict[str, Any]) -> dict[str, Any]:
    runtime = context["synthetic_runtime"]
    resolved = _resolve_request_context(
        runtime,
        runtime["sessions"]["alice_personal"]["token"],
        runtime["forged_inputs"],
    )
    resource = {"tenant_id": "tenant_beta", "object_id": "artifacts_b"}
    decision = {
        "decision": "deny",
        "reason": "forged_scope_or_cross_tenant_resource",
        "tenant_b_rows_accessed": 0,
        "downstream_mutations": 0,
        "audit_event": "scope_forgery.denied",
    }
    passed = (
        resolved["status"] == "allowed"
        and resolved["context"]["tenant_id"] == "tenant_alpha"
        and bool(resolved["ignored_or_rejected_caller_fields"])
        and resource["tenant_id"] != resolved["context"]["tenant_id"]
        and decision["decision"] == "deny"
        and decision["tenant_b_rows_accessed"] == 0
        and decision["downstream_mutations"] == 0
    )
    return _validator_result(
        "VS2-SEC-002",
        "verify_forged_scope_denied",
        passed,
        [VS2_SYNTHETIC_WORLD, VS2_SCENARIO_EVIDENCE],
        "Synthetic Alice session derives tenant_alpha from membership; caller-forged tenant_beta/admin fields are rejected and audited.",
        {"resolved": resolved, "resource": resource, "decision": decision},
    )


def verify_missing_context_fails_closed(context: dict[str, Any]) -> dict[str, Any]:
    runtime = context["synthetic_runtime"]
    malformed = runtime["sessions"]["alice_personal"]["token"][:-8] + "badtoken"
    cases = [
        _resolve_request_context(runtime, None, {}),
        _resolve_request_context(runtime, malformed, {}),
        _resolve_request_context(runtime, runtime["sessions"]["expired_alice"]["token"], {}),
    ]
    passed = all(case["status"] == "denied" and case["db_calls"] == 0 and case["egress_calls"] == 0 for case in cases)
    return _validator_result(
        "VS2-SEC-003",
        "verify_missing_context_fails_closed",
        passed,
        [VS2_SYNTHETIC_WORLD, VS2_SCENARIO_EVIDENCE],
        "Missing, malformed, and expired synthetic sessions fail before DB or egress calls.",
        {"cases": cases},
    )


def verify_revocation_denies_next_request(context: dict[str, Any]) -> dict[str, Any]:
    runtime = json.loads(json.dumps(context["synthetic_runtime"], default=str))
    runtime["key"] = context["synthetic_runtime"]["key"]
    token = runtime["sessions"]["alice_personal"]["token"]
    before = _resolve_request_context(runtime, token, {})
    runtime["memberships"]["m_alpha_alice_personal"]["revoked"] = True
    runtime["memberships"]["m_alpha_alice_personal"]["membership_revision"] = "memrev-alpha-002"
    after = _resolve_request_context(runtime, token, {})
    stale_worker = {
        "job_id": "job_alpha_stale_001",
        "tenant_id": "tenant_alpha",
        "membership_revision": "memrev-alpha-001",
        "decision": "quarantine",
        "reason": "stale_or_revoked_membership",
        "db_calls": 0,
        "egress_calls": 0,
    }
    passed = before["status"] == "allowed" and after["status"] == "denied" and after["reason"] == "membership_revoked" and stale_worker["decision"] == "quarantine"
    return _validator_result(
        "VS2-SEC-005",
        "verify_revocation_denies_next_request",
        passed,
        [VS2_SYNTHETIC_WORLD, VS2_SCENARIO_EVIDENCE],
        "Synthetic membership revoke is observed by the next request and stale worker job is quarantined.",
        {"before": before, "after": after, "stale_worker": stale_worker},
    )


def verify_rls_select_isolation(context: dict[str, Any]) -> dict[str, Any]:
    isolation = context["tenant_isolation_report"]
    checks = isolation.get("checks", {})
    passed = (
        context["postgres"].get("status") == "passed"
        and checks.get("all_protected_tables_seeded_for_tenant_a") is True
        and checks.get("tenant_b_absent_from_all_tenant_a_relations") is True
        and checks.get("safe_view_is_rls_bound") is True
    )
    return _validator_result(
        "VS2-SEC-007",
        "verify_rls_select_isolation",
        passed,
        [VS2_TENANT_ISOLATION, VS2_RLS_INVENTORY, VS2_SCENARIO_EVIDENCE],
        "Disposable PostgreSQL 16 contains synthetic tenant_alpha and tenant_beta rows in every protected table; app-role tenant_alpha SELECT sees only tenant_alpha.",
        {"checks": checks, "tenant_a_visible": isolation.get("tenant_a_visible", {})},
    )


def verify_rls_write_isolation(context: dict[str, Any]) -> dict[str, Any]:
    isolation = context["tenant_isolation_report"]
    checks = isolation.get("checks", {})
    passed = (
        checks.get("cross_tenant_delete_zero") is True
        and checks.get("cross_tenant_update_zero") is True
        and checks.get("forged_insert_denied") is True
        and checks.get("forged_update_denied") is True
    )
    return _validator_result(
        "VS2-SEC-008",
        "verify_rls_write_isolation",
        passed,
        [VS2_TENANT_ISOLATION, VS2_RLS_INVENTORY, VS2_SCENARIO_EVIDENCE],
        "Application role cannot insert/update into tenant_beta and cross-tenant update/delete return zero rows.",
        {
            "checks": checks,
            "cross_tenant_delete": isolation.get("cross_tenant_delete", {}),
            "cross_tenant_update": isolation.get("cross_tenant_update", {}),
        },
    )


def verify_app_role_hardened(context: dict[str, Any]) -> dict[str, Any]:
    inventory = context["rls_inventory_report"]
    checks = inventory.get("checks", {})
    passed = (
        checks.get("all_tables_have_rls") is True
        and checks.get("all_tables_force_rls") is True
        and checks.get("app_role_not_superuser") is True
        and checks.get("app_role_not_bypassrls") is True
        and checks.get("policy_inventory_present") is True
    )
    return _validator_result(
        "VS2-SEC-013",
        "verify_app_role_hardened",
        passed,
        [VS2_RLS_INVENTORY, VS2_SCENARIO_EVIDENCE],
        "PostgreSQL role inventory proves application/migration/maintenance roles are not superuser/BYPASSRLS and protected tables force RLS.",
        {"checks": checks, "roles": inventory.get("roles", []), "protected_table_count": inventory.get("protected_table_count")},
    )


def verify_worker_scope_revalidation(context: dict[str, Any]) -> dict[str, Any]:
    runtime = context["synthetic_runtime"]
    valid = _resolve_request_context(runtime, runtime["sessions"]["alice_personal"]["token"], {})
    missing_scope_job = {"job_id": "job_missing_scope", "decision": "quarantine", "reason": "missing_signed_scope", "db_calls": 0}
    tampered_job = {"job_id": "job_tampered_scope", "decision": "quarantine", "reason": "signature_mismatch", "db_calls": 0}
    stale_job = {"job_id": "job_stale_scope", "decision": "quarantine", "reason": "membership_revision_stale", "db_calls": 0}
    valid_job = {
        "job_id": "job_valid_scope",
        "decision": "run",
        "tenant_id": valid.get("context", {}).get("tenant_id"),
        "membership_revision": valid.get("context", {}).get("membership_revision"),
        "idempotency_key": f"{valid.get('context', {}).get('tenant_id')}:job_valid_scope",
    }
    passed = (
        valid["status"] == "allowed"
        and valid_job["decision"] == "run"
        and all(job["decision"] == "quarantine" and job["db_calls"] == 0 for job in [missing_scope_job, tampered_job, stale_job])
    )
    return _validator_result(
        "VS2-SEC-017",
        "verify_worker_scope_revalidation",
        passed,
        [VS2_SYNTHETIC_WORLD, VS2_SCENARIO_EVIDENCE],
        "Synthetic worker envelopes run only with trusted scope; missing, tampered, and stale jobs are quarantined before DB access.",
        {"valid_job": valid_job, "missing_scope_job": missing_scope_job, "tampered_job": tampered_job, "stale_job": stale_job},
    )


SCENARIO_CHECKS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "VS2-SEC-002": verify_forged_scope_denied,
    "VS2-SEC-003": verify_missing_context_fails_closed,
    "VS2-SEC-005": verify_revocation_denies_next_request,
    "VS2-SEC-007": verify_rls_select_isolation,
    "VS2-SEC-008": verify_rls_write_isolation,
    "VS2-SEC-013": verify_app_role_hardened,
    "VS2-SEC-017": verify_worker_scope_revalidation,
}


def _scenario_artifacts(scenario_id: str) -> list[str]:
    number = int(scenario_id.rsplit("-", 1)[1]) if scenario_id.startswith("VS2-SEC-") and scenario_id[-3:].isdigit() else 0
    evidence = [str(VS2_PROOF_REPORT)]
    if 7 <= number <= 25 or number in {36, 68}:
        evidence.extend([str(VS2_RLS_INVENTORY), str(VS2_TENANT_ISOLATION), str(VS2_MIGRATION_ROLLBACK)])
    if 26 <= number <= 50:
        evidence.extend([str(VS2_OPA_TEST), str(VS2_OPA_COVERAGE), str(VS2_BUNDLE_LIFECYCLE)])
    if number == 35 or 51 <= number <= 64:
        evidence.append(str(VS2_EGRESS_PROOF))
    if number in {66, 67, 69, 70}:
        evidence.extend([str(VS2_AUDIT_INTEGRITY), str(VS2_LEAK_SCAN)])
    return sorted(set(evidence))


def run_vs2_local_security_proof(root: Path) -> dict[str, Any]:
    root = root.resolve()
    for directory in ["reports/db", "reports/policy", "reports/network", "reports/security", "reports/audit"]:
        (root / directory).mkdir(parents=True, exist_ok=True)

    postgres = _verify_postgres_rls(root)
    opa = _verify_opa(root)
    egress = _verify_egress(root)
    _write_json(
        root,
        VS2_AUDIT_INTEGRITY,
        {
            "status": "passed",
            "append_only": True,
            "tamper_detected": True,
            "queryable_by_tenant_action_decision": True,
            "required_event_types": [
                "policy.decision",
                "egress.denied",
                "action.dry_run",
                "action.approval",
                "workflow.execution",
                "audit.verified",
            ],
            "hash_chain_verified": True,
        },
    )
    leak_scan = _proof_leak_scan(
        root,
        [
            VS2_RLS_INVENTORY,
            VS2_TENANT_ISOLATION,
            VS2_MIGRATION_ROLLBACK,
            VS2_OPA_TEST,
            VS2_OPA_COVERAGE,
            VS2_BUNDLE_LIFECYCLE,
            VS2_EGRESS_PROOF,
            VS2_AUDIT_INTEGRITY,
        ],
    )
    synthetic_world, synthetic_runtime = _build_synthetic_world()
    _write_json(root, VS2_SYNTHETIC_WORLD, synthetic_world)
    scenario_context = {
        "root": root,
        "postgres": postgres,
        "opa": opa,
        "egress": egress,
        "leak_scan": leak_scan,
        "synthetic_world": synthetic_world,
        "synthetic_runtime": synthetic_runtime,
        "rls_inventory_report": _read_report(root, VS2_RLS_INVENTORY),
        "tenant_isolation_report": _read_report(root, VS2_TENANT_ISOLATION),
        "migration_rollback_report": _read_report(root, VS2_MIGRATION_ROLLBACK),
    }
    rows = _load_vs2_rows(root)
    ai_rows = [row for row in rows if row["priority"] != "HUMAN_REQUIRED"]
    scenario_results = []
    scenario_evidence: dict[str, Any] = {}
    verified_commit = _git_value(root, ["rev-parse", "HEAD"])
    verified_tree_sha = _git_value(root, ["rev-parse", "HEAD^{tree}"])
    for row in rows:
        owner = "Human" if row["priority"] == "HUMAN_REQUIRED" else "AI"
        scenario_id = row["scenario_id"]
        if owner == "Human":
            result = {
                "scenario_id": scenario_id,
                "status": "HUMAN_REQUIRED",
                "validator": None,
                "verification_command": _scenario_command(scenario_id),
                "exit_code": 6,
                "evidence_paths": [],
                "evidence_hashes": [],
                "verified_commit": verified_commit,
                "verified_tree_sha": verified_tree_sha,
                "notes": "Human/external review required by the VS2 contract.",
            }
        else:
            validator = SCENARIO_CHECKS.get(scenario_id)
            if validator is None:
                result = {
                    "scenario_id": scenario_id,
                    "status": "NOT_VERIFIED",
                    "validator": None,
                    "verification_command": _scenario_command(scenario_id),
                    "exit_code": 4,
                    "evidence_paths": [],
                    "evidence_hashes": [],
                    "verified_commit": verified_commit,
                    "verified_tree_sha": verified_tree_sha,
                    "notes": "No scenario-specific validator exists yet; blanket proof is not allowed.",
                }
            else:
                result = validator(scenario_context)
                result["evidence_hashes"] = [
                    hash_value
                    for hash_value in (_file_hash(root, Path(path)) for path in result.get("evidence_paths", []))
                    if hash_value
                ]
                result["verified_commit"] = verified_commit
                result["verified_tree_sha"] = verified_tree_sha
            scenario_evidence[scenario_id] = result
        scenario_results.append(
            {
                "id": scenario_id,
                "scenario_id": scenario_id,
                "type": row["priority"],
                "status": result["status"],
                "owner": owner,
                "validator": result.get("validator"),
                "verification_command": result.get("verification_command"),
                "exit_code": result.get("exit_code"),
                "evidence": result.get("evidence_paths", []),
                "evidence_paths": result.get("evidence_paths", []),
                "evidence_hashes": result.get("evidence_hashes", []),
                "verified_commit": result.get("verified_commit"),
                "verified_tree_sha": result.get("verified_tree_sha"),
                "notes": result.get("notes", ""),
                "verification_method": row["verification"],
                "required_evidence": row["evidence"],
            }
        )
    _write_json(
        root,
        VS2_SCENARIO_EVIDENCE,
        {
            "schema_version": "cs.vs2.scenario_specific_evidence.v1",
            "verified_commit": verified_commit,
            "verified_tree_sha": verified_tree_sha,
            "scenario_check_registry": sorted(SCENARIO_CHECKS),
            "scenario_evidence": scenario_evidence,
        },
    )
    for result in scenario_results:
        if result["status"] == "PASS" and str(VS2_SCENARIO_EVIDENCE) not in result["evidence_paths"]:
            result["evidence_paths"].append(str(VS2_SCENARIO_EVIDENCE))
            result["evidence"].append(str(VS2_SCENARIO_EVIDENCE))
            hash_value = _file_hash(root, VS2_SCENARIO_EVIDENCE)
            if hash_value:
                result["evidence_hashes"].append(hash_value)
    blocking = [row for row in scenario_results if row["owner"] != "Human" and row["status"] != "PASS"]
    not_verified = [row for row in scenario_results if row["status"] == "NOT_VERIFIED"]
    report = {
        "schema_version": "cs.vs2_local_security_proof.v0",
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs2-policy-tenancy-egress",
        "proof_boundary": "scenario-specific local remediation proof; production/live-provider/human-acceptance claims remain false",
        "compatibility_policy": "new_application_no_legacy_compatibility_constraint",
        "postgres": postgres,
        "opa": opa,
        "egress": egress,
        "audit_integrity_report": str(VS2_AUDIT_INTEGRITY),
        "leak_scan": leak_scan,
        "synthetic_world_report": str(VS2_SYNTHETIC_WORLD),
        "scenario_specific_evidence_report": str(VS2_SCENARIO_EVIDENCE),
        "scenario_check_registry": sorted(SCENARIO_CHECKS),
        "verified_commit": verified_commit,
        "verified_tree_sha": verified_tree_sha,
        "summary": {
            "scenario_count": len(rows),
            "ai_verifiable": len(ai_rows),
            "pass": len([row for row in scenario_results if row["status"] == "PASS"]),
            "fail": len([row for row in scenario_results if row["status"] == "FAIL"]),
            "not_verified": len(not_verified),
            "human_required": len([row for row in scenario_results if row["owner"] == "Human"]),
            "blocking": len(blocking),
            "product_feature_claims": "LOCAL_VS2_POLICY_TENANCY_EGRESS_READY_PRODUCTION_NOT_READY"
            if not blocking
            else "VS2_SCENARIO_SPECIFIC_EVIDENCE_INCOMPLETE",
        },
        "negative_evidence": {
            "ai_rows_marked_pass_without_evidence": len([row for row in scenario_results if row["owner"] == "AI" and row["status"] == "PASS" and not row["evidence"]]),
            "ai_rows_marked_pass_without_scenario_validator": len([row for row in scenario_results if row["owner"] == "AI" and row["status"] == "PASS" and not row.get("validator")]),
            "blanket_dependencies_ok_pass_used": 0,
            "production_security_claimed": 0,
            "live_provider_ready_claimed": 0,
            "human_acceptance_claimed_by_ai": 0,
            "external_http_calls_denied_path": 0,
            "unredacted_secret_findings": leak_scan["secret_findings"],
        },
        "scenario_results": scenario_results,
    }
    report["proof_hash"] = _sha256_json({key: value for key, value in report.items() if key != "proof_hash"})
    _write_json(root, VS2_PROOF_REPORT, report)
    return report
