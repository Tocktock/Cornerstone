# VS3 Local Checkpoint Self Command Transcript Guard Checkpoint

**Date:** 2026-06-29 KST
**Status:** Local verifier/evidence-layout slice complete
**Scope:** `cornerstone security vs3-local-checkpoint --json` self command transcript metadata
**Claim boundary:** VS3-L local/dev verifier evidence only. VS3-P, production/on-prem readiness, live-provider readiness, real-IdP readiness, real-network readiness, migration/restore readiness, independent security acceptance, and human UX acceptance remain `NOT_CLAIMED` or `HUMAN_REQUIRED`.

## Slice Contract

Goal: harden the VS3 local checkpoint output so the checkpoint's own native CLI transcript carries verifiable command metadata required by the CLI-native-first and local verification contracts.

In scope:

- Add complete self transcript metadata to `cornerstone security vs3-local-checkpoint --json`.
- Preserve exact local/dev proof boundary in the transcript `stdout_json`.
- Add deterministic regression assertions in the existing VS3 local checkpoint test.
- Record before/after CLI evidence.

Out of scope:

- No production, on-prem, live provider, real IdP, real network, migration/restore, independent security review, or human UX acceptance evidence.
- No new VS3 subsystem implementation beyond the local checkpoint transcript surface.
- No broad dependency or runtime upgrade.

## Full Scenario Mapping

| Scenario | Type | Slice classification | Required proof surface / reason |
|---|---|---|---|
| VS3-GATE-001 | MUST_PASS | later_slice | Evidence reconciliation JSON, VS2 report hashes, rejected/superseded report list; not changed in this transcript slice. |
| VS3-GATE-002 | MUST_PASS | later_slice | Docs verifier and matrix structural checks; not the selected behavior, though docs checks are rerun as supporting evidence. |
| VS3-GATE-003 | MUST_PASS | later_slice | Static overclaim lint over VS3 reports/copy; this slice preserves boundary in transcript but does not replace full overclaim gate. |
| VS3-GATE-004 | MUST_PASS | in_this_slice | Native CLI transcript metadata for the VS3 verifier/checkpoint path; before/after CLI output and unit regression test required. |
| VS3-CTX-001 | MUST_PASS | later_slice | Cross-surface trusted RequestContext fixture and matching policy outcome evidence. |
| VS3-CTX-002 | MUST_PASS | later_slice | Forged-authority negative matrix with zero DB/egress/tool/provider side effects. |
| VS3-CTX-003 | MUST_PASS | later_slice | Allow -> revoke -> retry proof across cached sessions, workers, and tools. |
| VS3-CTX-004 | MUST_PASS | later_slice | Malformed/missing/expired RequestContext fail-closed matrix. |
| VS3-CTX-005 | MUST_PASS | later_slice | Workspace/mission policy fixture proving tenant membership alone is insufficient. |
| VS3-RLS-001 | MUST_PASS | later_slice | Durable table scope inventory and null-insert/create/read tests. |
| VS3-RLS-002 | MUST_PASS | later_slice | Two-tenant DB read/count/search matrix under application role. |
| VS3-RLS-003 | MUST_PASS | later_slice | Cross-tenant write/RETURNING mutation matrix with rollback assertions. |
| VS3-RLS-004 | MUST_PASS | later_slice | Pool/retry/worker/job/cancel tenant context reset stress test. |
| VS3-RLS-005 | MUST_PASS | later_slice | Migration ownership/quarantine/rollback/restore-read proof. |
| VS3-RLS-006 | MUST_PASS | later_slice | Backup/export/restore tenant-boundary and audit verification. |
| VS3-OPA-001 | MUST_PASS | later_slice | PolicyInput schema and source-of-attribute fixture proof. |
| VS3-OPA-002 | MUST_PASS | later_slice | OPA/Rego and HTTP decision tests with deterministic default deny. |
| VS3-OPA-003 | MUST_PASS | later_slice | OPA service management/decision API access hardening test. |
| VS3-OPA-004 | MUST_PASS | later_slice | Bundle lifecycle, invalid bundle, rollback, and first-start fail-closed proof. |
| VS3-OPA-005 | MUST_PASS | later_slice | Decision-log/audit redaction canary proof. |
| VS3-EGR-001 | MUST_PASS | later_slice | Runtime/network deny evidence and forbidden sink zero-request counters. |
| VS3-EGR-002 | MUST_PASS | later_slice | Governed allowed sink evidence with exactly one approved ConnectorHub call. |
| VS3-EGR-003 | MUST_PASS | later_slice | URL/DNS/redirect/IPv4/IPv6 bypass negative matrix. |
| VS3-EGR-004 | MUST_PASS | later_slice | Socket/proxy/subprocess/shell/filesystem/env sandbox adversarial suite. |
| VS3-EGR-005 | MUST_PASS | later_slice | Egress/sandbox outage fail-closed proof and degraded readiness evidence. |
| VS3-EGR-006 | MUST_PASS | later_slice | Prompt-injection fixture proving untrusted content cannot grant authority. |
| VS3-CON-001 | MUST_PASS | later_slice | Connector projection ack-after-immutable-artifact-commit crash/retry proof. |
| VS3-CON-002 | MUST_PASS | later_slice | GitHub read-only static mapping scan and runtime write denial. |
| VS3-CON-003 | MUST_PASS | later_slice | Connector credential redaction canary scan across outputs and durable state. |
| VS3-CON-004 | MUST_PASS | later_slice | SourcePolicy update/delivery/capture/revoke/retry scoped enforcement proof. |
| VS3-CON-005 | MUST_PASS | later_slice | WatchAgent/macOS/Chrome capture fixture with consent, pause, revoke, and no disallowed raw output. |
| VS3-CON-006 | MUST_PASS | later_slice | Connector fault/duplicate/stale/malicious payload quarantine proof. |
| VS3-TOOL-001 | MUST_PASS | later_slice | Tool package manifest/signature/SBOM/schema validation. |
| VS3-TOOL-002 | MUST_PASS | later_slice | Registry accept/reject tests for signed, unsigned, tampered, stale, revoked, and unknown-source packages. |
| VS3-TOOL-003 | MUST_PASS | later_slice | Installed-inactive execution denial across surfaces. |
| VS3-TOOL-004 | MUST_PASS | later_slice | Activation dry-run, approval, execution, and revoke least-privilege proof. |
| VS3-TOOL-005 | MUST_PASS | later_slice | Runtime sandbox negative suite for undeclared file/env/network/shell/model/connector/memory access. |
| VS3-TOOL-006 | MUST_PASS | later_slice | Pack update diff/evaluation/risk/migration/rollback dry-run gate. |
| VS3-TOOL-007 | MUST_PASS | later_slice | Rollback and emergency-patch simulation without authority expansion. |
| VS3-OBS-001 | MUST_PASS | later_slice | Fault-injection status plus CLI/API/UI comparison. |
| VS3-OBS-002 | MUST_PASS | later_slice | Audit contract and tamper-evident fixture proof. |
| VS3-OBS-003 | MUST_PASS | later_slice | Human-gate package generation/schema validation without marking H rows PASS. |
| VS3-REG-001 | REGRESSION | later_slice | Fresh VS0 loop reports on the same source tree; not rerun by this narrow slice. |
| VS3-REG-002 | REGRESSION | later_slice | Fresh VS1 ontology gate and cross-scope tests; not rerun by this narrow slice. |
| VS3-REG-003 | REGRESSION | later_slice | Red-team authority expansion fixture; not rerun by this narrow slice. |
| VS3-REG-004 | REGRESSION | in_this_slice | Checkpoint transcript coverage cannot silently drop; regression assertion now requires complete transcript fields. |
| VS3-REG-005 | REGRESSION | in_this_slice | Transcript `stdout_json.proof_boundary` preserves `NOT_CLAIMED` for VS3-P and external/human claims. |
| VS3-REG-006 | REGRESSION | later_slice | Normal-user UI/product-first browser or DOM review. |
| VS3-REG-007 | REGRESSION | in_this_slice | No new dependency or lockfile change is introduced by this slice; verified by diff scope. |
| VS3-REG-008 | REGRESSION | later_slice | Fresh/reset/partial-upgrade secure-default integration suite. |
| VS3-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires signed owner architecture/security/dependency/migration approval. |
| VS3-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires independent security review and retest evidence. |
| VS3-H03 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires real IdP/OIDC mapping and revocation evidence. |
| VS3-H04 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires target on-prem network/firewall/proxy/service-mesh evidence. |
| VS3-H05 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires approved live ConnectorHub/provider rehearsal with redacted evidence. |
| VS3-H06 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires human operator UX/trust acceptance or rejection evidence. |
| VS3-H07 | HUMAN_REQUIRED | HUMAN_REQUIRED | Requires human-supervised migration/backup/restore drill evidence. |

## Before Evidence

Command:

```bash
python3 - <<'PY'
import json, subprocess, pathlib
root=pathlib.Path('/Users/jiyong/playground/Cornerstone')
cmd=[str(root/'cornerstone'),'security','vs3-local-checkpoint','--json']
r=subprocess.run(cmd,cwd=root,text=True,capture_output=True,check=False,timeout=180)
payload=json.loads(r.stdout)
out=pathlib.Path('/tmp/vs3-local-checkpoint-self-transcript-before.json')
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print('returncode', r.returncode)
print('status', payload.get('status'))
print('final_verdict', payload.get('final_verdict'))
t=payload['command_transcripts'][0]
print('transcript_keys', sorted(t.keys()))
for k in ['schema_version','arguments','timed_out','output_mode','cli_schema_version','started_at','ended_at','evidence_refs','audit_refs','policy_decision_refs','scope','source_tree','stdout_json']:
    print(k, t.get(k))
PY
```

Observed output:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
transcript_keys ['command', 'exit_code', 'json_schema']
schema_version None
arguments None
timed_out None
output_mode None
cli_schema_version None
started_at None
ended_at None
evidence_refs None
audit_refs None
policy_decision_refs None
scope None
source_tree None
stdout_json None
```

Baseline artifact: `/tmp/vs3-local-checkpoint-self-transcript-before.json`.

## Change Summary

Changed [packages/cornerstone_cli/main.py](/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py) so `command_security_vs3_local_checkpoint` emits a complete self command transcript after checkpoint evidence, audit, and policy refs are finalized.

The transcript now includes:

- `schema_version=cs.command_transcript.v0`
- native `command` and `arguments`
- `started_at`, `ended_at`, `timed_out`, `elapsed_seconds`
- `output_mode=json`, `json_schema=cs.vs3_local_checkpoint.v0`, `cli_schema_version=cs.cli.v0`
- tenant/owner/namespace/workspace `scope` plus `scope_source=local_vs3_fixture`
- `evidence_refs`, `audit_refs`, `policy_decision_refs`, and `ref_summary`
- scenario report `source_tree`
- `stdout_json` with matching command, refs, scope, source tree, and proof boundary

Changed [tests/scenario/test_scaffold_cli.py](/Users/jiyong/playground/Cornerstone/tests/scenario/test_scaffold_cli.py) so the existing local checkpoint manifest test rejects a return to minimal transcript metadata.

## After Evidence

Command:

```bash
python3 - <<'PY'
import json, subprocess, pathlib
root=pathlib.Path('/Users/jiyong/playground/Cornerstone')
cmd=[str(root/'cornerstone'),'security','vs3-local-checkpoint','--json']
r=subprocess.run(cmd,cwd=root,text=True,capture_output=True,check=False,timeout=180)
payload=json.loads(r.stdout)
out=pathlib.Path('/tmp/vs3-local-checkpoint-self-transcript-after.json')
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print('returncode', r.returncode)
print('status', payload.get('status'))
print('final_verdict', payload.get('final_verdict'))
t=payload['command_transcripts'][0]
print('transcript_keys', sorted(t.keys()))
print('scope', t['scope'])
print('ref_summary', t['ref_summary'])
print('source_tree_keys', sorted(t['source_tree'].keys()))
print('stdout_boundary', t['stdout_json']['proof_boundary'])
PY
```

Observed output:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
transcript_keys ['arguments', 'audit_refs', 'cli_schema_version', 'command', 'elapsed_seconds', 'ended_at', 'evidence_refs', 'exit_code', 'json_schema', 'name', 'output_mode', 'policy_decision_refs', 'ref_summary', 'required', 'schema_version', 'scope', 'source', 'source_tree', 'started_at', 'stderr_tail', 'stdout_json', 'stdout_tail', 'timed_out']
scope {'namespace_id': 'personal', 'owner_id': 'local-user', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'local-dev', 'workspace_id': 'default'}
ref_summary {'audit_refs_count': 195, 'evidence_refs_count': 25, 'policy_decision_refs_count': 111}
source_tree_keys ['dirty_paths', 'final_commit', 'final_commit_pending_reason', 'generated_dirty_paths', 'report_generated_before_commit', 'verified_base_commit', 'verified_base_commit_full', 'verified_base_tree_hash', 'verified_source_snapshot_paths', 'verified_source_worktree_hash', 'worktree_dirty_at_verification']
stdout_boundary {'human_acceptance': 'NOT_CLAIMED', 'live_provider': 'NOT_CLAIMED', 'migration_restore': 'NOT_CLAIMED', 'production': 'NOT_CLAIMED', 'production_onprem': 'NOT_CLAIMED', 'real_idp': 'NOT_CLAIMED', 'real_network': 'NOT_CLAIMED', 'security_acceptance': 'NOT_CLAIMED', 'vs3_l': 'LOCAL_COMPONENT_PROOF_ONLY', 'vs3_p': 'NOT_CLAIMED'}
```

After artifact: `/tmp/vs3-local-checkpoint-self-transcript-after.json`.

## Verification

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code 0.

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe
```

Result:

```text
Ran 1 test in 26.779s
OK
```

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_secret_leak \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_secret_leak
```

Result:

```text
Ran 3 tests in 80.329s
OK
```

## Decision

This slice is locally verified for the checkpoint self command transcript guard.

It does not complete VS3-L by itself and does not claim VS3-P. Human-required rows remain `HUMAN_REQUIRED`.

Recommended next slice: add a deterministic validator for the local checkpoint's own self transcript shape so future report consumers can fail closed if the checkpoint transcript regresses outside the unit test surface.
