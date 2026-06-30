# VS3 Postgres/RLS Checkpoint

## Summary

- Verdict: PASS for the VS3-2 local Postgres/RLS rehearsal slice only.
- Scope: VS3-RLS-001 through VS3-RLS-006.
- Date: 2026-06-29 KST.
- Owner: AI local verification.
- Report: `reports/db/vs3-postgres-rls-proof.json`.
- Aggregate scenario report: `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.

This checkpoint does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, independent security acceptance, production migration/restore readiness, or human UX acceptance.

## Goal

Prove the VS3 Postgres/RLS and scoped durable-truth baseline in a local deterministic rehearsal:

- active VS0/VS1 durable object families have required scope fields and failed-null-insert evidence;
- tenant read paths hide foreign rows and metadata;
- cross-tenant writes, forged inserts, and scope mutation are denied or affect zero unauthorized rows;
- pool, retry, worker, scheduled, concurrent, error, timeout, and rollback paths reset context;
- migration imports known rows and quarantines ambiguous or invalid rows without ownerless truth;
- local backup, restore, RLS recheck, audit recheck, and scoped tenant export all pass.

## Full Scenario Mapping Gate

The frozen VS3 matrix currently contains:

| Type | Count | Classification for this slice |
|---|---:|---|
| MUST_PASS | 42 | VS3-RLS-001 through VS3-RLS-006 are in this slice. Earlier VS3-0 and VS3-1 rows have separate checkpoint evidence. Remaining MUST_PASS rows stay mapped to later slices or existing local proof reports. |
| REGRESSION | 8 | No REGRESSION row is in this slice; all eight remain final-gate coverage. |
| HUMAN_REQUIRED | 7 | VS3-H01 through VS3-H07 remain HUMAN_REQUIRED and are not promoted by this checkpoint. |
| Total | 57 | Full 57-row inventory remains the release coverage basis. |

## Scenario Verification

| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| VS3-RLS-001 | MUST_PASS | Active durable object families have tenant, owner, namespace, workspace, classification, provenance, and audit scope where required. | Run `cornerstone security vs3-postgres-rls --json` and `cornerstone tenant rls-inventory --json`; inspect schema inventory and null-insert denial. | `reports/db/vs3-postgres-rls-proof.json`; `/tmp/vs3_rls_inventory_after.json`. | PASS for local Postgres/RLS rehearsal. |
| VS3-RLS-002 | MUST_PASS | Tenant A reads only authorized tenant rows; tenant-B rows and existence metadata remain hidden. | Run `cornerstone tenant rls-isolation --json`; inspect read isolation matrix and foreign-row negative evidence. | `reports/db/vs3-postgres-rls-proof.json`; `/tmp/vs3_rls_isolation_after.json`. | PASS for local Postgres/RLS rehearsal. |
| VS3-RLS-003 | MUST_PASS | Cross-tenant INSERT, UPDATE, DELETE, bulk mutation, and RETURNING paths are denied or affect zero unauthorized rows. | Inspect mutation matrix from `cornerstone tenant rls-isolation --json`. | Forged insert denied, cross-tenant update zero, scope mutation denied. | PASS for local Postgres/RLS rehearsal. |
| VS3-RLS-004 | MUST_PASS | Pool, retry, worker, scheduled, concurrent, error, timeout, and rollback paths do not leak tenant context. | Inspect `cornerstone security vs3-postgres-rls --json` pool and worker reset matrix. | Pool reset matrix, worker scope evidence, zero pool context leaks. | PASS for local Postgres/RLS rehearsal. |
| VS3-RLS-005 | MUST_PASS | Migration imports known rows, quarantines missing/ambiguous/invalid/duplicate rows, preserves checksums, and proves rollback. | Run `cornerstone tenant migration-rehearsal --json`; inspect quarantine and rollback evidence. | `/tmp/vs3_migration_after.json`; ownerless truth count 0. | PASS for local Postgres/RLS rehearsal. |
| VS3-RLS-006 | MUST_PASS | Backup/restore preserves rows, RLS policies, audit integrity, and tenant export boundaries. | Run `cornerstone backup create --json` and `cornerstone restore verify --json`; inspect restore checks. | `/tmp/vs3_backup_after.json`; `/tmp/vs3_restore_after.json`. | PASS for local Postgres/RLS rehearsal only; human migration/restore readiness remains HUMAN_REQUIRED. |

## CLI Parity Summary

| Feature / Scenario | CLI Command(s) | JSON Schema | Exit-Code Tests | Evidence/Audit Refs | Same Backend Path | Status |
|---|---|---|---|---|---|---|
| RLS aggregate proof | `cornerstone security vs3-postgres-rls --json` | `cs.cli.v0` plus `cs.vs3_postgres_rls_proof.v0` | Exit 0 on local proof success. | `reports/db/vs3-postgres-rls-proof.json`; `reports/security/vs2-local-range.json`; 18 audit refs. | Native CLI calls `run_vs3_postgres_rls_proof`. | PASS for this slice. |
| RLS inventory | `cornerstone tenant rls-inventory --json` | `cs.vs3_tenant_rls_inventory.v0` | Exit 0 on local proof success. | Same report refs and audit refs. | Native CLI reads the VS3 RLS proof. | PASS for this slice. |
| RLS isolation and mutation | `cornerstone tenant rls-isolation --json` | `cs.vs3_tenant_rls_isolation.v0` | Exit 0 on local proof success. | Same report refs and audit refs. | Native CLI reads the VS3 RLS proof. | PASS for this slice. |
| Migration rehearsal | `cornerstone tenant migration-rehearsal --json` | `cs.vs3_tenant_migration_rehearsal.v0` | Exit 0 on local proof success. | Same report refs and audit refs. | Native CLI reads the VS3 RLS proof. | PASS for this slice. |
| Backup manifest proof | `cornerstone backup create --json` | `cs.vs3_backup_manifest.v0` | Exit 0 on local proof success. | Same report refs and audit refs. | Native CLI reads the VS3 RLS proof. | PASS for this slice. |
| Restore verification proof | `cornerstone restore verify --json` | `cs.vs3_restore_verification.v0` | Exit 0 on local proof success. | Same report refs and audit refs. | Native CLI reads the VS3 RLS proof. | PASS for this slice. |

## Command Evidence

### Aggregate Postgres/RLS proof

```text
PATH="$PWD:$PATH" cornerstone security vs3-postgres-rls --json
status success
schema_version cs.vs3_postgres_rls_proof.v0
scenario_status {'VS3-RLS-001': 'PASS', 'VS3-RLS-002': 'PASS', 'VS3-RLS-003': 'PASS', 'VS3-RLS-004': 'PASS', 'VS3-RLS-005': 'PASS', 'VS3-RLS-006': 'PASS'}
checks {'vs3_rls_001_schema_scope_non_null': True, 'vs3_rls_002_tenant_read_isolation': True, 'vs3_rls_003_cross_tenant_mutation_denied': True, 'vs3_rls_004_pool_worker_context_reset': True, 'vs3_rls_005_migration_quarantine_rollback': True, 'vs3_rls_006_backup_restore_tenant_safe': True}
proof_boundary {'human_migration_restore': 'HUMAN_REQUIRED', 'human_security_acceptance': 'HUMAN_REQUIRED', 'production_onprem': 'HUMAN_REQUIRED', 'real_idp': 'HUMAN_REQUIRED', 'surface': 'local_postgres_rls_rehearsal', 'vs3_l': 'NOT_CLAIMED', 'vs3_p': 'NOT_CLAIMED'}
audit_refs_count 18
evidence_refs ['reports/db/vs3-postgres-rls-proof.json', 'reports/security/vs2-local-range.json']
```

### Negative evidence

```text
negative_evidence {
  'nullable_or_missing_required_scope_columns': 0,
  'null_insert_attempts_allowed': 0,
  'foreign_tenant_rows_visible': 0,
  'unauthorized_mutation_effects': 0,
  'pool_context_leaks': 0,
  'ownerless_truth_rows': 0,
  'silent_default_tenant_assignments': 0,
  'destructive_migration_steps': 0,
  'restore_missing_rows': 0,
  'tenant_export_leaks': 0,
  'production_migration_claimed': 0,
  'vs3_p_claimed': 0,
  'human_migration_restore_marked_pass': 0
}
```

### Native subcommands

```text
PATH="$PWD:$PATH" cornerstone tenant rls-inventory --json
inventory_exit:0
inventory status success
inventory schema_version cs.cli.v0
inventory tenant_rls_inventory_schema_version cs.vs3_tenant_rls_inventory.v0
inventory vs3_postgres_rls_schema_version cs.vs3_postgres_rls_proof.v0
inventory audit_refs_count 18
inventory proof_vs3_p NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone tenant rls-isolation --json
isolation_exit:0
isolation status success
isolation tenant_rls_isolation_schema_version cs.vs3_tenant_rls_isolation.v0
isolation audit_refs_count 18
isolation proof_vs3_p NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone tenant migration-rehearsal --json
migration_exit:0
migration status success
migration tenant_migration_rehearsal_schema_version cs.vs3_tenant_migration_rehearsal.v0
migration audit_refs_count 18
migration proof_vs3_p NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone backup create --json
backup_exit:0
backup status success
backup backup_schema_version cs.vs3_backup_manifest.v0
backup audit_refs_count 18
backup proof_vs3_p NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone restore verify --json
restore_exit:0
restore status success
restore restore_schema_version cs.vs3_restore_verification.v0
restore audit_refs_count 18
restore proof_vs3_p NOT_CLAIMED
```

### Aggregate scenario report and gate

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
status success
postgres_rls_status success
postgres_rls_checks {'vs3_rls_001_schema_scope_non_null': True, 'vs3_rls_002_tenant_read_isolation': True, 'vs3_rls_003_cross_tenant_mutation_denied': True, 'vs3_rls_004_pool_worker_context_reset': True, 'vs3_rls_005_migration_quarantine_rollback': True, 'vs3_rls_006_backup_restore_tenant_safe': True}
rls_row_statuses {'VS3-RLS-001': 'PASS', 'VS3-RLS-002': 'PASS', 'VS3-RLS-003': 'PASS', 'VS3-RLS-004': 'PASS', 'VS3-RLS-005': 'PASS', 'VS3-RLS-006': 'PASS'}
proof_boundary_vs3_p NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
status success
scenario_count 57
coverage_validation_status passed
human_required_validation_status passed
claim_boundary_validation_status passed
row_ref_validation_status passed
error_codes []
```

### Automated checks

```text
python3 -m compileall packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit code: 0
```

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_postgres_rls_proof_is_local_and_negative_evidence_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_tenant_backup_restore_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows
Ran 3 tests in 28.183s
OK
```

## Implementation Evidence

- `packages/cornerstone_cli/main.py` now emits command-specific schema-version fields for VS3 RLS inventory, isolation, migration rehearsal, backup manifest, and restore verification native CLI payloads.
- `tests/scenario/test_scaffold_cli.py` asserts those command-specific schema versions and audit evidence for the native VS3-2 commands.
- `reports/db/vs3-postgres-rls-proof.json` records the refreshed local Postgres/RLS proof and keeps production/on-prem and human migration acceptance as HUMAN_REQUIRED.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS3-H01 through VS3-H07 | These require human/external/on-prem/security/operator evidence. | Complete the human review or external rehearsal named by each row. | Dated, redacted, signed approval/review/topology/provider/UX/migration evidence. | Blocks VS3-P, production/on-prem readiness, live readiness, security acceptance, UX acceptance, and migration/restore readiness. |

## Deliberately Not Done

- Did not claim VS3-P or production/on-prem readiness.
- Did not convert human rows to PASS.
- Did not claim real IdP, live provider, real network, or independent security acceptance.
- Did not claim production migration/restore readiness.
- Did not begin VS3-3 OPA/Rego work in this slice.
- Did not run full repository tests.

## Risks

- This proof uses local Docker/Postgres rehearsal evidence, not real on-prem topology, production network, production data, or human-approved migration/restore evidence.
- `cornerstone backup create --json` and `cornerstone restore verify --json` currently expose the local proof manifest and verification result; they do not create or certify a production backup.
- The aggregate VS3 report records a dirty worktree and remains local/dev evidence only.

## Verdict

- AI-verifiable scope: done for VS3-RLS-001 through VS3-RLS-006 local Postgres/RLS rehearsal proof.
- Human/release gate: needs-human-verification for VS3-H01 through VS3-H07.
- Recommendation: continue to the next VS3 slice only after accepting this checkpoint boundary.
