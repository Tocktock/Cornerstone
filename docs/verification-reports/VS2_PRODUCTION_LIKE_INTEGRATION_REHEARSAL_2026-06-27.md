# VS2 Production-Like Local Integration Rehearsal

**Date:** 2026-06-27
**Owner:** JiYong / Tars
**Status:** Local production-like integration rehearsal PASS
**Scope:** Local Docker Compose only. This report does not claim production readiness, live connector readiness, penetration-test completion, or human acceptance.

## Goal

Create and verify a production-like local VS2 environment for CornerStone using:

- real PostgreSQL;
- real OPA/Rego HTTP decision service;
- Docker internal network default egress denial;
- controlled local egress gateway;
- real VS2 migrations;
- tenant-isolation checks through the `cornerstone_app` role;
- backup/restore rehearsal;
- audit hash-chain tamper detection.

## Scenario Contract

| ID | Type | Expected Result | Status | Evidence |
|---|---|---|---|---|
| PLIKE-001 | MUST_PASS | Compose stack starts with Postgres, OPA, and egress gateway; network topology is inspectable. | PASS | `reports/security/vs2-production-like-integration-2026-06-27.json` |
| PLIKE-002 | MUST_PASS | Real VS2 migrations apply to a fresh Postgres database with hardened roles. | PASS | `migrations/vs2/*.sql`, migration transcript in report |
| PLIKE-003 | MUST_PASS | RLS hides tenant B and denies forged cross-tenant writes for `cornerstone_app`. | PASS | SQL transcript and `pg_policies` inventory in report |
| PLIKE-004 | MUST_PASS | OPA allows valid owner input and denies cross-tenant and invalid-schema input. | PASS | Internal `POST /v1/data/cornerstone/vs2/decision` probe |
| PLIKE-005 | MUST_PASS | Internal-network runtime cannot reach external egress directly; controlled gateway remains reachable. | PASS | Ephemeral internal-network probe in report |
| PLIKE-006 | MUST_PASS | `pg_dump`/`pg_restore` preserve tenant-scoped counts, RLS checks, and audit verification. | PASS | Backup path and restore database in report |
| PLIKE-R01 | REGRESSION_GUARD | Local rehearsal does not claim production readiness, live-provider readiness, or human acceptance. | PASS | Claim-boundary and human-required fields in report |

## Command Evidence

| Command | Result |
|---|---|
| `PATH="$(PWD):$PATH" cornerstone security vs2-production-like-integration --json > reports/security/vs2-production-like-integration-command.json` | PASS |
| `make verify-vs2-production-like` | PASS |
| `python3 -m py_compile packages/cornerstone_cli/vs2_production_like.py packages/cornerstone_cli/main.py` | PASS |
| `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs2_local_range_command_runs_real_first_slice` | PASS, 1 test in 44.967s |
| `git diff --check` | PASS |

## Result Summary

The latest report is:

- `reports/security/vs2-production-like-integration-2026-06-27.json`
- `reports/security/vs2-production-like-integration-command.json`
- `run_id`: `20260627T105721Z_29362`
- `report_payload_sha256_without_hash`: `b5907b345835c38999edf929e414c3385d84ce49f0b370f67b9c6fab76bc46bb`

All seven local production-like rehearsal rows are `PASS`.

## Human Required Boundary

`VS2-SEC-H04` and related production/human gates remain `HUMAN_REQUIRED`.

Required human/external evidence before production PASS:

- reviewed production topology;
- real network-control evidence;
- production backup/restore drill evidence;
- independent security review or owner approval;
- live-provider or connector evidence where applicable;
- redacted policy/audit transcripts from the reviewed environment.
