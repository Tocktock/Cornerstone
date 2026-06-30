# VS3 Overclaim Lint Report Semantics Guard Checkpoint

**Date:** 2026-06-29 KST
**Status:** Local deterministic VS3-L checkpoint hardening slice.
**Scope:** `cornerstone security vs3-local-checkpoint --json` validation of `reports/security/vs3-overclaim-lint.json` semantics.
**Verdict:** AI-verifiable slice PASS; VS3-P, production/on-prem, real IdP, real network, live-provider, migration/restore, security-acceptance, and human-acceptance claims remain `NOT_CLAIMED` / `HUMAN_REQUIRED`.

## Slice Contract

Goal:

- Make the VS3 local checkpoint reject an overclaim-lint report that still points at the current reconciliation report but whose own status, claim boundary, overclaim fields, or negative evidence are no longer clean.

In scope:

- Validate `reports/security/vs3-overclaim-lint.json` schema/status, claim boundary, overclaim fields, and negative evidence from `cornerstone security vs3-local-checkpoint --json`.
- Add a dedicated checkpoint condition and negative evidence counter.
- Add a direct regression test for a tampered failed/overclaiming lint report.
- Regenerate local checkpoint evidence after the code change.

Out of scope:

- New production, on-prem, real IdP, real network, live-provider, migration/restore, independent security review, or human UX acceptance proof.
- Promoting `VS3-L` to `VS3-P`.
- Treating generated human-gate templates, review kits, or structural validation as signed human evidence.

Done criteria:

- A clean lint report keeps `overclaim_lint_report_passed_without_overclaims=true`.
- A tampered lint report with matching source identity but `status=failed`, `claim_boundary.vs3_p=READY`, non-empty `claim_boundary_overclaim_fields`, or non-zero overclaim counters returns exit 4.
- The failed checkpoint sets `VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED`, keeps `vs3_l` and `vs3_p` as `NOT_CLAIMED`, and records exact semantic error codes.

## Full Scenario Mapping

Matrix scope:

```text
total=57
MUST_PASS=42
REGRESSION=8
HUMAN_REQUIRED=7
phases: VS3-0=4, VS3-1=5, VS3-2=6, VS3-3=5, VS3-4=6, VS3-5=6, VS3-6=7, VS3-7=3, Final gate=8, Human gate=7
```

Directly covered in this slice:

| Scenario ID | Priority | Phase | Slice classification | Pass condition for this slice |
|---|---:|---|---|---|
| VS3-GATE-003 | MUST_PASS | VS3-0 | direct | The local checkpoint cannot pass if the overclaim lint report itself is failed or contains VS3-P/production/human overclaim signals. |
| VS3-GATE-004 | MUST_PASS | VS3-0 | direct | Native `cornerstone security vs3-local-checkpoint --json` exposes the lint semantic condition, semantic error codes, and negative evidence. |
| VS3-REG-004 | REGRESSION | Final gate | direct | Local checkpoint coverage catches a tampered evidence artifact instead of silently preserving the local claim. |
| VS3-REG-005 | REGRESSION | Final gate | direct | Local/dev proof cannot be converted into VS3-P, production/on-prem, security acceptance, or human acceptance through report tampering. |

Supporting / existing evidence:

| Scenario IDs | Priority | Phase | Slice classification | Guard surface |
|---|---:|---|---|---|
| VS3-GATE-001 | MUST_PASS | VS3-0 | existing local evidence | Reconciliation report identity remains recorded and checked by overclaim lint and local checkpoint. |
| VS3-GATE-002 | MUST_PASS | VS3-0 | existing local evidence | Contract/matrix/goal prompt remain hash-backed by the local checkpoint manifest. |
| VS3-CTX-001 through VS3-CTX-005 | MUST_PASS | VS3-1 | existing local evidence | RequestContext proof remains component-identity guarded by the aggregate scenario report and checkpoint. |
| VS3-RLS-001 through VS3-RLS-006 | MUST_PASS | VS3-2 | existing local evidence | Postgres/RLS proof remains component-identity guarded by the aggregate scenario report and checkpoint. |
| VS3-OPA-001 through VS3-OPA-005 | MUST_PASS | VS3-3 | existing local evidence | OPA/Rego proof remains component-identity guarded by the aggregate scenario report and checkpoint. |
| VS3-EGR-001 through VS3-EGR-006 | MUST_PASS | VS3-4 | existing local evidence | Egress/sandbox proof remains component-identity guarded by the aggregate scenario report and checkpoint. |
| VS3-CON-001 through VS3-CON-006 | MUST_PASS | VS3-5 | existing local evidence | ConnectorHub/source proof remains component-identity guarded; physical-device/live capture still requires human evidence where applicable. |
| VS3-TOOL-001 through VS3-TOOL-007 | MUST_PASS | VS3-6 | existing local evidence | Tool SDK/signed registry proof remains component-identity guarded by the aggregate scenario report and checkpoint. |
| VS3-OBS-001 through VS3-OBS-003 | MUST_PASS | VS3-7 | existing local evidence | Observability/audit proof remains component-identity guarded by the aggregate scenario report and checkpoint. |
| VS3-REG-001, VS3-REG-002, VS3-REG-003, VS3-REG-006, VS3-REG-007, VS3-REG-008 | REGRESSION | Final gate | existing local evidence | Final regression proof remains component-identity guarded and no new UI/human/dependency claim is made. |

Human-required rows remain blocked:

| Scenario IDs | Priority | Phase | Required evidence before promotion |
|---|---:|---|---|
| VS3-H01 | HUMAN_REQUIRED | Human gate | Dated signed architecture/security/dependency/migration approval. |
| VS3-H02 | HUMAN_REQUIRED | Human gate | Independent security review and retest. |
| VS3-H03 | HUMAN_REQUIRED | Human gate | Real IdP mapping and revocation evidence. |
| VS3-H04 | HUMAN_REQUIRED | Human gate | Real on-prem network/firewall/proxy/service-mesh evidence. |
| VS3-H05 | HUMAN_REQUIRED | Human gate | Approved live-provider rehearsal. |
| VS3-H06 | HUMAN_REQUIRED | Human gate | Human operator UX/trust review. |
| VS3-H07 | HUMAN_REQUIRED | Human gate | Signed migration/backup/restore drill. |

## Implementation Summary

Changed:

- `packages/cornerstone_cli/main.py`
  - Extended `_vs3_local_checkpoint_overclaim_lint_source_identity(...)` with lint report semantic checks.
  - Added fields for lint schema validity, status, claim-boundary safety, empty overclaim fields, zero negative-evidence counters, and semantic error codes.
  - Added checkpoint condition `overclaim_lint_report_passed_without_overclaims`.
  - Added negative evidence counter `overclaim_lint_report_failed_or_overclaiming`.
- `tests/scenario/test_scaffold_cli.py`
  - Positive checkpoint test now asserts the lint report semantic fields.
  - Added `test_vs3_local_checkpoint_rejects_failed_overclaim_lint_report`.

## Verification Evidence

Compile check:

```text
python3 -m compileall packages/cornerstone_cli
```

Result:

```text
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_overclaim_lint_source_identity \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_failed_overclaim_lint_report \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_overclaim_lint_cli_preserves_no_claim_boundary
```

Result:

```text
Ran 4 tests in 81.400s
OK
```

Regenerated local checkpoint:

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
scenario_count=57
pass=50
human_required=7
blocking=0
vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p_claim=NOT_CLAIMED
overclaim_lint_source_reconciliation_matches_current_file=true
overclaim_lint_report_passed_without_overclaims=true
overclaim_lint_report_failed_or_overclaiming=0
```

Direct tamper reproduction after the fix:

```text
mutate reports/security/vs3-overclaim-lint.json:
  status=failed
  claim_boundary.vs3_p=READY
  claim_boundary_overclaim_fields=["vs3_p"]
  negative_evidence.claim_boundary_overclaim_count=1
  negative_evidence.vs3_p_claimed=1

cornerstone security vs3-local-checkpoint --json
```

Result:

```text
returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
claim_boundary {'vs3_l': 'NOT_CLAIMED', 'vs3_p': 'NOT_CLAIMED', 'production_onprem': 'NOT_CLAIMED', 'security_acceptance': 'NOT_CLAIMED', 'human_acceptance': 'NOT_CLAIMED'}
failed_conditions ['overclaim_lint_report_passed_without_overclaims']
overclaim_identity {'matches_current_source_file': True, 'report_semantics_passed': False, 'lint_report_status_passed': False, 'lint_report_claim_boundary_safe': False, 'lint_report_claim_boundary_overclaim_fields_empty': False, 'lint_report_negative_evidence_zero': False, 'error_code': 'CS_VS3_OVERCLAIM_LINT_REPORT_FAILED_OR_OVERCLAIMING', 'semantic_error_codes': ['CS_VS3_OVERCLAIM_LINT_STATUS_NOT_PASSED', 'CS_VS3_OVERCLAIM_LINT_CLAIM_BOUNDARY_UNSAFE', 'CS_VS3_OVERCLAIM_LINT_OVERCLAIM_FIELDS_PRESENT', 'CS_VS3_OVERCLAIM_LINT_NEGATIVE_EVIDENCE_NONZERO']}
negative {'overclaim_lint_source_reconciliation_mismatches': 0, 'overclaim_lint_source_reconciliation_missing_or_invalid': 0, 'overclaim_lint_report_failed_or_overclaiming': 1, 'vs3_p_claimed_by_checkpoint': 0, 'production_readiness_claimed_by_checkpoint': 0, 'security_acceptance_claimed_by_checkpoint': 0, 'human_acceptance_claimed_by_checkpoint': 0}
```

## Pass / Fail Criteria

PASS for this slice:

- `checkpoint_conditions.overclaim_lint_report_passed_without_overclaims == true` for the clean lint report.
- `negative_evidence.overclaim_lint_report_failed_or_overclaiming == 0` for the clean lint report.
- Tampered failed/overclaiming lint report returns exit 4.
- Tampered failed/overclaiming lint report leaves `vs3_l`, `vs3_p`, production/on-prem, security acceptance, and human acceptance as `NOT_CLAIMED`.
- Tampered failed/overclaiming lint report records `CS_VS3_OVERCLAIM_LINT_REPORT_FAILED_OR_OVERCLAIMING`.

FAIL for this slice:

- The checkpoint passes when the lint report says `status=failed`.
- The checkpoint passes when lint `claim_boundary` claims VS3-P, production/on-prem, real IdP, real network, live-provider, migration/restore, security acceptance, or human acceptance.
- The checkpoint passes when lint overclaim fields or negative overclaim counters are non-zero.
- The failed checkpoint preserves a local VS3-L claim after lint semantics fail.

## Remaining Human Gates

Still `HUMAN_REQUIRED`:

- `VS3-H01`: architecture/security/dependency/migration approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation.
- `VS3-H04`: real on-prem network controls.
- `VS3-H05`: live ConnectorHub/provider rehearsal.
- `VS3-H06`: operator UX/trust review.
- `VS3-H07`: migration/backup/restore drill.

This checkpoint validates local report semantics only. It does not promote any human-required row, dependency, production, or VS3-P claim.
