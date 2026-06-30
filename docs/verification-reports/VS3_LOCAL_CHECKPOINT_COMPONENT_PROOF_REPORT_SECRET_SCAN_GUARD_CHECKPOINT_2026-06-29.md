# VS3 Local Checkpoint - Component Proof Report Secret Scan Guard

**Date:** 2026-06-29 KST
**Status:** PASS for this AI-verifiable verifier slice
**Scope:** VS3-L local/dev component-proof report secret-scan hardening only
**Not claimed:** VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance

## Slice Contract

Goal:

- `cornerstone security vs3-local-checkpoint --json` must reject unredacted secret-like values anywhere in embedded and current component proof reports.
- The guard must cover report-level fields outside `command_transcripts`, including future debug/provider payload fields.
- A `ghp_*` or `sk-*` canary outside command transcripts must fail the component proof semantics gate.

In scope:

- Add report-level secret scan counts to each component proof identity.
- Fail component proof semantics when either the embedded scenario-report proof or current proof file contains an unredacted secret-like value.
- Add a focused regression test using a top-level ConnectorHub provider-token canary.

Out of scope:

- Production/on-prem deployment.
- Real IdP, real network, live provider, migration/restore, independent security review, or human operator acceptance.
- New connector/provider capability implementation beyond the local checkpoint verifier.

## Full Scenario Mapping

Counts from `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`:

- Total rows: 57
- MUST_PASS: 42
- REGRESSION: 8
- HUMAN_REQUIRED: 7
- Duplicate IDs: 0

Current slice classification:

| Classification | Scenario IDs | Required proof surface | Reason |
|---|---|---|---|
| `in_this_slice` | `VS3-GATE-004`, `VS3-CON-003`, `VS3-OPA-005`, `VS3-TOOL-005`, `VS3-REG-004`, `VS3-REG-005` | Native checkpoint CLI, focused unittest, manual tamper before/after evidence, final proof-report secret scan | This is a local verifier guard for native VS3 CLI proof, credential redaction, decision/tool secret safety, coverage retention, and overclaim-safe local evidence. |
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001`..`VS3-CTX-005`, `VS3-RLS-001`..`VS3-RLS-006`, `VS3-OPA-001`..`VS3-OPA-004`, `VS3-EGR-001`..`VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-004`..`VS3-CON-006`, `VS3-TOOL-001`..`VS3-TOOL-004`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`..`VS3-OBS-003`, `VS3-REG-001`..`VS3-REG-003`, `VS3-REG-006`..`VS3-REG-008` | Each row's native CLI/API/UI/DB/policy/audit evidence from the VS3 contract and matrix | Not required to close this checkpoint guard. |
| `HUMAN_REQUIRED` | `VS3-H01`..`VS3-H07` | Dated signed human/external evidence | Human/security/external proof cannot be converted to AI PASS. |
| `blocked` | none | n/a | No blocker for this slice. |
| `out_of_scope` | VS3-P and all production/live/human acceptance claims | Human/external proof surfaces | Explicitly outside this local verifier slice. |

## Baseline Gap

Before the patch, this controlled tamper passed:

- Tampered report: `reports/security/vs3-connectorhub-source-proof.json`
- Tamper: added top-level `debug_raw_provider_payload="accidental raw credential ghp_vs3reportsecret0000 should have been redacted"` outside `command_transcripts`.
- Captured evidence: `/tmp/vs3-component-proof-report-secret-leak-before.json`

Observed pre-patch result:

```text
checkpoint_returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
semantic_failures 0
connector_semantics_success True
connector_secret_scan_success None
connector_error_codes []
```

This was a proof-integrity gap: the checkpoint accepted a component proof report with a raw provider-token canary outside CLI command evidence.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
  - `_vs3_local_checkpoint_component_proof_identity` now computes `embedded_unredacted_secret_count` and `file_unredacted_secret_count` for each component proof report.
  - Each component proof identity now exposes `report_secret_scan_success`.
  - Component proof `semantics_success` now requires `report_secret_scan_success`.
  - A report-level secret finding adds `CS_VS3_COMPONENT_PROOF_UNREDACTED_SECRET`.
  - Local checkpoint summary and negative evidence now include `component_proof_report_secret_scan_failures`.
- `tests/scenario/test_scaffold_cli.py`
  - Added `test_vs3_local_checkpoint_rejects_component_proof_report_secret_leak`.
  - Updated the command-transcript secret test because transcript secret leaks now also fail the whole report secret scan.

## Verification Evidence

Syntax checks:

```text
$ python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...

$ python3 -m py_compile tests/scenario/test_scaffold_cli.py
# exit 0, no output
```

Focused regression tests:

```text
$ python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_secret_leak \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_secret_leak \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe
...
----------------------------------------------------------------------
Ran 3 tests in 79.798s

OK
```

Post-patch controlled tamper:

- Captured evidence: `/tmp/vs3-component-proof-report-secret-leak-after.json`

```text
checkpoint_returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
semantic_failures 1
secret_scan_failures 1
connector_semantics_success False
connector_report_secret_scan_success False
connector_embedded_secret_count 1
connector_file_secret_count 1
connector_semantic_error_codes ['CS_VS3_COMPONENT_PROOF_UNREDACTED_SECRET']
```

Untampered local checkpoint after restore:

```text
$ PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json > /tmp/vs3-local-checkpoint-report-secret-scan-output.json

status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
semantic_failures 0
secret_scan_failures 0
shape_failures 0
claim_boundary.vs3_l LOCAL_DEV_ASSURANCE_VERIFIED
claim_boundary.vs3_p NOT_CLAIMED
negative.vs3_p_claimed 0
```

Current component proof and scenario report secret scan:

```text
reports/security/vs3-connectorhub-source-proof.json unredacted_secret_count 0
reports/security/vs3-egress-sandbox-proof.json unredacted_secret_count 0
reports/security/vs3-final-regression-proof.json unredacted_secret_count 0
reports/security/vs3-request-context-proof.json unredacted_secret_count 0
reports/security/vs3-tool-registry-proof.json unredacted_secret_count 0
reports/db/vs3-postgres-rls-proof.json unredacted_secret_count 0
reports/policy/vs3-opa-policy-proof.json unredacted_secret_count 0
reports/observability/vs3-observability-proof.json unredacted_secret_count 0
reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json unredacted_secret_count 0
findings []
```

## Pass / Fail Criteria

PASS for this slice:

- A secret-like canary anywhere in an embedded or current component proof report must fail the local checkpoint.
- The failure must be classified as a component proof semantic failure.
- The checkpoint must keep `vs3_p`, production/on-prem readiness, security acceptance, and human acceptance as `NOT_CLAIMED`.
- Untampered current component reports must remain accepted by the checkpoint.
- Current component proof and scenario reports must scan with zero unredacted secret findings.

FAIL for this slice:

- A component proof report containing an unredacted secret-like value outside command transcripts can pass.
- The verifier reports VS3-P or production/security/human acceptance from local proof.
- Existing valid component proof reports fail due to the new guard.
- Current VS3 proof reports contain unredacted secret findings.

## Remaining Gates

This checkpoint does not complete VS3. The following remain open:

- All `later_slice` AI-owned rows listed above require their own concrete proof.
- `VS3-H01`..`VS3-H07` remain `HUMAN_REQUIRED`.
- VS3-P and all production/live/human/security/migration readiness claims remain unclaimed.
