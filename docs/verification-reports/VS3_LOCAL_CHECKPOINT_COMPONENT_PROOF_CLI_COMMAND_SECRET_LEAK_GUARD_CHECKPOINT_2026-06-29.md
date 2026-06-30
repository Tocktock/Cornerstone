# VS3 Local Checkpoint - Component Proof CLI Command Secret Leak Guard

**Date:** 2026-06-29 KST
**Status:** PASS for this AI-verifiable verifier slice
**Scope:** VS3-L local/dev checkpoint command-transcript evidence hardening only
**Not claimed:** VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance

## Slice Contract

Goal:

- `cornerstone security vs3-local-checkpoint --json` must reject any component-proof command transcript that contains an unredacted secret-like value.
- The guard must scan the whole transcript entry, including command, args, stdout/stderr tails, embedded `stdout_json`, refs, and future transcript fields.
- A canary shaped like `sk-*` or `ghp_*` in command evidence must fail as command-evidence shape failure.

In scope:

- Detect unredacted secret-like values in VS3 component-proof command transcripts.
- Add a focused regression test for a leaked provider-token canary in `stderr_tail`.
- Preserve the existing proof boundary: local component proof can support only VS3-L local/dev assurance and cannot claim VS3-P.

Out of scope:

- Production/on-prem deployment.
- Real IdP, real network, live provider, migration/restore, independent security review, or human operator acceptance.
- New product capability implementation beyond the local checkpoint verifier.

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
| `in_this_slice` | `VS3-GATE-004`, `VS3-CON-003`, `VS3-REG-004`, `VS3-REG-005` | Native checkpoint CLI, focused unittest, manual tamper before/after evidence, component proof secret scan | This is a verifier/evidence-shape guard for native VS3 CLI proof, ConnectorHub credential redaction, coverage retention, and overclaim-safe local evidence. |
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001`..`VS3-CTX-005`, `VS3-RLS-001`..`VS3-RLS-006`, `VS3-OPA-001`..`VS3-OPA-005`, `VS3-EGR-001`..`VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-004`..`VS3-CON-006`, `VS3-TOOL-001`..`VS3-TOOL-007`, `VS3-OBS-001`..`VS3-OBS-003`, `VS3-REG-001`..`VS3-REG-003`, `VS3-REG-006`..`VS3-REG-008` | Each row's native CLI/API/UI/DB/policy/audit evidence from the VS3 contract and matrix | Not required to close this checkpoint guard. |
| `HUMAN_REQUIRED` | `VS3-H01`..`VS3-H07` | Dated signed human/external evidence | Human/security/external proof cannot be converted to AI PASS. |
| `blocked` | none | n/a | No blocker for this slice. |
| `out_of_scope` | VS3-P and all production/live/human acceptance claims | Human/external proof surfaces | Explicitly outside this local verifier slice. |

## Baseline Gap

Before the patch, this controlled tamper passed:

- Tampered report: `reports/security/vs3-request-context-proof.json`
- Tamper: only `command_transcripts[0].stderr_tail` was changed to contain `provider error token sk-vs3secret0000 should have been redacted`.
- Captured evidence: `/tmp/vs3-component-transcript-secret-leak-before.json`

Observed pre-patch result:

```text
checkpoint_returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
shape_failures 0
request_context_shape_success True
request_context_invalid_entries []
```

This was a proof-integrity gap: the checkpoint accepted command evidence containing an unredacted secret-like canary.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
  - Imported `count_unredacted_secrets`.
  - `_vs3_cli_command_transcript_errors` now serializes each transcript entry and rejects it with `unredacted_secret_in_transcript` when the existing secret scanner finds an unredacted secret-like value.
  - The check runs before field-specific command validation, so it covers current and future transcript fields.
- `tests/scenario/test_scaffold_cli.py`
  - Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_secret_leak`.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_secret_leak \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_stdout_json_execution_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_stdout_json_overclaim
...
----------------------------------------------------------------------
Ran 3 tests in 94.633s

OK
```

Post-patch controlled tamper:

- Captured evidence: `/tmp/vs3-component-transcript-secret-leak-after.json`

```text
checkpoint_returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
shape_failures 1
request_context_shape_success False
request_context_invalid_entries [{'entry_id': '0', 'errors': ['unredacted_secret_in_transcript']}]
```

Untampered local checkpoint:

```text
$ PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json > /tmp/vs3-local-checkpoint-secret-leak-output.json

status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
shape_failures 0
missing_evidence_failures 0
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

- A secret-like canary in any component-proof command transcript field must fail the local checkpoint.
- The failure must be classified as command transcript evidence shape failure, not missing evidence.
- The checkpoint must keep `vs3_p`, production/on-prem readiness, security acceptance, and human acceptance as `NOT_CLAIMED`.
- Untampered current component reports must remain accepted by the checkpoint.
- Current component proof and scenario reports must scan with zero unredacted secret findings.

FAIL for this slice:

- A tampered command transcript containing an unredacted secret-like value can pass.
- The verifier reports VS3-P or production/security/human acceptance from local proof.
- Existing valid component proof reports fail due to the new guard.
- Current VS3 proof reports contain unredacted secret findings.

## Remaining Gates

This checkpoint does not complete VS3. The following remain open:

- All `later_slice` AI-owned rows listed above require their own concrete proof.
- `VS3-H01`..`VS3-H07` remain `HUMAN_REQUIRED`.
- VS3-P and all production/live/human/security/migration readiness claims remain unclaimed.
