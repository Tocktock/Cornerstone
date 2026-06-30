# VS3 Local Checkpoint Component Proof Policy Decision Refs Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint.
**Scope:** `cornerstone security vs3-local-checkpoint --json`
**Proof boundary:** Local deterministic verifier evidence only. This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

## Slice Contract

Goal:

- Make the VS3 local checkpoint reject a scenario-backed component proof report when its top-level `policy_decision_refs` is missing or empty, even if the proof file and embedded scenario-report copy still match by hash.

In scope:

- Validate top-level `policy_decision_refs` for scenario-backed component proofs in `packages/cornerstone_cli/main.py`.
- Expose per-proof checkpoint conditions named `component_proof_<key>_policy_decision_refs_present`.
- Expose aggregate counters named `component_proof_report_policy_decision_ref_failures`.
- Add the missing top-level policy decision refs to the local Postgres/RLS component proof generator in `packages/cornerstone_cli/scenarios.py`.
- Add a regression test that removes report-level `policy_decision_refs` while preserving file and embedded identity.

Out of scope:

- No new VS3 product behavior.
- No production, live provider, real IdP, real network, migration, security-review, or human-acceptance evidence.
- No conversion of human-required rows to PASS.
- No dependency changes.

## Full Scenario Mapping

Authoritative matrix:

- `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`
- Parsed total: 57 rows.
- MUST_PASS: 42 rows.
- REGRESSION: 8 rows.
- HUMAN_REQUIRED: 7 rows.
- Duplicate scenario IDs: 0.

Current slice classification:

- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`.
- `guarded_evidence_surface_only`: scenario-backed component proof rows for `VS3-CTX-*`, `VS3-RLS-*`, `VS3-OPA-*`, `VS3-EGR-*`, `VS3-CON-*`, `VS3-TOOL-*`, `VS3-OBS-*`, and `VS3-REG-*`.
- `later_slice`: product behavior implementation beyond this verifier guard for those component rows.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

This slice does not mark any scenario PASS by itself. It strengthens the evidence gate used by the existing VS3 local checkpoint.

## Implementation Notes

Changed verifier behavior:

- Added `_vs3_component_nonempty_string_refs`.
- Added component identity fields:
  - `embedded_policy_decision_refs_present`
  - `file_policy_decision_refs_present`
  - `embedded_policy_decision_ref_count`
  - `file_policy_decision_ref_count`
  - `policy_decision_refs_success`
- Added semantic error code:
  - `CS_VS3_COMPONENT_PROOF_POLICY_DECISION_REFS_MISSING`
- Added local checkpoint condition:
  - `component_proof_<key>_policy_decision_refs_present`
- Added summary and negative-evidence counter:
  - `component_proof_report_policy_decision_ref_failures`

Changed local Postgres/RLS proof behavior:

- `reports/db/vs3-postgres-rls-proof.json` now includes nonempty top-level `policy_decision_refs`.
- Its six command transcripts inherit the same policy decision refs through the existing transcript enrichment path.
- Current generated RLS proof evidence:
  - `policy_decision_refs_count`: 11
  - `transcript_policy_counts`: `[11, 11, 11, 11, 11, 11]`

## Verification Evidence

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_refs_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_policy_decision_refs_even_when_identity_matches

Ran 3 tests in 82.045s
OK
```

Native VS3 CLI evidence:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
scenario_count 57
pass 50
human_required 7
blocking 0
vs3_l LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p NOT_CLAIMED
```

```text
cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit 0
status success
final_verdict HUMAN_REQUIRED
scenario_count 57
pass 50
human_required 7
```

```text
cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit 0
status success
final_verdict HUMAN_REQUIRED
scenario_count 57
pass 50
human_required 7
```

```text
cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit 4
status blocked
final_verdict HUMAN_REQUIRED
unresolved_human_required_rows VS3-H01..VS3-H07
```

```text
cornerstone security vs3-local-checkpoint --output reports/security/vs3-local-checkpoint.json --json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
scenario_count 57
pass 50
human_required 7
blocking 0
component_proof_report_policy_decision_ref_failures 0
component_proof_report_reference_failures 0
vs3_l LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p NOT_CLAIMED
```

```text
cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
scenario_count 57
pass 50
human_required 7
blocking 0
vs3_l LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p NOT_CLAIMED
```

Docs and whitespace:

```text
scripts/verify_sot_docs.sh
PASS: CornerStone SoT docs verified (206 full scenarios, design system, VS-0 scaffold readiness, VS-0 scaffold gate, 58 VS-0 scenarios, CLI native-first gate, local verification plane).
```

```text
git diff --check -- packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py docs/verification-reports/VS3_LOCAL_CHECKPOINT_COMPONENT_PROOF_POLICY_DECISION_REFS_GUARD_CHECKPOINT_2026-06-30.md
exit 0
```

## Remaining Human Gates

Still HUMAN_REQUIRED:

- `VS3-H01` architecture/security ownership approval.
- `VS3-H02` independent security review.
- `VS3-H03` real IdP mapping and revocation evidence.
- `VS3-H04` real network/topology egress evidence.
- `VS3-H05` live ConnectorHub/provider rehearsal.
- `VS3-H06` human operator UX/trust review.
- `VS3-H07` migration/backup/restore drill evidence.

## Decision

The slice is locally verified as a verifier hardening checkpoint. Continue VS3 in small slices. Do not use this checkpoint as production/on-prem, live-provider, real-IdP, migration-readiness, independent-security-acceptance, or human-acceptance evidence.
