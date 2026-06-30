# VS3 Local Checkpoint Human-Gate Package Digest Guard Checkpoint

**Date:** 2026-06-29 KST
**Scope:** VS3 local checkpoint digest-integrity guard for individual `VS3-H01` through `VS3-H07` human-gate package bodies.
**Status:** Local deterministic verifier-hardening slice verified.
**Verdict:** `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` remains available only when generated human-gate package digests match the current package bodies. `VS3-P`, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Repair generated `package_digest_sha256` values after human-gate package audit refs are attached.
- Make `cornerstone security vs3-local-checkpoint --json` recompute every `VS3-H01` through `VS3-H07` package digest and reject stale package-body drift.

In scope:

- Recompute package digests after final package mutation in `run_vs3_observability_proof`.
- Validate every package digest in the local checkpoint by hashing the package body with `package_digest_sha256` removed.
- Add checkpoint condition `human_gate_package_body_digests_match`.
- Add negative-evidence counter `human_gate_package_digest_mismatches`.
- Add focused positive and stale-digest regression coverage.
- Regenerate observability, aggregate scenario, human-gate, VS3-P gate, and local checkpoint reports so persisted package digests are internally consistent.

Out of scope:

- Accepting or validating real human evidence.
- Promoting `VS3-H01` through `VS3-H07` from `HUMAN_REQUIRED` to `PASS`.
- Unlocking or claiming `VS3-P`.
- Production/on-prem, real IdP, real network, live-provider, migration/restore, independent security, or human UX acceptance claims.

## Full Scenario Mapping

The frozen VS3 matrix remains unchanged:

| Type | Count | Slice classification |
|---|---:|---|
| MUST_PASS | 42 | `VS3-OBS-003` directly hardened through digest validation for seven human-gate packages. `VS3-GATE-003` and `VS3-GATE-004` are guard rails for no-overclaim and native CLI checkpoint evidence. Other AI rows remain carried by the current aggregate VS3 report or later-slice scope. |
| REGRESSION | 8 | `VS3-REG-004` and `VS3-REG-005` are directly hardened because stale package-body drift now fails the checkpoint without creating any stronger readiness claim. Remaining regression rows stay final-gate coverage. |
| HUMAN_REQUIRED | 7 | `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED` and block VS3-P. |
| Total | 57 | No scenario rows were added, removed, or reclassified by this slice. |

Directly affected scenarios:

| ID | Type | Expected behavior | Verification | Result |
|---|---|---|---|---|
| `VS3-OBS-003` | MUST_PASS | Human-gate packages prepare evidence without treating generated artifacts as human evidence. | Verify all seven package body digests during local checkpoint. | Local deterministic digest guard added; human rows remain `HUMAN_REQUIRED`. |
| `VS3-GATE-003` | MUST_PASS | Local report/checkpoint cannot overclaim VS3-P, production/on-prem, security acceptance, or human acceptance. | Tamper package body without updating digest and rerun checkpoint. | Checkpoint exits 4 and keeps all readiness/acceptance claims `NOT_CLAIMED`. |
| `VS3-GATE-004` | MUST_PASS | Native CLI checkpoint emits deterministic JSON conditions, counters, evidence refs, and exit codes. | Run `cornerstone security vs3-local-checkpoint --json`. | CLI path emits digest condition and negative evidence. |
| `VS3-REG-004` | REGRESSION | Scenario/evidence coverage cannot silently drift. | Mutate package body after package generation. | Drift is detected by digest mismatch. |
| `VS3-REG-005` | REGRESSION | Local/dev proof cannot be described as production, human-accepted, security-accepted, or migration-ready. | Digest-drift fixture. | Failure emits no stronger readiness or acceptance claim. |

## Before Evidence

Before this slice, all seven persisted package digests were stale, but the local checkpoint still passed and had no digest condition:

```text
package_digest_results [('VS3-H01.json', False, 'a5cebdda3c1e', '6059d3aacbc9'), ('VS3-H02.json', False, 'd06cb6f58146', '2a8599809005'), ('VS3-H03.json', False, '494357ea60cb', '509dafbfc256'), ('VS3-H04.json', False, '683bb1ede44c', '773d9edbefc3'), ('VS3-H05.json', False, 'b99a5a012ffb', 'b30ed7df4285'), ('VS3-H06.json', False, 'e84181d2e595', '6d45031fab70'), ('VS3-H07.json', False, '133c8b306130', '62d161b618d5')]
checkpoint_rc 0
checkpoint_status success
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
digest_condition_present False
digest_negative_present False
```

Root cause:

- `_vs3_human_gate_package` set `package_digest_sha256`.
- `run_vs3_observability_proof` appended `audit_refs` afterward.
- The package file was written after the mutation, but the digest was not recomputed.
- The local checkpoint did not validate `package_digest_sha256`, so the drift was invisible.

## Implementation Summary

Updated `packages/cornerstone_cli/scenarios.py`:

- Recomputes `package_digest_sha256` after appending the generated audit ref and before writing each `VS3-Hxx.json` package.

Updated `packages/cornerstone_cli/main.py`:

- Recomputes each package digest with `package_digest_sha256` removed.
- Adds per-package fields:
  - `package_digest_present`
  - `package_digest_matches`
  - `package_digest_sha256`
  - `computed_package_digest_sha256`
- Adds semantic error `CS_VS3_HUMAN_GATE_PACKAGE_DIGEST_MISMATCH`.
- Adds `digest_mismatch_ids`.
- Adds checkpoint condition `human_gate_package_body_digests_match`.
- Adds negative-evidence counter `human_gate_package_digest_mismatches`.

Updated `tests/scenario/test_scaffold_cli.py`:

- Positive checkpoint test now asserts every package digest is present and matches.
- Added `test_vs3_local_checkpoint_rejects_human_gate_package_digest_drift`.

## Verification Evidence

Code compile:

```text
python3 -m compileall packages/cornerstone_cli
exit 0
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
Compiling 'packages/cornerstone_cli/scenarios.py'...
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_human_gate_package_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_human_gate_package_digest_drift \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_scenario_report_product_feature_claim_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_failed_overclaim_lint_report
```

Result:

```text
exit 0
Ran 5 tests in 137.448s
OK
```

Regenerated package digest check:

```text
VS3-H01.json True f8be9344c97e f8be9344c97e
VS3-H02.json True 5e50fd1accb9 5e50fd1accb9
VS3-H03.json True 2a6762671ce8 2a6762671ce8
VS3-H04.json True cfdccef4b165 cfdccef4b165
VS3-H05.json True 3424ef4bf1ce 3424ef4bf1ce
VS3-H06.json True ae198bba604c ae198bba604c
VS3-H07.json True 45065982b730 45065982b730
```

Aggregate and human-gate regeneration:

```text
PATH="$PWD:$PATH" cornerstone security vs3-observability --json
exit 0
status=success
output_path=/Users/jiyong/playground/Cornerstone/reports/observability/vs3-observability-proof.json
package_count=7
```

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0
status=success
summary={'scenario_count': 57, 'pass': 50, 'human_required': 7, 'blocking': 0, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED'}
```

```text
PATH="$PWD:$PATH" cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --force --use-existing --json --output reports/human-gates/vs3/record-scaffold.json
exit 0
final_verdict=HUMAN_REQUIRED
template_count=7
```

```text
PATH="$PWD:$PATH" cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit 0
final_verdict=HUMAN_REQUIRED
blank_template_pending_count=7
filled_record_count=0
evidence_acceptance_candidate_count=0
vs3_p_claim=NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit 0
status=success
final_verdict=HUMAN_REQUIRED
review_queue_count=7
package_count=7
template_count=7
vs3_p_claim=NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit 4
status=blocked
final_verdict=HUMAN_REQUIRED
unresolved_human_required_rows=7
vs3_p_ready=false
vs3_p_claim=NOT_CLAIMED
```

Clean local checkpoint:

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit 0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
checkpoint_id=vs3l_manifest_9800c370c48fe80b
scenario_count=57
pass=50
human_required=7
blocking=0
human_gate_package_semantic_violations=0
human_gate_package_body_digests_match=true
human_gate_package_digest_mismatches=0
vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p_claim=NOT_CLAIMED
```

Direct stale-digest tamper reproduction:

```text
returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['human_gate_package_body_semantics_safe', 'human_gate_package_body_digests_match']
claim_boundary {'vs3_l': 'NOT_CLAIMED', 'vs3_p': 'NOT_CLAIMED', 'production_onprem': 'NOT_CLAIMED', 'security_acceptance': 'NOT_CLAIMED', 'human_acceptance': 'NOT_CLAIMED'}
package_semantics {'packages_ready': False, 'semantic_error_count': 1, 'invalid_package_ids': ['VS3-H03'], 'digest_mismatch_ids': ['VS3-H03'], 'marked_pass_package_ids': [], 'unlock_or_acceptance_claim_ids': [], 'filled_blank_approval_record_ids': []}
package_row {'package_digest_present': True, 'package_digest_matches': False, 'package_digest_sha256': '2a6762671ce8318ac3e449b08507b777a5767c8dbac0d617df9e8ece8df79599', 'computed_package_digest_sha256': '837bfb96bb1f90e91b406cc9dad3255aa971d9111025da700e1e9d340305c112', 'semantic_error_codes': ['CS_VS3_HUMAN_GATE_PACKAGE_DIGEST_MISMATCH']}
negative_selected {'human_gate_package_semantic_violations': 1, 'human_gate_package_digest_mismatches': 1, 'human_gate_package_status_overclaims': 0, 'human_gate_package_claim_boundary_violations': 0, 'human_gate_package_unlock_or_acceptance_claims': 0, 'human_gate_package_blank_approval_records_filled': 0, 'vs3_p_claimed_by_checkpoint': 0, 'production_readiness_claimed_by_checkpoint': 0, 'security_acceptance_claimed_by_checkpoint': 0, 'human_acceptance_claimed_by_checkpoint': 0}
```

## Pass / Fail Criteria

PASS for this slice:

- All seven generated `VS3-Hxx.json` package digests match their package bodies.
- Clean checkpoint exits 0 and records `human_gate_package_body_digests_match=true`.
- Clean checkpoint records `human_gate_package_digest_mismatches=0`.
- A package body changed after digest generation fails the checkpoint with exit 4.
- Failed digest-drift checkpoint keeps `vs3_l`, `vs3_p`, production/on-prem readiness, security acceptance, and human acceptance unclaimed.

FAIL for this slice:

- Any generated package digest can drift while the checkpoint exits 0.
- The generator writes stale package digests after audit refs are attached.
- Any digest mismatch emits a stronger product, production, security, or human claim.

## Claim Boundary

- `VS3-L`: local/dev assurance remains the maximum current AI-verifiable claim.
- `VS3-P`: `NOT_CLAIMED`.
- Production/on-prem readiness: `NOT_CLAIMED`.
- Security acceptance: `NOT_CLAIMED`.
- Migration/restore readiness: `NOT_CLAIMED`.
- Live-provider readiness: `NOT_CLAIMED`.
- Real IdP and real-network readiness: `NOT_CLAIMED`.
- Human acceptance: `NOT_CLAIMED`.

## Remaining Human Gates

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`. This checkpoint validates only local package digest integrity and does not provide the human/on-prem evidence needed for VS3-P.
