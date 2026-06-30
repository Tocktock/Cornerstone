# VS3 Source Tree Source Document Boundary Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint.
**Scope:** Source/evidence boundary regression for VS3 source-tree freshness.
**Proof boundary:** Local deterministic proof only. This checkpoint does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:

- Guard the previous generated-evidence exclusion so it cannot silently exclude source-bearing docs.
- Prove `docs/verification-reports/` is treated as generated evidence while `docs/scenario-contracts/` remains hash-bearing source.

Non-scope:

- No VS3-P or on-prem readiness claim.
- No live provider, real network, real IdP, migration/restore, independent security review, or human UX acceptance proof.
- No broad source hashing redesign.

## Full Scenario Mapping

| Classification | Scenario rows | Reason |
|---|---|---|
| In this slice | `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005` | This slice hardens local report/gate source freshness, CLI-native scenario verification, coverage guard behavior, and overclaim-safe evidence treatment. |
| Guarded by current aggregate report, not changed here | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | Their behavior and pass/fail criteria remain exactly as defined in the VS3 matrix and current component proofs. |
| Human required | `VS3-H01` through `VS3-H07` | These rows require signed human or external evidence and remain `HUMAN_REQUIRED`. |
| Out of scope | VS3-P, production/on-prem, real IdP, live provider, real network, migration/restore, independent security review, human UX acceptance | No matching human/external proof surface was used. |

## Change

- `tests/scenario/test_scaffold_cli.py` now creates both a generated verification report and a source contract in the snapshot fixture.
- The regression asserts that `docs/scenario-contracts/VS3_TEST_CONTRACT.md` remains in the hashable source snapshot while `docs/verification-reports/...` is excluded.

## Evidence

Focused unit test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_source_snapshot_excludes_generated_report_roots
.
----------------------------------------------------------------------
Ran 1 test in 0.002s

OK
```

Source-snapshot probe:

```text
snapshot_count 16
verification_report_paths_included []
scenario_contract_paths_included ['docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md']
test_file_included True
```

Pre-regeneration stale gate, proving source-bearing test changes still invalidate the report:

```text
status failed
errors ['CS_VS3_COMPONENT_PROOF_INVALID', 'CS_VS3_SOURCE_TREE_METADATA_STALE']
source_tree_current failed
mismatches ['verified_source_worktree_hash_mismatch']
```

Native VS3 verify/gate after regenerating the report:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
status success
errors []
source_hash 863f1d35e6bf1fc25350179f5f4b9cd5b23f3b0c5cfa738d986e8184251bf1bc
snapshot_count 16
verification_report_paths []
scenario_contract_paths ['docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md']

./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
status success
errors []
component passed
source_tree_current passed
source_hash 863f1d35e6bf1fc25350179f5f4b9cd5b23f3b0c5cfa738d986e8184251bf1bc
snapshot_count 16
verification_report_paths []
scenario_contract_paths ['docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md']
```

## Decision

The source-tree boundary now has a two-sided regression: generated verification-report evidence is excluded from source freshness, but scenario contracts remain source-bearing. This keeps checkpoint evidence from self-invalidating reports without weakening contract/code freshness.

## Remaining Human Gates

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`. This slice does not supply architecture approval, independent security review, real IdP proof, real network proof, live provider proof, human UX acceptance, or migration/restore evidence.

## Next Recommended Slice

Continue with the next narrow VS3 gate/component-proof guard after rechecking the current scenario gate. Do not widen into VS3-P or any human/external readiness claim.
