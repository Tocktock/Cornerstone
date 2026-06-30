# VS3 Source Tree Generated Evidence Boundary Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint.
**Scope:** VS3 source-tree freshness guard for generated evidence artifacts.
**Proof boundary:** Local deterministic proof only. This checkpoint does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:

- Prevent VS3 local/dev reports from becoming stale only because a checkpoint artifact under `docs/verification-reports/` is written after verification.
- Preserve strict source freshness for code, scenario contracts, fixtures, policies, and other source-bearing files.

Non-scope:

- No VS3-P or on-prem readiness claim.
- No live provider, real network, real IdP, migration/restore, independent security review, or human UX acceptance proof.
- No broad refactor of VS3 component proof generation.

## Full Scenario Mapping

| Classification | Scenario rows | Reason |
|---|---|---|
| In this slice | `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005` | The slice hardens report/gate claim boundaries, native CLI verification, scenario coverage guard behavior, and overclaim-safe local evidence handling. |
| Guarded by current aggregate report, not changed here | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | Their behavior and pass/fail criteria remain exactly as defined in the VS3 matrix and current component proofs. This slice only changes how generated checkpoint evidence participates in source-tree freshness. |
| Human required | `VS3-H01` through `VS3-H07` | These rows require signed human or external evidence and remain `HUMAN_REQUIRED`. |
| Out of scope | VS3-P, production/on-prem, real IdP, live provider, real network, migration/restore, independent security review, human UX acceptance | No matching human/external proof surface was used. |

## Current Behavior Found

Before this slice, the current VS3 scenario report contained all 57 matrix rows and had row statuses `50 PASS` and `7 HUMAN_REQUIRED`, but the scenario gate rejected it:

```text
gate_status failed
error_codes ['CS_VS3_COMPONENT_PROOF_INVALID', 'CS_VS3_SOURCE_TREE_METADATA_STALE']
component_proof_validation failed
source_tree_current_validation failed
```

The only added source-snapshot path relative to the recorded report was:

```text
docs/verification-reports/VS3_SCENARIO_GATE_COMPONENT_PROOF_GUARD_CHECKPOINT_2026-06-30.md
```

That means a generated checkpoint Markdown file could invalidate VS3 source-tree freshness after the report was generated.

## Change

- `packages/cornerstone_cli/acceptance.py` now treats `docs/verification-reports/` as generated evidence for source-snapshot purposes.
- `tests/scenario/test_scaffold_cli.py` extends the source-snapshot regression so generated verification-report Markdown is excluded while source files such as `packages/cornerstone_cli/sample.py` remain hash-bearing.

## Evidence

Focused unit test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_source_snapshot_excludes_generated_report_roots
.
----------------------------------------------------------------------
Ran 1 test in 0.002s

OK
```

Source-snapshot probe after the patch:

```text
recorded snapshot paths 100 current 16
added source paths []
changed source paths ['packages/cornerstone_cli/acceptance.py', 'tests/scenario/test_scaffold_cli.py']
contains checkpoint? False
```

Native VS3 verify/gate after regenerating the report:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
status success
errors []
source_hash c3cf0de3922fceca1684f55708c584a17315f571a78719b8ee2f61b83773d6d3
dirty_count 132
snapshot_count 16

./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
status success
errors []
component passed
source_tree_current passed
source_hash c3cf0de3922fceca1684f55708c584a17315f571a78719b8ee2f61b83773d6d3
dirty_count 132
snapshot_count 16
```

## Decision

The generated verification-report checkpoint surface is evidence, not source. Excluding `docs/verification-reports/` from the source snapshot prevents a self-invalidating evidence loop while preserving source freshness for code, contracts, fixtures, policies, and other non-generated dirty paths.

## Remaining Human Gates

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`. No local deterministic check in this slice can satisfy those rows.

## Next Recommended Slice

Continue with a narrow VS3 gate guard or component-proof guard only after rechecking the current scenario gate. Do not widen into VS3-P, live-provider, real-IdP, real-network, migration/restore, security-acceptance, or human-acceptance claims.
