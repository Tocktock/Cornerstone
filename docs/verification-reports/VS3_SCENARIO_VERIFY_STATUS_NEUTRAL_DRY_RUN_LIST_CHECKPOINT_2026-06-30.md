# VS3 Scenario Verify Status-Neutral Dry-Run/List Checkpoint - 2026-06-30

**Status:** Local verifier CLI parity checkpoint.
**Scope:** `VS3-GATE-004`.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-L completion, VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Provide the native status-neutral `cornerstone scenario verify vs3-onprem-trusted-extension --dry-run --json` and `--list --json` paths required by the VS3 contract before full implementation or local-dev assurance claims.

In scope:
- Parser support for `scenario verify ... --dry-run` and `scenario verify ... --list`.
- Status-neutral full-row VS3 mapping output.
- Human-required rows preserved as `HUMAN_REQUIRED`.
- CLI transcript, counts, proof boundary, and gate metadata in JSON.

Out of scope:
- New VS3 runtime security substrate behavior.
- Full VS3-L completion.
- Any production, live-provider, real-network, real-IdP, migration, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-004`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001` through `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Before Evidence

Commands:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --dry-run --json
./cornerstone scenario verify vs3-onprem-trusted-extension --list --json
```

Observed result before this checkpoint:

```text
dry_run_exit=2
cornerstone: error: unrecognized arguments: --dry-run

list_exit=2
cornerstone: error: unrecognized arguments: --list
```

Interpretation:
- `VS3-GATE-004` explicitly requires a native verifier dry-run/list/coverage path.
- The prior CLI supported full verification but not the status-neutral dry-run/list overlay.

## Change

`cornerstone scenario verify vs3-onprem-trusted-extension` now supports:

- `--dry-run --json`
- `--list --json`

Both paths emit:

- all 57 VS3 rows;
- 42 `MUST_PASS`, 8 `REGRESSION`, 7 `HUMAN_REQUIRED` counts;
- 50 AI-verifiable rows as `NOT_RUN`;
- 7 human rows as `HUMAN_REQUIRED`;
- `PASS=0`;
- status-neutral `product_feature_claims`;
- CLI transcript;
- gate metadata;
- proof boundaries that leave VS3-L and VS3-P unclaimed.

## After Evidence

Focused unittest:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_dry_run_and_list_are_status_neutral
```

Result:

```text
Ran 1 test in 0.198s
OK
```

Direct CLI probes:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --dry-run --json
./cornerstone scenario verify vs3-onprem-trusted-extension --list --json
```

Dry-run result:

```text
status success
mode dry_run
schema_version cs.vs3_onprem_trusted_extension.dry_run.v0
summary {'blocking': 50, 'fail': 0, 'human_required': 7, 'not_run': 50, 'not_verified': 0, 'pass': 0, 'product_feature_claims': 'STATUS_NEUTRAL_VS3_VERIFY_DRY_RUN_ONLY', 'scenario_count': 57}
counts {'HUMAN_REQUIRED': 7, 'MUST_PASS': 42, 'REGRESSION': 8}
proof_boundary {'vs3_l': 'NOT_CLAIMED_BY_DRY_RUN', 'vs3_p': 'NOT_CLAIMED', 'production_onprem': 'NOT_CLAIMED', 'security_acceptance': 'NOT_CLAIMED', 'human_acceptance': 'NOT_CLAIMED'}
command ['cornerstone', 'scenario', 'verify', 'vs3-onprem-trusted-extension', '--dry-run', '--json']
```

List result:

```text
status success
mode list
schema_version cs.vs3_onprem_trusted_extension.dry_run.v0
summary {'blocking': 50, 'fail': 0, 'human_required': 7, 'not_run': 50, 'not_verified': 0, 'pass': 0, 'product_feature_claims': 'STATUS_NEUTRAL_VS3_VERIFY_DRY_RUN_ONLY', 'scenario_count': 57}
counts {'HUMAN_REQUIRED': 7, 'MUST_PASS': 42, 'REGRESSION': 8}
proof_boundary {'vs3_l': 'NOT_CLAIMED_BY_DRY_RUN', 'vs3_p': 'NOT_CLAIMED', 'production_onprem': 'NOT_CLAIMED', 'security_acceptance': 'NOT_CLAIMED', 'human_acceptance': 'NOT_CLAIMED'}
command ['cornerstone', 'scenario', 'verify', 'vs3-onprem-trusted-extension', '--list', '--json']
```

Adjacent tests:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_dry_run_and_list_are_status_neutral \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript
```

Result:

```text
Ran 3 tests in 26.983s
OK
```

Broader gate subset:

```bash
python3 -m unittest $(cat /tmp/cs-vs3-gate-tests.args)
```

Result:

```text
Ran 34 tests in 19.183s
OK
```

## Verification Note

Artifact-generating VS3 checks should not be run in parallel with gate tests that inspect shared generated report/browser paths. A parallel run produced transient row-reference failures; rerunning the same gate subset sequentially passed.

## Remaining Proof Surfaces

- VS3-H01 through VS3-H07 remain `HUMAN_REQUIRED`.
- This checkpoint proves only native status-neutral dry-run/list CLI coverage for `VS3-GATE-004`.
- It does not replace runtime proof for RequestContext, RLS, OPA, egress, ConnectorHub, trusted tool registry, operator status, or final VS0/VS1 regression evidence.

## Decision

Continue to the next small VS3 verifier or runtime substrate slice. Do not widen from this checkpoint into VS3-L completion, VS3-P, or human/on-prem claims.
