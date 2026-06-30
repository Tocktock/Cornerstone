# VS3 Local Checkpoint Component Proof Ref Taxonomy Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local verifier hardening checkpoint
**Scope:** VS3-L local/dev checkpoint evidence validation only

## Summary

This slice hardens `cornerstone security vs3-local-checkpoint --json` so a component proof cannot satisfy VS3 local checkpoint reference requirements with arbitrary nonempty strings in `evidence_refs`, `audit_refs`, or `policy_decision_refs`.

This checkpoint does not claim VS3-P, production/on-prem readiness, real IdP readiness, live provider readiness, migration/restore readiness, independent security acceptance, or human UX acceptance.

## Full VS3 Mapping

| Classification | Scenario IDs |
|---|---|
| In this slice | `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005` |
| Later slice | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001..005`, `VS3-RLS-001..006`, `VS3-OPA-001..005`, `VS3-EGR-001..006`, `VS3-CON-001..006`, `VS3-TOOL-001..007`, `VS3-OBS-001..003`, `VS3-REG-001..003`, `VS3-REG-006..008` |
| Human required | `VS3-H01..H07` |
| Out of scope | VS3-P, production/on-prem readiness, real IdP, real network, live provider, migration/restore readiness, independent security review, human UX acceptance |

## Slice Contract

| Item | Contract |
|---|---|
| Goal | Reject forged component-proof refs even when embedded and file copies are internally consistent. |
| Current gap | A reversible pre-patch probe showed malformed refs were not counted as component-proof failures. |
| Expected behavior | Bogus nonempty refs fail local checkpoint semantics and reference counters. |
| CLI path | `cornerstone security vs3-local-checkpoint --json` |
| Evidence required | Reversible tamper probe, focused unittest, adjacent component-proof regression tests, clean checkpoint, doc verifier, whitespace check. |
| Non-scope | No product feature expansion, no human-row PASS, no production/live-provider/security/migration claim. |

## Pre-Patch Probe

The reversible probe changed `reports/security/vs3-request-context-proof.json` and the embedded `request_context_proof` in `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` so all top-level and nested refs became:

```text
evidence_refs = ["not-a-valid-evidence-ref"]
audit_refs = ["not-a-valid-audit-ref"]
policy_decision_refs = ["not-a-valid-policy-ref"]
```

Observed output before this slice:

```text
returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions [
  "human_gate_evidence_status_scenario_report_hash_matches",
  "human_gate_review_kit_scenario_report_hash_matches",
  "vs3_p_gate_scenario_report_hash_matches"
]
component_proof_report_evidence_ref_failures 0
component_proof_report_audit_ref_failures 0
component_proof_report_policy_decision_ref_failures 0
component_proof_report_reference_failures 0
component_proof_report_semantic_failures 0
request_context_proof evidence_refs_success True
request_context_proof audit_refs_success True
request_context_proof policy_decision_refs_success True
request_context_proof semantic_error_codes []
```

Interpretation: the checkpoint failed for derived hash mismatches caused by the tamper, but the component-proof reference validators still accepted bogus nonempty refs.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Verifier behavior added:

- component proof `evidence_refs` must use known local evidence prefixes such as `reports/`, `docs/`, `fixtures/`, `policies/`, `config/`, `artifact:`, and ConnectorHub/audit-state typed refs;
- component proof `audit_refs` must use `audit:` or `audit_`;
- component proof `policy_decision_refs` must use `policy:` or `policy_`;
- malformed refs now produce `CS_VS3_COMPONENT_PROOF_REFS_MALFORMED`;
- the JSON identity exposes malformed ref lists and allowed prefixes;
- shared CLI transcript taxonomy checks are scoped to component-proof transcript validation so unrelated human-gate self transcripts are not over-constrained.

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages:. python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_malformed_refs_even_when_identity_matches
Ran 1 test in 52.248s
OK
```

Adjacent component-proof regressions:

```text
PYTHONPATH=packages:. python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_refs_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_policy_decision_refs_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_malformed_refs_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_scope_extra_key_even_when_all_copies_match \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_source_tree_extra_key_even_when_all_copies_match
Ran 5 tests in 262.113s
OK
```

Clean checkpoint:

```text
PYTHONPATH=packages:. python3 -m cornerstone_cli.main security vs3-local-checkpoint --json
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary {
  "pass": 50,
  "human_required": 7,
  "blocking": 0,
  "component_proof_report_evidence_ref_failures": 0,
  "component_proof_report_audit_ref_failures": 0,
  "component_proof_report_policy_decision_ref_failures": 0,
  "component_proof_report_reference_failures": 0,
  "component_proof_report_semantic_failures": 0,
  "component_proof_report_cli_command_evidence_shape_failures": 0
}
claim_boundary {
  "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED",
  "vs3_p": "NOT_CLAIMED",
  "production_onprem": "NOT_CLAIMED",
  "security_acceptance": "NOT_CLAIMED",
  "human_acceptance": "NOT_CLAIMED"
}
negative_evidence {
  "vs3_p_claimed_by_checkpoint": 0,
  "production_readiness_claimed_by_checkpoint": 0,
  "security_acceptance_claimed_by_checkpoint": 0,
  "human_acceptance_claimed_by_checkpoint": 0
}
```

## Pass / Fail Criteria

| Criterion | Result |
|---|---|
| Malformed component-proof refs are rejected through native CLI checkpoint | PASS |
| Existing missing-ref, scope-key, and source-tree component-proof guards still pass | PASS |
| Clean VS3 local checkpoint still succeeds | PASS |
| Human-required rows remain `HUMAN_REQUIRED` | PASS |
| VS3-P / production / security / human acceptance remains unclaimed | PASS |

## Remaining Proof Surfaces

This is local deterministic verifier proof only. It is not production, live-provider, real IdP, real network, migration/restore, independent security-review, or human UX evidence.
