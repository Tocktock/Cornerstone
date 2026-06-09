# VS-0 Memory Truth Boundary Batch 19 Report - 2026-06-09

Status: PASS for deterministic CLI-native raw-memory source-of-truth regression guard only.
Scope: `CS-REG-005`.

This report does not mark production UI runtime, production API runtime, RBAC/ABAC enforcement, namespace promotion, personal-to-organization memory leakage prevention, or full 206-scenario completion as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies that raw agent memory cannot become canonical truth when it conflicts with durable archive evidence and owner-approved evidence-backed memory.

## Research Checkpoint

- Letta documents persistent agent memory as explicitly managed memory, including archival storage: <https://docs.letta.com/concepts/memory-management>
- MemGPT frames long-running memory around managed context and archival stores rather than treating transient model recall as truth: <https://shishirpatil.github.io/publications/memgpt-2023.pdf>
- W3C PROV frames provenance as evidence for trustworthiness: <https://www.w3.org/TR/prov-overview/>

Best fit for this batch remains the existing deterministic local runtime. It adds a raw-memory candidate state and an explicit conflict-resolution record instead of installing a memory framework or making raw agent recall authoritative.

## Assumptions

- Native CLI JSON is the scaffold verification surface until production UI/API surfaces exist.
- Raw agent memory may exist as a candidate record, but it is not owner-approved and cannot be canonical truth.
- Archive evidence plus owner-approved evidence-backed memory outranks raw agent memory in conflict resolution.

## Out Of Scope

- Production UI/browser walkthrough, production API runtime, cross-namespace memory leakage, explicit namespace promotion, RBAC/ABAC, and any real provider/connector execution.
- `CS-NS-004`, `CS-SEC-004`, and `CS-REG-006` remain `NOT_VERIFIED`.

## Checklist

- [x] Frozen `CS-REG-005` wording inspected.
- [x] README read before coding.
- [x] Research checkpoint completed for source-backed memory and provenance.
- [x] Archive evidence is created from a conversation-turn artifact.
- [x] Owner-approved memory requires an Evidence Bundle and keeps `archive_evidence` as truth foundation.
- [x] Raw agent memory is stored only as `raw_agent_memory`, `unverified`, non-owner-approved, and non-canonical.
- [x] Conflict resolution selects archive evidence instead of raw memory.
- [x] Conflict resolution records evidence refs and an audit event.
- [x] Matrix PASS row backed by a JSON report artifact.
- [x] Unit and shell-gate coverage added.
- [x] No destructive action, external call, secret access, tenant/security mutation, new dependency, or production state change.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-REG-005 | REGRESSION_GUARD | PASS | `reports/scenario/vs0-memory-truth-boundary-2026-06-09.json`, raw-memory conflict transcript |

## Human Required

No human-required item was introduced for this batch. Production visual acceptance remains outside this scaffold slice.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-memory-truth-boundary --json --output reports/scenario/vs0-memory-truth-boundary-2026-06-09.json
# status: success
# scenario_set: vs0-memory-truth-boundary
# summary.pass: 1
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_MEMORY_TRUTH_BOUNDARY_ONLY
# memory_truth_evidence.owner_memory_status: owner_approved
# memory_truth_evidence.owner_memory_truth_foundation: archive_evidence
# memory_truth_evidence.raw_memory_status: raw_agent_memory
# memory_truth_evidence.raw_memory_canonical: false
# memory_truth_evidence.conflict_selected_truth_foundation: archive_evidence
# memory_truth_evidence.conflict_raw_memory_used_as_truth: false
# memory_truth_evidence.conflict_answer_based_on: archive_evidence
# memory_truth_evidence.audit_event_count: 7
# negative_evidence.owner_memory_without_evidence: 0
# negative_evidence.raw_agent_memory_canonical: 0
# negative_evidence.raw_agent_memory_owner_approved: 0
# negative_evidence.conflict_selected_raw_memory: 0
# negative_evidence.conflict_truth_foundation_not_archive_evidence: 0
# negative_evidence.conflict_without_audit: 0
# negative_evidence.real_external_http_calls: 0
```

## Evidence Summary

- `conversation start` creates an immutable artifact stating the Project Atlas launch date is Friday.
- `memory create` creates owner-approved memory from the Evidence Bundle with `canonical_truth_foundation=archive_evidence`.
- `memory raw-agent-note` creates a contradictory Monday memory as `raw_agent_memory`, `trust_state=unverified`, `raw_agent_memory_canonical=false`, and no evidence refs.
- `memory conflict-test` resolves the conflict with `selected_truth_foundation=archive_evidence`, `raw_agent_memory_used_as_truth=false`, and audit refs.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` now marks `CS-REG-005` as `PASS`.

Current full matrix after this batch:

- `PASS`: 62
- `NOT_VERIFIED`: 144
- `FAIL`: 0
- `NOT_RUN`: 0

Current VS-0 subset after this batch:

- `PASS`: 55
- `NOT_VERIFIED`: 3

Remaining VS-0 rows:

- `CS-NS-004`
- `CS-SEC-004`
- `CS-REG-006`

## Gaps

- Full 206-scenario PASS remains incomplete.
- Production UI/API product surfaces remain missing; this batch uses CLI-native JSON as scaffold evidence.
- Explicit namespace promotion, RBAC/ABAC, and personal-memory leakage tests remain `NOT_VERIFIED` and require approval before tenant/security semantics are changed.

## Risks

- Raw memory conflict behavior is deterministic local scaffold behavior, not production persistence.
- This batch proves source-of-truth priority within one scope; it does not prove cross-namespace leakage prevention.
- Future UI/API implementations must preserve raw memory as non-canonical and keep archive/evidence plus owner approval as the durable truth foundation.
