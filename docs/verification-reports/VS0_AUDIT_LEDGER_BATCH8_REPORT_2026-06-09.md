# VS-0 Audit Ledger Batch 8 Report - 2026-06-09

Status: PASS for the first tamper-evident audit-ledger runtime slice only.
Scope: `CS-SEC-006`.

This report does not mark RBAC/ABAC, policy-denial UX, action dry-run/execution, connector calls, model routing, Agent Pack activation, autonomy changes, production append-only storage, external anchoring, or Merkle transparency-log infrastructure as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies only the implemented local audit behavior: critical VS-0 scaffold events are appended with reviewable detail, event hashes, previous hashes, and scope; a clean audit verifies; controlled tampering is detected.

## Research Choice

The dominant tamper-evident pattern is append-only logging with cryptographic linkage, often implemented as a Merkle transparency log. RFC 6962 Certificate Transparency defines an append-only Merkle tree model, and Sigstore Rekor is the dominant OSS transparency-log implementation. For this scaffold batch, the safest fit is the existing dependency-free JSONL hash chain because it is deterministic, auditable, and sufficient for local scenario proof without introducing new production infrastructure or dependencies.

References:

- RFC 6962 Certificate Transparency: https://datatracker.ietf.org/doc/rfc6962/
- Sigstore Rekor overview: https://docs.sigstore.dev/logging/overview/

## Assumptions

- The current implemented critical event types are artifact ingestion, artifact read, search snapshot creation, evidence bundle creation, and draft claim creation.
- Local JSONL hash-chain verification is acceptable scaffold evidence for `CS-SEC-006`; production anchoring, Merkle inclusion proofs, and retention controls remain future work.
- Controlled tampering is performed only inside a temporary `tmp/scenario/vs0-audit-ledger-*` state directory.

## Out Of Scope

- Memory writes, claim approval, action dry-run, action execution, connector calls, policy decisions, tool runs, model routing, Agent Pack activation, and autonomy changes that are not yet implemented.
- Production append-only storage, external witness/anchor, cryptographic signing, Merkle inclusion proofs, and export APIs.
- `CS-SEC-004` RBAC/ABAC matrix and `CS-SEC-005` denial-resolution UX.

## Checklist

- [x] Frozen `CS-SEC-006` wording inspected.
- [x] Batch scope limited to implemented audit events.
- [x] CLI-native verifier added.
- [x] Matrix PASS row backed by a JSON report artifact.
- [x] Clean audit verification transcript captured.
- [x] Tamper-detection transcript captured.
- [x] Unit test added for report shape and negative evidence.
- [x] No destructive action, external call, secret access, tenant/security mutation, new dependency, or production state change.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-SEC-006 | MUST_PASS | PASS | `reports/scenario/vs0-audit-ledger-2026-06-09.json`, clean audit verification and controlled tamper-detection transcripts |

## Human Required

No human-required item was introduced for this batch. Production audit anchoring and retention evidence remain outside the current scaffold scope.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-audit-ledger --json --output reports/scenario/vs0-audit-ledger-2026-06-09.json
# status: success
# scenario_set: vs0-audit-ledger
# summary.pass: 1
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_AUDIT_LEDGER_ONLY
# audit_evidence.clean_audit_event_count: 9
# audit_evidence.event_types: artifact.ingested, artifact.read, search.snapshot.created, evidence_bundle.created, brief.created, claim.draft.created, claim.approved, policy.egress.denied, policy.sandbox_access.denied
# audit_evidence.event_scopes_complete: true
# audit_evidence.event_hashes_present: true
# audit_evidence.event_details_present: true
# audit_evidence.tamper_detection_exit_code: 5
# audit_evidence.tamper_detection_errors[0].code: AUDIT_EVENT_HASH_MISMATCH
# negative_evidence.tamper_accepted: 0
```

## Evidence Summary

- Artifact ingestion appends `artifact.ingested` with subject, details, owner/namespace/workspace scope, `previous_hash`, and `event_hash`.
- Artifact read appends `artifact.read`.
- Search appends `search.snapshot.created`.
- Evidence bundle creation appends `evidence_bundle.created`.
- Brief creation appends `brief.created`.
- Draft claim creation appends `claim.draft.created`.
- Claim approval appends `claim.approved`.
- Default egress denial appends `policy.egress.denied`.
- Undeclared sandbox access denial appends `policy.sandbox_access.denied`.
- Clean `cornerstone audit verify --json` returns `status=success`.
- Controlled mutation of the first audit JSONL event makes `cornerstone audit verify --json` return exit code `5` with `AUDIT_EVENT_HASH_MISMATCH`.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` marks only `CS-SEC-006` as `PASS` in this batch.

## Gaps

- `CS-SEC-004` remains `NOT_VERIFIED`; broader RBAC/ABAC access-control matrix is not implemented.
- `CS-SEC-005` remains `NOT_VERIFIED`; policy denial examples and resolution-path UX/API are not implemented.
- Action, connector, model-routing, Agent Pack, and autonomy audit events remain `NOT_VERIFIED` until those surfaces exist.
- Production-grade append-only storage, external anchoring, and Merkle inclusion proofs remain future hardening work.

## Risks

- A local hash chain detects mutation but does not prevent deletion, truncation, or host compromise by itself.
- The verifier proves scaffold event integrity only; production audit must add storage isolation, retention policy, and operational monitoring.
- Future critical event types must be added to this scenario verifier as new product surfaces are implemented.
