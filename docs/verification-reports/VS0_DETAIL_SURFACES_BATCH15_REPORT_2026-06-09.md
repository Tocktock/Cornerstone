# VS-0 Detail Surfaces Batch 15 Report - 2026-06-09

Status: PASS for deterministic CLI-native detail, trust, workspace, evidence-inspection, and denial-explanation surfaces only.
Scope: `CS-UND-004`, `CS-CLAIM-005`, `CS-CLAIM-008`, `CS-NS-002`, and `CS-SEC-005`.

This report does not mark production UI runtime, production API runtime, RBAC/ABAC enforcement, namespace promotion, memory, learning, or full product-loop completion as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies that users can inspect active workspace boundaries, artifact details, trust states, supporting evidence, and safe denial explanations through replayable local scaffold JSON.

## Research Checkpoint

- W3C PROV keeps provenance as an interoperable model for connecting entities, activities, and agents: <https://www.w3.org/TR/prov-overview/>
- Open Policy Agent decision-log documentation reinforces decision records as auditable outputs that may require careful redaction: <https://www.openpolicyagent.org/docs/management-decision-logs>

Best fit for this batch remains the existing no-new-dependency deterministic local runtime. It exposes the evidence and policy shape through native CLI JSON instead of adding a UI/API framework before the scaffold scenarios justify it.

## Assumptions

- Native CLI JSON is the scaffold API-style verification surface for this VS-0 detail slice.
- `workspace show` is a local product walkthrough proxy for active workspace visibility until a production UI exists.
- Denial examples are local policy outcomes and do not perform real egress, shell, provider, or connector calls.

## Out Of Scope

- Production UI/API runtime, browser screenshots, RBAC/ABAC access-control matrix, namespace promotion, personal memory boundaries, human visual acceptance, and full 206-scenario completion.
- `CS-SEC-004`, `CS-NS-004`, `CS-REG-005`, and `CS-REG-006` remain `NOT_VERIFIED`.

## Checklist

- [x] Frozen SoT wording inspected for the five target rows.
- [x] Workspace detail exposes active tenant, owner, namespace, workspace, mode, navigation, and context boundary.
- [x] Artifact detail shows original storage, derived metadata/text preview, source, provenance, related claim, and related mission.
- [x] Claim examples show Draft, Evidence-backed, and Approved trust states plus authority limits.
- [x] Evidence viewer opens source artifact metadata, relevant snippet, derived representation, and query-linked bundle data.
- [x] Denied egress, sandbox access, unsupported claim approval, and high-risk action execution include cause, safe resolution path, and audit evidence.
- [x] Matrix PASS rows backed by a JSON report artifact.
- [x] Unit and shell-gate coverage added.
- [x] No destructive action, external call, secret access, tenant/security mutation, new dependency, or production state change.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-UND-004 | MUST_PASS | PASS | `reports/scenario/vs0-detail-surfaces-2026-06-09.json`, `cornerstone artifact show <artifact_id> --json` transcript |
| CS-CLAIM-005 | MUST_PASS | PASS | `reports/scenario/vs0-detail-surfaces-2026-06-09.json`, Draft/Evidence-backed/Approved claim show transcripts |
| CS-CLAIM-008 | MUST_PASS | PASS | `reports/scenario/vs0-detail-surfaces-2026-06-09.json`, claim show plus evidence view transcripts |
| CS-NS-002 | MUST_PASS | PASS | `reports/scenario/vs0-detail-surfaces-2026-06-09.json`, personal and organization workspace show transcripts |
| CS-SEC-005 | MUST_PASS | PASS | `reports/scenario/vs0-detail-surfaces-2026-06-09.json`, denial examples with resolution paths and audit refs |

## Human Required

No human-required item was introduced for this batch. Production UI screenshot/browser acceptance remains outside this scaffold slice.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-detail-surfaces --json --output reports/scenario/vs0-detail-surfaces-2026-06-09.json
# status: success
# scenario_set: vs0-detail-surfaces
# summary.pass: 5
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_DETAIL_SURFACES_ONLY
# detail_surface_evidence.audit_event_count: 19
# detail_surface_evidence.trust_states: draft, evidence_backed, approved
# negative_evidence.workspace_boundary_implicit_cross_namespace_context: 0
# negative_evidence.artifact_detail_missing_related_claims: 0
# negative_evidence.artifact_detail_missing_related_missions: 0
# negative_evidence.trust_ladder_missing_states: 0
# negative_evidence.evidence_viewer_missing_sources: 0
# negative_evidence.policy_denials_missing_resolution_path: 0
# negative_evidence.policy_denials_without_audit: 0
```

## Evidence Summary

- `workspace show` reports `personal / default` and `organization / ops` active labels with `implicit_cross_namespace_context=false`.
- `artifact show` reports the original SHA-256 storage ref, derived text metadata, source path, provenance transformations, a related approved claim, and a related mission.
- Claim detail transcripts cover `draft`, `evidence_backed`, and `approved` trust states with authority limits.
- `evidence view` opens the cited source artifact and derived text preview for the claim evidence bundle.
- Denial examples include `CS_EGRESS_DENIED`, `CS_SANDBOX_ACCESS_DENIED`, `CS_CLAIM_EVIDENCE_REQUIRED`, and `CS_ACTION_POLICY_DENIED`, each with a resolution path and audit refs.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` now marks the 5 rows in this batch as `PASS`.

Current full matrix after this batch:

- `PASS`: 51
- `NOT_VERIFIED`: 155
- `FAIL`: 0
- `NOT_RUN`: 0

## Gaps

- Full 206-scenario PASS remains incomplete.
- Production UI/API product surfaces remain missing; this batch uses CLI-native JSON as scaffold evidence.
- RBAC/ABAC enforcement, namespace promotion, memory source-of-truth conflict tests, and personal-to-organization memory leakage tests remain `NOT_VERIFIED`.

## Risks

- The detail surfaces are deterministic local JSON, not production UI/API screens.
- Workspace visibility is proven as structured CLI output; human visual acceptance still needs production UI work.
- Future UI/API implementations must preserve the same active-scope, provenance, evidence, trust, denial-resolution, and audit fields.
