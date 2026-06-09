# VS-0 Conversation Onboarding Batch 16 Report - 2026-06-09

Status: PASS for deterministic conversation-first onboarding and manual claim promotion only.
Scope: `CS-PROD-005`, `CS-CLAIM-001`, `CS-CLAIM-003`, `CS-CLAIM-004`, and `CS-CLAIM-009`.

This report does not mark production chat UI, production API runtime, memory, learning, RBAC/ABAC, namespace promotion, or full product-loop completion as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies that a user can start from natural messy input, reach an evidence-backed brief, see optional durable outputs, manually promote a source-linked claim, and receive insufficient-evidence labeling for unsupported assertions.

## Research Checkpoint

- W3C PROV frames provenance as information about entities, activities, and people involved in producing data, useful for trustworthiness assessment: <https://www.w3.org/TR/prov-overview/>
- LlamaIndex's Document/Node model is a dominant OSS reference for source-node style RAG traceability: <https://docs.llamaindex.ai/en/v0.10.19/module_guides/loading/documents_and_nodes/root.html>

Best fit for this batch remains the existing no-new-dependency deterministic runtime. It keeps source artifact refs, evidence bundles, and conversation provenance explicit without adding a RAG framework before the frozen scaffold scenarios require one.

## Assumptions

- Native CLI JSON is the scaffold E2E verification surface for this VS-0 conversation slice.
- A conversation turn is captured as an immutable artifact so conversation work keeps the same artifact/evidence/audit safety model.
- Unsupported-answer handling must distinguish raw search hits from supporting evidence; stopword-only matches are auditable but not enough to present a claim as fact.

## Out Of Scope

- Production chat UI, streaming conversation UX, production API runtime, memory creation, knowledge capsule persistence, organization promotion, RBAC/ABAC, and full learning loop.
- `CS-PROD-001`, `CS-PROD-002`, `CS-REG-001`, `CS-REG-005`, and `CS-REG-006` remain `NOT_VERIFIED`.

## Checklist

- [x] Frozen conversation/onboarding scenario wording inspected.
- [x] Conversation start captures messy input without connector, model-provider, ontology, organization policy, case, mission, or document setup.
- [x] Conversation turn persists as an immutable artifact with source type `conversation_turn`.
- [x] Search, Evidence Bundle, and Brief use the conversation artifact.
- [x] Conversation start suggests Mission Card, Knowledge Capsule, Claim, Action Card, Memory, and Playbook Candidate as optional promotions.
- [x] Manual conversation promotion creates a source-linked evidence-backed claim.
- [x] Unsupported answer is labeled `insufficient_evidence` and is not presented as fact.
- [x] Matrix PASS rows backed by a JSON report artifact.
- [x] Unit and shell-gate coverage added.
- [x] No destructive action, external call, secret access, tenant/security mutation, new dependency, or production state change.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-PROD-005 | MUST_PASS | PASS | `reports/scenario/vs0-conversation-onboarding-2026-06-09.json`, conversation-to-brief-to-claim E2E transcript |
| CS-CLAIM-001 | MUST_PASS | PASS | `reports/scenario/vs0-conversation-onboarding-2026-06-09.json`, natural conversation start to brief and claim transcript |
| CS-CLAIM-003 | MUST_PASS | PASS | `reports/scenario/vs0-conversation-onboarding-2026-06-09.json`, optional durable output suggestions |
| CS-CLAIM-004 | MUST_PASS | PASS | `reports/scenario/vs0-conversation-onboarding-2026-06-09.json`, promoted claim with source conversation and evidence |
| CS-CLAIM-009 | MUST_PASS | PASS | `reports/scenario/vs0-conversation-onboarding-2026-06-09.json`, insufficient-evidence answer transcript |

## Human Required

No human-required item was introduced for this batch. Production chat UI/browser acceptance remains outside this scaffold slice.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-conversation-onboarding --json --output reports/scenario/vs0-conversation-onboarding-2026-06-09.json
# status: success
# scenario_set: vs0-conversation-onboarding
# summary.pass: 5
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_CONVERSATION_ONBOARDING_ONLY
# conversation_evidence.source_artifact_source_type: conversation_turn
# conversation_evidence.brief_status: evidence_backed
# conversation_evidence.promoted_claim_trust_state: evidence_backed
# conversation_evidence.unsupported_answer_label: insufficient_evidence
# conversation_evidence.unsupported_answer_presented_as_fact: false
# conversation_evidence.unsupported_answer_supporting_result_count: 0
# negative_evidence.pre_modeling_required: 0
# negative_evidence.required_connector_setup: 0
# negative_evidence.required_model_provider_setup: 0
# negative_evidence.required_ontology_setup: 0
# negative_evidence.forced_conversion: 0
# negative_evidence.promoted_objects_without_scope: 0
# negative_evidence.promoted_objects_without_evidence: 0
# negative_evidence.unsupported_assertions_presented_as_fact: 0
# negative_evidence.real_external_http_calls: 0
```

## Evidence Summary

- `conversation start` creates a `cs.conversation.v0` record and an immutable `conversation_turn` artifact.
- Search and evidence bundle creation use the conversation artifact as the source node.
- Brief creation produces `status=evidence_backed` without requiring connector/model/ontology setup.
- `conversation promote` creates a claim with `source_conversation`, `source_artifact_ref`, Evidence Bundle refs, owner namespace, `trust_state=evidence_backed`, provenance, and audit refs.
- Unsupported answering records meaningful question terms `approved`, `budget`, `project`, and `zeta`; the only raw match is stopword `is`, so supporting evidence count is `0` and the answer remains `insufficient_evidence`.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` now marks the 5 rows in this batch as `PASS`.

Current full matrix after this batch:

- `PASS`: 56
- `NOT_VERIFIED`: 150
- `FAIL`: 0
- `NOT_RUN`: 0

## Gaps

- Full 206-scenario PASS remains incomplete.
- Production UI/API product surfaces remain missing; this batch uses CLI-native JSON as scaffold evidence.
- One coherent product walkthrough, full evidence-first product loop, three-domain generality, namespace promotion, Autopilot readiness, RBAC/ABAC, memory source-of-truth conflict, and personal-memory leakage tests remain `NOT_VERIFIED`.

## Risks

- Conversation behavior is deterministic local JSON, not a production chat runtime.
- The retrieval guard now prevents stopword-only support, but future semantic retrieval must preserve the same distinction between raw matches and supporting evidence.
- Future UI/API implementations must preserve conversation source artifacts, evidence bundle provenance, manual promotion audit, and insufficient-evidence labeling.
