# Cornerstone v1.9.0 — End-to-end Proof and Operator UX Checklist

## Purpose

`v1.9.0` proves the full ontology Single Source of Truth loop with an operator-oriented checklist instead of a visual UI. The release gives operators a repeatable way to verify that Cornerstone can ingest manual source material, create evidence, queue and run ontology re-extraction, review evidence, approve candidates, serve an official graph, and evaluate that graph.

This version is intentionally focused on checklist behavior because UI/UX quality is hard to evaluate without a dedicated frontend surface.

## Product Goal

The goal is to make this flow demonstrable and auditable:

```text
manual proof source
→ Artifact
→ EvidenceFragment
→ OntologyReExtractionRun
→ OntologyExtractionRun
→ ConceptCandidates / RelationCandidates
→ evidence review
→ candidate review
→ official Concepts / official Relations
→ explainable depth-1 graph
→ ontology graph evaluation result
```

The proof validates that the reviewed ontology graph can be used as the Single Source of Truth while preserving the existing trust boundary.

## Non-goals

`v1.9.0` does not implement:

```text
- frontend graph UI
- visual review UX
- live external LLM provider
- automatic approval of normal production data
- automatic evidence review
- semantic/vector candidate scoring
- graph depth above 1
- graph database
- new connector types
- PDF/Office parsing
```

The proof checklist is an operator backend workflow, not a frontend product experience.

## New API Contract

### Run ontology proof checklist

```http
POST /v1/ontology/proof-runs
```

Request:

```json
{
  "focusConcept": "Settlement",
  "reviewer": "reviewer@example.com",
  "createdBy": "operator@example.com",
  "dryRun": false,
  "confirmMutation": true,
  "runEvaluation": true
}
```

Response:

```json
{
  "status": "passed",
  "focusConcept": "Settlement",
  "sourceId": "...",
  "artifactIds": ["..."],
  "evidenceFragmentIds": ["..."],
  "reextractionRunId": "...",
  "extractionRunIds": ["..."],
  "conceptCandidateIds": ["..."],
  "relationCandidateIds": ["..."],
  "approvedConceptIds": ["..."],
  "approvedRelationIds": ["..."],
  "graphResponseId": "...",
  "evaluationTaskId": "...",
  "evaluationResultId": "...",
  "summary": {
    "requiredPassed": 8,
    "requiredFailed": 0,
    "officialGraphAvailable": true,
    "officialGraphMutated": false,
    "evaluationSuccess": true
  },
  "checklist": []
}
```

### Dry run

Operators can inspect the checklist without mutation:

```json
{
  "focusConcept": "Settlement",
  "dryRun": true
}
```

Dry-run response status is `planned`; no DataSource, Artifact, EvidenceFragment, candidate, Concept, Relation, graph, or evaluation object is created.

### Mutation confirmation

A non-dry proof run requires:

```json
{
  "confirmMutation": true
}
```

This is required because the proof creates explicit proof source data and official proof ontology objects.

## Checklist Steps

The v1.9.0 proof checklist contains these required steps:

| Step | Category | Goal | Success signal |
|---|---|---|---|
| `create_manual_source` | source | Create explicit proof source material. | Manual production-enabled source exists. |
| `sync_manual_seed` | ingestion | Create Artifact and EvidenceFragment records. | Artifact, evidence, and queued re-extraction run ids exist. |
| `run_reextraction` | extraction | Generate candidate-only ontology output. | ConceptCandidates and RelationCandidates exist; `officialGraphMutated=false`. |
| `review_evidence` | review | Mark proof evidence reviewed. | All proof EvidenceFragments have reviewer provenance. |
| `approve_concepts` | review | Promote/merge ConceptCandidates through officialization gates. | Official Concept ids exist. |
| `approve_relations` | review | Promote/merge RelationCandidates after endpoint Concepts are official. | Official Relation ids exist. |
| `serve_explainable_graph` | graph | Serve depth-1 official graph. | `officialGraphAvailable=true`, `trustLabel=official`, edges exist. |
| `run_ontology_evaluation` | evaluation | Verify deterministic SSOT graph gates. | Evaluation result success is true. |

## Safety and Trust Rules

These rules must remain true:

```text
1. Dry runs create no objects.
2. Non-dry runs require confirmMutation=true.
3. Reviewer identity must be authorized.
4. Evidence must be reviewed before candidate approval can promote official objects.
5. Re-extraction output remains candidate-only.
6. officialGraphMutated remains false for re-extraction.
7. Official graph mutation happens only through candidate review/officialization gates.
8. The proof response must expose checklist status, object ids, limitations, and next actions.
```

## CLI Operator Usage

The existing proof runner gains an ontology-loop scope:

```bash
cornerstone proof run \
  --ontology-loop \
  --confirm-ontology-mutation \
  --base-url http://localhost:8000 \
  --markdown auto
```

The flag is explicit because it mutates the running API's store with proof data.

Dry-run planning remains available:

```bash
cornerstone proof run --ontology-loop --dry-run --markdown auto
```

## Settlement Reference Behavior

Default proof seed text uses `Settlement` as the focus concept:

```text
Settlement is the process of finalizing obligations.
Clearing is the process of calculating obligations before settlement.
Reconciliation is the process of verifying settlement outcomes.
Clearing precedes Settlement.
Reconciliation validates Settlement.
```

Expected graph after proof approval:

```text
Clearing --precedes--> Settlement
Reconciliation --validates--> Settlement
```

The exact ids are generated at runtime. The proof asserts graph availability, relation edges, citations, reviewer provenance, and evaluation success rather than relying on fixed ids.

## Implementation Checklist

```text
[x] Add proof request/response/checklist schemas.
[x] Add OntologyProofService.
[x] Add POST /v1/ontology/proof-runs.
[x] Add CLI proof --ontology-loop flag.
[x] Require confirmMutation for non-dry proof runs.
[x] Preserve authorized reviewer gate.
[x] Use existing manual source sync pipeline.
[x] Use existing re-extraction service.
[x] Use existing evidence review model.
[x] Use existing candidate review officialization gates.
[x] Use existing ontology graph service.
[x] Use existing ontology graph evaluation service.
[x] Add integration tests for dry run, confirmation gate, full proof, and reviewer gate.
[x] Update API contract, API freeze review, limitations, README, readiness docs, and release notes.
```

## Exit Criteria

`v1.9.0` is complete when:

```text
- dry-run proof returns a planned checklist and creates no source data
- non-dry proof without confirmMutation is rejected
- unauthorized reviewer is rejected
- confirmed proof returns status=passed
- proof creates source/artifact/evidence/re-extraction/extraction/candidate ids
- evidence is reviewed before candidate approval
- Concepts and Relations are promoted through officialization gates
- graph response is official and depth=1
- ontology evaluation succeeds
- release docs and release checker pass
```

## Known Limitations

```text
- The proof is backend/API/CLI only.
- It creates explicit proof data when confirmed.
- It uses deterministic local ontology extraction.
- It does not evaluate visual graph UX.
- It is not a substitute for live connector proof with real Notion or Google Drive data.
- Repeated proof runs may create additional proof Concepts unless the operator scopes focus/source names carefully.
```

## Next Version Handoff

The next version should move toward `v2.0.0 — Ontology SSOT release` by tightening operator readiness, adding stronger proof reports, and optionally preparing a frontend review/graph UX after backend proof gates are stable.

## Chronicle position and measurable release checklist

This section was added during the v2.0.0 documentation chronicle pass so the version goal and checklist are explicit, measurable, and easy to audit.

**Version goal:** Give operators a checklist-driven proof flow for the full ontology SSOT loop without requiring a frontend UI.

**Confirmed non-goal:** No visual UI, no live external LLM provider, no normal-data auto-approval.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V190-01 | `POST /v1/ontology/proof-runs` supports dry run and confirmed mutation modes. | API contract and tests. | complete |
| V190-02 | Non-dry proof execution requires explicit `confirmMutation=true`. | Safety rules and tests. | complete |
| V190-03 | The proof checklist covers ingestion, re-extraction, evidence review, candidate approval, graph serving, and evaluation. | Checklist Steps section. | complete |
| V190-04 | CLI supports `cornerstone proof run --ontology-loop`. | CLI operator usage. | complete |
| V190-05 | Successful proof reports official graph availability and evaluation success. | Smoke report and exit criteria. | complete |

**Chronicle handoff:** This version is recorded in `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. Later versions may build on this release only within the confirmed non-goal boundary above.

