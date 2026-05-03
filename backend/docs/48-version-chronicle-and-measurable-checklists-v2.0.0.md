# Cornerstone Version Chronicle and Measurable Checklists

## Purpose

This document is the canonical version-by-version chronicle for Cornerstone documentation through `v2.5.0`.

It was introduced during the `v2.0.0` documentation hardening pass and remains the audit source for release goals, non-goals, measurable acceptance checks, and handoffs. The filename keeps `v2.0.0` because that is when the chronicle was introduced; the content now records later patch releases.

## Current final version

```text
v2.0.4 — Forward Roadmap Goals and Measurable Checklists
```

`v2.0.4` adds a documentation-first roadmap for `v2.1.0` through `v2.5.0`, with each future version defined by a clear goal, confirmed non-goal, domain boundary, measurable checklist, verification plan, and handoff.

## What this chronicle guarantees

Every version row must answer four questions:

```text
1. What was the version goal?
2. What was intentionally not part of the version?
3. What measurable checklist proves the version was complete?
4. What did the version hand off to the next version?
```

## Canonical product boundary

```text
Raw source data is not the Single Source of Truth.
Connector sync output is not the Single Source of Truth.
LLM or extractor output is not the Single Source of Truth.
Pending candidates are not the Single Source of Truth.
The reviewed official ontology graph is the Single Source of Truth.
```

## Documentation layers

As of `v2.0.4`, Cornerstone documentation has three layers:

```text
Layer 1 — Product documentation
  docs/product/*
  Purpose: explain the product, user value, trust model, and settlement walkthrough.

Layer 2 — Technical documentation
  docs/00-05 and feature/API docs
  Purpose: explain architecture, APIs, implementation, and operations.

Layer 3 — Roadmap, release chronicle, and readiness
  docs/roadmap/*, docs/48-*, and docs/release/*
  Purpose: preserve future release plans, version goals, acceptance checks, readiness, and audit proof.
```

## Complete version chronicle

| Doc # | Version | Document | Version goal | Measurable completion gate | Status |
|---:|---|---|---|---|---|
| 06 | v0.2 | `docs/06-backend-hardening-v0.2.md` | Harden initial backend trust contract and remove demo/fake confidence risk. | Evidence, officialization gates, auditability, and product-safety notes are documented. | documented |
| 07 | v0.3 | `docs/07-postgres-persistence-v0.3.md` | Move from memory-only prototype to PostgreSQL persistence foundation. | Durable persistence supports evidence, provenance, freshness, review state, officialization, and auditability. | documented |
| 08 | v0.4.0 | `docs/08-connector-framework-notion-v0.4.md` | Create connector foundation and Notion skeleton before ingestion. | Connector catalog, OAuth boundary, credential storage, source test, selection, sync lifecycle, and audit visibility are documented. | documented |
| 09 | v0.5.0 | `docs/09-source-state-notion-discovery-v0.5.md` | Add runtime source state and Notion discovery/selection. | Discovery snapshots and source state exist without fabricating Artifacts or EvidenceFragments. | documented |
| 10 | v0.6.0 | `docs/10-generic-ingestion-notion-v0.6.md` | Define generic ingestion from provider objects into Artifacts and EvidenceFragments. | Provider objects map through SourceObject → Artifact → EvidenceFragment without Notion-specific core coupling. | documented |
| 11 | v0.6.1 | `docs/11-notion-sdk-gateway-v0.6.1.md` | Replace handwritten provider plumbing with SDK-backed Notion gateway. | Notion API access is isolated behind gateway/mapper modules. | documented |
| 12 | v0.7.0 | `docs/12-durable-sync-worker-v0.7.md` | Move sync from request-time work toward durable worker/checkpoint execution. | Sync jobs are queued by default; failures do not mark data fresh. | documented |
| 13 | v0.8.0 | `docs/13-external-worker-scheduled-postgres-v0.8.md` | Add external worker, scheduled sync, and PostgreSQL CI assets. | Schedules enqueue due jobs and external worker path is documented. | documented |
| 14 | v0.8.1 | `docs/14-manual-source-sync-safety-v0.8.1.md` | Remove generic source sync bypass and make manual sync manual-source-only. | Provider-backed sources cannot ingest arbitrary caller-supplied objects. | documented |
| 15 | v0.8.2 | `docs/15-provider-object-ingestion-safety-v0.8.2.md` | Prevent unsupported provider objects from creating false ingestion confidence. | Unsupported provider objects are not silently ingested. | documented |
| 16 | v0.8.3 | `docs/16-production-config-fail-closed-v0.8.3.md` | Make production configuration fail closed. | Production rejects in-memory persistence, mocked APIs, placeholder identities, localhost callbacks, and weak secrets. | documented |
| 17 | v0.8.4 | `docs/17-atomic-sync-write-boundaries-v0.8.4.md` | Make connector worker writes atomic at service boundary. | Artifact/evidence writes, source state, cursor advancement, and job success commit consistently. | documented |
| 18 | v0.8.5 | `docs/18-worker-lease-primitives-v0.8.5.md` | Prepare worker runtime for multi-worker safety. | Claim/lease primitives and scheduled enqueue idempotency are documented. | documented |
| 19 | v0.9.0 | `docs/19-live-postgres-multi-worker-v0.9.md` | Harden live PostgreSQL and multi-worker safety. | Ownership, heartbeat, cursor advancement, and audit events remain consistent under concurrency. | documented |
| 20 | v0.9.1 | `docs/20-live-postgres-verification-v0.9.1.md` | Turn PostgreSQL/concurrency assets into stricter verification. | Live PostgreSQL runner and mypy stability checks are documented. | documented |
| 21 | v0.9.2 | `docs/21-live-notion-e2e-v0.9.2.md` | Add gated live Notion end-to-end pilot path. | Operator-supplied token/page/secret are required; normal tests do not contact Notion. | documented |
| 22 | v0.10.0 | `docs/22-evidence-review-officialization-v0.10.0.md` | Implement evidence review queue and officialization hardening. | Evidence review and official Concept/Relation/DecisionRecord gates are documented. | documented |
| 23 | v0.11.0 | `docs/23-grounded-serving-contract-v0.11.0.md` | Harden grounded serving contract. | Unsupported/stale/conflicted/partial states are explicit in responses. | documented |
| 24 | v0.12.0 | `docs/24-evaluation-framework-v0.12.0.md` | Add grounded-context evaluation framework. | Responses are evaluated for correctness, evidence, provenance, trust label, freshness, and unsupported claims. | documented |
| 25 | v0.12.1 | `docs/25-product-trust-cleanup-v0.12.1.md` | Clean product trust risks before live proof. | Fake provider creation and unsafe routes are removed or constrained. | documented |
| 26 | v0.12.2 | `docs/26-live-proof-fixes-v0.12.2.md` | Package fixes found during manual live proof. | Artifact line matches the code that passed proof gate. | documented |
| 27 | v0.13.1 | `docs/27-backend-release-candidate-v0.13.0.md` | Package release-candidate cleanup material after live proof. | Operational docs, limitations, release checklist, and live-proof records are complete. Historical exception: filename keeps v0.13.0 while title records v0.13.1. | documented |
| 28 | v1.0.0-rc.1 | `docs/28-backend-v1.0.0-rc.1.md` | Create backend MVP release-candidate tag from verified line. | RC label is tied to verified v0.13.1 implementation. | documented |
| 29 | v1.0.0 | `docs/29-backend-v1.0.0.md` | Promote verified backend MVP to final release. | PostgreSQL proof, Notion proof, product-loop proof, safety checks, secret scan, and signoff are recorded. | documented |
| 30 | v1.1.0 | `docs/30-cli-macos-starter-v1.1.0.md` | Make backend MVP easier to run on macOS. | CLI wrapper, macOS setup scripts, and starter docs exist. | documented |
| 31 | v1.1.1 | `docs/31-cli-product-loop-v1.1.1.md` | Make product loop operable without curl. | CLI commands cover source, evidence, concept, context, eval, and status. | documented |
| 32 | v1.1.2 | `docs/32-one-command-proof-runner-v1.1.2.md` | Add one-command proof runner. | Proof command executes backend loop and produces proof output. | documented |
| 33 | v1.1.3 | `docs/33-cross-platform-starter-v1.1.3.md` | Expand starter docs beyond macOS. | macOS, Linux, and Windows PowerShell starter paths are documented. | documented |
| 34 | v1.1.4 | `docs/34-cli-maintainability-v1.1.4.md` | Split CLI code for maintainability. | CLI package structure is modular with behavior preserved. | documented |
| 35 | v1.2.0 | `docs/35-google-drive-connector-v1.2.0.md` | Add Google Drive connector. | OAuth, discovery, selection, Google Doc/text ingestion, Artifact, and EvidenceFragment paths are documented. | documented |
| 36 | v1.2.1 | `docs/36-ontology-ssot-product-contract-v1.2.1.md` | Define ontology SSOT product contract. | Reviewed official ontology graph is defined as SSOT; LLM output is candidate-only. | documented |
| 37 | v1.2.1 | `docs/37-ontology-domain-model-proposal-v1.2.1.md` | Propose ontology domain model. | OntologyExtractionRun, ConceptCandidate, RelationCandidate, ConceptAlias, graph response, and evidence rules are specified. | documented |
| 38 | v1.2.1 | `docs/38-ontology-versioned-implementation-plan-v1.2.1.md` | Plan versioned ontology implementation. | Each planned version has goal, scope, deferred scope, checklist, test plan, proof, and handoff. | documented |
| 39 | v1.3.0 | `docs/39-ontology-graph-runtime-v1.3.0.md` | Implement first depth-1 graph/search runtime. | Search and graph endpoints return official depth-1 graph with citations and depth limit. | documented |
| 40 | v1.3.1 | `docs/40-manual-upload-ingestion-v1.3.1.md` | Add manual text-like upload ingestion. | Uploaded text creates SourceObjects, Artifacts, and EvidenceFragments; no official graph mutation. | documented |
| 41 | v1.4.0 | `docs/41-llm-ontology-extraction-v1.4.0.md` | Add candidate-only ontology extraction. | Extraction run creates pending ConceptCandidates and RelationCandidates with evidence ids. | documented |
| 42 | v1.5.0 | `docs/42-ontology-candidate-review-workflow-v1.5.0.md` | Add human review workflow for ontology candidates. | Reviewers can edit, approve, reject, or merge candidates; officialization gates remain enforced. | documented |
| 43 | v1.6.0 | `docs/43-explainable-graph-serving-v1.6.0.md` | Enhance graph serving explainability. | Graph response includes support summary, provenance, candidate summary, and explanation fields. | documented |
| 44 | v1.7.0 | `docs/44-ontology-evaluation-v1.7.0.md` | Add read-only ontology graph evaluation. | Tasks/results/summary measure graph evidence, provenance, trust, freshness, safety, and candidate boundary. | documented |
| 45 | v1.8.0 | `docs/45-connector-driven-reextraction-v1.8.0.md` | Queue re-extraction when source data changes. | New/changed Artifacts queue candidate-only re-extraction; official graph remains unchanged. | documented |
| 46 | v1.9.0 | `docs/46-end-to-end-proof-operator-ux-v1.9.0.md` | Add checklist-driven end-to-end ontology proof. | Proof validates ingestion → candidates → review → official graph → evaluation with explicit mutation confirmation. | documented |
| 47 | v2.0.0 | `docs/47-ontology-ssot-release-v2.0.0.md` | Stabilize ontology SSOT backend contract. | Readiness endpoint and CLI checklist verify official graph safety for a focus concept. | documented |
| 48 | v2.0.0 | `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md` | Create canonical chronicle with measurable version checklists. | Chronology records goals, non-goals, checklists, historical exceptions, and release handoffs. | documented |
| 49 | v2.0.1 | `docs/49-refactor-domain-boundary-v2.0.1.md` | Refactor duplicate logic and clarify domain boundaries. | Duplicate audit passes; versioning checks target 2.0.1; no runtime contract change. | documented |
| 50 | v2.0.2 | `docs/50-product-documentation-layer-v2.0.2.md` | Add separate product documentation layer. | Product docs explain overview, value, workflow, settlement, graph, trust, roles, positioning, quickstart, and glossary. | documented |
| 51 | v2.0.3 | `docs/51-dependency-complete-verification-v2.0.3.md` | Add dependency-complete verification and CI hardening. | Verification runner, CI workflow, command plan, version checks, and readiness docs exist without product behavior change. | documented |
| 52 | v2.0.4 | `docs/52-forward-roadmap-goals-checklists-v2.0.4.md` | Document the forward roadmap for v2.1.0 through v2.5.0 before implementation. | Separate roadmap docs exist for v2.1.0, v2.2.0, v2.3.0, v2.4.0, and v2.5.0, each with measurable checklists. | documented |

## Planned-but-not-released item

`docs/38-ontology-versioned-implementation-plan-v1.2.1.md` originally listed a possible `v1.4.1 — Alias and deduplication` release.

That version was not released separately. Its intended scope was absorbed into implemented releases:

```text
v1.3.0 → ConceptAlias, alias normalization, alias search, alias conflict checks
v1.5.0 → candidate merge workflow and explicit review/promotion gates
```

Canonical ontology implementation chronology:

```text
v1.2.1 → v1.3.0 → v1.3.1 → v1.4.0 → v1.5.0 → v1.6.0 → v1.7.0 → v1.8.0 → v1.9.0 → v2.0.0 → v2.0.1 → v2.0.2 → v2.0.3 → v2.0.4
```

There is no missing runtime release between `v1.4.0` and `v1.5.0`.

## Ontology release measurable checklist ledger

### v1.2.1 — `36-ontology-ssot-product-contract-v1.2.1.md`

**Version goal:** Define the ontology SSOT product contract and trust boundary before runtime implementation.

**Confirmed non-goal:** No runtime API, persistence, extraction, or graph-serving behavior changes.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V121-CONTRACT-01 | Official reviewed ontology graph is named as SSOT. | Product contract doc. | complete |
| V121-CONTRACT-02 | LLM/extractor output is candidate-only. | Product contract and ADR. | complete |
| V121-CONTRACT-03 | Default graph depth is 1. | Product contract. | complete |
| V121-CONTRACT-04 | Settlement is the reference example. | Product contract and implementation plan. | complete |
| V121-CONTRACT-05 | Future versions are sequenced. | Versioned implementation plan. | complete |

### v1.3.0 — `39-ontology-graph-runtime-v1.3.0.md`

**Version goal:** Serve existing official Concepts and Relations as a depth-1 ontology graph.

**Confirmed non-goal:** No LLM extraction, candidate review, or automatic graph construction.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V130-01 | Ontology search endpoint exists. | API contract and tests. | complete |
| V130-02 | Ontology graph endpoint exists. | API contract and tests. | complete |
| V130-03 | Official mode excludes non-official objects. | Graph service tests. | complete |
| V130-04 | Alias support exists. | ConceptAlias implementation. | complete |
| V130-05 | Depth above 1 is rejected. | Graph API tests. | complete |

### v1.3.1 — `40-manual-upload-ingestion-v1.3.1.md`

**Version goal:** Allow text-like manual uploads to become source-backed evidence.

**Confirmed non-goal:** No ontology construction, no PDF/Office parsing, no official graph mutation.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V131-01 | File upload endpoint exists. | API contract and tests. | complete |
| V131-02 | Text upload endpoint exists. | API contract and tests. | complete |
| V131-03 | UTF-8 text-like content is accepted. | Upload service tests. | complete |
| V131-04 | Binary/unsupported content is rejected. | Upload tests. | complete |
| V131-05 | Upload creates Artifacts and EvidenceFragments only. | Trust boundary docs and tests. | complete |

### v1.4.0 — `41-llm-ontology-extraction-v1.4.0.md`

**Version goal:** Turn evidence into candidate-only ontology objects.

**Confirmed non-goal:** No official Concept/Relation creation, no live external LLM provider.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V140-01 | OntologyExtractionRun exists. | Schema/store docs. | complete |
| V140-02 | ConceptCandidates are created. | Extraction tests. | complete |
| V140-03 | RelationCandidates are created. | Extraction tests. | complete |
| V140-04 | Candidates carry evidence ids and confidence. | Candidate schemas. | complete |
| V140-05 | Official graph remains unchanged after extraction. | Trust boundary tests. | complete |

### v1.5.0 — `42-ontology-candidate-review-workflow-v1.5.0.md`

**Version goal:** Let reviewers approve, reject, edit, or merge ontology candidates.

**Confirmed non-goal:** No automatic approval or evidence-review bypass.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V150-01 | Concept candidate approve/reject/merge endpoints exist. | API contract and tests. | complete |
| V150-02 | Relation candidate approve/reject/merge endpoints exist. | API contract and tests. | complete |
| V150-03 | Approval uses officialization gates. | Review service tests. | complete |
| V150-04 | Rejected candidates cannot be approved later. | Review tests. | complete |
| V150-05 | Approved Relations appear in official graph only after endpoint Concepts are official. | Integration tests. | complete |

### v1.6.0 — `43-explainable-graph-serving-v1.6.0.md`

**Version goal:** Make graph responses explain why they are official, supported, limited, or candidate-adjacent.

**Confirmed non-goal:** No new extraction, review, graph mutation, or graph depth above 1.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V160-01 | Graph response includes supportSummary, candidateSummary, and explanation. | Schema/service tests. | complete |
| V160-02 | Nodes include reviewProvenance and supportSummary. | Graph tests. | complete |
| V160-03 | Edges include direction, provenance, support, and explanation. | Graph tests. | complete |
| V160-04 | Citations include source and review metadata. | Citation schema/tests. | complete |
| V160-05 | `GET /v1/ontology/explain` exists. | API contract and tests. | complete |

### v1.7.0 — `44-ontology-evaluation-v1.7.0.md`

**Version goal:** Add read-only evaluation tasks/results/summary for ontology graph quality.

**Confirmed non-goal:** No graph mutation, semantic scoring, extraction, or review change.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V170-01 | Ontology evaluation endpoints exist. | API contract and tests. | complete |
| V170-02 | Evaluation checks evidence validity, provenance, trust, freshness, official safety, and candidate boundary. | Result fields and tests. | complete |
| V170-03 | Evaluation results persist. | Store/migration docs. | complete |
| V170-04 | Evaluation calls graph service read-only. | Service tests. | complete |
| V170-05 | Summary reports measurable rates. | Metrics docs/tests. | complete |

### v1.8.0 — `45-connector-driven-reextraction-v1.8.0.md`

**Version goal:** Queue candidate-only ontology re-extraction when source data changes.

**Confirmed non-goal:** No official graph mutation from connector sync or re-extraction.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V180-01 | OntologyReExtractionRun exists. | Schema/store docs. | complete |
| V180-02 | Manual sync/upload can queue re-extraction. | Manual source tests. | complete |
| V180-03 | Connector sync can queue re-extraction. | Worker docs/tests. | complete |
| V180-04 | Running re-extraction creates candidates only. | Service/API tests. | complete |
| V180-05 | Sync response exposes created/reused/changed Artifact metadata. | API contract. | complete |

### v1.9.0 — `46-end-to-end-proof-operator-ux-v1.9.0.md`

**Version goal:** Add checklist-driven proof of the full ontology loop.

**Confirmed non-goal:** No frontend UI, no normal-data auto-approval, no live LLM provider.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V190-01 | Proof endpoint exists. | API contract and tests. | complete |
| V190-02 | Dry run creates no data. | Proof tests. | complete |
| V190-03 | Non-dry mutation requires confirmation. | Proof tests. | complete |
| V190-04 | Proof checklist covers ingestion, extraction, review, graph, and evaluation. | Proof response. | complete |
| V190-05 | CLI can run ontology proof. | CLI docs/tests. | complete |

### v2.0.0 — `47-ontology-ssot-release-v2.0.0.md`

**Version goal:** Stabilize backend contract for ontology SSOT readiness.

**Confirmed non-goal:** No UI, live LLM provider, deeper graph traversal, or automatic mutation.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V200-01 | SSOT readiness endpoint exists. | API contract and tests. | complete |
| V200-02 | Readiness endpoint is read-only. | Service/docs/tests. | complete |
| V200-03 | CLI readiness scope exists. | CLI docs/tests. | complete |
| V200-04 | Readiness checks official graph safety and evaluation. | Readiness docs/tests. | complete |
| V200-05 | Operator sequence is documented. | Operator checklist. | complete |

### v2.0.1 — `49-refactor-domain-boundary-v2.0.1.md`

**Version goal:** Preserve v2.0.0 behavior while removing duplicate logic and clarifying boundaries.

**Confirmed non-goal:** No new API endpoint, migration, graph behavior, automatic approval, or UI.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V201-01 | Package/runtime/readiness/checker/test versions target 2.0.1. | Version files and release checker. | complete |
| V201-02 | Ontology policy is centralized. | `src/cornerstone/domain/ontology.py`. | complete |
| V201-03 | Evidence/provenance duplication is consolidated. | `services/evidence_support.py`. | complete |
| V201-04 | Evaluation rules are shared. | `services/evaluation_rules.py`. | complete |
| V201-05 | Sync/source/provider helpers are shared. | Refactor modules and tests. | complete |
| V201-06 | Duplicate function-body audit passes. | `reports/refactor-duplicate-audit-v2.0.1.txt`. | complete |

### v2.0.2 — `50-product-documentation-layer-v2.0.2.md`

**Version goal:** Add product-first documentation separate from release chronology.

**Confirmed non-goal:** No runtime product behavior change beyond version metadata.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V202-01 | Product docs folder exists and is separate from release docs. | `docs/product/`. | complete |
| V202-02 | Product overview explains Cornerstone and product loop. | `docs/product/00-product-overview.md`. | complete |
| V202-03 | Settlement walkthrough shows source → graph story. | `docs/product/03-settlement-walkthrough.md`. | complete |
| V202-04 | Trust model explains SSOT boundary. | `docs/product/06-trust-model.md`. | complete |
| V202-05 | README is product-first. | `README.md`. | complete |
| V202-06 | Release checker and docs tests require product docs. | `scripts/check_release_candidate.py`, tests. | complete |

### v2.0.3 — `51-dependency-complete-verification-v2.0.3.md`

**Version goal:** Add dependency-complete verification and CI hardening for the stable ontology SSOT backend.

**Confirmed non-goal:** No new endpoint, schema, migration, ontology behavior, connector behavior, live LLM behavior, automatic approval, or UI.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V203-01 | Version metadata reports 2.0.3. | `pyproject.toml`, `__init__.py`, readiness schema. | complete |
| V203-02 | Dependency-complete command plan exists. | `src/cornerstone/verification/dependency_complete.py`. | complete |
| V203-03 | Strict runner requires explicit live DB confirmation. | `scripts/run_dependency_complete_verification.py --strict --confirm-live-db`. | complete |
| V203-04 | CI workflow provisions PostgreSQL and runs verification. | `.github/workflows/dependency-complete-verification.yml`. | complete |
| V203-05 | Release checker and tests require v2.0.3 docs. | `scripts/check_release_candidate.py`, tests. | complete |
| V203-06 | Plan-only verification produces measurable command-plan reports. | `reports/dependency-complete-command-plan-v2.0.3.md`. | complete |

### v2.0.4 — `52-forward-roadmap-goals-checklists-v2.0.4.md`

**Version goal:** Document the future implementation path for `v2.1.0` through `v2.5.0` before new product behavior is added.

**Confirmed non-goal:** No live LLM provider, review operator UX changes, graph visualization schema, connector expansion, frontend UI, integration package, endpoint, migration, or runtime behavior change.

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V204-01 | Roadmap README lists v2.1.0 through v2.5.0. | `docs/roadmap/README.md`. | complete |
| V204-02 | v2.1.0 live LLM provider roadmap is measurable. | `docs/roadmap/v2.1.0-live-llm-ontology-provider.md`. | complete |
| V204-03 | v2.2.0 review operator roadmap is measurable. | `docs/roadmap/v2.2.0-review-operator-experience.md`. | complete |
| V204-04 | v2.3.0 graph visualization roadmap is measurable. | `docs/roadmap/v2.3.0-graph-visualization-contract.md`. | complete |
| V204-05 | v2.4.0 connector hardening roadmap is measurable. | `docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md`. | complete |
| V204-06 | v2.5.0 frontend/integration roadmap is measurable. | `docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md`. | complete |
| V204-07 | Release checker and docs tests require roadmap docs. | `scripts/check_release_candidate.py`, tests. | complete |

## Forward roadmap after v2.0.4

The planned implementation sequence is:

```text
v2.1.0 — Live LLM ontology provider
v2.2.0 — Review operator experience
v2.3.0 — Graph visualization contract
v2.4.0 — Connector expansion / live connector proof hardening
v2.5.0 — Frontend MVP or external integration package
```

Each future implementation release should use its roadmap document as the starting contract and create its own release document, readiness document, release notes, tests, and verification report.

## Documentation completeness checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| DOC-01 | Every ontology version has a clear version goal. | Version docs 36-52 include goal rows or detailed chronicle goals. | complete |
| DOC-02 | Every ontology version has a clear non-goal boundary. | Version docs 36-52 include non-goals or this chronicle states them. | complete |
| DOC-03 | Every ontology version has a measurable checklist. | Version docs 36-52 and this chronicle include check IDs and measurable conditions. | complete |
| DOC-04 | The chronology is strictly ordered. | Complete version chronicle is ordered by document number 06-52 and release sequence. | complete |
| DOC-05 | Historical naming exceptions are explicit. | Doc 27 filename/title mismatch is called out. | complete |
| DOC-06 | Planned but skipped versions are explained. | Planned `v1.4.1` is recorded as absorbed, not missing. | complete |
| DOC-07 | The ontology SSOT trust boundary is repeated. | See Canonical product boundary. | complete |
| DOC-08 | Product docs are separate from release docs. | `docs/product/*` is Layer 1; release docs remain Layer 3. | complete |
| DOC-09 | README leads with product clarity. | README product overview appears before release chronology. | complete |
| DOC-10 | Documentation-only releases do not claim runtime feature changes. | v2.0.2, v2.0.3, and v2.0.4 non-goals and release notes state no new endpoint/migration/behavior. | complete |

## How to use this chronicle

Use this document when reviewing release completeness:

```text
1. Open the version document listed in the chronicle.
2. Confirm the version goal.
3. Confirm the non-goal boundary.
4. Confirm the measurable checklist rows.
5. Confirm the handoff to the next version.
```

Use product docs when onboarding a new reader:

```text
1. docs/product/00-product-overview.md
2. docs/product/03-settlement-walkthrough.md
3. docs/product/06-trust-model.md
4. docs/product/08-operator-quickstart.md
```

## Exit criteria for the current documentation state

```text
[x] Complete version chronicle records docs 06-52.
[x] Historical v0.13.0/v0.13.1 naming exception is explicit.
[x] Planned-but-not-released v1.4.1 is explained.
[x] Ontology implementation sequence is chronological.
[x] v2.0.1 refactor goal and measurable checklist are recorded.
[x] v2.0.2 product documentation layer is recorded.
[x] v2.0.3 Dependency-Complete Verification hardening is recorded.
[x] v2.0.4 forward roadmap goals and measurable checklists are recorded.
[x] Product docs are separate from technical and release docs.
[x] Release checker requires the product docs.
[x] Documentation tests require the product docs, v2.0.3 verification docs, v2.0.4 roadmap docs, and v2.5.0 implementation docs.
```

## Implemented roadmap releases through v2.5.0

The v2.0.4 roadmap has now been implemented through `v2.5.0`:

| Version | Document | Version goal | Measurable acceptance condition | Status |
|---|---|---|---|---|
| v2.1.0 | `docs/53-live-llm-ontology-provider-v2.1.0.md` | Add a gated live LLM ontology provider. | `provider=live_llm` is strict, evidence-bound, and candidate-only. | complete |
| v2.2.0 | `docs/54-review-operator-experience-v2.2.0.md` | Improve review operator queue and preview experience. | Queue summary and preview APIs expose blockers without mutating graph state. | complete |
| v2.3.0 | `docs/55-graph-visualization-contract-v2.3.0.md` | Add visualization-ready graph response metadata. | Graph responses include visualization nodes, edges, legends, panels, and empty state. | complete |
| v2.4.0 | `docs/56-connector-expansion-live-proof-hardening-v2.4.0.md` | Expose connector support and proof boundaries. | Support matrix lists object states, proof guards, and `mutatesOfficialGraph=false`. | complete |
| v2.5.0 | `docs/57-external-integration-package-v2.5.0.md` | Ship an external integration package. | Integration endpoints wrap official graph, SSOT readiness, citations, and trust state while rejecting candidate bypass. | complete |

`v2.5.0` remains backend-only. The frontend MVP path is deliberately deferred.
