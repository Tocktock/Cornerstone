# Known Limitations

This document lists known backend limitations for the backend MVP release candidate.

## Accepted for backend MVP

These do not block backend v1.0.0 if the documented pilot loop passes.

```text
1. Notion page ingestion is supported and live-proofed.
2. Notion database/data_source objects are discoverable but intentionally not selectable for ingestion yet.
3. Slack and GitHub connectors are not included in the backend MVP release.
4. Google Drive supports selected Google Docs and text files; Sheets, Slides, PDFs, folders, and unsupported binary files are discoverable but not ingestible yet.
5. Manual source ingestion is available for controlled pilot data, including v1.3.1 UTF-8 text-like file/text uploads.
6. Runtime vector retrieval is not implemented; evidence_embeddings is prepared for future use.
7. Evaluation is deterministic/rule-based, not LLM-graded.
8. Clarification-reduction measurement is represented as a metric target but not instrumented from real Slack/user behavior yet.
9. Full enterprise RBAC/SSO is not implemented; reviewer authorization uses a configured allow-list.
10. Notion webhooks and full incremental provider cursor integration are deferred.
11. Ontology extraction in v1.4.0 creates pending candidates only and uses the bundled local deterministic provider.
12. Ontology candidate review in v1.5.0 supports edit, approve, reject, and merge workflows, but batch review and a frontend review UI are deferred.
13. Explainable graph serving in v1.6.0 is backend/API-only and depth-limited to 1.
14. Frontend/UI is not included in this backend release.
```

## Must remain true despite limitations

```text
1. No fake provider source creation path.
2. No fake OAuth completion path.
3. No generic source sync bypass for provider-backed sources.
4. No official Concept without reviewed eligible evidence or valid DecisionRecord.
5. No grounded official response without valid citations.
6. Unsupported queries must return unsupported, not fabricated answers.
7. Source sync failure must not mark data fresh.
8. Production mode must fail closed on unsafe defaults.
```

## Post-v1.0 candidate work

```text
1. Notion database/data_source ingestion semantics.
2. Slack connector.
3. GitHub App connector.
4. Google Sheets / Slides / PDF ingestion semantics.
5. Manual PDF/Office/binary upload extraction semantics.
6. Runtime vector retrieval and ranking.
7. Batch evidence review operations.
8. Full RBAC/SSO integration.
9. KMS/secret-manager-backed credential provider.
10. LLM-assisted evaluation and human adjudication workflow.
11. Frontend Source Studio / Evidence Review / Glossary UI.
```



## Google Drive v1.2.0

Google Drive supports selected Google Docs and plain text files. Google Sheets, Slides, PDFs, folders, and unsupported binary files are discoverable but not ingestible yet.

## Ontology SSOT v1.2.1

`v1.2.1` defines the ontology product contract but does not implement ontology runtime behavior yet.

Accepted limitations for this documentation baseline:

```text
1. Ontology graph APIs are not implemented yet.
2. LLM ConceptCandidate and RelationCandidate extraction is not implemented yet.
3. Manual file upload is not implemented yet; manual source sync remains available.
4. ConceptAlias is proposed but not implemented yet.
5. Graph visualization UI is not included.
6. The reviewed ontology graph is defined as the future Single Source of Truth, but v1.2.1 only documents the contract.
```

Must remain true:

```text
1. LLM output must not directly become official truth.
2. Official Concepts and Relations must remain evidence-backed.
3. Unsupported grounded responses must not fabricate answers.
```

## Ontology Graph Runtime v1.3.0

`v1.3.0` implements ontology search and depth-1 graph serving over existing Concepts and ConceptRelations.

Accepted limitations:

```text
1. Ontology graph serving exists, but only for Concepts and ConceptRelations that already exist.
2. LLM ConceptCandidate and RelationCandidate extraction is still not implemented.
3. Manual file upload is still not implemented; manual source sync remains available.
4. Candidate graph mode only exposes existing candidate/reviewing Concepts and Relations.
5. Search is lexical over Concept names and aliases, not semantic/vector search.
6. Graph depth above 1 is intentionally rejected.
7. Graph visualization UI is not included.
```

Must remain true:

```text
1. Official ontology graph mode must exclude candidate Concepts and candidate ConceptRelations.
2. Official graph output must cite eligible EvidenceFragments.
3. Unsupported graph responses must not invent Concepts or Relations.
4. LLM output must remain candidate-only when added in a future version.
```


## Manual Uploaded Data Ingestion v1.3.1

`v1.3.1` implements manual uploaded data ingestion for UTF-8 text-like files and pasted text.

Accepted limitations:

```text
1. Manual upload accepts text-like UTF-8 content only.
2. PDFs, Office documents, images, binary files, and non-UTF-8 files are rejected.
3. Large files are rejected by configured request limits instead of chunked.
4. Upload source identity is filename/title based plus content hash idempotency.
5. Upload endpoints create only Artifacts and EvidenceFragments.
6. Upload endpoints do not create Concepts, ConceptRelations, candidates, or ontology graph edges.
7. LLM extraction is still deferred to v1.4.0.
```

Must remain true:

```text
1. Manual upload must be manual-source-only.
2. Provider-backed sources must still use connector discovery, selection, and sync jobs.
3. Rejected uploads must not create Artifacts or EvidenceFragments.
4. Uploaded evidence must still be reviewed before official ontology use.
5. Upload endpoints must not imply that the uploaded document itself is the Single Source of Truth.
```


## LLM Ontology Extraction v1.4.0

`v1.4.0` implements ontology extraction runs and candidate persistence.

Accepted limitations:

```text
1. The bundled extraction provider is deterministic and local, not a live external LLM.
2. Candidate quality depends on explicit evidence wording.
3. Candidate review workflows are implemented in v1.5.0, but extraction itself still creates candidates only.
4. Candidate promotion requires explicit reviewer approval through v1.5.0 review endpoints.
5. Concept deduplication is limited to exact normalized names and aliases.
6. Relation extraction supports explicit relation phrases only.
7. Candidates do not appear in the official ontology graph until approved through review.
```

Must remain true:

```text
1. Ontology extraction must create ConceptCandidates and RelationCandidates only.
2. Candidates must remain pending until reviewed by a later workflow.
3. Extraction must not create official Concepts.
4. Extraction must not create official ConceptRelations.
5. Extraction must not mutate the official graph.
6. Every candidate must cite source EvidenceFragments.
```


## Ontology Candidate Review Workflow v1.5.0

`v1.5.0` implements human review for ConceptCandidates and RelationCandidates.

Accepted limitations:

```text
1. Review endpoints are backend-only; there is no frontend review queue UI yet.
2. Batch approve/reject/merge operations are not included.
3. Review conflict handling is API-first and returns 409 responses instead of guided UI resolution.
4. Relation approval requires endpoints to resolve to existing Concepts; it does not auto-create missing Concepts.
5. Candidate merge is exact/id-based, not semantic duplicate clustering.
6. The bundled extraction provider remains deterministic/local; no live external LLM provider is introduced.
7. Official graph depth remains limited to 1 from v1.3.0.
```

Must remain true:

```text
1. Only pending candidates can be edited, approved, rejected, or merged.
2. Reviewer identity must be authorized before review actions in production mode.
3. Approval must pass existing officialization gates.
4. Rejection must not create Concepts, Relations, or graph edges.
5. Merge into official objects must preserve reviewed evidence requirements.
6. Candidate review must not call an LLM or fabricate ontology content.
7. Unsupported candidate content must not enter the official graph.
```

## Explainable Graph Serving v1.6.0

`v1.6.0` improves graph explanations, provenance, and support summaries.

Accepted limitations:

```text
1. The graph is still backend/API-only; no visual graph UI is included.
2. Graph depth remains limited to 1.
3. CandidateSummary is lexical/focus-based and not semantic duplicate clustering.
4. Pending candidates are summarized but not rendered as official graph nodes/edges.
5. Explainability is based on stored review/evidence metadata; it does not judge semantic quality with an LLM.
6. No graph database or recursive traversal engine is introduced.
```

Must remain true:

```text
1. Official graph mode must exclude candidate objects.
2. Explanations must not hide unsupported, stale, or conflicted graph state.
3. Graph serving must not call an LLM or infer missing edges.
4. Candidate summaries must not imply candidate content is official truth.
```

## Ontology Evaluation v1.7.0

`v1.7.0` adds read-only evaluation for ontology graph serving.

Known limitations:

```text
- Evaluation is deterministic and rule-based, not semantic.
- Evaluation does not call an external LLM.
- Evaluation does not judge whether a reviewed definition is domain-perfect.
- Evaluation does not create, repair, approve, or promote Concepts or Relations.
- Evaluation does not mutate the official graph.
- Evaluation is API-only; there is no evaluation dashboard UI yet.
- Graph depth remains limited to 0 or 1.
- Persistent-store and migration verification require a dependency-complete environment with SQLAlchemy and Alembic installed.
```

## Connector-driven Re-extraction v1.8.0

`v1.8.0` adds safe re-extraction queueing after source changes.

Accepted limitations:

```text
1. Change detection is based on source object identity and content hash, not semantic meaning.
2. First-time Artifacts queue re-extraction because they introduce new evidence.
3. Exact reused Artifacts do not queue re-extraction by default.
4. Re-extraction uses the deterministic/local ontology extractor from v1.4.0.
5. Re-extraction creates ConceptCandidates and RelationCandidates only.
6. Re-extraction does not clean up stale previous candidates.
7. Re-extraction does not automatically merge duplicate candidates.
8. Re-extraction queueing is API/worker driven; no queue UI is included.
9. Inline re-extraction is supported but default-off to avoid increasing sync latency.
10. Persistent-store and migration verification require SQLAlchemy and Alembic in the local environment.
```

Must remain true:

```text
1. Source sync and manual upload must not create official ontology objects.
2. OntologyReExtractionRun must not mutate the official graph.
3. officialGraphMutated must remain false for v1.8.0 runs.
4. Candidate review must remain required before SSOT graph mutation.
5. Pending candidates must not appear as official graph nodes or edges.
6. Re-extraction must preserve evidence provenance for every generated candidate.
```

## End-to-end Proof and Operator UX Checklist v1.9.0

`v1.9.0` adds an operator proof checklist rather than a frontend UX.

Accepted limitations:

```text
1. The proof workflow is API/CLI-driven; no visual UI is included.
2. Confirmed proof runs create explicit proof source data and official proof graph objects.
3. Dry-run mode is required for no-mutation checklist inspection.
4. The proof uses deterministic local ontology extraction, not a live external LLM provider.
5. The proof is not a substitute for real connector/live-source acceptance testing.
6. Repeated proof runs may create additional proof Concepts and Relations unless operators choose isolated focus/source names.
7. Graph depth remains limited to 1.
```

Must remain true:

```text
1. Non-dry proof execution must require confirmMutation=true.
2. Reviewer authorization must be enforced.
3. Evidence review must precede candidate promotion.
4. Re-extraction output must remain candidate-only.
5. Official graph changes must happen only through candidate review/officialization gates.
```


## Ontology SSOT Release v2.0.0

`v2.0.0` stabilizes the backend SSOT contract and adds a read-only readiness endpoint.

Accepted limitations:

```text
1. The readiness endpoint checks current state only; it does not repair missing setup.
2. The readiness result is focus-concept specific, not whole-tenant certification.
3. A successful ontology evaluation result must already exist for readiness to pass.
4. No frontend graph visualization is included.
5. No live external LLM provider is included.
6. Graph depth remains limited to 1.
7. Persistent-store proof should be rerun in dependency-complete environments.
```

Must remain true:

```text
1. GET /v1/ontology/ssot/readiness must be read-only.
2. Raw source data must not be treated as SSOT.
3. Candidate output must not be treated as SSOT.
4. Pending candidates must not appear in official graph mode.
5. Official graph safety must require valid evidence and review provenance.
```


## v2.0.1 refactor limitations

`v2.0.1` improves maintainability and domain boundaries only. It intentionally does not add a frontend UI, live external LLM provider, graph depth above `1`, automatic candidate approval, or new database migrations. Persistent-store verification should still be rerun in an environment with SQLAlchemy and Alembic installed.

## v2.0.2 Product Documentation Limitations

`v2.0.2` improves product explanation only. It intentionally does not add a frontend UI, live external LLM provider, graph depth above `1`, automatic candidate approval, new database migrations, or new API behavior.

Accepted limitations:

```text
1. Product docs are Markdown-only.
2. The operator quickstart is backend/API/CLI-oriented.
3. The settlement walkthrough is illustrative and depends on actual source evidence during runtime.
4. The backend still has no visual graph UI.
5. Persistent-store and live connector verification remain separate operational checks.
```

## v2.0.3 Dependency-Complete Verification Limitations

`v2.0.3` improves verification and CI readiness only. It intentionally does not add a frontend UI, live external LLM provider, graph depth above `1`, automatic candidate approval, new database migrations, new connector behavior, or new API behavior.

The strict verification runner requires a dependency-complete environment with PostgreSQL and dev dependencies installed. Minimal sandboxes can run the plan-only mode, but they cannot prove the full dependency-complete contract unless SQLAlchemy, Alembic, Ruff, mypy, PostgreSQL, and the package dev dependencies are available.

## v2.0.4 Forward Roadmap Documentation Limitations

`v2.0.4` improves planning clarity only. It intentionally does not add a live external LLM provider, review operator UX changes, graph visualization response contract, connector expansion, frontend UI, integration package, graph depth above `1`, automatic candidate approval, new database migrations, new connector behavior, or new API behavior.

Accepted limitations:

```text
1. Roadmap documents are planning contracts, not implementation proof.
2. Future releases may refine scope, but must preserve the SSOT trust boundary unless a new ADR explicitly changes it.
3. v2.1.0 through v2.5.0 remain unimplemented at v2.0.4.
4. The future frontend/integration path is intentionally undecided until graph and connector maturity improve.
5. Persistent-store, live connector, live LLM, UI, and integration behavior must be verified in their own future releases.
```

## v2.5.0 External Integration Package Limitations

`v2.5.0` implements the external integration package path and intentionally does not ship a frontend MVP.

Accepted limitations:

```text
1. Live LLM extraction is disabled by default and must be explicitly configured.
2. Extractor output remains candidate-only until a reviewer approves or merges it.
3. Graph depth remains bounded to 1.
4. The connector support matrix exposes capabilities; it does not expand every connector object into ingestion support.
5. Integration endpoints expose candidates only as summaries and reject candidate bypass requests.
6. No database migration is included in this release.
```
