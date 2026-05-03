# Cornerstone Backend

Cornerstone turns scattered organizational knowledge into a reviewed, explainable ontology graph.

It gathers source material from manual uploads and connectors, extracts evidence, proposes Concepts and Relations, lets humans review those proposals, and serves the reviewed official graph as the organization's Single Source of Truth.

## Why it exists

Organizations often have important knowledge spread across documents, tools, policies, runbooks, and people's memory. A user may ask:

```text
What is settlement?
How is settlement related to clearing, reconciliation, and ledger updates?
Which source proves that answer?
Is the answer official or only a model-generated guess?
```

Search can find documents. A chatbot can summarize text. Cornerstone is different: it builds reviewed meaning from evidence.

## How Cornerstone works

```text
Scattered Sources
    ↓
Artifacts
    ↓
EvidenceFragments
    ↓
ConceptCandidates / RelationCandidates
    ↓ human review
Official Concepts / Official Relations
    ↓
Explainable Ontology Graph
    ↓
Single Source of Truth
```

The core trust boundary is unchanged:

```text
Raw documents are inputs.
Extractor or LLM output is a proposal.
Pending candidates are not official truth.
The reviewed official ontology graph is the Single Source of Truth.
```

## Example: Settlement

A user uploads settlement notes:

```text
Settlement is the process of finalizing financial obligations.
Clearing happens before settlement.
Reconciliation validates settlement results.
Settlement updates the ledger after obligations are finalized.
```

Cornerstone can create evidence, propose candidates, and after review serve this official depth-1 graph:

```text
Clearing --------precedes--------> Settlement
Reconciliation --validates-------> Settlement
Settlement ------updates---------> Ledger
```

Every official node and edge should be explainable with evidence, review provenance, trust state, and limitations.

## Product documentation

Start here if you want to understand the product before the implementation history:

```text
docs/product/README.md
docs/product/00-product-overview.md
docs/product/01-user-problem-and-value.md
docs/product/02-how-cornerstone-works.md
docs/product/03-settlement-walkthrough.md
docs/product/04-ontology-graph-explained.md
docs/product/05-user-roles-and-workflows.md
docs/product/06-trust-model.md
docs/product/07-product-vs-chatbot-rag-wiki.md
docs/product/08-operator-quickstart.md
docs/product/09-product-glossary.md
```

Forward roadmap planning docs:

```text
docs/roadmap/README.md
docs/roadmap/v2.1.0-live-llm-ontology-provider.md
docs/roadmap/v2.2.0-review-operator-experience.md
docs/roadmap/v2.3.0-graph-visualization-contract.md
docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md
docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md
```

## Quickstart

Install and check the backend:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cornerstone doctor
cornerstone api --reload
```

Check health:

```bash
curl http://localhost:8000/healthz
```

Run the ontology proof loop:

```bash
cornerstone proof run \
  --ontology-loop \
  --confirm-ontology-mutation \
  --ontology-focus-concept Settlement \
  --reviewer reviewer@example.com \
  --json
```

Run SSOT readiness:

```bash
cornerstone proof run \
  --ssot-readiness \
  --ontology-focus-concept Settlement \
  --json
```

## Shared synthetic test corpus

The project-wide synthetic corpus lives in:

```text
test-data/shared-synthetic-corpus/
```

It contains 49 generated source documents for a fictional temperature-controlled specialty pharmacy logistics organization, plus `manifest.json`, `source_objects.jsonl`, `expected_ontology.json`, and `evaluation_tasks.json`.

Regenerate it deterministically with:

```bash
python scripts/generate_shared_synthetic_corpus.py
```

Validate the corpus and full ingestion path with:

```bash
python -m pytest tests/unit/test_shared_synthetic_corpus.py
```

## Core APIs

```http
POST /v1/manual-sources/{sourceId}/uploads
POST /v1/manual-sources/{sourceId}/uploads/text
POST /v1/ontology/extraction-runs
GET  /v1/ontology/concept-candidates
GET  /v1/ontology/relation-candidates
POST /v1/ontology/concept-candidates/{candidateId}/approve
POST /v1/ontology/relation-candidates/{candidateId}/approve
GET  /v1/ontology/graph?concept=Settlement&depth=1&mode=official
GET  /v1/ontology/explain?concept=Settlement&depth=1&mode=official
POST /v1/evaluations/ontology/run
GET  /v1/ontology/ssot/readiness?focusConcept=Settlement&depth=1&mode=official
```

Default graph depth is `1`.

## Documentation map

Use the docs in this order:

```text
Layer 1 — Product docs
  docs/product/*

Layer 2 — Technical docs
  docs/00-backend-architecture.md
  docs/01-api-contract.md
  docs/05-development-standards.md
  docs/integration-starter-kit/local-quickstart.md
  docs/integration-starter-kit/macos-quickstart.md
  docs/integration-starter-kit/google-drive-quickstart.md

Layer 3 — Roadmap, release chronicle, and readiness docs
  docs/roadmap/README.md
  docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md
  docs/50-product-documentation-layer-v2.0.2.md
  docs/51-dependency-complete-verification-v2.0.3.md
  docs/52-forward-roadmap-goals-checklists-v2.0.4.md
  docs/53-live-llm-ontology-provider-v2.1.0.md
  docs/54-review-operator-experience-v2.2.0.md
  docs/55-graph-visualization-contract-v2.3.0.md
  docs/56-connector-expansion-live-proof-hardening-v2.4.0.md
  docs/57-external-integration-package-v2.5.0.md
  docs/release/v2.0.2-product-documentation-readiness.md
  docs/release/v2.0.3-verification-readiness.md
  docs/release/v2.0.4-forward-roadmap-readiness.md
  docs/release/v2.5.0-external-integration-package-readiness.md
  docs/release/v2.5.0-release-notes.md
```

## Current version

```text
v2.5.0 — External Integration Package
```

`v2.5.0` completes the roadmap through the external integration package path. It adds a gated live LLM ontology provider, review queue and preview APIs, visualization-ready graph payloads, connector support matrix, and integration package endpoints for external consumers.

It still does not add automatic approval, frontend UI, graph depth above `1`, direct connector-to-official-graph mutation, or candidate bypass. The package/runtime version and readiness metadata now report `2.5.0`.

Current verification plan from `v2.0.3` remains available:

```bash
python scripts/run_dependency_complete_verification.py --plan-only
```

Strict verification for CI or a disposable PostgreSQL environment remains:

```bash
RUN_POSTGRES_TESTS=1 \
PERSISTENCE_BACKEND=postgres \
DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone' \
python scripts/run_dependency_complete_verification.py --strict --confirm-live-db
```

Forward roadmap docs:

```text
docs/52-forward-roadmap-goals-checklists-v2.0.4.md
docs/roadmap/README.md
docs/roadmap/v2.1.0-live-llm-ontology-provider.md
docs/roadmap/v2.2.0-review-operator-experience.md
docs/roadmap/v2.3.0-graph-visualization-contract.md
docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md
docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md
docs/53-live-llm-ontology-provider-v2.1.0.md
docs/54-review-operator-experience-v2.2.0.md
docs/55-graph-visualization-contract-v2.3.0.md
docs/56-connector-expansion-live-proof-hardening-v2.4.0.md
docs/57-external-integration-package-v2.5.0.md
```

## Release chronicle

The version chronicle remains the audit source for goals, non-goals, measurable acceptance checks, and handoffs:

```text
docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md
```

Previous roadmap milestone:

```text
v2.0.4 — Forward Roadmap Goals and Measurable Checklists
```

Important recent release docs:

```text
docs/47-ontology-ssot-release-v2.0.0.md
docs/49-refactor-domain-boundary-v2.0.1.md
docs/50-product-documentation-layer-v2.0.2.md
docs/51-dependency-complete-verification-v2.0.3.md
docs/52-forward-roadmap-goals-checklists-v2.0.4.md
docs/53-live-llm-ontology-provider-v2.1.0.md
docs/54-review-operator-experience-v2.2.0.md
docs/55-graph-visualization-contract-v2.3.0.md
docs/56-connector-expansion-live-proof-hardening-v2.4.0.md
docs/57-external-integration-package-v2.5.0.md
docs/release/v2.0.0-ontology-ssot-readiness.md
docs/release/v2.0.1-refactor-readiness.md
docs/release/v2.0.2-product-documentation-readiness.md
docs/release/v2.0.3-verification-readiness.md
docs/release/v2.0.4-forward-roadmap-readiness.md
docs/release/v2.5.0-external-integration-package-readiness.md
```

## What Cornerstone is not

Cornerstone is not just a wiki, search engine, RAG chatbot, vector database, or automatic truth generator.

It can work with those systems, but its product value is the reviewed, evidence-backed ontology graph.
