# Product Overview

## What Cornerstone is

Cornerstone is an evidence-grounded ontology system for organizational knowledge.

It gathers knowledge from manual uploads and connected sources, turns source material into evidence, proposes Concepts and Relations, lets humans review those proposals, and serves the reviewed ontology graph as the organization's explainable Single Source of Truth.

A shorter version:

```text
Cornerstone turns scattered company knowledge into a reviewed, explainable knowledge graph.
```

## The problem it solves

Organizations usually know important things in many places:

```text
- Google Drive documents
- Notion pages
- internal policies
- operations runbooks
- Slack decisions
- manually uploaded notes
- people’s memory
```

That creates a recurring problem:

```text
A user asks, “What does settlement mean here?”
Search returns documents.
A chatbot may summarize text.
But nobody knows which answer is official, current, reviewed, or supported by evidence.
```

Cornerstone is designed to answer that question with reviewed meaning, not just retrieved text.

## The product loop

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

## What becomes the Single Source of Truth

The Single Source of Truth is not the raw document repository and not the LLM output.

```text
Raw documents are inputs.
Extractor or LLM output is a proposal.
The reviewed official ontology graph is the Single Source of Truth.
```

That boundary is the core product idea.

## What users get

For a concept like `Settlement`, users should get:

```text
- the reviewed definition
- directly related Concepts
- the type of each Relation
- citations back to evidence
- review and provenance metadata
- warnings for pending, stale, unsupported, or conflicted knowledge
```

A depth-1 graph might look like:

```text
Clearing --------precedes--------> Settlement
Reconciliation --validates-------> Settlement
Settlement ------updates---------> Ledger
Settlement ------governed_by-----> Settlement Policy
```

Every displayed node and edge should be explainable.

## What Cornerstone is not

Cornerstone is not just:

```text
- a document search engine
- a chatbot
- a wiki
- a vector database
- an automatic truth generator
```

It can work with search, LLMs, and connectors, but its product value is the reviewed ontology graph.

## Who uses it

```text
End User      asks what a concept means and why.
Reviewer      approves, rejects, edits, or merges candidates.
Source Admin  connects or uploads source material.
Operator      runs proof, readiness, and evaluation checklists.
Developer     integrates the graph and readiness APIs.
```

## Where to go next

```text
New to Cornerstone:       docs/product/03-settlement-walkthrough.md
Need trust details:       docs/product/06-trust-model.md
Need operator steps:      docs/product/08-operator-quickstart.md
Need release history:     docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md
Need technical API shape: docs/01-api-contract.md
```

## Product overview acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-OVERVIEW-01 | Product identity is understandable in one sentence. | Defines Cornerstone as an evidence-grounded ontology system. | complete |
| PROD-OVERVIEW-02 | Product problem is clear. | Explains scattered knowledge and unclear official truth. | complete |
| PROD-OVERVIEW-03 | Product loop is visible. | Includes the source → evidence → candidates → review → graph loop. | complete |
| PROD-OVERVIEW-04 | SSOT boundary is explicit. | States raw documents and LLM output are not the SSOT. | complete |
| PROD-OVERVIEW-05 | Reader knows the next document to open. | Includes navigation section. | complete |
