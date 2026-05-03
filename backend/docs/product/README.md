# Cornerstone Product Documentation

This folder is the product-first documentation layer for Cornerstone. It explains what the product is, why it exists, how users experience it, and how the ontology Single Source of Truth works.

Read these before the release chronicle if you are new to the project.

## Recommended reading order

```text
00-product-overview.md
01-user-problem-and-value.md
02-how-cornerstone-works.md
03-settlement-walkthrough.md
04-ontology-graph-explained.md
05-user-roles-and-workflows.md
06-trust-model.md
07-product-vs-chatbot-rag-wiki.md
08-operator-quickstart.md
09-product-glossary.md
```

## Core product sentence

Cornerstone turns scattered organizational knowledge into a reviewed, explainable ontology graph.

## Core trust sentence

Raw documents are inputs. Extractor output is a proposal. The reviewed official graph is the Single Source of Truth.

## Product documentation acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-DOC-README-01 | New readers can choose the correct first document. | Reading order lists all product docs. | complete |
| PROD-DOC-README-02 | Product identity is stated without implementation jargon. | Core product sentence appears above. | complete |
| PROD-DOC-README-03 | Trust boundary is stated before technical docs. | Core trust sentence appears above. | complete |
| PROD-DOC-README-04 | Product docs are separate from release chronology. | This folder is independent from `docs/48-*`. | complete |
| PROD-DOC-README-05 | Product docs are measurable. | Each product doc has an acceptance checklist. | complete |
