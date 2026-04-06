# P0 Implementation Specs

These specs define the **P0 functional feature contract** for the first implementation milestone.

They are intentionally functional only:
- no framework choices
- no programming-language choices
- no database choices
- no transport-specific implementation choices

## P0 feature set

- [P0-001 Source Ingestion and Sync](./001-source-ingestion-and-sync/spec.md)
- [P0-002 Curated Concepts and Relations](./002-curated-concepts-and-relations/spec.md)
- [P0-003 Decision Context and Officialization](./003-decision-context-and-officialization/spec.md)
- [P0-004 Retrieval and Source-Backed Answers](./004-retrieval-and-source-backed-answers/spec.md)
- [P0-005 Workspace Governance and Access](./005-workspace-governance-and-access/spec.md)
- [P0-006 Serving Contract](./006-serving-contract/spec.md)


## Upstream dependency

The P0 specs assume the canonical object model, predicate taxonomy, and projection semantics defined in [`../ontology/spec.md`](../ontology/spec.md).
