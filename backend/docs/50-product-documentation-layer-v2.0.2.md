# v2.0.2 — Product Documentation Layer

## Purpose

`v2.0.2` is a documentation-first release after `v2.0.1`. Its purpose is to make Cornerstone understandable as a product before readers enter the technical release chronicle.

The prior documentation was strong for auditability, versioning, and engineering verification, but it was too release-centric for a new reader. This release adds a separate product documentation layer that explains the user problem, product value, ontology graph, trust model, settlement walkthrough, roles, and operator path.

## Version goal

Create a product-first documentation layer that clearly explains:

```text
- what Cornerstone is
- why it exists
- who uses it
- how the product loop works
- why the reviewed ontology graph is the Single Source of Truth
- how the settlement example works end to end
- how Cornerstone differs from chatbots, RAG, search, wikis, and generic knowledge graphs
```

## Confirmed non-goal

`v2.0.2` does not add product runtime behavior.

Specifically, it does not add:

```text
- new API endpoints
- new database migrations
- new graph behavior
- graph depth above 1
- live external LLM integration
- automatic candidate approval
- automatic official graph mutation
- frontend UI or graph visualization
```

The only runtime-facing change is version metadata moving from `2.0.1` to `2.0.2`.

## Product boundary preserved

The SSOT boundary remains unchanged:

```text
Raw source data is not the Single Source of Truth.
Connector sync output is not the Single Source of Truth.
Ontology extraction output is candidate-only.
Pending candidates are not the Single Source of Truth.
Reviewed official Concepts and ConceptRelations form the ontology Single Source of Truth.
```

## Documentation architecture after v2.0.2

Cornerstone documentation now has three layers:

```text
Layer 1 — Product documentation
For new users, reviewers, operators, stakeholders, and product discussion.

Layer 2 — Technical documentation
For developers and integrators who need APIs, architecture, and implementation details.

Layer 3 — Release chronicle
For audit, versioning, measurable acceptance checks, and historical proof.
```

## Product docs added

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

## README change

The README now leads with product explanation instead of release history.

The new README order is:

```text
1. What Cornerstone is
2. Why it exists
3. How it works
4. Settlement example
5. Trust boundary
6. Quickstart
7. Core APIs
8. Documentation map
9. Current version and release chronicle
```

Release history is still preserved in the chronicle, but it is no longer the first thing a new reader sees.

## Chronicle position and measurable release checklist

`v2.0.2` follows `v2.0.1` as a product-documentation release. It is documentation-only except for version metadata.

## Measurable acceptance checklist

| Check ID | Measurable acceptance condition | Evidence / verification source | Status |
|---|---|---|---|
| V202-01 | Product documentation folder exists and is separate from release docs. | `docs/product/` with README and product docs. | complete |
| V202-02 | Product overview explains Cornerstone in one sentence and one product loop. | `docs/product/00-product-overview.md`. | complete |
| V202-03 | User problem/value doc explains why documents, search, and chatbots are insufficient. | `docs/product/01-user-problem-and-value.md`. | complete |
| V202-04 | Settlement walkthrough shows source text → evidence → candidates → review → official graph. | `docs/product/03-settlement-walkthrough.md`. | complete |
| V202-05 | Ontology graph doc defines Concept, Relation, Evidence, candidate, official, and depth 1. | `docs/product/04-ontology-graph-explained.md`. | complete |
| V202-06 | Trust model doc states the SSOT trust boundary and official graph mode rules. | `docs/product/06-trust-model.md`. | complete |
| V202-07 | Operator quickstart shows manual source → review → graph → evaluation → readiness. | `docs/product/08-operator-quickstart.md`. | complete |
| V202-08 | README is product-first and points to product docs before chronicle docs. | `README.md`. | complete |
| V202-09 | Chronicle records v2.0.2 with goal, non-goal, checklist, and handoff. | `docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md`. | complete |
| V202-10 | Release checker and documentation tests require the product docs. | `scripts/check_release_candidate.py`, `tests/unit/test_release_candidate_docs.py`. | complete |

## Implementation checklist

```text
[x] Add product documentation folder.
[x] Add product documentation index.
[x] Add product overview.
[x] Add user problem and value document.
[x] Add how-it-works document.
[x] Add settlement walkthrough.
[x] Add ontology graph explanation.
[x] Add user roles and workflows.
[x] Add trust model.
[x] Add product positioning comparison.
[x] Add operator quickstart.
[x] Add product glossary.
[x] Rewrite README into product-first structure.
[x] Update release checker and docs tests.
[x] Update chronicle with v2.0.2.
[x] Add v2.0.2 readiness and release notes.
```

## Test plan

```text
python -m compileall -q src tests scripts
python scripts/check_release_candidate.py
python -m pytest tests/unit/test_release_candidate_docs.py -q
python -m pytest tests/integration/test_ontology_ssot_readiness_api.py -q
```

## Proof / verification

This release is verified by static documentation checks and compile checks. It does not require a new migration, live connector proof, or UI proof because no runtime product behavior is added.

## Known limitations

```text
- Product docs are Markdown-only.
- There is still no frontend graph UI.
- The operator quickstart is backend/API/CLI-oriented.
- Live connector proof remains outside this documentation-only release.
```

## Exit criteria

`v2.0.2` is complete when a new reader can understand Cornerstone from product docs without reading version history first, and release checks require those docs to remain present.

## Next version handoff

Future product or technical work should keep the three-layer documentation structure:

```text
1. Product docs explain user value.
2. Technical docs explain implementation.
3. Chronicle docs record version goals and measurable checks.
```
