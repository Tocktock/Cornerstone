# Retrieval and Answers

## Summary

Cornerstone serves organizational context back to humans and AI through search, browse, glossary, graph, and grounded answers.

Retrieval should not return isolated fragments when a better context reconstruction is possible. The product should help users understand not only the answer, but also the meaning, rationale, and provenance behind it.

## Scope and owned behavior

This spec owns:
- search and browse principles
- grounded answer behavior
- glossary and graph consumption behavior
- answer explainability expectations for human and AI consumers
- interaction between visible support and restricted support

## Current behavior

- Retrieval is always scoped to the current workspace.
- Member-facing retrieval defaults to `member_visible` content only.
- Review retrieval may include `evidence_only` support when needed for diagnosis or officialization.
- Retrieval should prefer curated meaning when it exists: concepts, relations, and linked decision context.
- Retrieval may fall back to artifacts or source-native paths when curated meaning is incomplete.
- The same underlying truth layer must serve both human UI and AI integrations.
- Grounded answers must be labeled according to the current consumer’s `support_visibility`:
  - `source_backed`
  - `restricted_support`
  - `insufficient_support`

## Grounded answer rules

A member-facing grounded answer must expose, where relevant:
- answer text or structured sections
- cited concepts
- cited relations
- cited decisions
- visible support items available to the current consumer
- provenance summary
- freshness state where relevant
- trust and review state

Mandatory trust rules:
- A member-facing answer may be labeled `source_backed` only when the current member can inspect at least one visible support item.
- If the answer is official but depends partly or wholly on hidden support, the answer must be labeled `restricted_support`.
- `restricted_support` answers may include provenance summary, review status, and visible linked concepts, but they must not imply that all decisive support is inspectable by the current consumer.
- If workspace policy disallows member-facing restricted-support publication, such answers must not appear in ordinary member-facing retrieval.

## Retrieval modes

Useful retrieval modes include:
- browse concepts
- search across artifacts and curated meaning
- inspect graph relationships
- ask for a grounded answer
- follow provenance back to source
- hand off to source-native search or the original source when necessary

## Canonical response kinds

Consumer-facing retrieval must use one of the canonical response kinds defined in [../serving-contract/spec.md](../serving-contract/spec.md):
- `concept`
- `relation`
- `decision`
- `answer`
- `search_results`
- `graph_slice`
- `provenance`
- `no_match`

## Constraints and non-goals

- Retrieval is not a generic public web search product.
- Retrieval is not only chat UX.
- Retrieval should not become a scoring-debug console for ordinary users.
- Cross-workspace retrieval is out of scope.

## Related docs

- [../concepts/spec.md](../concepts/spec.md)
- [../graph-and-relations/spec.md](../graph-and-relations/spec.md)
- [../decision-context/spec.md](../decision-context/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md](../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md)
- [../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md](../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md)
