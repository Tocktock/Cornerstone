# Ontology Worked Example

This file is **non-normative**.

It exists to make the canonical ontology easier to read by showing how one small workspace slice maps to the class model defined in [`spec.md`](./spec.md).

## Example scenario

A workspace wants to preserve its shared understanding of customer-health scoring.

### Concepts

| Concept ref | Concept kind | Meaning |
| --- | --- | --- |
| `concept:customer_health_score` | `term` | Canonical definition of customer health score |
| `concept:renewal_risk_policy` | `policy` | Policy used to assess renewal risk |
| `concept:quarterly_customer_review` | `workflow` | Recurring review workflow |
| `concept:account_expansion_score` | `metric` | One of the metrics used in the score |
| `concept:customer_success_team` | `role` | Operational owner of the practice |

### Semantic graph relations

| Subject | Predicate | Object |
| --- | --- | --- |
| `customer_health_score` | `depends_on` | `account_expansion_score` |
| `renewal_risk_policy` | `applies_to` | `quarterly_customer_review` |
| `quarterly_customer_review` | `owned_by` | `customer_success_team` |

### Decision context

One `DecisionRecord` may state:

> Use customer health score as an early renewal-risk signal for enterprise accounts, because churn indicators appear earlier in health trends than in renewal-stage deal notes.

That decision may:
- affect the `customer_health_score` concept
- affect the `renewal_risk_policy` concept
- justify the `applies_to` relation between the policy and the workflow

### Support items

The slice may be supported by:
- one `EvidenceFragment` from a shared playbook
- one `EvidenceFragment` from a synced CRM policy artifact
- one `PromotedSupport` item created from a private analyst note and explicitly promoted into the workspace

### What appears in the graph

The graph view shows:
- concepts as nodes
- concept relations as semantic edges

It does **not** treat:
- support links
- decision applicability links
- review-scope grants
- promotion lineage

as ordinary semantic graph edges.

Those remain linked context around the graph, not replacements for graph predicates.
