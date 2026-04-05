---
kind: decision
title: Model Decision Record as a first-class object
status: accepted
problem: A glossary and relation graph alone are not enough to reconstruct why the organization made specific choices.
decision: Decision Record becomes a first-class entity linked to evidence, concepts, and relations.
rationale: The main pain is the loss of organizational rationale, not only the loss of terms.
constraints:
  - Must be reviewable.
  - Must support evidence lineage.
impact:
  - Review workflows can approve or reject decision context explicitly.
  - Answers can explain why a concept or relation exists.
concepts:
  - Decision Record
  - Cornerstone
---

Decision context is represented as a first-class record so the product can preserve what changed, why it changed, and what evidence supported the change.
