---
kind: decision
title: Treat Cornerstone as the official context layer
status: accepted
problem: Documents, conversations, and systems are scattered, so both humans and AI fail to reconstruct organizational context.
decision: Cornerstone will act as the official context layer above source systems instead of replacing them.
rationale: This keeps provenance intact, avoids source replacement scope creep, and aligns humans and AI on the same shared memory.
constraints:
  - Must preserve provenance.
  - Must support eventual sync.
  - Must remain reviewable by humans.
impact:
  - Official knowledge becomes source-backed.
  - API consumers and AI workers read the same context model.
concepts:
  - Cornerstone
  - Shared Organizational Context Layer
---

This decision anchors the product as a context layer rather than a document replacement system.

It also clarifies that glossary, relation graph, and decision context are curated projections built on top of artifacts and evidence.
