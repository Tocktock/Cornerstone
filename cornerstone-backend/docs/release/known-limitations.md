# Known Limitations

This document lists known backend limitations for the backend MVP release candidate.

## Accepted for backend MVP

These do not block backend v1.0.0 if the documented pilot loop passes.

```text
1. Notion page ingestion is supported and live-proofed.
2. Notion database/data_source objects are discoverable but intentionally not selectable for ingestion yet.
3. Slack, Google Docs, and GitHub connectors are not included in the backend MVP release.
4. Manual source ingestion is available for controlled pilot data.
5. Runtime vector retrieval is not implemented; evidence_embeddings is prepared for future use.
6. Evaluation is deterministic/rule-based, not LLM-graded.
7. Clarification-reduction measurement is represented as a metric target but not instrumented from real Slack/user behavior yet.
8. Full enterprise RBAC/SSO is not implemented; reviewer authorization uses a configured allow-list.
9. Notion webhooks and full incremental provider cursor integration are deferred.
10. Frontend/UI is not included in this backend release.
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
3. Google Docs/Drive connector.
4. GitHub App connector.
5. Runtime vector retrieval and ranking.
6. Batch evidence review operations.
7. Full RBAC/SSO integration.
8. KMS/secret-manager-backed credential provider.
9. LLM-assisted evaluation and human adjudication workflow.
10. Frontend Source Studio / Evidence Review / Glossary surfaces.
```

