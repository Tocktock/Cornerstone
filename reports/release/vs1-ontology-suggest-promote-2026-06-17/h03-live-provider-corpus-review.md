# VS1 H03 Live-Provider Corpus Review

Status: H03_DEFERRED_LLM_REVIEW_REQUIRED
Owner: JiYong / Tars
Date: 2026-06-17

## Decision

`VS1-ONT-H03` is **not accepted as production/live-provider ready yet**.

The current review selects **OpenAlex** as the safest public read-only corpus for a future H03 rehearsal, but final H03 acceptance is deferred until CornerStone uses the actual live-provider or LLM-provider path that will be released.

## Selected Corpus

| Provider | Decision | Why |
|---|---|---|
| OpenAlex Works API | Recommended | Public, read-only, no basic credentials, graph-shaped data, and natural evidence-map entities: works, authors, institutions, topics, sources, publishers, funders. |
| Crossref REST API | Fallback | Public DOI metadata corpus; useful for bibliographic fallback but less graph-shaped than OpenAlex. |
| GitHub public issues API | Fallback | Good operational issue/label corpus; more volatile and includes public usernames. |

Recommended query:

```text
https://api.openalex.org/works?search=software%20supply%20chain%20security&per-page=10
```

## Captured Corpus Evidence

Evidence file: `reports/release/vs1-ontology-suggest-promote-2026-06-17/h03-corpus-evidence.json`

Observed live-read summary:

- OpenAlex returned `219331` matching works for `software supply chain security`.
- First OpenAlex result: `Top Five Challenges in Software Supply Chain Security: Observations From 30 Industry and Government Organizations`.
- Crossref and GitHub public issues were checked as fallback read-only corpora.
- No credentials were used.
- No mutating requests were made.
- No private or customer data was used.
- No LLM provider was used in this corpus-selection step.

## Future H03 Pass Criteria

To accept `VS1-ONT-H03` later, capture a separate live-provider or production-data proof with:

- explicit human approval record;
- provider and corpus/data-source name;
- redacted request/response transcript;
- evidence-map output showing source evidence, edges, weights, descriptions, and why;
- audit refs from the product runtime;
- execution/result evidence;
- redaction confirmation;
- explicit note that secrets, private data, and unsupported production readiness were not exposed.

## Future LLM Review

When an LLM provider is introduced, run a separate LLM-specific review before accepting H03:

- hallucinated entities or edges;
- unsupported relationship weights;
- missing evidence spans;
- prompt-injection handling;
- overconfident descriptions;
- drift between deterministic local extraction and LLM extraction;
- redacted transcript and audit refs for every provider call.

## Current Boundary

H01 and H02 are accepted for the local vendor-risk evidence-map UX. H03 remains deferred for future live-provider or LLM-provider proof.

