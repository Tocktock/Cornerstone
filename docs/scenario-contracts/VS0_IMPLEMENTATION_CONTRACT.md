# VS-0+ Implementation Contract — Zero-Base CornerStone

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** First implementation contract to freeze before coding

## Feature / Task

Build the first zero-base CornerStone vertical slice.

## Goal

Implement the first complete product identity loop:

```text
Personal messy input
→ immutable artifact
→ searchable derived representation
→ evidence-backed brief
→ draft/evidence-backed claim
→ action card dry-run
→ approval/execution
→ audit trail
```

## Success criteria

A new user can start CornerStone locally, create a personal workspace, drop messy input, search it, generate an evidence-backed brief, create a claim, create an action card, run dry-run, approve/execute a safe internal action, and inspect audit/evidence.

## Constraints

### Product / UX

- One visible product: CornerStone.
- Do not expose `Cornerstone`, `KnowledgeBase`, or `ConnectorHub` as required user mental models.
- First value must not require connector setup, paid model provider, organization admin setup, or ontology modeling.
- Primary path: `Inbox → Brief → Claim → Action → Learn`.

### Data / State

- Preserve original artifact before derived processing.
- Every truth-bearing object has tenant, namespace, owner, provenance, and trust state where relevant.
- Search snapshots must be reproducible evidence.
- Claims cannot become approved without evidence.

### Permission / Security

- Owner-scoped namespace by default.
- No cross-namespace context mixing.
- Default egress deny.
- Untrusted content cannot instruct the system.
- Secrets are redacted before durable generated knowledge.
- Actions require Workflow/Action path; no direct mutation by agents.

### Compatibility / Format

- Postgres-first durable data model.
- OpenAPI for public APIs.
- Deterministic local/test model provider for CI.
- ConnectorHub integration deferred behind capability interface.

### Operational / Environment

- One-command local run target.
- Tests must not require live external credentials.
- Verification report must list every scenario and evidence.

## Assumptions

- This is a zero-base implementation, not a direct merge of the existing three repos.
- Existing `Cornerstone`, `KnowledgeBase`, and `Connector-Hub` projects are references/adapters until imported through scenario-verified boundaries.
- Live external writeback is out of scope for VS-0+.
- Full permanent wiki/memory sovereignty is started later after artifact/evidence/claim/action/audit foundation passes.

## Out of scope before coding

- Live ConnectorHub provider E2E.
- External writeback to Slack/Notion/GitHub/email/source systems.
- Full Agent Pack registry.
- Full Memory Sovereignty Center.
- Full Experience Library and Product Learning namespace.
- Multi-brain ensemble optimization.
- Production SSO/Kubernetes/Helm/hardened backup.
- Tool marketplace and full supply-chain registry.

## Scenario Contract

| ID | Type | Trigger / Action | Expected Result | Affected Layers | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|---|
| CS-PROD-001..005 | MUST_PASS | New user opens product, follows first-use path | One coherent CornerStone service; first value via messy input → evidence-backed brief → draft claim | frontend, API, onboarding, docs | browser/E2E/API walkthrough | screenshots, E2E output, transcript | AI + human UX acceptance |
| CS-ARCH-001 | MUST_PASS | Upload any input | Original artifact preserved before processing | API, storage, DB, worker | integration test | artifact row, checksum, storage ref | AI |
| CS-ARCH-002 | MUST_PASS | Derived processor fails | Original remains accessible with failed/partial derived state | worker, API, UI | fixture test | failure record + accessible original | AI |
| CS-ARCH-003 | MUST_PASS | Upload identical and changed content | Identical content dedupes; changed content gets new identity/version/link | storage, DB | integration test | artifact IDs and lineage output | AI |
| CS-ARCH-004 | MUST_PASS | Open artifact/claim/action | Provenance visible | API, UI | API/UI test | provenance fields/panel | AI |
| CS-ARCH-005 | MUST_PASS | Upload unknown file type | Original archived; parsing deferred | API, storage, UI | fixture test | preserved unknown artifact | AI |
| CS-ARCH-006 | MUST_PASS | Input contains secret/token | Generated outputs/logs redact secrets | redaction, logging, UI | security test | redacted output/log | AI |
| CS-ARCH-007 | MUST_PASS | Artifact contains malicious prompt | Treated as untrusted evidence; no tool/action follows it | RAG, policy, tools | red-team fixture | blocked tool/action record | AI |
| CS-ARCH-008 | MUST_PASS | User creates claim from search | Query/filter/result snapshot attached to Evidence Bundle | search, evidence, claims | integration test | evidence bundle JSON | AI |
| CS-ARCH-009 | MUST_PASS | Claim cites excerpt/derived text | Evidence opens original and derived representation | evidence viewer, API | UI/API test | original + derived links | AI |
| CS-UND-001..005 | MUST_PASS | Upload/search/query without ontology setup | Keyword + semantic search work; active workspace respected; artifact viewer works; no forced modeling | search, DB, UI | integration/E2E tests | search result snapshots | AI |
| CS-CLAIM-001..010 | MUST_PASS | User chats/asks “what matters?” and promotes output | Brief cites evidence; trust ladder visible; unsupported assertions labeled; claim can become action input | conversation, brief, claims, action | API/E2E tests | brief/claim/action records | AI |
| CS-NS-001..004 | MUST_PASS | Create/use/promote items across personal/org workspaces | Every item has owner namespace; active workspace clear; no automatic context crossing; promotion is explicit/audited | authz, DB, UI, audit | isolation tests | allowed/denied outputs, promotion audit | AI |
| CS-AUTO-001..011 | MUST_PASS | Create action from claim and try to execute | Workspace mode exists; action card uses dry-run; high-risk requires approval; direct write path denied | workflow, policy, action, audit | integration/security tests | dry-run, policy, approval, audit records | AI |
| CS-SEC-001..008 | MUST_PASS | Run local stack and security fixtures | One-command local start; egress deny; RBAC/ABAC starter; policy explanation; tamper-evident audit; injection/secret tests pass | deploy, security, policy, audit | compose/tests/security suite | logs/test output | AI |
| CS-REG-001..006 | REGRESSION_GUARD | Review release behavior | Product does not regress into chatbot-only/search-only/connector-only/logistics-only; memory not truth; personal context does not leak | product, UX, search, namespaces | scenario report + tests | regression outputs | AI |
| H-UX-001 | HUMAN_REQUIRED | Owner reviews first-run product feel | Owner confirms it feels like one CornerStone product | UX/product | human walkthrough | owner approval/screenshot notes | Human |

## Observable VS-0+ verification checklist

- [ ] `docker compose config` or equivalent local stack validation passes.
- [ ] API health/ready endpoints work.
- [ ] Web shell opens as CornerStone.
- [ ] Personal workspace is created by default.
- [ ] Upload creates immutable artifact with hash and storage ref.
- [ ] Derived extraction can fail without losing original.
- [ ] Unknown file is still archived.
- [ ] Redaction fixture hides secrets in generated outputs/logs.
- [ ] Prompt-injection fixture does not trigger tool/action execution.
- [ ] Search finds uploaded content.
- [ ] Search snapshot is stored.
- [ ] Evidence bundle contains artifact refs and search snapshot refs.
- [ ] Brief includes evidence links, uncertainty, contradictions/gaps where applicable.
- [ ] Unsupported assertion is labeled Draft/Assumption/Insufficient Evidence.
- [ ] Claim without evidence cannot be approved.
- [ ] Claim with evidence can become Evidence-backed/Approved through allowed policy.
- [ ] ActionCard shows goal, evidence, diff, impact, policy, risk, status, audit link.
- [ ] Action execution requires dry-run first.
- [ ] Risky/external action requires approval.
- [ ] Direct mutation outside Workflow/Action path is denied.
- [ ] Audit ledger records critical lifecycle events.
- [ ] Audit tamper test detects modification.
- [ ] Personal/org namespace isolation test passes.
- [ ] Scenario report generated with PASS/FAIL/NOT_VERIFIED/HUMAN_REQUIRED statuses.

## Verdict rule

VS-0+ cannot be marked done if any AI-verifiable MUST_PASS or REGRESSION_GUARD scenario is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
