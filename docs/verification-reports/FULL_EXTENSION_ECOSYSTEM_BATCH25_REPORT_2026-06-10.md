# Full Extension Ecosystem Batch 25 Report - 2026-06-10

Status: PASS for deterministic CLI-native Agent Pack / extension ecosystem scaffold only.
Scope: `CS-EXT-001` through `CS-EXT-016`, plus directly covered `CS-SEC-015`, `CS-SEC-016`, `CS-REG-014`, and `CS-REG-015`.

This report does not mark production UI runtime, production API runtime, public marketplace behavior, real package installation, real signature verification, real ConnectorHub provider calls, hosted registry certification, or full 206-scenario completion as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies the local Agent Pack registry, install-vs-activation separation, explicit activation grants, organization-admin shortcut governance, certification cards, version pinning, update diff/evaluation gates, rollback, ConnectorHub-mediated access, direct-provider quarantine, untrusted activation denial, and emergency security patch policy.

## Research Checkpoint

- SLSA provenance models where, when, and how artifacts were produced and captures builder identity, inputs, dependencies, and subjects: <https://slsa.dev/spec/v1.1/provenance>
- Sigstore/Cosign supports signature and in-toto attestation verification workflows: <https://docs.sigstore.dev/cosign/verifying/>
- OpenSSF Scorecard treats automated security results as heuristics for risk decisions, not a definitive universal trust judgment: <https://github.com/ossf/scorecard>
- VS Code extension manifests separate declared contributions from activation events, supporting explicit activation design: <https://code.visualstudio.com/api/references/extension-manifest> and <https://code.visualstudio.com/api/references/activation-events>
- MCP authorization requires bearer authorization on requests and forbids access tokens in query strings, reinforcing ConnectorHub credential custody boundaries: <https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization>
- AgentDojo shows tool-using agents are vulnerable to prompt injection through external tools and need security evaluations: <https://arxiv.org/abs/2406.13352>
- Recent MCP/tool-security work highlights gateway policy, descriptor integrity, and deterministic tool-boundary enforcement as dominant defensive directions: <https://research.ibm.com/publications/securing-mcp-based-agent-workflows>, <https://arxiv.org/abs/2512.06556>, and <https://arxiv.org/abs/2604.11790>

Best fit for this batch is the existing deterministic local runtime. It records local JSON registry, install, activation, certification, update, rollback, patch, playbook, policy, and audit records rather than adding a public marketplace, package manager, signature-verification dependency, or live provider integration.

## Assumptions

- Native CLI JSON is the scaffold verification surface until production UI/API surfaces exist.
- Fixture signatures, attestations, SBOMs, and provenance are deterministic local evidence metadata, not real cryptographic verification.
- Mocked ConnectorHub requests are acceptable local proof only when they record zero real external HTTP calls and zero credential exposure.
- Installation is availability only; activation grants mission/workspace authority.

## Out Of Scope

- Production UI/browser pack registry, production API, public marketplace, real dependency/package installation, real signing infrastructure, real SBOM scanner, live ConnectorHub providers, real credentials, external network calls, tenant/security policy changes, new dependencies, and full 206-scenario completion.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-EXT-001 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, core ingest/search/bundle/brief and pack activation transcripts |
| CS-EXT-002 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, `pack playbook propose/approve` transcripts |
| CS-EXT-003 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, `pack show` Agent Pack detail |
| CS-EXT-004 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, Agent Pack component counts |
| CS-EXT-005 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, `pack list` trusted registry view |
| CS-EXT-006 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, install inactive plus connector request denial |
| CS-EXT-007 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, explicit activation grants |
| CS-EXT-008 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, org-admin shortcut policy activation |
| CS-EXT-009 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, certification card |
| CS-EXT-010 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, certification input-not-replacement evidence |
| CS-EXT-011 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, pinned install and no silent update evidence |
| CS-EXT-012 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, update dry-run diff/evaluation gate |
| CS-EXT-013 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, rollback transcript |
| CS-EXT-014 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, ConnectorHub-mediated request |
| CS-EXT-015 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, direct-provider pack quarantine |
| CS-EXT-016 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, emergency patch and behavior-change denial |
| CS-SEC-015 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, supply-chain trust checks |
| CS-SEC-016 | MUST_PASS | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, untrusted activation denial |
| CS-REG-014 | REGRESSION_GUARD | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, direct provider denial and ConnectorHub mediation |
| CS-REG-015 | REGRESSION_GUARD | PASS | `reports/scenario/full-extension-ecosystem-2026-06-10.json`, scoped playbook proposal approval |

## Human Required

No human-required item was introduced for this local batch. Production registry approval, public marketplace review, real signing key policy, real SBOM scanner selection, live ConnectorHub provider review, and UI/API review remain future human-required surfaces before production PASS.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify full-extension-ecosystem --json --output reports/scenario/full-extension-ecosystem-2026-06-10.json
# status: success
# scenario_set: full-extension-ecosystem
# summary.pass: 20
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_FULL_EXTENSION_ECOSYSTEM_ONLY
# extension_evidence.pack_id: pack_ops_recovery_agent
# extension_evidence.untrusted_activation_exit_code: 8
# extension_evidence.direct_provider_import_exit_code: 8
# extension_evidence.audit_event_count: 37
# negative_evidence: all integer counters are 0
```

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` now marks `CS-EXT-001` through `CS-EXT-016`, `CS-SEC-015`, `CS-SEC-016`, `CS-REG-014`, and `CS-REG-015` as `PASS`.

Current full matrix after this batch:

- `PASS`: 132
- `NOT_VERIFIED`: 74
- `FAIL`: 0
- `NOT_RUN`: 0

## Gaps And Risks

- Full 206-scenario PASS remains incomplete.
- Production UI/API surfaces remain missing; this batch uses CLI-native JSON as scaffold evidence.
- Signature, attestation, SBOM, and provenance checks are deterministic fixture checks, not real cryptographic verification.
- Future registry/provider work must preserve install-vs-activation separation, explicit grants, ConnectorHub credential custody, update approval, rollback, emergency patch review, prompt-injection defenses, and tamper-evident audit refs.
