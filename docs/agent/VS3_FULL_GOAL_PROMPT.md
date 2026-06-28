# VS3 Full Goal Prompt

**Status:** Ready-to-use `/goal` prompt for VS3 continuation.
**Owner:** JiYong / Tars
**Date:** 2026-06-29 KST

## Source Inputs

- `docs/agent/PRIMARY_GOAL_INSTRUCTION_FOLLOWING.md`
- `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md`
- `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`

This prompt intentionally maps the full VS3 `MUST_PASS`, `REGRESSION`, and
`HUMAN_REQUIRED` inventory before narrowing execution to a delivery slice.

## Prompt

```markdown
/goal

Continue CornerStone VS3: On-Prem Security and Trusted Extension development, but operate in small verified slices.

Primary objective:
Build the VS3 on-prem security and trusted extension foundation from the frozen VS3 contract, preserving strict proof boundaries and never converting local/dev proof into production, live-provider, security-acceptance, migration-readiness, or human-acceptance claims.

Primary references to read before planning or implementation:
- docs/agent/PRIMARY_GOAL_INSTRUCTION_FOLLOWING.md
- AGENTS.md
- docs/sot/README.md
- docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md
- docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md
- docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md
- docs/scenario-contracts/SCENARIO_MATRIX_FULL.md
- docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md
- docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md
- docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md
- docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv
- docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md
- docs/agent/PROJECT_OPERATING_CONSTITUTION.md

Non-negotiable proof boundaries:
- Treat documentation, local deterministic checks, local browser/API observations, local integration rehearsals, staging/prod evidence, external-provider evidence, human review, PR state, and release state as separate proof surfaces.
- Do not claim VS3-P, production/on-prem readiness, security acceptance, real IdP readiness, live provider readiness, migration/restore readiness, or human UX acceptance unless the matching human/external proof exists.
- Every product feature or verification gate must expose a native `cornerstone ... --json` path where applicable. No CLI path, no feature PASS.
- AI-verifiable rows start as `NOT_RUN`. Human rows remain `HUMAN_REQUIRED` until signed or explicitly approved evidence exists.
- Do not start implementation until the full scenario mapping below and the current slice contract are explicit.

Full VS3 MUST_PASS mapping:

VS3-0 evidence reconciliation and contract freeze - first slice:
- VS3-GATE-001 (MUST_PASS, in_this_slice): Expected: one canonical current status exists; conflicting VS2 claims are rejected, superseded, or reconciled with exact report paths and hashes. Verify: run VS3 evidence reconciliation, inspect VS2 JSON summaries, and lint claim strings. Evidence: reconciliation JSON, report hashes, rejected/superseded artifact list, final product claim string. Pass/fail: PASS only if one canonical status remains and no report can claim broader readiness; FAIL if contradictory PASS/NOT_VERIFIED signals remain unclassified.
- VS3-GATE-002 (MUST_PASS, in_this_slice): Expected: Markdown and CSV exist, row counts match, and every row has expected behavior, verification, evidence, pass/fail criteria, owner, and initial status. Verify: run docs verifier and matrix structural checks. Evidence: `scripts/verify_sot_docs.sh`, row-count output, duplicate-ID check. Pass/fail: PASS only if matrix and contract are internally consistent; FAIL on duplicate IDs, missing criteria, or status-bearing contract claims.
- VS3-GATE-003 (MUST_PASS, in_this_slice): Expected: report distinguishes VS3-L local/dev assurance from VS3-P production/on-prem candidate and cannot overclaim real IdP, live provider, real network, migration, or human acceptance. Verify: static report lint over docs, reports, README, and UI copy touched by VS3. Evidence: overclaim lint report, negative evidence counters, reviewed allowlist if any. Pass/fail: PASS only if forbidden overclaim strings are absent or explicitly historical; FAIL if local proof is described as production/on-prem/live readiness.
- VS3-GATE-004 (MUST_PASS, in_this_slice): Expected: `cornerstone scenario verify vs3-onprem-trusted-extension --json` emits status, counts, per-row evidence, human rows, and gate metadata. Verify: execute dry-run/list/coverage path before full implementation and validate JSON schema. Evidence: CLI transcript, JSON schema validation, coverage matrix. Pass/fail: PASS only if CLI is native and status-neutral before implementation; FAIL if raw scripts replace CLI parity.

VS3-1 trusted RequestContext - later slice:
- VS3-CTX-001 (MUST_PASS, later_slice): Expected: all surfaces derive RequestContext from trusted identity/membership and produce matching context digests and policy outcomes. Verify: normalized fixture across all surfaces. Evidence: CLI/API/UI/worker/tool transcripts, context digest, policy decision ID, audit refs. Pass/fail: PASS only if no surface accepts caller-controlled scope; FAIL on digest mismatch or missing audit refs.
- VS3-CTX-002 (MUST_PASS, later_slice): Expected: forged fields are ignored or denied before protected DB access, egress, connector call, or tool execution. Verify: parameterized negative tests for each field and entry point. Evidence: denied responses, zero protected DB rows touched, zero egress, policy/audit refs. Pass/fail: PASS only if every forged-authority path is denied or neutralized; FAIL if any forged value influences authority.
- VS3-CTX-003 (MUST_PASS, later_slice): Expected: new requests deny within the documented revocation window, including cached sessions, workers, and tool runtimes. Verify: allow -> revoke -> retry across API, CLI, worker, and pack/tool path. Evidence: before/after decision revisions, cache invalidation record, denial audit, zero post-revocation side effects. Pass/fail: PASS only if stale allow cannot produce data access or side effects; FAIL on stale success beyond documented bound.
- VS3-CTX-004 (MUST_PASS, later_slice): Expected: protected operations fail closed with helpful redacted errors and no downstream DB/egress/tool/provider access. Verify: fault matrix over gateway, service, worker, CLI, and tool entry points. Evidence: stable exit/status codes, zero downstream counters, sanitized logs, audit refs. Pass/fail: PASS only if no protected boundary is reached; FAIL if any path falls back open.
- VS3-CTX-005 (MUST_PASS, later_slice): Expected: tenant membership alone is insufficient; mission/workspace policy controls memory, connector, model, tool, and action use. Verify: same-tenant personal/org/project fixture with allowed and denied operations. Evidence: policy decisions, zero implicit context use, allowed promotion provenance, audit refs. Pass/fail: PASS only if namespace/workspace/mission scope is enforced above tenant RLS; FAIL on implicit cross-context use.

VS3-2 Postgres RLS and scoped durable truth - later slice:
- VS3-RLS-001 (MUST_PASS, later_slice): Expected: required tenant, owner, namespace, workspace, classification, provenance, and audit fields exist and are non-null where required. Verify: schema inventory plus create/read/null-insert tests. Evidence: schema report, failed-null insert transcript, representative API/CLI payloads. Pass/fail: PASS only if every active truth-bearing table is scoped; FAIL on ownerless global truth or nullable required scope.
- VS3-RLS-002 (MUST_PASS, later_slice): Expected: application role can read only authorized tenant rows; tenant-B data and existence metadata remain hidden from tenant A. Verify: two-tenant DB integration matrix using real app role. Evidence: SQL transcript, pg_policies inventory, row counts, zero foreign canaries. Pass/fail: PASS only if SELECT/count/join/search paths hide foreign data; FAIL on row, count, ID, cursor, or timing/error leak.
- VS3-RLS-003 (MUST_PASS, later_slice): Expected: RLS/WITH CHECK/constraints deny or affect zero unauthorized rows without partial mutation. Verify: database mutation matrix with rollback assertions. Evidence: SQLSTATE or neutral API error, before/after snapshots, audit/anomaly event. Pass/fail: PASS only if unauthorized mutations have no effect; FAIL on partial cross-tenant write or revealing error.
- VS3-RLS-004 (MUST_PASS, later_slice): Expected: tenant context is transaction-local or otherwise reset; no request contaminates another request or report. Verify: pool stress test with alternating tenants and injected errors/timeouts. Evidence: connection IDs, tenant sequence, reset assertions, parallel report hash stability. Pass/fail: PASS only if repeated parallel tests show zero contamination; FAIL on leaked context or false PASS.
- VS3-RLS-005 (MUST_PASS, later_slice): Expected: known rows migrate deterministically; ambiguous rows quarantine or block; rollback is deterministic; no ownerless truth is created. Verify: forward migration, failed migration, quarantine, rollback, and restored-read tests. Evidence: migration report, counts, checksums, quarantine reasons, rollback transcript. Pass/fail: PASS only if data integrity and scope are preserved; FAIL on silent default tenant assignment or destructive migration.
- VS3-RLS-006 (MUST_PASS, later_slice): Expected: backup/restore preserves artifacts, evidence, claims, ontology, policy decisions, audit integrity, RLS policies, and tenant boundaries. Verify: local backup -> restore -> verify suite with two tenants. Evidence: backup manifest, restored hashes/counts, RLS inventory before/after, audit verify output. Pass/fail: PASS only if restored system reproduces evidence and audit safely; FAIL on missing rows, broken policies, or leaked tenant export.

VS3-3 OPA/Rego policy plane - later slice:
- VS3-OPA-001 (MUST_PASS, later_slice): Expected: policy input schema includes trusted subject, scope, resource, action, classification, mission authority, connector/tool capability, model policy, risk, data scope, and environment. Verify: contract tests and golden fixtures for every operation family. Evidence: schema file, valid/invalid fixture results, source-of-attribute map, input digest. Pass/fail: PASS only if unknown or caller-authoritative fields fail closed; FAIL on schema drift or unvalidated authority.
- VS3-OPA-002 (MUST_PASS, later_slice): Expected: PolicyDecision records action, reason codes, safe resolution, bundle revision/hash, decision ID, input digest, tenant/namespace, evidence refs, and audit refs. Verify: `opa test`, HTTP decision tests, golden JSON validation. Evidence: OPA JSON output, coverage, PolicyDecision fixtures, audit linkage, redaction proof. Pass/fail: PASS only if decisions are deterministic and default deny; FAIL on implicit allow or missing reason/audit refs.
- VS3-OPA-003 (MUST_PASS, later_slice): Expected: OPA is bound to intended interfaces, authenticated/authorized, and management APIs are not anonymously exposed. Verify: network/container tests from allowed and denied peers. Evidence: OPA config, listening sockets, authorized success, unauthorized denial, audit/status refs. Pass/fail: PASS only if unauthorized access is denied; FAIL on anonymous policy or data API exposure.
- VS3-OPA-004 (MUST_PASS, later_slice): Expected: activation is atomic; invalid bundles fail without replacing last known good; first-start failure fails closed and is visible. Verify: bundle lifecycle tests and rollback injection. Evidence: active revision before/after, invalid-bundle error, degraded readiness, policy decisions under failure. Pass/fail: PASS only if no invalid policy becomes active; FAIL on stale permissive allow or hidden degraded state.
- VS3-OPA-005 (MUST_PASS, later_slice): Expected: decision logging and audit mirror mask secrets/protected values while preserving correlation. Verify: secret-canary fixture through allow, deny, and malformed paths. Evidence: decision log sample, CornerStone audit refs, zero secret scanner findings. Pass/fail: PASS only if logs are useful and redacted; FAIL on raw secret or protected payload leak.

VS3-4 default-deny egress and runtime containment - later slice:
- VS3-EGR-001 (MUST_PASS, later_slice): Expected: runtime/network boundary blocks connection; forbidden sink records zero requests and zero bytes. Verify: container/process integration test with sink counters. Evidence: denied tool result, sink logs, network counters, policy/audit refs. Pass/fail: PASS only if no packet/request reaches forbidden sink; FAIL if app merely skips call without runtime proof.
- VS3-EGR-002 (MUST_PASS, later_slice): Expected: exactly one approved call reaches allowed sink; evidence, connector result, policy, approval, and audit are linked. Verify: end-to-end local controlled sink test. Evidence: one sink request, sanitized metadata, Action/Workflow/Connector result, decision/audit refs. Pass/fail: PASS only if exactly the declared call occurs; FAIL on missing call, duplicate call, or undeclared call.
- VS3-EGR-003 (MUST_PASS, later_slice): Expected: every normalized destination and redirect hop is re-authorized; denied addresses are never contacted. Verify: table-driven URL, redirect, fake-DNS, IPv4/IPv6 tests. Evidence: per-case decisions, DNS transcript, redirect hop decisions, sink logs, zero denied-address contact. Pass/fail: PASS only if all bypass variants deny safely; FAIL on broadened allowlist or header leak.
- VS3-EGR-004 (MUST_PASS, later_slice): Expected: undeclared protocols and host access are blocked by sandbox/capability boundary. Verify: adversarial sandbox suite. Evidence: zero unauthorized connections/processes/files/env reads, denied capability records, runtime audit. Pass/fail: PASS only if no undeclared access succeeds; FAIL on arbitrary host/shell access.
- VS3-EGR-005 (MUST_PASS, later_slice): Expected: protected capabilities fail closed; readiness degrades; no fallback direct connection or host access occurs. Verify: disable component and rerun allowed/denied cases. Evidence: degraded readiness, denied operations, zero sink/host counters, operator guidance. Pass/fail: PASS only if outage denies safely; FAIL on fallback open or misleading ready state.
- VS3-EGR-006 (MUST_PASS, later_slice): Expected: content remains untrusted evidence and cannot create egress grants, action approvals, policy changes, or tool execution. Verify: prompt-injection fixtures across artifact and connector paths. Evidence: zero tool/action/egress calls, blocked-attempt audit, untrusted label, evidence refs. Pass/fail: PASS only if malicious content produces no authority; FAIL on content-driven authority expansion.

VS3-5 ConnectorHub read-only source and trusted capture boundary - later slice:
- VS3-CON-001 (MUST_PASS, later_slice): Expected: CornerStone commits immutable Artifact and evidence metadata before ack; crash/retry cannot acknowledge uncommitted source truth. Verify: projection delivery crash/retry fixture. Evidence: Artifact ID/hash/provenance, delivery attempt log, ack-after-commit proof, audit refs. Pass/fail: PASS only if ack follows durable commit; FAIL on lost projection or ack-before-commit.
- VS3-CON-002 (MUST_PASS, later_slice): Expected: connector is read-only: zero write mappings, zero mutation commands, write attempts denied or quarantined. Verify: static mapping scan plus runtime write-denial tests. Evidence: capability manifest, denied write transcript, zero external mutation counters, audit refs. Pass/fail: PASS only if no write-capable path is exposed; FAIL on mutation mapping or direct write client.
- VS3-CON-003 (MUST_PASS, later_slice): Expected: credentials stay in ConnectorHub; Product outputs use credential references only and redact sensitive payload fields. Verify: secret canary scan across CLI, logs, screenshots, reports, audit, and durable state. Evidence: zero secret scanner findings, credential-ref-only payload, redacted sink metadata. Pass/fail: PASS only if raw secrets never appear; FAIL on raw credential exposure.
- VS3-CON-004 (MUST_PASS, later_slice): Expected: SourcePolicySnapshot is tenant/workspace scoped, auditable, revocable, and enforced on next delivery/capture. Verify: policy update -> delivery/capture -> revoke -> retry fixture. Evidence: snapshot diff, before/after delivery decisions, revoked denial, audit refs. Pass/fail: PASS only if source policy cannot bleed across scopes; FAIL on stale or cross-scope delivery.
- VS3-CON-005 (MUST_PASS, later_slice): Expected: capture is consented, bounded, pauseable, revocable, scope-visible, and summary-only where raw capture is disallowed. Verify: fixture runtime and browser/CLI checks with pause/revoke. Evidence: consent record, scope config, capture summary, pause/revoke transcript, zero disallowed raw output. Pass/fail: PASS only if capture obeys explicit scope and stop controls; FAIL on silent or unbounded capture.
- VS3-CON-006 (MUST_PASS, later_slice): Expected: retry/quarantine is idempotent, evidence-safe, and cannot create memory, policy, action, egress, or authority without product approval. Verify: fault injection and malicious payload fixture. Evidence: retry/quarantine record, zero unauthorized side effects, evidence refs, audit refs. Pass/fail: PASS only if failed/malicious delivery remains contained; FAIL on duplicate truth or side effect.

VS3-6 Tool SDK, signed registry, Agent Packs, activation, update, rollback - later slice:
- VS3-TOOL-001 (MUST_PASS, later_slice): Expected: package contains manifest, capabilities, file/network/env/model grants, ConnectorHub requirements, risk, version, evaluation rubric, signature, SBOM, and provenance metadata. Verify: build and manifest/schema verification. Evidence: package artifact, manifest JSON, signature, SBOM, schema validation, audit refs. Pass/fail: PASS only if required metadata exists and validates; FAIL on missing grants, signature, or SBOM.
- VS3-TOOL-002 (MUST_PASS, later_slice): Expected: registry accepts only trusted signed package metadata and rejects unsigned, tampered, stale, revoked, or unknown-source packages. Verify: registry positive/negative package tests. Evidence: acceptance record, rejection transcripts, signature verification output, registry audit. Pass/fail: PASS only if tampered/untrusted packages are rejected; FAIL on silent trust or unsigned install.
- VS3-TOOL-003 (MUST_PASS, later_slice): Expected: install makes package available but gives no mission/workspace authority, connector access, model route, egress, file/env grant, or action capability. Verify: installed-inactive execution attempts across surfaces. Evidence: install record, denied execution transcripts, zero side effects, audit refs. Pass/fail: PASS only if installed package cannot act; FAIL on install-as-activation.
- VS3-TOOL-004 (MUST_PASS, later_slice): Expected: activation grants are explicit, least-privilege, pinned, reversible, auditable, and tied to policy decision and RequestContext. Verify: activation dry-run, approval, execution, revoke test. Evidence: grant record, policy decision, audit refs, capability list, revocation transcript. Pass/fail: PASS only if only granted capabilities work; FAIL on ungranted capability or hidden activation.
- VS3-TOOL-005 (MUST_PASS, later_slice): Expected: sandbox denies undeclared access and records negative evidence without leaking secrets. Verify: runtime sandbox negative suite. Evidence: denied records, zero file/env/network/process access counters, audit refs, secret scan. Pass/fail: PASS only if every undeclared access is denied; FAIL on sandbox bypass.
- VS3-TOOL-006 (MUST_PASS, later_slice): Expected: update shows diff of role contract, capabilities, playbooks, model policy, connector requirements, risk, evaluation results, migration notes, and rollback path before activation. Verify: update dry-run and evaluation gate tests. Evidence: diff artifact, evaluation result, blocked/approved update transcript, audit refs. Pass/fail: PASS only if behavior-changing update cannot silently activate; FAIL on unreviewed update.
- VS3-TOOL-007 (MUST_PASS, later_slice): Expected: rollback returns to previous pinned version; emergency patch is policy-governed, audited, and does not expand authority without review. Verify: rollback and emergency-patch simulation. Evidence: previous/new version records, rollback transcript, policy decision, affected mission list. Pass/fail: PASS only if rollback is deterministic and auditable; FAIL on irreversible or authority-expanding patch.

VS3-7 observability, audit integrity, and human evidence packages - later slice:
- VS3-OBS-001 (MUST_PASS, later_slice): Expected: status distinguishes Postgres/RLS, OPA, egress, ConnectorHub, tool runtime, registry, audit, backup/restore, migration, and human gates. Verify: fault-injection status tests and CLI/API/UI comparison. Evidence: observe status JSON, UI/DOM snapshot, component fault results, audit refs. Pass/fail: PASS only if degraded components are visible and scoped; FAIL on misleading green state.
- VS3-OBS-002 (MUST_PASS, later_slice): Expected: audit records are append-only, tamper-evident, tenant-safe, queryable, and link evidence/policy/action refs. Verify: audit contract tests and tamper fixture. Evidence: event inventory, hash/checkpoint verification, tamper detection failure, query transcript. Pass/fail: PASS only if required event classes are covered; FAIL on missing or mutable critical event.
- VS3-OBS-003 (MUST_PASS, later_slice): Expected: package contains scope, why AI cannot verify, required human action, expected evidence, redaction rules, release impact, and blank approval/rejection record. Verify: generate and validate evidence packages. Evidence: seven package files, schema validation, redaction guidance, no PASS status. Pass/fail: PASS only if packages prepare evidence without marking H rows PASS; FAIL if generated package is treated as human evidence.

Full VS3 REGRESSION mapping:
- VS3-REG-001 (REGRESSION, final_gate): Expected: VS0 Artifact -> Search -> Evidence -> Claim -> Action -> Audit loop still passes on the same tree. Verify: rerun accepted VS0 gates. Evidence: scenario reports, command transcripts, zero weakened assertions. Pass/fail: PASS only if fresh VS0 reports pass; FAIL on stale report reuse or regression.
- VS3-REG-002 (REGRESSION, final_gate): Expected: VS1 suggestions remain draft until promotion; search/profile/claim/action/audit integration remains green under VS3 policy/RLS boundaries. Verify: rerun VS1 gate and cross-scope ontology tests. Evidence: VS1 scenario report, zero auto-promotions, policy/audit refs. Pass/fail: PASS only if fresh VS1 report passes; FAIL on auto-promotion or cross-scope leakage.
- VS3-REG-003 (REGRESSION, final_gate): Expected: authority remains governed by RequestContext, Mission/Agent/Pack contracts, ConnectorHub capability, and policy. Verify: red-team fixture suite. Evidence: zero authority changes, zero tool/action/egress calls, blocked-attempt audit. Pass/fail: PASS only if prompt cannot expand authority; FAIL on content-driven grant.
- VS3-REG-004 (REGRESSION, final_gate): Expected: audit and scenario coverage cannot silently drop; missing coverage fails gate before release claim. Verify: coverage and audit mutation tests. Evidence: failing omission fixture, passing corrected inventory, matrix coverage output. Pass/fail: PASS only if coverage gates detect omissions; FAIL on silent scenario disappearance.
- VS3-REG-005 (REGRESSION, final_gate): Expected: claims do not describe local/dev proof as production, real IdP, live provider, penetration-tested, human-accepted, or migration-ready. Verify: static overclaim lint and evidence manifest review. Evidence: zero overclaim findings, historical-claim allowlist if needed, human-required table. Pass/fail: PASS only if wording is no stronger than evidence; FAIL on unqualified readiness claim.
- VS3-REG-006 (REGRESSION, final_gate): Expected: Product still appears as one calm CornerStone workspace; connector/tool/policy admin detail is progressively disclosed in admin context. Verify: UI/nav review and browser/DOM check. Evidence: screenshots/DOM, nav map, absence of repo-name mental model. Pass/fail: PASS only if normal flow remains product-first; FAIL if first value is connector/tool admin setup.
- VS3-REG-007 (REGRESSION, final_gate): Expected: supply-chain change is justified, pinned, scanned, and approved where production dependency or security-sensitive gate applies. Verify: dependency diff, lockfile review, approval-gate check. Evidence: diff, version pins, security notes, H01 approval if required. Pass/fail: PASS only if no unapproved production dependency enters; FAIL on broad/churn dependency change.
- VS3-REG-008 (REGRESSION, final_gate): Expected: default egress deny, default policy deny, no arbitrary shell, RLS enforced, activation inactive, and high-risk actions require approval. Verify: fresh/reset/partial-config integration suite. Evidence: default config snapshot, denial suite, role/policy inventory, audit refs. Pass/fail: PASS only if conservative defaults hold everywhere; FAIL on permissive default.

Full VS3 HUMAN_REQUIRED mapping:
- VS3-H01 (HUMAN_REQUIRED): JiYong/Tars or authorized owner must approve architecture, dependency scope, migration/rollback plan, and security ownership. Required evidence: dated APPROVE/REJECT record with scope, exceptions, rollback owner, dependency decision. Blocks VS3-P and security-sensitive implementation gates as applicable.
- VS3-H02 (HUMAN_REQUIRED): independent reviewer must validate tenant, policy, connector, tool, extension, and egress bypass resistance. Required evidence: signed review, test scope/topology, findings, remediation, retest evidence. Blocks VS3-P.
- VS3-H03 (HUMAN_REQUIRED): real user/group/role/attribute/revocation mapping must be validated without promoting synthetic fixture proof. Required evidence: redacted login, mapping, revocation transcript, owner/security approval. Blocks VS3-P.
- VS3-H04 (HUMAN_REQUIRED): real topology must enforce deny-by-default and approved exceptions outside local harness. Required evidence: topology diagram, firewall/proxy/network-policy evidence, packet/log transcript, approval. Blocks VS3-P.
- VS3-H05 (HUMAN_REQUIRED): low-risk read-only or explicitly approved live provider rehearsal must succeed without secret exposure. Required evidence: redacted provider transcript, approval, ConnectorHub/CornerStone audit refs, result/evidence refs. Blocks live readiness claim.
- VS3-H06 (HUMAN_REQUIRED): operator must accept or reject whether boundaries, causes, risks, and recovery paths are understandable and not misleading. Required evidence: ACCEPT/REJECT note, screenshots/recording, task outcomes, issue list if rejected. Blocks UX acceptance claim.
- VS3-H07 (HUMAN_REQUIRED): migration, backup, restore, rollback, quarantine, RLS, policy, and audit drill must pass under approved data rules. Required evidence: signed transcript, before/after counts/hashes, restore/rollback result, RLS/policy inventory. Blocks migration/restore readiness claim.

Execution loop for every slice:
1. Freeze a short slice contract before coding: goal, scope, non-scope, full scenario mapping, scenarios selected for this slice, proof needed, human-required items, CLI parity, and done criteria.
2. Inspect the relevant product/docs/code context. Use senior lenses for product/domain, architecture/data contracts, security/reliability, observability/performance, verification, migration, and maintainability, but keep research bounded to the current slice.
3. Implement the smallest complete AI-verifiable solution for the selected slice only.
4. Refactor only where it directly improves current-slice correctness or prevents obvious follow-on risk.
5. Run targeted automated checks yourself, including native `cornerstone ... --json` checks where applicable.
6. Document evidence, remaining HUMAN_REQUIRED gates, and the decision before moving on.
7. Stop at a checkpoint before starting the next slice.

First delivery slice:
- Start with VS3-0 only: VS3-GATE-001, VS3-GATE-002, VS3-GATE-003, and VS3-GATE-004.
- Do not begin VS3-1 RequestContext implementation until the VS3-0 checkpoint exists.
- The VS3-0 checkpoint must prove canonical VS2/VS3 evidence status, internal contract/matrix consistency, overclaim prevention, and status-neutral native CLI coverage output.

Checkpoint format after each slice:
- What changed
- What passed
- What remains HUMAN_REQUIRED
- What was deliberately not done
- Evidence: commands, files, reports, browser/API observations, CLI transcripts, JSON outputs, audit refs
- Recommendation: continue / pause / ask human review / open PR

Final VS3-L local/dev assurance criteria:
- All 42 VS3 MUST_PASS rows have fresh AI-verifiable local/dev evidence where local proof is allowed.
- All 8 VS3 REGRESSION guards pass on the same tree without stale report reuse.
- All 7 human rows remain explicit and are not marked PASS unless matching signed evidence exists.
- `cornerstone scenario verify vs3-onprem-trusted-extension --json` reports status, counts, per-row evidence, human rows, and gate metadata.
- Documentation, reports, UI copy, and release notes do not overclaim VS3-P or live readiness.

Final VS3-P production/on-prem candidate criteria:
- VS3-L criteria are met.
- VS3-H01 through VS3-H07 have matching human/external evidence.
- Real IdP, real topology, live-provider, independent security review, operator UX/trust review, and migration/restore drill proof are present and scoped.
- No local-only report is used as production/on-prem readiness evidence.
```
