# CornerStone Local Verification Plane v0

**Status:** Planning and verification contract; documentation only.
**Owner:** JiYong / Tars.
**Scope:** Local scenario verification, fixture corpus, local model harness, deterministic validators, CLI-native evidence, and release gating.
**Non-goal:** This document does not implement runtime code, tests, model providers, fixtures, or CLI commands.

## 1. Purpose

CornerStone local verification must prove required scenarios with replayable evidence. A local corpus and a local LLM are useful, but they are not sufficient alone.

The verification plane exists to make this rule enforceable:

```text
Every required scenario
-> has a local verifier or explicit HUMAN_REQUIRED classification
-> has CLI/API/UI/policy evidence where applicable
-> emits machine-readable evidence
-> cannot be marked PASS without concrete output
```

The product-level entry point should be:

```bash
cornerstone scenario verify <contract> --json
```

Pytest, OPA tests, Playwright traces, model outputs, API responses, and database checks may run underneath, but the release-facing proof surface is the CornerStone scenario report.

## 2. Authority and relationship to existing contracts

This document is subordinate to:

1. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
2. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
3. `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`
4. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` for VS-0 work
5. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
6. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`

If this document conflicts with the product goal, scenario standard, CLI-native-first contract, or a frozen task scenario contract, the higher-priority document wins.

## 3. Core principle

A local LLM may generate a brief, classify uncertainty, or draft a claim. It must never be the judge of scenario `PASS`.

`PASS` must come from deterministic validators over durable product evidence:

- artifact records and hashes;
- original storage references;
- derived representations and extraction statuses;
- search snapshots;
- evidence bundles;
- brief and claim records;
- trust-state transitions;
- policy decisions;
- workflow/action records;
- audit events and audit integrity checks;
- CLI transcripts;
- UI traces where relevant;
- scenario reports.

## 4. Local Verification Plane architecture

```text
Scenario Contract
      ↓
Scenario Registry
      ↓
Verification Corpus
      ↓
Scenario Runner
      ↓
CLI / API / Worker / UI / Policy execution
      ↓
Evidence Collectors
      ↓
Deterministic Validators
      ↓
Scenario Report
      ↓
Release Gate
```

### 4.1 Scenario Registry

The registry maps each scenario ID to:

- scenario type: `MUST_PASS`, `REGRESSION_GUARD`, `HUMAN_REQUIRED`, or declared `OUT_OF_SCOPE`;
- verification class;
- required command path;
- required fixtures;
- required validators;
- required evidence artifacts;
- expected status rules;
- owner: `AI`, `Human`, or `AI+Human`.

The registry must fail coverage checks if any required scenario is missing.

### 4.2 Verification Corpus

The corpus is a versioned, local, content-addressed set of fixture packs. It must test happy paths and failure paths.

Suggested layout:

```text
fixtures/
  vs0/
    manifest.yaml
    packs/
      01_artifact_basic/
      02_dedup_versioning/
      03_unknown_and_failed_extraction/
      04_search_exact_and_semantic/
      05_evidence_bundle/
      06_brief_claim_trust_ladder/
      07_action_dry_run_policy_approval/
      08_namespace_isolation/
      09_redaction_secrets/
      10_prompt_injection/
      11_audit_integrity/
      12_contradiction_and_uncertainty/
      13_ui_onboarding/
```

Each pack should declare:

- fixture ID;
- scenario IDs covered;
- input files;
- expected outputs;
- forbidden outputs;
- expected CLI commands;
- expected exit codes;
- required evidence refs;
- required audit refs;
- required policy decisions;
- negative evidence requirements.

Example:

```yaml
id: pack_10_prompt_injection
scenario_ids:
  - CS-ARCH-007
  - CS-SEC-007
classes:
  - security
  - policy
inputs:
  - path: corpus/security/prompt_injection_tool_call.md
expected:
  artifact_tagged_untrusted: true
  tool_calls_created: 0
  action_cards_created: 0
  external_calls_created: 0
  required_policy_decisions:
    - prompt_injection_blocked
  required_audit_events:
    - artifact.ingested
    - unsafe_instruction.detected
```

## 5. Scenario verification classes

Use these classes before implementation:

| Class | Meaning | Local verification shape |
|---|---|---|
| `D` | Deterministic local | CLI/API/db/object-store assertions |
| `M` | Model-assisted local | Model output plus deterministic validators |
| `S` | Security/adversarial local | malicious fixtures plus negative evidence |
| `P` | Policy local | OPA/service policy tests plus product records |
| `U` | UI local | Playwright or browser trace plus product assertions |
| `E` | External provider required | fixture provider locally; real provider is separate |
| `H` | Human judgment required | explicit required human action and evidence |

Example mapping:

| Scenario area | Class | Rule |
|---|---|---|
| Artifact preservation | `D` | Verify artifact record, hash, storage ref, audit event. |
| Search snapshot as evidence | `D` | Verify query, filters, results, and evidence bundle refs. |
| Brief generation | `M+D` | Model drafts; validators check schema, evidence refs, gaps, redaction. |
| Claim approval without evidence | `D+P` | CLI/API must fail with evidence/trust-state violation. |
| Prompt injection | `S+P` | Verify no tool/action/egress and record blocked attempt where relevant. |
| Egress denied | `S+P` | Verify policy denial and audit/policy event. |
| Action dry-run required | `D+P` | Verify dry-run record before execution and approval need where applicable. |
| Audit tamper detection | `D` | Mutate controlled local audit data and verify detection. |
| Onboarding quality | `U+H` | UI path can be automated; subjective quality remains human-required. |
| Live connector E2E | `E+H` | Fixture provider locally; real credentials/provider checks separately. |

## 6. Model harness

The model harness must support at least three provider modes.

### 6.1 `local_test` provider

Required baseline for local/CI gates.

- deterministic;
- no external credentials;
- fixed schema-valid responses for known fixture prompts;
- proves product workflow independent of model randomness.

### 6.2 `ollama` provider

Optional but recommended for local semantic smoke tests.

Rules:

- use pinned model tags and, where available, a recorded digest;
- use low temperature for repeatability;
- request structured JSON output where supported;
- never allow the model to mark scenarios `PASS`;
- validate outputs with deterministic validators;
- if Ollama is unavailable, core deterministic verification must still run with `local_test`.

### 6.3 `external_optional` provider

Reserved for later comparison or release-candidate quality review.

VS-0 local verification must not require live external model credentials.

## 7. Deterministic validators

The verification plane should define reusable validators. Initial validator families:

```text
ArtifactPreservedValidator
ContentHashValidator
DedupVersioningValidator
DerivedFailureDoesNotLoseOriginalValidator
SearchSnapshotReproducibleValidator
EvidenceBundleResolvableValidator
BriefEvidenceRefValidator
UnsupportedAssertionLabelValidator
ClaimEvidenceRequiredValidator
TrustStateTransitionValidator
ActionDryRunRequiredValidator
PolicyDecisionValidator
ApprovalRequiredValidator
WorkflowOnlyMutationValidator
AuditEventRequiredValidator
AuditHashChainValidator
NamespaceIsolationValidator
SecretRedactionValidator
PromptInjectionNoToolCallValidator
EgressDeniedValidator
CLIJsonSchemaValidator
CLIExitCodeValidator
```

Model-output validators must avoid brittle golden-string comparison. Prefer structural and evidence checks:

- JSON schema is valid;
- every factual claim has an evidence ref;
- every evidence ref resolves to an artifact, chunk, search snapshot, policy decision, tool result, workflow result, or audit record;
- unsupported statements are labeled `Assumption`, `Hypothesis`, or `Insufficient Evidence`;
- contradictions and gaps from fixture metadata appear in the brief;
- forbidden secret patterns do not appear in generated output, logs, CLI stdout/stderr, screenshots, or reports;
- untrusted document instructions were not followed.

## 8. CLI-native evidence requirements

Every local product-feature scenario must run through a native CLI path unless explicitly classified as non-feature internal, `HUMAN_REQUIRED`, or `OUT_OF_SCOPE` before coding.

A CLI transcript should include:

```json
{
  "schema_version": "cs.cli_transcript.v0",
  "scenario_id": "CS-ARCH-001",
  "command": ["cornerstone", "artifact", "ingest", "fixtures/vs0/...", "--json"],
  "exit_code": 0,
  "stdout_json": {
    "schema_version": "cs.cli.v0",
    "status": "success",
    "tenant_id": "...",
    "owner_id": "...",
    "namespace_id": "...",
    "workspace_id": "...",
    "ids": {},
    "evidence_refs": [],
    "audit_refs": []
  },
  "stderr_redacted": "",
  "started_at": "...",
  "ended_at": "..."
}
```

The CLI gate must fail when:

- `--json` output is not valid JSON;
- `schema_version` is missing;
- tenant/owner/namespace/workspace scope is missing where relevant;
- evidence refs are missing where required;
- audit refs are missing for state-changing or security-relevant behavior;
- exit code does not match the documented contract;
- secrets appear in stdout, stderr, logs, reports, or transcripts;
- mutation commands lack dry-run support or required approval behavior.

## 9. Negative evidence for safety scenarios

Security scenarios often require proving that something did not happen. Scenario reports must support negative evidence fields such as:

```json
{
  "negative_evidence": {
    "tool_calls_created": 0,
    "action_cards_created_from_untrusted_artifact": 0,
    "external_http_calls": 0,
    "unredacted_secret_occurrences": 0,
    "cross_namespace_results": 0
  }
}
```

Negative evidence is required for prompt injection, egress-deny, secret-redaction, namespace-isolation, and direct-mutation-denial scenarios.

## 10. Traceability

Each scenario run should have:

- `scenario_run_id`;
- `trace_id`;
- `scenario_id`;
- `corpus_pack_id`;
- `model_provider` and `model_name` where relevant;
- tenant, owner, namespace, and workspace scope;
- audit refs;
- policy decision refs;
- evidence refs;
- transcript paths.

This enables a single chain:

```text
scenario -> CLI transcript -> API request -> worker job -> model run -> policy decision -> workflow/action -> audit event -> scenario report
```

## 11. Local verification commands

Planned commands for future implementation:

```bash
make verify-local-fast
make verify-local-llm
make verify-local-full
```

`make verify-local-fast` should run deterministic checks:

```bash
scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh
docker compose config

cornerstone scenario verify vs0   --corpus fixtures/vs0   --model-provider local_test   --json   --output reports/scenario/vs0-local-test.json

opa test policies/opa tests/policy --format=json --fail-on-empty   > reports/policy/opa-test.json

pytest tests/scenario   --junitxml=reports/pytest/scenario.xml
```

`make verify-local-llm` should run pinned local semantic smoke tests:

```bash
cornerstone scenario verify vs0-llm   --corpus fixtures/vs0   --model-provider ollama   --model <pinned-model-tag-or-digest>   --temperature 0   --json   --output reports/scenario/vs0-ollama.json
```

`make verify-local-full` should merge evidence and gate release readiness:

```bash
make verify-local-fast
make verify-local-llm
pnpm --filter web test:e2e
cornerstone scenario report merge reports/**/* --output reports/scenario/final-local-verification.json
cornerstone scenario gate reports/scenario/final-local-verification.json
```

These commands are planned verification targets. They must not be reported as `PASS` until implemented and run.

## 12. Scenario report status rules

Allowed statuses:

| Status | Meaning | Gate behavior |
|---|---|---|
| `PASS` | Verified with concrete evidence. | Does not block. |
| `FAIL` | Checked and expected result was not met. | Blocks. |
| `NOT_VERIFIED` | Evidence is insufficient. | Blocks AI-verifiable done. |
| `NOT_RUN` | Planned check was not executed. | Blocks AI-verifiable done. |
| `HUMAN_REQUIRED` | Truly needs human, unavailable external access, production access, credentials, or subjective judgment. | Excluded from AI gate but must be reported. |
| `OUT_OF_SCOPE` | Explicitly excluded before coding. | Does not block only if declared before coding. |

Gate rules:

```text
AI-verifiable MUST_PASS + REGRESSION_GUARD scenarios require PASS.
HUMAN_REQUIRED requires reason, required human action, expected evidence, and release impact.
OUT_OF_SCOPE must be declared before implementation.
NOT_RUN, NOT_VERIFIED, and FAIL block AI-verifiable completion.
```

## 13. Scenario coverage matrix

The verification plane should include a machine-readable matrix, for example:

```text
docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv
```

Suggested columns:

```csv
scenario_id,type,local_required,verification_class,verification_owner,verification_command,evidence_artifact,human_required_reason,status
```

Coverage checks must fail when:

- a required scenario is missing from the matrix;
- an AI-verifiable scenario has no verification command;
- a required command did not run;
- a `PASS` row lacks evidence;
- a `HUMAN_REQUIRED` row lacks reason, required human action, expected evidence, or release impact.

## 14. UI evidence

CLI-native is mandatory, but some product scenarios require UI observation. Use UI evidence only where the scenario requires product-surface proof:

- one coherent CornerStone product experience;
- onboarding guide;
- active workspace visibility;
- claim-to-evidence path;
- Action Card clarity;
- policy-denial cause and resolution guide.

Suggested evidence artifacts:

```text
reports/ui/<scenario_id>/screenshot.png
reports/ui/<scenario_id>/trace.zip
reports/ui/<scenario_id>/assertions.json
```

Subjective UX quality remains `HUMAN_REQUIRED` unless a frozen scenario defines objective UI assertions.

## 15. Fault injection and mutation checks

Local verification must include controlled failures:

| Fault | Expected behavior |
|---|---|
| Extraction throws | Original remains preserved; derived status is partial/failed. |
| Embedding provider unavailable | Keyword search still works; model gap is logged. |
| Ollama unavailable | `local_test` verification still runs; Ollama scenario is optional or `NOT_RUN`. |
| Policy denies action | No execution; helpful denial and policy/audit record exist. |
| Egress attempt | Blocked and recorded. |
| Audit record tampered | Audit verification fails. |
| Duplicate artifact upload | Same content identity or explicit duplicate link. |
| Changed artifact upload | New version or lineage link. |
| Prompt injection asks to call tool | No tool call or action is created. |
| Secret appears in input | Original is controlled; generated outputs/logs are redacted. |

Security mutation checks should intentionally weaken policies or validators in controlled test mode and verify that scenario gates fail, for example:

- allow egress by default;
- allow claim approval without evidence;
- allow cross-namespace reads;
- allow action execution without dry-run;
- disable redaction;
- disable audit writes.

## 16. Initial VS-0 local gates

Before any VS-0 release claim, require these local gates:

1. Scenario coverage: every VS-0 row mapped.
2. Deterministic provider: VS-0 passes with `local_test` provider.
3. Local LLM smoke: model-dependent cases pass with pinned Ollama/local model where configured.
4. Security fixtures: prompt injection, secret redaction, egress deny, namespace isolation pass.
5. Policy tests: OPA/service policy tests pass and are non-empty.
6. CLI parity: product-feature scenarios have `cornerstone ... --json` transcripts.
7. Evidence/audit integrity: every `PASS` has evidence refs and audit refs where relevant.
8. Human-required clarity: external, production, or subjective checks are explicitly marked with required evidence.

## 17. Implementation order for the verification plane

Do not implement this entire plane at once. Use these slices:

1. Document scenario report schema and coverage matrix format.
2. Add `cornerstone scenario list --json` and `cornerstone scenario coverage --json`.
3. Add deterministic `local_test` provider contract.
4. Add first corpus packs for artifact, redaction, prompt injection, namespace, action, and audit.
5. Add validators for artifact preservation, CLI JSON, evidence refs, and audit refs.
6. Add `cornerstone scenario verify vs0 --json` for a narrow first subset.
7. Add report merge and gate commands.
8. Add Ollama semantic smoke path.
9. Add UI trace evidence where required.
10. Add mutation checks and dashboard.

Each slice needs its own frozen scenario contract and CLI parity section before implementation.

## 18. Non-negotiables

- Local verification must not require live external credentials.
- Local verification must not require paid model providers.
- LLM output alone never proves `PASS`.
- Raw SQL, curl-only scripts, or ad-hoc Python scripts do not satisfy CLI-native parity.
- Safety checks need positive evidence and negative evidence.
- Scenario reports are durable evidence and must redact secrets.
- A scenario cannot disappear from the matrix to make a gate pass.

## 19. Required final report additions for verification-plane work

Any task that changes this verification plane must report:

```markdown
Local Verification Plane:
- Scenario registry changes:
- Corpus packs changed:
- Validators changed:
- Model providers changed:
- CLI transcript schema changes:
- Evidence artifacts generated:
- Scenario report/gate result:
- Human-required rows:
```

## 20. Verdict rule

The local verification plane is not complete until:

```text
cornerstone scenario coverage <contract> --json
cornerstone scenario verify <contract> --json
cornerstone scenario gate <report> --json
```

exist, are CLI-native, emit evidence/audit refs where relevant, and reject incomplete or unevidenced `PASS` claims.
