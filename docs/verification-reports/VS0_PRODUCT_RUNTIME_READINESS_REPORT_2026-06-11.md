# VS0 Product Runtime Readiness Implementation Report - 2026-06-11

## Summary

- Verdict: local deterministic VS0 product runtime is implemented and AI-verifiable rows pass.
- Scenario set: `vs0-product-runtime`
- Scenario result: 12 PASS, 2 HUMAN_REQUIRED, 0 blocking AI-verifiable rows.
- Runtime claim: local VS0 runtime ready.
- Production claim: not ready; `production_release_ready=false` and live-provider/usability items remain human-required.
- Primary machine evidence: `reports/scenario/vs0-product-runtime-2026-06-11.json`

## Goal Implemented

The local user can complete this VS0 loop through shared CLI/API/UI runtime paths:

```text
Artifact ingest
-> searchable derived representation
-> reproducible search snapshot
-> Evidence Bundle
-> Draft/Evidence-backed Claim
-> Action Card dry-run
-> approval
-> local/mock ConnectorHub-style execution
-> audit timeline and tamper verification
```

## Scenario Table

| ID | Type | Owner | Status | Evidence |
|---|---|---:|---|---|
| VS0-RT-001 | MUST_PASS | AI | PASS | `cornerstone ready --json`; `GET /health`; `GET /ready`; `GET /` |
| VS0-RT-002 | MUST_PASS | AI | PASS | `cornerstone artifact ingest`; `cornerstone artifact show`; `POST /artifacts`; `GET /artifacts/{id}` |
| VS0-RT-003 | MUST_PASS | AI | PASS | `cornerstone search query`; `cornerstone search snapshot show`; `POST /search` |
| VS0-RT-004 | MUST_PASS | AI | PASS | `cornerstone evidence bundle create`; `cornerstone claim create`; `cornerstone claim approve` |
| VS0-RT-005 | MUST_PASS | AI | PASS | `cornerstone action propose`; `cornerstone action dry-run`; `POST /actions/{id}/dry-run` |
| VS0-RT-006 | MUST_PASS | AI | PASS | `cornerstone action approve`; `cornerstone action execute`; `POST /actions/{id}/execute` |
| VS0-RT-007 | MUST_PASS | AI | PASS | `cornerstone audit list`; `cornerstone audit verify`; `GET /audit-events`; `POST /audit/verify` |
| VS0-RT-008 | MUST_PASS | AI | PASS | UI summary for Home/Ops Inbox, Artifact Viewer, Search, Claim Builder, Action Card, Audit Detail |
| VS0-RT-R01 | REGRESSION_GUARD | AI | PASS | Prompt-injection fixture negative evidence counters |
| VS0-RT-R02 | REGRESSION_GUARD | AI | PASS | Cross-namespace access denial with policy/audit refs |
| VS0-RT-R03 | REGRESSION_GUARD | AI | PASS | Zero-evidence claim approval denial |
| VS0-RT-R04 | REGRESSION_GUARD | AI | PASS | readiness JSON keeps `production_release_ready=false` and `human_required=true` |
| VS0-RT-H01 | HUMAN_REQUIRED | Human | HUMAN_REQUIRED | Live ConnectorHub/provider verification later with redacted evidence |
| VS0-RT-H02 | HUMAN_REQUIRED | Human | HUMAN_REQUIRED | Human usability acceptance later |

## Runtime Surfaces

### CLI

Implemented or verified CLI paths:

- `cornerstone ready --json`
- `cornerstone runtime serve --host 127.0.0.1 --port 8787`
- `cornerstone artifact ingest <path> --json`
- `cornerstone artifact show <artifact_id> --json`
- `cornerstone search query "<query>" --json`
- `cornerstone search snapshot show <search_snapshot_id> --json`
- `cornerstone evidence bundle create --search-snapshot-id <id> --json`
- `cornerstone evidence bundle show <evidence_bundle_id> --json`
- `cornerstone claim create --evidence-bundle-id <id> --json`
- `cornerstone claim approve <claim_id> --json`
- `cornerstone mission create --claim-id <claim_id> --json`
- `cornerstone mission activate <mission_id> --mode autopilot --json`
- `cornerstone action propose --mission-id <mission_id> --claim-id <claim_id> --json`
- `cornerstone action show <action_id> --json`
- `cornerstone action dry-run <action_id> --json`
- `cornerstone action approve <action_id> --json`
- `cornerstone action execute <action_id> --json`
- `cornerstone audit list --json`
- `cornerstone audit export --json`
- `cornerstone audit verify --json`
- `cornerstone scenario verify vs0-product-runtime --json`

### API

Implemented stdlib local API routes:

- `GET /health`
- `GET /ready`
- `GET /`
- `POST /artifacts`
- `GET /artifacts/{artifact_id}`
- `POST /search`
- `GET /search-snapshots/{snapshot_id}`
- `POST /evidence-bundles`
- `GET /evidence-bundles/{evidence_bundle_id}`
- `POST /claims`
- `GET /claims/{claim_id}`
- `POST /claims/{claim_id}/approve`
- `POST /actions`
- `GET /actions/{action_id}`
- `POST /actions/{action_id}/dry-run`
- `POST /actions/{action_id}/approve`
- `POST /actions/{action_id}/execute`
- `GET /audit-events`
- `POST /audit/verify`

### UI

Implemented minimal Calm Surface UI at `/` with these asserted surfaces:

- Home/Ops Inbox
- Artifact Viewer
- Search
- Claim Builder
- Action Card
- Audit Detail

The UI also exposes:

- `local_scenario_ready=true`
- `vs0_runtime_ready=true`
- `production_release_ready=false`
- `external_calls=0`

## Safety / Negative Evidence

The verifier records zero-valued negative evidence for:

- `tool_calls_from_untrusted_artifact=0`
- `action_cards_from_untrusted_artifact=0`
- `egress_from_untrusted_artifact=0`
- `authority_expanded_from_untrusted_artifact=0`
- `cross_namespace_read_allowed=0`
- `zero_evidence_claim_approved=0`
- `production_release_overclaimed=0`
- `real_external_http_calls=0`
- `connector_credentials_exposed=0`

The mocked ConnectorHub-style action records `mock_connector_calls=1` and `external_http_calls=0`.

## Research Rationale

Implementation intentionally avoids new dependencies and agent-framework expansion. The chosen approach is a small stdlib API/UI facade over the existing `LocalRuntimeStore`.

Research inputs considered:

- OWASP Top 10 for LLM Applications: prompt injection and excessive agency require explicit tool/action boundaries.
- UK NCSC prompt-injection guidance: LLMs do not enforce a security boundary between instructions and data, so impact reduction and control-plane integrity matter.
- CaMeL: control/data-flow separation and capability boundaries are safer than trusting model text to drive actions.
- InjecAgent: tool-integrated agents need explicit indirect prompt-injection regression tests.
- W3C PROV: provenance records help assess quality, reliability, and trustworthiness.
- OPA decision logs: policy decisions should be auditable and replayable.

Decision:

- Use existing deterministic store and validators.
- Keep untrusted artifact text as evidence, never authority.
- Keep local/mock ConnectorHub execution inside governed Action Card path.
- Record policy, approval, execution, evidence, and audit refs in every product path.
- Avoid FastAPI/React/LangChain/LlamaIndex or MCP runtime dependencies for this gate because they add supply-chain and approval scope without improving the frozen scenario pass condition.

## Commands / Logs

Focused checks:

```sh
python3 -m py_compile packages/cornerstone_cli/product_runtime.py packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py
PATH="$PWD:$PATH" cornerstone ready --json
PATH="$PWD:$PATH" cornerstone scenario verify vs0-product-runtime --json --output tmp/vs0-product-runtime-first.json
```

Final checks:

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-product-runtime --json --output reports/scenario/vs0-product-runtime-2026-06-11.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-product-runtime-2026-06-11.json --json
make verify-vs0-runtime
make verify-local-fast
scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh
scripts/verify_local_verification_plane_docs.sh
scripts/verify_design_system_docs.sh
scripts/verify_vs0_scaffold_readiness_docs.sh
python3 scripts/verify_scenario_matrix.py
python3 -m unittest tests.scenario.test_scaffold_cli
git diff --check
```

Observed final results:

- `cornerstone scenario verify vs0-product-runtime --json`: `status=success`, `blocking=0`, `pass=12`, `human_required=2`.
- `cornerstone scenario gate reports/scenario/vs0-product-runtime-2026-06-11.json --json`: `status=success`, `blocking_count=0`, `scenario_count=14`.
- `make verify-vs0-runtime`: exit 0; regenerated report, gate passed, 42 unit tests `OK`.
- `make verify-local-fast`: exit 0; docs, matrix, scaffold CLI, existing scenario slices, and 42 unit tests `OK`.
- `python3 -m py_compile packages/cornerstone_cli/product_runtime.py packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py`: exit 0.
- `git diff --check`: exit 0.
- Real-browser Playwright proof: NOT_RUN. The Playwright CLI was available, but the local browser binary was missing and requested `playwright-cli install-browser chrome-for-testing`. Browser binary installation was not performed because it would require an external download. The scenario verifier still captured deterministic UI HTTP assertions for the required surfaces.

## Changed Files

Runtime and verifier:

- `packages/cornerstone_cli/product_runtime.py`
- `packages/cornerstone_cli/main.py`
- `packages/cornerstone_cli/scenarios.py`
- `tests/scenario/test_scaffold_cli.py`
- `Makefile`

Documentation and evidence:

- `README.md`
- `docs/sot/README.md`
- `docs/sot/sot_manifest.yaml`
- `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md`
- `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_MATRIX.csv`
- `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_REPORT_2026-06-11.md`
- `reports/scenario/vs0-product-runtime-2026-06-11.json`

Historical scenario-freeze docs preserved:

- `docs/verification-reports/VS0_PRODUCT_RUNTIME_READINESS_SCENARIO_FREEZE_REPORT_2026-06-11.md`

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-RT-H01 | Live connector/provider verification requires credentials and may mutate third-party state. | Run approved live-provider dry-run/execution with fake-safe or approved data, then redact evidence. | Redacted provider transcript, policy approval, execution result, and audit refs. | Blocks production release claim; does not block local VS0 runtime readiness. |
| VS0-RT-H02 | Usability acceptance is subjective. | Review the VS0 UI/API/CLI loop and record accept/reject with screenshots or recording. | Human acceptance note, screenshots or recording, and follow-up issue list if rejected. | Blocks human acceptance; does not block deterministic local scenario readiness. |

## Gaps / Risks

- Live Gmail/Slack/Notion/GitHub or ConnectorHub provider writeback is not implemented or verified.
- Production tenant/security proof is not implemented or verified.
- Human usability acceptance is not complete.
- The local API/UI is intentionally minimal and stdlib-only; it is a scenario gate runtime, not a production web architecture.
- Existing full scenario verification remains deterministic/local; it does not replace production CI, deployment, or operator acceptance.

## Verdict

- AI-verifiable local runtime: PASS.
- Production release: blocked by human-required/live-provider/production proof.
- Human-only items: listed with required action and expected evidence.
