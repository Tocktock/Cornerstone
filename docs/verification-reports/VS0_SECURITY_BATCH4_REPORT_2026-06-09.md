# VS-0 Security Batch 4 Report - 2026-06-09

Status: PASS for the first redaction and prompt-injection runtime slice only.
Scope: `CS-ARCH-006`, `CS-ARCH-007`, `CS-SEC-007`, `CS-SEC-008`, and `CS-REG-013`.

This report does not mark broad policy, access control, egress, connector, action, UI, API, or full audit lifecycle scenarios as complete.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-ARCH-006 | MUST_PASS | PASS | `reports/scenario/vs0-security-2026-06-09.json`, secret fixture artifact ingest transcript |
| CS-ARCH-007 | MUST_PASS | PASS | `reports/scenario/vs0-security-2026-06-09.json`, prompt-injection artifact ingest transcript |
| CS-SEC-007 | MUST_PASS | PASS | `reports/scenario/vs0-security-2026-06-09.json`, prompt-injection blocked-attempt metadata and audit verification |
| CS-SEC-008 | MUST_PASS | PASS | `reports/scenario/vs0-security-2026-06-09.json`, generated-output redaction and no-leak check |
| CS-REG-013 | REGRESSION_GUARD | PASS | `reports/scenario/vs0-security-2026-06-09.json`, authority-expanded negative evidence |

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-security --json --output reports/scenario/vs0-security-2026-06-09.json
# status: success
# scenario_set: vs0-security
# summary.pass: 5
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_SECURITY_ONLY
# negative_evidence.unredacted_secret_occurrences: 0
# negative_evidence.tool_calls_created: 0
# negative_evidence.action_cards_created_from_untrusted_artifact: 0
# negative_evidence.external_http_calls: 0
# negative_evidence.authority_expanded: 0
```

```sh
rg -n "sk-test-|ghp_" reports/scenario/vs0-security-2026-06-09.json
# no matches
```

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
# Ran 12 tests
# OK
```

## Evidence Summary

- Secret fixture ingest produces a redacted derived output and keeps raw original access marked as policy-controlled.
- The security report and CLI transcript do not contain fake secret prefixes.
- Prompt-injection fixture ingest keeps the artifact as untrusted evidence, records `unsafe_instruction_detected: true`, and emits a prompt-injection policy denial ref.
- Negative evidence records zero tool calls, zero action cards from untrusted artifacts, zero external HTTP calls, and no authority expansion.
- The scenario run verifies audit hash-chain integrity after artifact ingestion and unsafe-instruction detection events.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` marks only `CS-ARCH-006`, `CS-ARCH-007`, `CS-SEC-007`, `CS-SEC-008`, and `CS-REG-013` as `PASS` in this batch.

## Gaps

- Full 206-scenario product PASS remains incomplete.
- Egress-deny, access-control, broader policy-denial UX/API examples, connector credential boundaries, action dry-run/approval, and UI/API flows remain `NOT_VERIFIED`.
- This batch uses the local standard-library runtime and deterministic fixtures; it does not add production dependencies or live external checks.
