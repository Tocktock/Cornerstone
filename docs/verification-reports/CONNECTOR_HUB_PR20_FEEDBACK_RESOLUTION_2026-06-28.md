# ConnectorHub PR20 Feedback Resolution

**Status:** Current response record for PR #20 review feedback.
**PR:** `codex/connectorhub-vs2-production-like` into `main`.
**Boundary:** Local ConnectorHub fixture/current local VS2 topology evidence only.
**Verdict:** Keep PR #20 draft. Do not treat this branch as merge-ready or production-ready.

## Why This Report Exists

The review correctly found that PR #20 was too broad to merge safely and that several proof surfaces were hard to review: contract status wording, VS2 supersession language, local path portability, generated evidence size, and workflow-dependent confidence. This report records the current response so reviewers can separate:

- fixed local proof-boundary issues;
- local verification commands replacing workflow-dependent checks for now;
- follow-up split/refactor/security work that still blocks merge confidence.

## Feedback Resolution Matrix

| Finding | Current resolution | Evidence | Merge impact |
| --- | --- | --- | --- |
| B1: draft/no visible CI | PR remains draft. No GitHub Actions workflow is added. `scripts/verify_connectorhub_local_evidence.sh` is the local review gate for non-Docker checks; strict mode adds the local Docker VS2 rehearsal. | `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md`; PR body; `scripts/verify_connectorhub_local_evidence.sh` | Still blocks merge confidence until maintainer accepts local logs or a future CI plan. |
| B2: contract mixed status/PASS | Contract status is separated from implementation status; README also points to reports instead of carrying ConnectorHub row counts as authority. | `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md`; `README.md`; `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json`; `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md` | Fixed for current local review. |
| B3: VS2 status inconsistent | Current VS2 status is delegated to `VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`; README no longer repeats generated counts. | `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`; `README.md` | Fixed for status authority; production/human gates remain outside this PR. |
| B4: non-portable evidence paths | Source documents are repo-archived, compact reports have `path_portability`, and the delivery-unit manifest marks `tmp/scenario/...` as regenerable local transcript refs. | `docs/archive/research/`; `reports/scenario/connector-contract-adapter/`; `reports/scenario/connectorhub-scenario-delivery-unit-manifest-2026-06-24.json`; `scripts/verify_connectorhub_engineering_trail.py` | Fixed for committed ConnectorHub review artifacts. |
| B5: PR too large | ADR-0008 defines the split sequence and target package/test module map. The split-readiness manifest makes the proposed slices locally verifiable. PR #20 stays a draft evidence branch. | `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md`; `docs/verification-reports/CONNECTOR_HUB_REVIEW_SPLIT_MANIFEST_2026-06-28.json`; `scripts/verify_connectorhub_review_split.py`; `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md` | Still blocks merge as one broad PR unless maintainer explicitly chooses a scoped slice. |
| M1: `connector.py`/test monolith | ADR-0008 records the target module/test map and the split-readiness verifier guards that map. This branch does not claim the split is complete. | `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md`; `docs/verification-reports/CONNECTOR_HUB_REVIEW_SPLIT_MANIFEST_2026-06-28.json`; `scripts/verify_connectorhub_review_split.py` | Follow-up implementation before merge or split PRs. |
| M2: Action lane boundary | `CS-CH-031` wording is local fixture Action and fixture outcome only; live side-effecting connector Actions remain out of scope. | `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md`; `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md` | Fixed for current local review. |
| M3: weak simulated egress checks | Strict local VS2 rehearsal now adds controlled `provider-sink` and `forbidden-sink` containers. `egress-gateway` records an OPA-mediated allowed provider request and a policy-denied forbidden request with Docker log evidence. | `compose.vs2.yml`; `packages/cornerstone_cli/vs2_production_like.py`; `reports/security/vs2-production-like-integration-2026-06-27.json` | Fixed for local/manual Docker rehearsal; still no production egress enforcement claim. |
| M4: milestone wording | PR/body/docs use ConnectorHub adoption overlay wording, not milestone-complete wording. | PR body; `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md`; `README.md` | Fixed for wording boundary. |
| M5: generated evidence hard to review | Reviewer guide plus local verifier define how to validate generated trail without reading every large report. The 41 source reports are represented by compact envelopes, a shared evidence index, and section-level SHA-256-addressed JSON objects. | `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md`; `reports/scenario/connector-contract-adapter/shared-evidence-index-2026-06-23.json`; `scripts/verify_connectorhub_local_evidence.sh`; `scripts/verify_connectorhub_engineering_trail.py` | Fixed for local review; still no CI artifact by design. |
| N1: README as status source | README now points to authoritative reports for ConnectorHub and VS2 generated status. | `README.md` | Fixed. |
| N2: downloaded source docs | Sanitized source inputs are archived under repo docs and pinned by hash/line count. | `docs/archive/research/CornerStone_ConnectorHub_Application_Guide_2026-06-24.md`; `docs/archive/research/CornerStone_ConnectorHub_Test_Scenario_Implementation_Document_2026-06-24.md`; engineering-trail manifest | Fixed for current source register. |
| N3: H04 substitution ADR | H04 local-baseline substitution is a formal ADR and remains review input only. | `docs/adr/ADR-0007-connectorhub-h04-local-baseline-substitution.md` | Fixed. |
| N4: trimmed stdout proof risk | Filtered single-scenario stdout includes `full_report_path` and `full_report_sha256`; tests cover the field presence and hash match. | `packages/cornerstone_cli/main.py`; `tests/scenario/test_connectorhub_cli.py`; PR body filtered stdout check | Fixed for current CLI stdout path. |

## Current Required Local Commands

Default local evidence gate:

```sh
scripts/verify_connectorhub_local_evidence.sh
```

Default mode runs `git diff --check`, SoT docs verification, PR20 feedback-response verification, ConnectorHub engineering-trail verification, ConnectorHub CLI unittest coverage with `CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1`, compact-report tests, scaffold CLI unittest coverage with `CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1`, and `compileall`. The default gate is intentionally non-Docker; VS2-heavy regression subtests move to the strict local/manual gate.

Feedback-response gate:

```sh
python3 scripts/verify_connectorhub_pr20_feedback.py
```

This command validates the B1-B5, M1-M5, and N1-N4 response surfaces: local-gate behavior, status delegation, VS2 current-report reconciliation, compact evidence hashes, controlled egress sink evidence, action-lane wording, path portability boundaries, and explicit remaining split/monolith blockers.

Split-readiness gate:

```sh
python3 scripts/verify_connectorhub_review_split.py
```

This command validates that the proposed PR slices cover `CS-CH-001` through `CS-CH-040`, keep the contract/human-gate and VS2 rehearsal boundaries explicit, avoid workflow-dependent commands, and preserve the target module/test map as follow-up work rather than completed refactor evidence.

Strict local/manual gate:

```sh
scripts/verify_connectorhub_local_evidence.sh --strict
```

Strict mode adds `make verify-vs2-production-like`. It is not represented as GitHub Actions in this branch because it depends on current local VS2 reusable proof state, Docker/network support, and a clean review workspace.

## Local Merge-Gate Rehearsal - 2026-06-28

The review-required local command set was rerun after adding the PR20 feedback-response verifier and wiring it into the default local evidence gate. The latest clean-clone default gate was run from `/tmp/cornerstone-pr20-feedback-verify.L5KlLB`.

| Command | Result | Evidence summary |
| --- | --- | --- |
| `PATH="$PWD:$PATH" cornerstone security vs2-local-proof --json` | PASS | `status=success`; `86 PASS`, `0 blocking`, `7 HUMAN_REQUIRED`; overclaim scan `status=passed`. |
| `PATH="$PWD:$PATH" cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json` | PASS | `status=success`; `86 PASS`, `0 blocking`, `7 HUMAN_REQUIRED`. |
| `PATH="$PWD:$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json` | PASS | `status=success`; `scenario_count=1`, `pass=1`, `blocking=0`; all egress topology checks true. |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_default_deny_egress_topology_cs_ch_036` | PASS | `Ran 1 test in 320.978s`; `OK`. |
| `scripts/verify_connectorhub_local_evidence.sh` | PASS | Clean clone default gate included `git diff --check`, SoT docs, PR20 feedback-response verifier, ConnectorHub engineering trail, review-split verifier, ConnectorHub CLI suite with VS2-heavy subtests skipped, compact report unittest, scaffold suite with VS2-heavy subtests skipped, and `compileall`. |
| `python3 scripts/verify_connectorhub_pr20_feedback.py` | PASS | `14` findings covered; local gate guarded; VS2 status reconciled; compact evidence hashed; controlled sink proof present; unresolved split/monolith boundaries explicit. |
| `env CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1 python3 -m unittest tests.scenario.test_connectorhub_cli` | PASS | Covered by the clean-clone default gate: `Ran 84 tests in 83.707s`; `OK (skipped=2)`. |
| `python3 -m unittest tests.scenario.test_connectorhub_cli.ConnectorHubCliTests.test_connectorhub_default_deny_egress_topology_cs_ch_036` | PASS | Targeted clean-clone regression check after removing the bind mount: `Ran 1 test in 109.410s`; `OK`. |
| `env CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1 python3 -m unittest tests.scenario.test_scaffold_cli` | PASS | Covered by the clean-clone default gate: `Ran 53 tests in 87.498s`; `OK (skipped=5)`. |
| `make verify-vs2-production-like` | PASS | Local command report `status=passed`; controlled provider/forbidden sink proof included; 8 scenario rows, all `PASS`. |

Proof boundary remains unchanged: this is local deterministic and local production-like rehearsal evidence only. It does not claim live connector readiness, production security readiness, or human acceptance.

## Remaining Non-Local Decisions

- Maintainer decides whether to split PR #20 or select a smaller merge slice.
- Human/external gates `CS-CH-H01` through `CS-CH-H07` require dated human records.
- Controlled forbidden-sink egress proof is now part of the strict local VS2 rehearsal; production network-control review remains human/external.
- `connector.py` and `tests/scenario/test_connectorhub_cli.py` still need the ADR-0008 module/test split before this becomes a clean implementation merge candidate.

## Current Verdict

The review feedback is addressed for local evidence review and proof-boundary clarity. PR #20 should remain draft and `needs-follow-up` until split, merge strategy, and human-gate decisions are made.
