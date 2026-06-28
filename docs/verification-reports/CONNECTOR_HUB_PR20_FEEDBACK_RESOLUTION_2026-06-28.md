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
| B1: draft/no visible CI | PR remains draft. No GitHub Actions workflow is added. `scripts/verify_connectorhub_local_evidence.sh` is the local review gate; strict mode remains local/manual. | `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md`; PR body; `scripts/verify_connectorhub_local_evidence.sh` | Still blocks merge confidence until maintainer accepts local logs or a future CI plan. |
| B2: contract mixed status/PASS | Contract status is separated from implementation status; README also points to reports instead of carrying ConnectorHub row counts as authority. | `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md`; `README.md`; `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json`; `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md` | Fixed for current local review. |
| B3: VS2 status inconsistent | Current VS2 status is delegated to `VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`; README no longer repeats generated counts. | `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`; `README.md` | Fixed for status authority; production/human gates remain outside this PR. |
| B4: non-portable evidence paths | Source documents are repo-archived, compact reports have `path_portability`, and the delivery-unit manifest marks `tmp/scenario/...` as regenerable local transcript refs. | `docs/archive/research/`; `reports/scenario/connector-contract-adapter/`; `reports/scenario/connectorhub-scenario-delivery-unit-manifest-2026-06-24.json`; `scripts/verify_connectorhub_engineering_trail.py` | Fixed for committed ConnectorHub review artifacts. |
| B5: PR too large | ADR-0008 defines the split sequence and target package/test module map. PR #20 stays a draft evidence branch. | `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md`; `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md` | Still blocks merge as one broad PR unless maintainer explicitly chooses a scoped slice. |
| M1: `connector.py`/test monolith | ADR-0008 records the target module/test map. This branch does not claim the split is complete. | `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md` | Follow-up implementation before merge or split PRs. |
| M2: Action lane boundary | `CS-CH-031` wording is local fixture Action and fixture outcome only; live side-effecting connector Actions remain out of scope. | `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md`; `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md` | Fixed for current local review. |
| M3: weak simulated egress checks | Current reports keep local-only/synthetic boundaries visible. Controlled forbidden-sink hardening remains a future VS2 local/manual proof slice. | `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md`; `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md` | Follow-up security hardening; do not claim production egress enforcement. |
| M4: milestone wording | PR/body/docs use ConnectorHub adoption overlay wording, not milestone-complete wording. | PR body; `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md`; `README.md` | Fixed for wording boundary. |
| M5: generated evidence hard to review | Reviewer guide plus local verifier define how to validate generated trail without reading every large report. | `docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md`; `scripts/verify_connectorhub_local_evidence.sh`; `scripts/verify_connectorhub_engineering_trail.py` | Fixed for local review; still no CI artifact by design. |
| N1: README as status source | README now points to authoritative reports for ConnectorHub and VS2 generated status. | `README.md` | Fixed. |
| N2: downloaded source docs | Sanitized source inputs are archived under repo docs and pinned by hash/line count. | `docs/archive/research/CornerStone_ConnectorHub_Application_Guide_2026-06-24.md`; `docs/archive/research/CornerStone_ConnectorHub_Test_Scenario_Implementation_Document_2026-06-24.md`; engineering-trail manifest | Fixed for current source register. |
| N3: H04 substitution ADR | H04 local-baseline substitution is a formal ADR and remains review input only. | `docs/adr/ADR-0007-connectorhub-h04-local-baseline-substitution.md` | Fixed. |
| N4: trimmed stdout proof risk | Filtered single-scenario stdout includes `full_report_path` and `full_report_sha256`; tests cover the field presence and hash match. | `packages/cornerstone_cli/main.py`; `tests/scenario/test_connectorhub_cli.py`; PR body filtered stdout check | Fixed for current CLI stdout path. |

## Current Required Local Commands

Default local evidence gate:

```sh
scripts/verify_connectorhub_local_evidence.sh
```

Strict local/manual gate:

```sh
scripts/verify_connectorhub_local_evidence.sh --strict
```

Strict mode is not represented as GitHub Actions in this branch. It depends on current local VS2 reusable proof state, Docker/network support, and a clean review workspace.

## Remaining Non-Local Decisions

- Maintainer decides whether to split PR #20 or select a smaller merge slice.
- Human/external gates `CS-CH-H01` through `CS-CH-H07` require dated human records.
- Controlled forbidden-sink egress proof remains a future VS2 hardening slice.
- `connector.py` and `tests/scenario/test_connectorhub_cli.py` still need the ADR-0008 module/test split before this becomes a clean implementation merge candidate.

## Current Verdict

The review feedback is addressed for local evidence review and proof-boundary clarity. PR #20 should remain draft and `needs-follow-up` until split/strict-gate/human-gate decisions are made.
