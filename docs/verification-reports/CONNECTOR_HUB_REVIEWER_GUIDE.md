# ConnectorHub Reviewer Guide

**Status:** Review guide for PR #20 while it remains draft.
**Scope:** Local ConnectorHub adoption overlay evidence, compact report layout, and review commands.

## Review Boundary

This package is a ConnectorHub adoption overlay, not a production-readiness milestone. It demonstrates 40 AI-owned local fixture rows and prepares 7 human/external gates. Live GitHub readiness, physical macOS capture, Chrome privacy acceptance, production-like VS2 topology acceptance, live non-GitHub Action execution, UX/trust acceptance, and recovery exercise evidence remain separate human-required gates.

## Compact Evidence Layout

The 41 previous full `connector-contract-adapter*.json` reports are represented by:

- `reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json`
- `reports/scenario/connector-contract-adapter/scenarios/CS-CH-001.json` through `CS-CH-040.json`
- `reports/scenario/connector-contract-adapter/shared-evidence-index-2026-06-23.json`
- `reports/scenario/connector-contract-adapter/objects/sha256/`
- `reports/scenario/connector-contract-adapter/manifest-2026-06-23.json`

Each compact report keeps a repo-relative portable report path and a `path_portability` block. The shared evidence index points to section-level SHA-256-addressed JSON objects so repeated command/evidence payloads are stored once and referenced by digest. The current index has 7 shared section refs backed by 5 unique content objects. Any remaining absolute `output_path` value is historical transcript metadata only. `tmp/scenario/...` refs are regenerable local transcript refs, not committed durable evidence.

The scenario delivery-unit manifest also carries a top-level `path_portability` block. Its `tmp/scenario/...` values are local replay transcript references only; reviewers should rely on the committed compact reports and manifest hashes for durable evidence.

## Required Local Review Commands

```sh
scripts/verify_connectorhub_local_evidence.sh
```

This default local gate runs whitespace, SoT docs, the PR20 feedback-response verifier, the ConnectorHub engineering-trail verifier, the ConnectorHub CLI unittest suite with VS2-heavy regression subtests skipped, compact-report tests, the scaffold CLI unittest suite with VS2-heavy regression subtests skipped, and Python compile checks without depending on GitHub Actions. The skip is explicit through `CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1`; Docker/VS2 rehearsal evidence belongs to the strict local gate below.

For the feedback-response boundary specifically, run:

```sh
python3 scripts/verify_connectorhub_pr20_feedback.py
```

This validates the B1-B5, M1-M5, and N1-N4 response surfaces: local-gate behavior, status delegation, VS2 current-report reconciliation, compact evidence hashes, controlled egress sink evidence, action-lane wording, path portability boundaries, and explicit remaining split/monolith blockers.

For the review-split plan specifically, run:

```sh
python3 scripts/verify_connectorhub_review_split.py
```

This validates `docs/verification-reports/CONNECTOR_HUB_REVIEW_SPLIT_MANIFEST_2026-06-28.json`, including slice order, scenario coverage, repo-relative path references, local-command boundaries, and the target module/test map. It is planning evidence only; it does not claim the PR has already been split.

For a clean review workspace with current VS2 reusable proof state and Docker/network support, run:

```sh
scripts/verify_connectorhub_local_evidence.sh --strict
```

Strict mode additionally runs `make verify-vs2-production-like`. This remains a local/manual Docker rehearsal and must not be replaced by a README claim.

## Review Sequence

1. Confirm the PR remains draft until local gate evidence and split strategy are agreed.
2. Inspect `docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md` for neutral contract status. PASS/HUMAN_REQUIRED counts belong to reports.
3. Inspect `docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md` for source reconciliation, remaining proof surfaces, and compact evidence refs.
4. Inspect `docs/verification-reports/CONNECTOR_HUB_PR20_FEEDBACK_RESOLUTION_2026-06-28.md` for the finding-by-finding response, current local fixes, and remaining merge blockers.
5. Run `python3 scripts/verify_connectorhub_engineering_trail.py` to validate matrix rows, compact hashes, focused gates, source-input hashes, path portability, and manifest exactness.
6. Spot-check one compact focused report, the shared evidence index, and one referenced object before reviewing scenario behavior.

## Split Recommendation

PR #20 should stay draft unless intentionally kept as a single evidence branch. The formal split and module map is recorded in `docs/adr/ADR-0008-connectorhub-review-split-and-module-map.md`. A safer merge sequence is:

1. Contract and matrix cleanup.
2. Compact evidence layout and verifier support.
3. ConnectorHub focused CLI/runtime scenarios.
4. Human-gate preparation artifacts.
5. VS2 current verification report and H04 local-baseline handoff.
6. Local reviewer-guide and verification-script wiring.
7. Any later production-like integration test work.

The current branch does not claim the `connector.py` or scenario-test split is already complete. The ADR defines the target package/test map and is guarded by `scripts/verify_connectorhub_engineering_trail.py`.

## H04 Substitution Boundary

The H04 local-baseline substitution is documented in `docs/adr/ADR-0007-connectorhub-h04-local-baseline-substitution.md`. It is review input only. It does not promote H04, production-like VS2 readiness, or full ConnectorHub acceptance to PASS.
