# VS4-H01 UI Recovery Owner Review Package

Date: 2026-07-05T00:01:26.031011Z

## Scope

This package covers the current verified recovery slice for the rejected VS4-H01 UI:

- R0: token-to-CSS pipeline, shared server-rendered product shell, reusable render helpers, real HTML routes, and language mapping.
- R1: Home rebuilt around Drop and Ask with real local records, day-zero copy, and internal owner material moved to `/review`.
- R2/R3: Search, Artifacts, Briefs, Claims, Actions, Inbox, Audit, owner connector governance, and record detail routes are represented in the screenshot pack.
- R4: product surfaces are scanned for forbidden internal language and raw runtime labels.
- R5: desktop, mobile, day-zero, not-found, and Home validation captures check horizontal overflow, mobile first-value ordering, and inline interaction states.
- R6: screenshot pack and automated checks exist; owner acceptance remains human-required.

## Evidence Files

- `screenshot-pack-manifest.json`
- `screenshots/` (24 desktop, 23 mobile captures, including day-zero, not-found, and Home validation states)
- `dom/` captured HTML for each screenshot route

## Screenshot Coverage

- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-search-desktop.png`: PASS / `/search` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-artifacts-desktop.png`: PASS / `/artifacts` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-briefs-desktop.png`: PASS / `/briefs` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-claims-desktop.png`: PASS / `/claims` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-actions-desktop.png`: PASS / `/actions` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-inbox-desktop.png`: PASS / `/inbox` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-audit-desktop.png`: PASS / `/audit` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-artifacts-mobile.png`: PASS / `/artifacts` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-briefs-mobile.png`: PASS / `/briefs` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-claims-mobile.png`: PASS / `/claims` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-actions-mobile.png`: PASS / `/actions` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-inbox-mobile.png`: PASS / `/inbox` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/day-zero-audit-mobile.png`: PASS / `/audit` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/not-found-page-desktop.png`: PASS / `/missing-product-route` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/not-found-source-desktop.png`: PASS / `/artifacts/missing-source?view=html` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/not-found-page-mobile.png`: PASS / `/missing-product-route` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/not-found-source-mobile.png`: PASS / `/artifacts/missing-source?view=html` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/home-validation-desktop.png`: PASS / `/` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/home-validation-mobile.png`: PASS / `/` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/home-desktop.png`: PASS / `/` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/search-desktop.png`: PASS / `/search?q=vendor%20renewal` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/artifacts-desktop.png`: PASS / `/artifacts` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/briefs-desktop.png`: PASS / `/briefs` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/claims-desktop.png`: PASS / `/claims` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/actions-desktop.png`: PASS / `/actions` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/inbox-desktop.png`: PASS / `/inbox` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/audit-desktop.png`: PASS / `/audit` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/owner-admin-desktop.png`: PASS / `/review` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/reference-gallery-desktop.png`: PASS / `/review/reference-images` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/artifact-detail-desktop.png`: PASS / `/artifacts/art_ede571327ab3ec20?view=html` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/brief-detail-desktop.png`: PASS / `/briefs/brief_723f2e01d403ffe4?view=html` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/claim-detail-desktop.png`: PASS / `/claims/claim_636afa656ba71367?view=html` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/action-detail-desktop.png`: PASS / `/actions/action_24051f108953750a?view=html` / desktop
- `output/playwright/vs4-h01-ui-recovery/screenshots/home-mobile.png`: PASS / `/` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/search-mobile.png`: PASS / `/search?q=vendor%20renewal` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/artifacts-mobile.png`: PASS / `/artifacts` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/briefs-mobile.png`: PASS / `/briefs` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/claims-mobile.png`: PASS / `/claims` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/actions-mobile.png`: PASS / `/actions` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/inbox-mobile.png`: PASS / `/inbox` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/audit-mobile.png`: PASS / `/audit` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/owner-admin-mobile.png`: PASS / `/review` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/reference-gallery-mobile.png`: PASS / `/review/reference-images` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/artifact-detail-mobile.png`: PASS / `/artifacts/art_ede571327ab3ec20?view=html` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/brief-detail-mobile.png`: PASS / `/briefs/brief_723f2e01d403ffe4?view=html` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/claim-detail-mobile.png`: PASS / `/claims/claim_636afa656ba71367?view=html` / mobile
- `output/playwright/vs4-h01-ui-recovery/screenshots/action-detail-mobile.png`: PASS / `/actions/action_24051f108953750a?view=html` / mobile

## Checks Run

- `python3 scripts/capture_vs4_h01_ui_recovery_screenshots.py --json`
  - Result: PASS; 47 pass, 0 fail.

## Companion Checks Before Owner Review

- `python3 -m unittest tests.scenario.test_product_ui_routes`
  - Purpose: focused route regression check.
- `make verify-vs4-product-alpha-shell`
  - Purpose: filtered scenario gate for shell proof.

## Human-Required Gates

- Owner subjective UI/UX acceptance remains not claimed.
- VS5 external-user readiness remains not claimed.
- The proposed Brief surface direction still needs owner acceptance because the redesign guidance notes no prior reference image exists for that centerpiece surface.

## Known Follow-Up

- Clean up the now-unreachable legacy VS4 browser-proof branch after the new multi-page route-scan proof has settled.
- Add real interaction captures for Drop, Ask, lane switching, and action approval preview before retrying VS4-H01 owner acceptance.
