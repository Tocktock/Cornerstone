# VS6 Daily Loop Contract — The Habit Gate

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Frozen milestone contract; status-neutral; intentionally lighter than VS5 (its detail freezes at VS6 kickoff, informed by VS5 external findings). This is not implementation evidence.
**Depends on:** VS5 reaching `VALUE_VERIFIED_EXTERNAL` (or a dated owner exception decision).
**Acceptance authority:** `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` + `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
**Matrix:** `docs/scenario-contracts/VS6_DAILY_LOOP_MATRIX.csv`

## Purpose

VS5 proves one great brief. VS6 proves the loop runs on the user's real life daily: sources flow in without manual pasting, the Ops Inbox fills itself, a morning digest gives a reason to return, and ~20 external users generate retention evidence. Urgency comes from the loop running on real, recurring input — not from features.

## Goals

1. **Exactly one read-only ingest source**, chosen at VS6 kickoff from: watched local folder (default candidate — fully local, no credentials), Gmail read-only, or Slack export import. One. The choice is recorded with rationale as a dated decision; the other candidates stay out of scope.
2. **Self-filling inbox:** new source items become artifacts and surface as reviewable work without user action; dedupe and trust states hold on continuous ingest.
3. **Morning digest:** a daily summary — new since yesterday, briefs awaiting review, decisions going stale (e.g., an extracted deadline approaching) — via the product surface (no email sending; sending is a write action and stays dormant).
4. **Retrieval quality at scale:** eval-corpus retrieval metrics don't degrade as the artifact store grows from continuous ingest; measured, not assumed.
5. **Habit evidence from ~20 external users:** activation (first brief <10 min), briefs/user/week, week-2 return rate, digest-open behavior — measured with local, privacy-respecting instrumentation (owner-visible, no third-party analytics).
6. **Decision resurfacing:** a recorded Decision with an extracted future date resurfaces before that date.

## Non-Goals

Second ingest source; write actions of any kind; team/sharing features; memory promotion machinery; tenancy/SSO/on-prem; any dormant system reactivation; pricing/billing implementation (pricing conversations are VS7 evidence work, not code).

## Scenario Rows (summary — full Given/When/Then freezes at VS6 kickoff)

| ID | Priority | Scenario | Verification mode |
|---|---|---|---|
| VS6-SRC-001 | MUST_PASS | Chosen source ingests read-only into immutable artifacts with provenance, dedupe, and untrusted-by-default trust state | AUTOMATED |
| VS6-SRC-002 | MUST_PASS | Zero writeback: negative evidence that the source connection never mutates the source system | AUTOMATED |
| VS6-SRC-003 | MUST_PASS | Source failure degrades honestly: outage/permission loss is visible, resumable, and loses no already-ingested evidence | AUTOMATED |
| VS6-LOOP-001 | MUST_PASS | Inbox fills itself: new source items appear as reviewable work with no manual step | AUTOMATED |
| VS6-LOOP-002 | MUST_PASS | Morning digest lists new items, pending reviews, and stale decisions, each linking into the loop | AUTOMATED + HUMAN_REQUIRED |
| VS6-LOOP-003 | MUST_PASS | Decision resurfacing: a Decision with an extracted deadline resurfaces before the deadline | AUTOMATED |
| VS6-QUAL-001 | MUST_PASS | Retrieval quality holds at volume: eval retrieval metrics within frozen tolerance as store grows 10x via continuous ingest | AUTOMATED |
| VS6-QUAL-002 | MUST_PASS | Plane 2 re-run: CS-VAL-001..007 hold on briefs generated from connector-ingested (not pasted) content | AUTOMATED + HUMAN_REQUIRED |
| VS6-EXT-001 | MUST_PASS | ~20 external users activated (first brief <10 min each), with dated session/onboarding records | HUMAN_REQUIRED (external) |
| VS6-EXT-002 | MUST_PASS | Habit evidence: week-2 return rate and briefs/user/week measured and reported honestly (targets set from VS5 findings at kickoff, then frozen) | HUMAN_REQUIRED (external) |
| VS6-EXT-003 | MUST_PASS | Keep/kill usage report: per-surface usage (Claims/Decisions, Memory candidates, Action drafts, Learn) with an explicit keep/fix/kill recommendation each | HUMAN_REQUIRED |
| VS6-REG-001 | REGRESSION | VS5 Plane 2 evidence remains valid or is re-earned after any generation-path change | AUTOMATED + HUMAN_REQUIRED |
| VS6-REG-002 | REGRESSION | Plane 1 structural sets pass under continuous ingest | AUTOMATED |

## Success Criteria

All MUST_PASS rows evidenced; retention numbers reported as measured (misses are findings, not failures to hide); VS6-EXT-003 keep/kill report exists — it is the primary input to VS7 scoping.

## Risks

| Risk | Mitigation |
|---|---|
| Source choice bikesheds | Decision is a dated one-day call at kickoff; watched folder is the default absent strong contrary evidence |
| Continuous ingest floods retrieval quality | VS6-QUAL-001 measures at 10x volume before external rollout |
| 20 users is beyond solo recruiting reach | VS5 participants + their referrals first; report BLOCKED honestly if stalled — do not lower the definition of "external" |
| Instrumentation becomes surveillance | Local-only metrics, documented fields, user-visible; no third-party analytics |

## Out of Scope (explicit)

Everything in the ADR-0007 dormancy register; second sources; write paths; sharing; billing.
