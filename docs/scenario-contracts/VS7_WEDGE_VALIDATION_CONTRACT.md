# VS7 Wedge Validation Contract — The Market Gate

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Frozen milestone contract; status-neutral; intentionally the lightest of the three (detail freezes at VS7 kickoff from VS6 evidence). This is not implementation evidence.
**Depends on:** VS6 habit evidence and the VS6-EXT-003 keep/kill report.
**Acceptance authority:** `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` + `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
**Matrix:** `docs/scenario-contracts/VS7_WEDGE_VALIDATION_MATRIX.csv`

## Purpose

VS5 proved the brief. VS6 proved the habit. VS7 proves someone specific needs this enough to commit: the wedge decision (personal-pro tool vs. small-team tool) is made **from usage data, not taste**, and validated with design partners including willingness-to-pay evidence. VS7 is also where dormant systems earn reactivation or stay dormant — by user pull only.

## Goals

1. **Wedge decision:** a dated decision record choosing personal-pro or small-team as the wedge, justified by VS6 usage/retention data and external-user statements.
2. **Three design partners:** external individuals/teams using CornerStone on real recurring work for ≥4 weeks, with dated evidence of recurring value.
3. **Willingness-to-pay evidence:** pricing conversations with each partner recorded (paid pilot, LOI, or documented refusal with reasons). Refusals are first-class evidence.
4. **Keep/kill execution:** VS6-EXT-003 recommendations executed — surfaces users ignored are removed or shelved from the default experience, with a dated record per surface.
5. **Conditional capability (at most one, by pull only):**
   - If the wedge is **team**: minimal sharing — a brief shareable read-only to a named workspace member, with real (non-hardcoded) identity for members and approval visibility. This is the first genuine activation of the governance layer for multiple humans.
   - If the wedge is **personal**: the **first real external action** — one provider, one reversible action type (e.g., draft an email / create a ticket in the partner's tool), strictly behind the existing dry-run → approval → audit gate.
   - Neither is built without a recorded partner request naming the need.
6. **Dormancy disposition:** a dated decision per dormant system (ConnectorHub expansion, VS2 tenancy, VS3 on-prem, brain/agents/ontology/autopilot/capsules/packs): reactivate-with-rationale, keep dormant, or delete. Silence is not a disposition.

## Non-Goals

Building both conditional capabilities; multi-provider actions; marketplace/packs; ontology/brain/agent reactivation absent partner pull; on-prem/compliance posture absent a paying regulated partner; billing infrastructure (a paid pilot may be invoiced manually).

## Scenario Rows (summary — full rows freeze at VS7 kickoff)

| ID | Priority | Scenario | Verification mode |
|---|---|---|---|
| VS7-WEDGE-001 | MUST_PASS | Dated wedge decision citing VS6 data and user statements | HUMAN_REQUIRED |
| VS7-PART-001 | MUST_PASS | 3 design partners, ≥4 weeks real recurring use each, dated evidence | HUMAN_REQUIRED (external) |
| VS7-PART-002 | MUST_PASS | Willingness-to-pay evidence per partner (pilot/LOI/documented refusal) | HUMAN_REQUIRED (external) |
| VS7-PART-003 | MUST_PASS | ≥1 partner decision made with CornerStone that they defended to someone else using the citation trail | HUMAN_REQUIRED (external) |
| VS7-KILL-001 | MUST_PASS | Keep/kill executed with dated per-surface records; default UX contains no ignored surfaces | AUTOMATED + HUMAN_REQUIRED |
| VS7-COND-001 | MUST_PASS (conditional) | The one pulled capability ships behind existing trust/approval/audit boundaries with Plane 1 + Plane 2 evidence; the trigger request is recorded | AUTOMATED + HUMAN_REQUIRED |
| VS7-DORM-001 | MUST_PASS | Dormancy disposition recorded per dormant system | HUMAN_REQUIRED |
| VS7-REG-001 | REGRESSION | Plane 1 + Plane 2 hold through VS7 changes | AUTOMATED + HUMAN_REQUIRED |

## Success Criteria

All rows evidenced; the wedge decision and partner evidence exist as dated records; at most one conditional capability shipped, and only by pull. If partners cannot be recruited or refuse to pay, VS7 closes honestly with that finding — **a documented negative market result is a valid VS7 outcome** and triggers a strategy review, not a claim adjustment.

## Risks

| Risk | Mitigation |
|---|---|
| Wedge decision made by preference, not data | VS7-WEDGE-001 requires cited VS6 evidence; the decision record names the data |
| First external action creates real-world damage | Reversible action types only; single provider; dry-run + approval + audit mandatory; partner sandbox first |
| Partner recruitment fails | VS6 cohort is the funnel; report BLOCKED/negative honestly |
| Dormant systems creep back without pull | VS7-DORM-001 makes every disposition explicit and dated |

## Out of Scope (explicit)

Everything not named in Goals; both conditional capabilities at once; any claim of production/enterprise readiness (that conversation reopens only against VS2/VS3 with a paying partner who needs it).
