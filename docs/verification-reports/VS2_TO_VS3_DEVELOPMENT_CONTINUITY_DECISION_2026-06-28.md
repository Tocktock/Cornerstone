# VS2 To VS3 Development Continuity Decision - 2026-06-28

**Status:** VS3 development-continuity decision.
**Owner:** JiYong / Tars.
**Decision date:** 2026-06-28.
**Commit observed:** `a09a395`.
**Scope:** Local/on-prem development continuity only.
**Non-goal:** This document does not promote any VS2 human-required scenario to official `PASS`.

## Decision

CornerStone may proceed to VS3 development under a constrained local/on-prem development scope.

Official VS2 scenario status remains unchanged:

- `86` AI-owned rows are `PASS`.
- `7` human-owned rows remain `HUMAN_REQUIRED`.
- `0` rows are `FAIL`.

The VS3 development entry decision is:

```text
AI-verifiable VS2 local scope: acceptable to continue from
Production/security-release scope: not approved
VS3 development: allowed with constraints
Production/live/customer/on-prem release claim: blocked until the relevant VS2 human gates close
```

## Source Inputs

| Source | Role |
|---|---|
| `/Users/jiyong/.codex/attachments/d6539d6b-a256-4e7c-b57f-9e5f11912f6c/pasted-text.txt` | User-provided feedback that requests an explicit VS2-to-VS3 go-forward decision. |
| `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md` | Official VS2 scenario contract: 70 MUST_PASS, 16 REGRESSION, 7 HUMAN_REQUIRED rows. |
| `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_MATRIX.csv` | Machine-readable VS2 row inventory. |
| `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md` | Current generated-status authority for VS2 local deterministic evidence. |
| `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json` | Current machine-readable scenario report: 86 PASS, 7 HUMAN_REQUIRED. |
| `docs/verification-reports/VS2_SEC_H01_OWNER_APPROVAL_2026-06-20.md` | Existing `APPROVE WITH CONDITIONS` record for local/on-prem VS2 implementation. |
| `docs/verification-reports/VS2_PRODUCTION_LIKE_INTEGRATION_REHEARSAL_2026-06-27.md` | Local Docker production-like rehearsal evidence, explicitly not production readiness or human acceptance. |

## Status Model

This decision uses two separate status tracks:

| Track | Meaning | Authority |
|---|---|---|
| Official scenario status | Scenario report status used by release and proof gates. | VS2 contract, matrix, and generated scenario report. |
| Practical VS3-development status | Whether VS3 local development may proceed without claiming production/security release readiness. | This decision document, constrained by VS2 human gates. |

The practical status track must never overwrite official scenario status.

## VS2 Human Gate Decision Table

| Scenario | Official scenario status | Practical VS3-development status | Blocks VS3 local/dev? | Blocks production/security readiness? | Decision rationale |
|---|---|---|---|---|---|
| `VS2-SEC-H01` Architecture approval | `HUMAN_REQUIRED` | `CONDITIONAL PASS` for local/on-prem development | No | Yes, for broader production claims | A dated owner approval exists with explicit conditions and no production/security-readiness claim. |
| `VS2-SEC-H02` Independent security review | `HUMAN_REQUIRED` | `NEEDS HUMAN VERIFICATION` | No, if local/dev only | Yes | Local automated tests are useful engineering evidence but are not an independent signed security review. |
| `VS2-SEC-H03` Real IdP / SSO | `HUMAN_REQUIRED` | `NEEDS HUMAN VERIFICATION` | No, if synthetic/local identity only | Yes | Synthetic identity is acceptable for local VS3 development; real IdP mapping and revocation still require human/external proof. |
| `VS2-SEC-H04` Network/topology | `HUMAN_REQUIRED` | `CONDITIONAL PASS` for local topology baseline | No, if local/mock egress only | Yes | Local Docker production-like rehearsal passed, but real production/on-prem network evidence is still absent. |
| `VS2-SEC-H05` Live provider | `HUMAN_REQUIRED` | `NEEDS HUMAN VERIFICATION` | No, if mock/local providers only | Yes | No live-provider credential/custody/rehearsal evidence is accepted by this decision. |
| `VS2-SEC-H06` UX/trust | `HUMAN_REQUIRED` | `NEEDS HUMAN VERIFICATION` | No, for internal development | Yes, before real operator release | Automated UI evidence cannot replace a human operator trust/usability judgment. |
| `VS2-SEC-H07` Migration/restore | `HUMAN_REQUIRED` | `CONDITIONAL PASS` for local rehearsal only | No, if no real data migration | Yes | Local backup/restore rehearsal is development evidence only; production-like data and rollback authority still require human-supervised proof. |

No VS2 human-required scenario is considered `FAIL` by this decision. Missing human/external evidence means the scenario remains not verified for that scope.

## VS3 Entry Constraints

VS3 work may start only under these constraints:

1. Use local or synthetic tenants, owners, namespaces, identities, memberships, and roles.
2. Use mock/local providers unless `VS2-SEC-H05` is explicitly completed.
3. Preserve default egress deny, OPA/Rego policy evaluation, PostgreSQL RLS boundaries, audit evidence, and secret redaction.
4. Do not use real customer data, private production data, or production credentials.
5. Do not run production deployment, production migration, or destructive migration.
6. Do not claim enterprise SSO / real IdP readiness unless `VS2-SEC-H03` is completed.
7. Do not claim production network security unless `VS2-SEC-H04` is completed.
8. Do not claim independent security review unless `VS2-SEC-H02` is completed.
9. Do not claim live provider readiness unless `VS2-SEC-H05` is completed.
10. Do not claim production migration/restore readiness unless `VS2-SEC-H07` is completed.
11. Do not claim real operator UX/trust acceptance unless `VS2-SEC-H06` is completed.
12. Add VS3 supply-chain controls as part of the next scenario contract: tool build/sign/register/install/execute boundaries, SBOM, signed registry direction, and SLSA/Rekor/TUF-compatible proof where applicable.

## Explicit Non-Claims

This decision does not claim:

- VS2 production readiness;
- production tenant isolation readiness;
- production network-control readiness;
- real IdP / SSO readiness;
- live provider readiness;
- independent penetration/security-review completion;
- human operator UX/trust acceptance;
- production migration, backup, restore, or rollback acceptance;
- VS3 implementation completion.

## Required Next Step For VS3

Before VS3 feature implementation starts, freeze a VS3 scenario contract that:

- names the VS3 scope without colliding with historical archived VS-3 wording;
- keeps this VS2 continuity decision as an entry constraint;
- defines all VS3 MUST_PASS, REGRESSION_GUARD, and HUMAN_REQUIRED rows;
- requires native `cornerstone ... --json` CLI paths for every product feature;
- includes deterministic local verification for AI-owned rows;
- marks live provider, production, external approval, signing infrastructure, or subjective review as `HUMAN_REQUIRED` where applicable.

## Verification Gaps

The following remain open and must stay visible in VS3 reports:

- `VS2-SEC-H02`: independent security review.
- `VS2-SEC-H03`: real IdP / SSO mapping and revocation.
- `VS2-SEC-H04`: real production/on-prem network topology and egress-control evidence.
- `VS2-SEC-H05`: approved live-provider rehearsal.
- `VS2-SEC-H06`: human operator UX/trust review.
- `VS2-SEC-H07`: human-supervised production-like migration/restore rehearsal.

## Verdict

- **Official VS2 scenario status:** unchanged; 86 `PASS`, 7 `HUMAN_REQUIRED`.
- **VS3 development continuity:** allowed with constraints.
- **Production/security/live-provider/customer release:** blocked until the applicable VS2 human gates close.
