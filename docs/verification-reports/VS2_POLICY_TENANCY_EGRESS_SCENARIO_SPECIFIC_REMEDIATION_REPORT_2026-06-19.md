# VS2 Policy, Tenant Isolation, and Default Egress Deny Scenario-Specific Remediation Report - 2026-06-19

## Verdict

VS2 is not complete yet.

This remediation slice corrects the previous blanket-PASS behavior. The VS2 verifier now marks a scenario `PASS` only when that scenario has its own validator and concrete evidence. Missing validators produce `NOT_VERIFIED`.

- Scenario set: `vs2-policy-tenancy-egress`
- Scenario count: 93
- PASS: 7
- FAIL: 0
- NOT_VERIFIED: 79
- HUMAN_REQUIRED: 7
- Blocking AI-verifiable rows: 79
- Claim boundary: `VS2_SCENARIO_SPECIFIC_EVIDENCE_INCOMPLETE`

The current canonical scenario-by-scenario result is:

`reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json`

## What Changed

Implemented the first local VS2 remediation slice:

- added a scenario-specific `SCENARIO_CHECKS` registry in `packages/cornerstone_cli/vs2_security.py`;
- removed the blanket `dependencies_ok` PASS assignment for AI-verifiable rows;
- generated a synthetic local world with two tenants, three principals, three memberships, signed fixture sessions, forged caller fields, revocation state, and worker-envelope cases;
- expanded the disposable PostgreSQL 16 RLS proof to seed tenant A and tenant B rows across every protected table;
- added per-scenario result fields: validator, verification command, exit code, evidence paths, evidence hashes, verified commit, and verified tree;
- updated `cornerstone scenario verify ... --scenario ...` summary recomputation for filtered scenario runs;
- updated `make verify-vs2-security` so it still writes incomplete evidence reports before failing the gate;
- updated tests so VS2 full verification is expected to fail until all scenario validators exist.

## Passing Scenario-Specific Validators

| Scenario | Validator | Evidence |
|---|---|---|
| `VS2-SEC-002` | `verify_forged_scope_denied` | Synthetic signed session derives tenant A; caller-forged tenant B/admin fields are rejected and audited. |
| `VS2-SEC-003` | `verify_missing_context_fails_closed` | Missing, malformed, and expired sessions fail before DB or egress calls. |
| `VS2-SEC-005` | `verify_revocation_denies_next_request` | Membership revocation denies the next request and stale worker job is quarantined. |
| `VS2-SEC-007` | `verify_rls_select_isolation` | PostgreSQL RLS tenant A SELECT sees only tenant A rows across protected tables. |
| `VS2-SEC-008` | `verify_rls_write_isolation` | Forged tenant B insert/update are denied; cross-tenant update/delete affect zero rows. |
| `VS2-SEC-013` | `verify_app_role_hardened` | App/migration/maintenance roles are not superuser/BYPASSRLS and protected tables force RLS. |
| `VS2-SEC-017` | `verify_worker_scope_revalidation` | Worker envelopes run only with trusted scope; missing/tampered/stale jobs quarantine before DB access. |

## Evidence Artifacts

| Artifact | SHA-256 |
|---|---|
| `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json` | `51636c8d10fd94a65a736cc4b4ad095ee26db0870ef57c131f72f0d20d3d7b64` |
| `reports/security/vs2-local-security-proof.json` | `b1ba45729eedd5f6c83202becaae0794e69395ff9de7256fbe3fdd729231a3da` |
| `reports/security/vs2-scenario-specific-evidence.json` | `d98ef9bef21c7aab3208f094dca66eb82320b7eeacf63dbcf7b910445ef3f821` |
| `reports/security/vs2-synthetic-world.json` | `6fce229bf6b93fba13fd936c4bc91d286b3340d1e3e940eeaa484014426c2781` |
| `reports/db/vs2-rls-inventory.json` | `299ba08ddda25bf0aba577707e5b448a53d2006355a43d7d4dddf31e153c40ee` |
| `reports/db/vs2-tenant-isolation.json` | `1da87ff42e480190e84fc2712fb5a19d4fc45c765cec82fbd161221cc2a2cd48` |
| `reports/network/vs2-egress-proof.json` | `34fd904a4fc38561820b34fd2015ba0105bac0f7a13106f6d03cf04c2f429976` |

## Verification Commands

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs2-policy-tenancy-egress --json --output reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json
PATH="$PWD:$PATH" cornerstone scenario verify vs2-policy-tenancy-egress --scenario VS2-SEC-002 --scenario VS2-SEC-003 --scenario VS2-SEC-005 --scenario VS2-SEC-007 --scenario VS2-SEC-008 --scenario VS2-SEC-013 --scenario VS2-SEC-017 --json
make verify-vs2-security
python3 -m unittest tests.scenario.test_scaffold_cli
python3 -m compileall packages/cornerstone_cli
python3 scripts/verify_scenario_matrix.py
scripts/verify_sot_docs.sh
git diff --check
```

Observed results:

- full VS2 scenario verify: exit 4, expected incomplete gate; 7 PASS, 79 NOT_VERIFIED, 7 HUMAN_REQUIRED;
- filtered seven-scenario verify: exit 0, 7 PASS, 0 blocking;
- `make verify-vs2-security`: expected failure at scenario gate after writing reports; `make` exit 2 because the gate command exits 4;
- `python3 -m unittest tests.scenario.test_scaffold_cli`: exit 0, 50 tests, 74.047 seconds;
- `python3 -m compileall packages/cornerstone_cli`: exit 0;
- `python3 scripts/verify_scenario_matrix.py`: exit 0;
- `scripts/verify_sot_docs.sh`: exit 0;
- `git diff --check`: exit 0.

## Remaining Work

The next remediation gates are still open:

- add validators for the remaining 79 AI-verifiable VS2 scenarios;
- replace remaining synthetic adapter checks with real API/UI/OPA-service/network-boundary validators where each scenario requires those surfaces;
- add real audit mutation/deletion/reordering verification before audit-integrity scenarios can PASS;
- add realistic migration, rollback, backup, and restore drill evidence before migration/restore scenarios can PASS;
- rerun VS0 and VS1 regression gates against the final VS2 tree before any broader VS2 readiness claim.

## Human Required

The seven `HUMAN_REQUIRED` rows remain separate from the local remediation claim:

- `VS2-SEC-H01`;
- `VS2-SEC-H02`;
- `VS2-SEC-H03`;
- `VS2-SEC-H04`;
- `VS2-SEC-H05`;
- `VS2-SEC-H06`;
- `VS2-SEC-H07`.

## Final Verdict

Ship this remediation slice as a corrective verification change. Do not claim VS2 local readiness until all 70 MUST_PASS and 16 REGRESSION scenarios are PASS with scenario-specific evidence.
