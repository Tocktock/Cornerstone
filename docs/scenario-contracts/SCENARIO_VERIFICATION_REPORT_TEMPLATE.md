# CornerStone Scenario Verification Report Template

Summary:
- Verdict:
- Scope:
- Date:
- Owner:

Goal:
- TODO

Scenario Verification:
| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| CS-... | MUST_PASS | ... | ... | ... | PASS/FAIL/NOT_VERIFIED/NOT_RUN/HUMAN_REQUIRED/OUT_OF_SCOPE |
| CS-REG-... | REGRESSION_GUARD | ... | ... | ... | PASS/FAIL/NOT_VERIFIED/NOT_RUN/HUMAN_REQUIRED/OUT_OF_SCOPE |

CLI Parity Summary:
| Feature / Scenario | CLI Command(s) | JSON Schema | Exit-Code Tests | Evidence / Audit Refs | Same Backend Path | Status |
|---|---|---|---|---|---|---|
| ... | `cornerstone ... --json` | ... | ... | ... | ... | PASS/FAIL/NOT_VERIFIED/NOT_RUN/HUMAN_REQUIRED/OUT_OF_SCOPE |

Human Required:
| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H-... | ... | ... | ... | blocks release / needs approval / non-blocking |

Tool / Process Evidence:
- Inputs inspected:
- Current behavior reverse-engineered:
- Files or artifacts changed:
- Checks run:
- Failed checks and fixes:
- CLI commands/transcripts checked:
- Checks not run:

Failure Reverse Engineering:
| Scenario | Expected | Actual / Missing Evidence | First Failing Layer | Root Cause | Fix or Blocker | Re-verification Plan |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

Verification Gaps:
- TODO

CLI Parity Gaps:
- TODO

Risks:
- TODO

Verdict:
- AI-verifiable scope: done / needs-follow-up / blocked
- Human/release gate: clear / needs-human-verification / blocked
