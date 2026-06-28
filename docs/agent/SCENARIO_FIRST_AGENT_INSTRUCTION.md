# Scenario-First Agent Instruction — Canonical Process Copy

**Replacement status:** Active implementation-agent process document.  
**Source file:** `scenario_first_agent_instruction_final_en.md` uploaded by JiYong / Tars.

**Goal-following overlay:** For long-running `/goal` work, use
`docs/agent/PRIMARY_GOAL_INSTRUCTION_FOLLOWING.md` to keep the scenario-first
direction while limiting execution to small verified slices with explicit
checkpoints.

---

# Scenario-First Agent Instruction — Final

## Purpose

This instruction is for an AI coding agent. It is **not** a repository packaging, patch bundle, or zip-output strategy.

The required output is a **verification-centered final report**. The agent must prove that the agreed checklist and scenarios were verified, or clearly state what could not be verified and why.

---

## Core Rule

Do not start implementation until the `Goal`, `Constraints`, `Checklist`, `MUST_PASS Scenarios`, `REGRESSION Scenarios`, and `Out of Scope` items are frozen.

A task is complete only when every AI-verifiable `MUST_PASS` and `REGRESSION` scenario is verified as `PASS` with concrete evidence.

If verification is impossible without a human, unavailable credentials, external approval, production access, irreversible action, physical device, real user, or subjective human judgment, do **not** mark it `PASS`. Move it to `Human Required` and state exactly what the human must do.

---

## 1. Non-Negotiable Rules

1. **Freeze the scenario contract before coding.**  
   The agent must first define the goal, constraints, checklist, scenarios, regressions, assumptions, and out-of-scope items.

2. **Scenarios are acceptance criteria.**  
   Completion is judged by scenario verification results, not by implementation effort.

3. **Every checklist/scenario item must appear in the final output.**  
   Missing items mean incomplete work.

4. **AI-verifiable items must be verified by the agent.**  
   Do not ask the human to verify something the agent can safely verify with tools.

5. **`PASS` requires evidence.**  
   Valid evidence includes test output, build/typecheck/lint output, command result, runtime observation, logs, screenshots, or specific source review references. “Looks fine”, “should work”, and “probably passes” are not evidence.

6. **Human-required is not an escape hatch.**  
   Use it only when AI truly cannot verify the item without a person or unavailable external access.

7. **Reverse-engineer before editing.**  
   The agent must inspect the current system with tools, reconstruct the relevant behavior, and map scenarios to changes before implementation. Do not code from memory.

8. **Failures require root-cause analysis.**  
   If a scenario fails or cannot be verified, identify the failed layer, root cause, why it was missed, fix or blocker, and recheck plan.

9. **Do not claim `done` if verification is incomplete.**  
   If any AI-verifiable `MUST_PASS` or `REGRESSION` item is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`, the AI-verifiable scope is not done.

---

## 2. Pre-Coding Scenario Contract

Before implementation, write this contract:

```markdown
Feature / Task:

Goal:

Success Criteria:

Constraints:
- Product / UX:
- Data / State:
- Permission / Security:
- Compatibility / Format:
- Operational / Environment:

Assumptions:

Out of Scope before coding:

Scenario Contract:
| ID | Type | Trigger / Action | Expected Result | Affected Layers | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|---|---|
| S01 | MUST_PASS | ... | ... | backend/frontend/storage/etc. | test/build/review/browser/etc. | ... | AI |
| R01 | REGRESSION | ... | ... | ... | ... | ... | AI |
| H01 | HUMAN_REQUIRED_CANDIDATE | ... | ... | ... | human check/approval | ... | Human |
```

Rules:

- Write scenarios as observable user/system behavior, not implementation tasks.
- Each scenario must have an expected result and verification method.
- `OUT_OF_SCOPE` must be declared before coding.
- Human-required candidates must be justified, not assumed.

---

## 3. Tool-Aware Coding Process

The agent must follow this flow and summarize the evidence in the final report.

### Step 1 — Intake & Freeze

Read the request and provided materials. Freeze goal, constraints, checklist, scenarios, regressions, assumptions, and out-of-scope items. Do not implement yet.

### Step 2 — Context Inventory

Use tools to inspect relevant context:

- files and directories;
- current implementation patterns;
- entry points and data flow;
- permission/security checks;
- UI state and labels;
- storage/schema/migrations;
- tests and test utilities;
- scripts, build commands, CI hints, and environment limits.

### Step 3 — Current Behavior Reverse Engineering

Reconstruct the current behavior from observable tool evidence:

- where input enters;
- where validation and authorization happen;
- where state is read or written;
- where UI state is derived;
- where errors are surfaced;
- which tests/checks already exist;
- which verification gaps remain.

This is not hidden reasoning. It is a concise reconstruction from inspected files, commands, outputs, errors, and observed behavior.

### Step 4 — Scenario-to-Work Mapping

Map each scenario to concrete changes and checks:

```markdown
S01:
  Expected Behavior:
  Current Behavior:
  Change Points:
  Verification Method:
  Evidence Needed:
  Regression Risk:
```

### Step 5 — Baseline Check

When safe and available, run or inspect baseline tests, typecheck, build, lint, or relevant scripts before editing. If something already fails, mark it as `pre-existing` and explain impact.

### Step 6 — Implement Smallest Complete Change

Implement only what is necessary to satisfy the frozen scenario contract. Prefer backend enforcement for security and permissions. Use frontend controls for UX, but never rely on frontend-only security. Keep changes traceable to scenario IDs.

### Step 7 — Verify Scenario by Scenario

For every AI-verifiable `MUST_PASS` and `REGRESSION` item, execute the planned verification when safe and possible.

```markdown
Scenario ID:
Status:
Verification Performed:
Evidence:
Notes:
```

### Step 8 — Failure Reverse Engineering Loop

For `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`, do this before claiming completion:

```markdown
Scenario:
Expected:
Actual / Missing Evidence:
Evidence Observed:
First Failing Layer:
Root Cause:
Why It Was Missed:
Fix or Blocker:
Re-verification Plan:
Related Regressions:
Updated Status:
```

After a fix, re-run the failed scenario and related regressions. If re-verification is not performed, status remains `NOT_VERIFIED` or `NOT_RUN`.

### Step 9 — Final Verification Gate

Before final output, confirm:

- every checklist/scenario item is included;
- every `PASS` has evidence;
- every AI-verifiable `MUST_PASS` is `PASS`;
- every AI-verifiable `REGRESSION` is `PASS`;
- all failures, gaps, and human-required items are explicit;
- the verdict is no stronger than the evidence supports.

---

## 4. Status Definitions

| Status | Meaning | Blocks AI-verifiable `done`? |
|---|---|---|
| `PASS` | Verified with concrete evidence. | No |
| `FAIL` | Checked and expected result was not met. | Yes |
| `NOT_VERIFIED` | AI could potentially verify, but evidence is insufficient. | Yes |
| `NOT_RUN` | Planned check/test/command was not executed. | Yes |
| `HUMAN_REQUIRED` | Verification truly requires a person or unavailable external access. | Excluded from AI gate, but must be reported |
| `OUT_OF_SCOPE` | Explicitly excluded before coding. | No, if declared before coding |

`OUT_OF_SCOPE` cannot be silently added after implementation starts. If scope changes during work, report it as a scope change.

---

## 5. Human Required Rules

Allowed only when verification requires one of these:

- unavailable credentials or private account access;
- production access or external third-party state;
- payment, deletion, deployment, account change, or another irreversible/sensitive action;
- legal, security, compliance, product, customer, or stakeholder approval;
- physical device, real user, or subjective UX/brand/policy judgment.

Not allowed when the reason is convenience, complexity, skipped tests, long-running checks, unclear commands, or poor planning.

Each human-required item must include:

```markdown
Human Required:
| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H01 | ... | ... | approval/screenshot/log/etc. | blocks release / blocks confidence / needs approval / non-blocking |
```

---

## 6. Required Final Output Format

The agent’s final output must use this structure:

```markdown
Summary:
- [What was done and the evidence-based verdict]

Goal:
- [Frozen goal]

Scenario Verification:
| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| S01 | MUST_PASS | ... | ... | ... | PASS/FAIL/NOT_VERIFIED/NOT_RUN |
| R01 | REGRESSION | ... | ... | ... | PASS/FAIL/NOT_VERIFIED/NOT_RUN |

Human Required:
| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H01 | ... | ... | ... | ... |

Tool / Process Evidence:
- Inputs inspected:
- Current behavior reverse-engineered:
- Files or artifacts changed:
- Commands/checks run:
- Failed checks and fixes:
- Checks not run:

Failure Reverse Engineering:
- Scenario:
- Root Cause:
- Fix or Blocker:
- Re-verification:
- Related regressions:

Verification Gaps:
- [Remaining NOT_VERIFIED / NOT_RUN / HUMAN_REQUIRED items and impact]

Risks:
- [Remaining technical, product, security, release, or evidence risks]

Verdict:
- AI-verifiable scope: done / needs-follow-up / blocked
- Human/release gate: clear / needs-human-verification / blocked
```

If a section has no items, write `None`. Do not omit it.

---

## 7. Verdict Rules

| Condition | Verdict |
|---|---|
| All AI-verifiable `MUST_PASS` and `REGRESSION` items are `PASS`; no release-blocking human item remains. | `AI-verifiable scope: done` / `Human/release gate: clear` |
| All AI-verifiable items are `PASS`, but human approval or external verification is required. | `AI-verifiable scope: done` / `Human/release gate: needs-human-verification` |
| Any AI-verifiable `MUST_PASS` or `REGRESSION` item is `FAIL`. | `AI-verifiable scope: blocked` |
| Any AI-verifiable `MUST_PASS` or `REGRESSION` item is `NOT_VERIFIED` or `NOT_RUN`. | `AI-verifiable scope: needs-follow-up` or `blocked`, depending on severity |
| A scenario is removed after coding starts. | Report as scope change; do not silently mark `OUT_OF_SCOPE` |

---

## 8. Compact Agent Instruction

```markdown
Follow Scenario-First Agent Instruction.

Before coding:
1. Freeze goal, constraints, assumptions, out-of-scope items, checklist, MUST_PASS scenarios, and REGRESSION scenarios.
2. For each scenario, define expected result, affected layer, verification method, evidence required for PASS, and verification owner.
3. Mark human-required candidates only when AI truly cannot verify without a person, unavailable credential, external approval, production access, irreversible action, physical device, real user, or subjective human judgment.

During coding:
1. Do not implement from memory. Inspect the current system with tools first.
2. Reverse-engineer current behavior from observable evidence: files inspected, commands run, outputs, errors, tests, UI/state paths, permission paths, and storage paths.
3. Map each scenario to concrete change points and verification checks.
4. Implement the smallest complete change that satisfies the frozen contract.
5. Run available safe checks yourself.
6. If a check fails, reverse-engineer the root cause from tool/process evidence before patching.
7. Re-run the failed scenario and related regressions after the fix.

Final output:
1. Include every checklist/scenario item.
2. Mark each item as PASS, FAIL, NOT_VERIFIED, NOT_RUN, HUMAN_REQUIRED, or OUT_OF_SCOPE.
3. Do not mark PASS without concrete evidence.
4. If human intervention is required, state why AI cannot verify it, the required human action, expected evidence, and release impact.
5. Include Tool / Process Evidence: inspected inputs, reverse-engineered current behavior, changed artifacts, checks run, failed checks/fixes, and checks not run.
6. Include Failure Reverse Engineering for failed or unverified scenarios.
7. Base the verdict only on verified evidence.
8. Never claim AI-verifiable scope is done if any AI-verifiable MUST_PASS or REGRESSION item is FAIL, NOT_VERIFIED, or NOT_RUN.
```
