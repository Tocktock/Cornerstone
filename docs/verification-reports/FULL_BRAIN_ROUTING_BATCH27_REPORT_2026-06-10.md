# Full Brain Routing Batch 27 Report - 2026-06-10

Status: PASS for deterministic CLI-native full brain routing scaffold only.
Scope: `CS-BRAIN-001` through `CS-BRAIN-016`, `CS-ARCH-012`, `CS-NS-009`, `CS-NS-010`, `CS-REG-009`, and `CS-REG-010`.

This report does not mark production UI runtime, production API runtime, live external model/provider compatibility, real provider calls, real credentials, production tenant/security policy, Ollama generation quality, or full 206-scenario completion as complete.

## Frozen Goal

Make the frozen scenario set pass through evidence-backed batches without adding unrelated features. This batch verifies replaceable model brains, policy-aware routing, provider override governance, Brain Performance Ledger records, namespace-local learning and opt-in aggregation, ensemble gating, LLM-as-judge support limits, objective outcome and owner acceptance precedence, judge recommendation governance, disagreement escalation, calibration/bias tracking, auditable routing, and provider-switch evidence continuity.

## Research Checkpoint

- OpenAI Agents SDK frames agent apps around agents, handoffs, guardrails, sessions, tools, and tracing: <https://developers.openai.com/api/docs/guides/agents>
- OpenAI Agents SDK guardrails reinforce that application code should validate and gate model-driven paths: <https://openai.github.io/openai-agents-python/guardrails/>
- LangGraph workflow guidance distinguishes predictable code-driven workflows from dynamic agentic paths, matching CornerStone's deterministic PASS baseline: <https://docs.langchain.com/oss/python/langgraph/workflows-agents>
- Recent LLM routing research treats model choice as a cost, latency, quality, and policy tradeoff rather than a hard-coded provider decision: <https://arxiv.org/html/2505.12601v2>
- LLM-as-judge research warns about overconfidence and supports calibration, disagreement tracking, and objective evidence precedence: <https://arxiv.org/html/2508.06225v2>
- LangChain guidance for LLM-as-judge evaluation emphasizes calibration against human labels and domain rubrics: <https://www.langchain.com/articles/llm-as-a-judge>
- IBM/ACL human-centered LLM-as-judge recommendations emphasize user-centered calibration and governance: <https://research.ibm.com/publications/human-centered-design-recommendations-for-llm-as-a-judge>

Best fit for this batch is the existing deterministic local runtime. It records registry metadata, routing decisions, ledger entries, policy denials, provider-switch continuity checks, judge records, objective conflicts, owner acceptance, recommendation candidates, adjudication records, calibration reports, and audit events without adding a live model-routing framework dependency or making real provider calls.

## Assumptions

- Native CLI JSON is the scaffold verification surface until production UI/API surfaces exist.
- `local_test` deterministic behavior remains the scenario PASS baseline.
- Ollama/local LLM availability is a smoke surface only; it is not the PASS judge.
- External providers are registry metadata only in this batch.
- No live secrets, real credentials, production tenant/security policy, or external calls are used.

## Out Of Scope

Production UI/browser model-routing views, production API, real model generation, real external provider/tool calls, real credentials, production tenant/security policy changes, new dependencies, full semantic model quality evaluation, and full 206-scenario completion.

## Checklist

- [x] Goal, assumptions, out-of-scope, applicable MUST_PASS rows, applicable REGRESSION rows, and human-required items frozen before implementation.
- [x] Relevant docs, scenario matrix, current behavior, and failure evidence inspected.
- [x] Recent routing, agent guardrail, LangGraph, and LLM-as-judge approaches reviewed.
- [x] Deterministic CLI-native implementation added without new dependencies.
- [x] Artifact, evidence, policy, workflow/action, judge, and audit boundaries preserved.
- [x] Scenario report saved under `reports/scenario/`.
- [x] Verification matrix updated for only the 21 covered rows.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-BRAIN-001 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, provider switch continuity |
| CS-BRAIN-002 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, route factor record |
| CS-BRAIN-003 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, allowed local override and denied external override |
| CS-BRAIN-004 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, Brain Performance Ledger |
| CS-BRAIN-005 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, static capability registry fallback |
| CS-BRAIN-006 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, namespace-local ledger and opt-in aggregation |
| CS-BRAIN-007 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, high-risk ensemble trigger |
| CS-BRAIN-008 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, routine single-brain route |
| CS-BRAIN-009 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, supporting judge record |
| CS-BRAIN-010 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, objective evidence override |
| CS-BRAIN-011 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, owner acceptance record |
| CS-BRAIN-012 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, governed judge recommendation candidate |
| CS-BRAIN-013 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, disagreement adjudication |
| CS-BRAIN-014 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, high-risk escalation card |
| CS-BRAIN-015 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, calibration and bias tracking |
| CS-BRAIN-016 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, auditable routing decision |
| CS-ARCH-012 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, evidence and mission readable after switch |
| CS-NS-009 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, namespace-local brain ledger |
| CS-NS-010 | MUST_PASS | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, workspace policy routing control |
| CS-REG-009 | REGRESSION_GUARD | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, provider swap evidence continuity |
| CS-REG-010 | REGRESSION_GUARD | PASS | `reports/scenario/full-brain-routing-2026-06-10.json`, judge cannot become unquestionable authority |

## Human Required

No human-required item was introduced for this local deterministic batch. Production UI/API review, real model/provider review, real Ollama generation-quality review, real tool execution review, and production tenant/security review remain future human-required surfaces before production PASS.

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify full-brain-routing --json --output reports/scenario/full-brain-routing-2026-06-10.json
# status: success
# scenario_set: full-brain-routing
# summary.pass: 21
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_FULL_BRAIN_ROUTING_ONLY
# brain_evidence.model_count: 5
# brain_evidence.override_denied_exit_code: 8
# brain_evidence.aggregation_denied_exit_code: 8
# brain_evidence.personal_ledger_entry_count: 3
# brain_evidence.org_ledger_entry_count: 1
# negative_evidence: all integer counters are 0
```

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` now marks `CS-BRAIN-001` through `CS-BRAIN-016`, `CS-ARCH-012`, `CS-NS-009`, `CS-NS-010`, `CS-REG-009`, and `CS-REG-010` as `PASS`.

Current full matrix after this batch:

- `PASS`: 167
- `NOT_VERIFIED`: 39
- `FAIL`: 0
- `NOT_RUN`: 0

## Gaps And Risks

- Full 206-scenario PASS remains incomplete.
- Production UI/API surfaces remain missing; this batch uses CLI-native JSON as scaffold evidence.
- The model registry includes external provider metadata, but no real external provider call is made.
- Ollama/local LLM availability is a smoke check only and does not determine PASS.
- LLM-as-judge records are supporting evidence, not scenario PASS authority.
- Cross-namespace brain performance aggregation is only allowed in a deterministic opt-in test path.
