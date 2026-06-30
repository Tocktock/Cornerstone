# VS3 Scenario Gate Negative Evidence Coverage Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-REG-003, VS3-REG-005, and VS3-REG-008.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent a successful VS3 local/dev scenario report from passing `cornerstone scenario gate` when it preserves only a partial top-level `negative_evidence` counter set.

In scope:
- Required top-level VS3 negative-evidence counter inventory.
- Stable gate failure code `CS_VS3_NEGATIVE_EVIDENCE_COVERAGE_INVALID`.
- Focused regression test for the one-counter tamper case.
- Direct before/after CLI tamper probes.

Out of scope:
- New VS3 runtime behavior.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX behavior.
- Any production, live-provider, real-network, real-IdP, migration, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-008`.
- `unchanged_ai_rows`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-004`, `VS3-REG-006`, `VS3-REG-007`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Before Evidence

Pre-change tamper probe:

```bash
python3 - <<'PY'
import json, subprocess
from pathlib import Path

source = Path('/tmp/cs-vs3-before-coverage-source.json')
tampered = Path('/tmp/cs-vs3-before-coverage-one-counter.json')
seed = subprocess.run(
    ['./cornerstone', 'scenario', 'verify', 'vs3-onprem-trusted-extension', '--json', '--output', str(source)],
    text=True,
    capture_output=True,
)
print('seed exit', seed.returncode)
data = json.loads(source.read_text())
data['negative_evidence'] = {'untrusted_content_egress_calls': 0}
tampered.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
gate = subprocess.run(['./cornerstone', 'scenario', 'gate', str(tampered), '--json'], text=True, capture_output=True)
payload = json.loads(gate.stdout)
print('tampered exit', gate.returncode)
print('tampered status', payload.get('status'))
print('errors', [error.get('code') for error in payload.get('errors', [])])
print('negative_validation', payload.get('negative_evidence_validation'))
PY
```

Observed result:

```text
seed exit 0
tampered exit 0
tampered status success
errors []
negative_validation {'count': 1, 'malformed_entries': {}, 'nonzero_entries': {}, 'present': True, 'required': True, 'status': 'passed'}
```

Interpretation:
- The gate correctly required non-empty zero-valued top-level `negative_evidence`.
- The gate incorrectly accepted a successful VS3 local/dev report after 189 of 190 required safety counters were removed.

## Change

The VS3 scenario gate now validates required top-level negative-evidence coverage:

- `VS3_REQUIRED_NEGATIVE_EVIDENCE_KEYS` defines the current 190 required safety counters.
- `negative_evidence_validation` records `required_key_count`, `observed_key_count`, `missing_required_keys`, and `unexpected_keys`.
- Missing required keys fail the gate with `CS_VS3_NEGATIVE_EVIDENCE_COVERAGE_INVALID`.
- Extra counters remain visible as `unexpected_keys` but do not fail the gate by themselves.
- Existing missing, malformed, or nonzero counter validation still uses `CS_VS3_NEGATIVE_EVIDENCE_INVALID`.

## After Evidence

Syntax check:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result:

```text
exit 0
```

Focused new unittest:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_incomplete_negative_evidence
```

Result:

```text
Ran 1 test in 0.579s
OK
```

Adjacent negative-evidence unittest set:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_incomplete_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_nonzero_negative_evidence
```

Result:

```text
Ran 3 tests in 1.855s
OK
```

Broader VS3 scenario-gate unittest subset:

```bash
python3 - <<'PY'
import unittest
from tests.scenario.test_scaffold_cli import ScaffoldCliTests

names = sorted(name for name in dir(ScaffoldCliTests) if name.startswith('test_vs3_scenario_gate'))
suite = unittest.TestSuite(ScaffoldCliTests(name) for name in names)
print('selected', len(names))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
```

Result:

```text
Ran 38 tests in 21.509s
OK
selected 38
```

Untampered canonical CLI probe after the change:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
```

Observed result:

```text
verify status success
verify summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate status success
gate negative_validation {'count': 190, 'malformed_entries': {}, 'missing_required_keys': [], 'nonzero_entries': {}, 'observed_key_count': 190, 'present': True, 'required': True, 'required_key_count': 190, 'status': 'passed', 'unexpected_keys': []}
gate errors []
```

Direct one-counter tamper probe after the change:

```text
tampered exit 4
tampered status failed
errors ['CS_VS3_NEGATIVE_EVIDENCE_COVERAGE_INVALID']
negative_validation {'count': 1, 'malformed_entries': {}, 'nonzero_entries': {}, 'observed_key_count': 1, 'present': True, 'required': True, 'required_key_count': 190, 'status': 'failed', 'unexpected_keys': []}
```

## Remaining Proof Surfaces

- VS3-H01 through VS3-H07 remain `HUMAN_REQUIRED`.
- This checkpoint proves only local scenario-gate hardening for top-level VS3 negative-evidence coverage completeness.
- It does not replace scenario-specific runtime proof for RequestContext, Postgres/RLS, OPA, egress sandboxing, ConnectorHub source policy, trusted tool registry, operator status, or final VS0/VS1 regression gates.

## Decision

Continue VS3 in the next small verified slice. Do not widen this checkpoint into VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human acceptance.
