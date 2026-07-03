.PHONY: verify-docs verify-scenario-matrix verify-scaffold-cli verify-vs0-runtime verify-vs0-acceptance verify-vs0-evux verify-vs0-operator-ui verify-vs1-ontology verify-vs4-product-alpha-shell verify-vs4-product-alpha-brief-detail verify-vs4-product-alpha-slice-003 verify-vs4-product-alpha-human-package verify-vs4-product-alpha-ux-polish-learn verify-vs4-product-alpha-responsive-mobile verify-vs4-product-alpha-keyboard-focus verify-vs4-product-alpha-ask-readability verify-vs4-product-alpha-decision-pages verify-vs4-product-alpha-ask-injection-boundary verify-vs4-product-alpha-ops-inbox-triage verify-vs4-product-alpha-action-execution-boundary verify-vs4-product-alpha-desktop-overflow verify-vs4-product-alpha-human-review-handoff verify-vs4-product-alpha-gate-integrity verify-vs4-product-alpha-evidence-audit-detail verify-vs4-product-alpha-user-drop-ask-source verify-vs4-product-alpha-drop-ask-trust-boundary verify-vs4-product-alpha-interactive-ops-inbox verify-vs4-product-alpha-runtime-backed-ops-inbox verify-vs4-product-alpha-runtime-loop-coherence verify-vs2-local-range verify-vs2-production-like verify-vs2-security verify-connector-contract-adapter generate-connectorhub-human-gate-artifacts generate-connectorhub-engineering-trail-manifest verify-connectorhub-engineering-trail verify-local-fast

verify-docs:
	scripts/verify_sot_docs.sh

verify-scenario-matrix:
	python3 scripts/generate_scenario_verification_matrix.py --check
	python3 scripts/verify_scenario_matrix.py

verify-scaffold-cli:
	scripts/verify_scaffold_cli.sh

verify-vs0-runtime:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs0-product-runtime --json --output reports/scenario/vs0-product-runtime-2026-06-11.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs0-product-runtime-2026-06-11.json --json
	python3 -m unittest tests.scenario.test_scaffold_cli

verify-vs0-acceptance:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs0-runtime-acceptance --json --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs0-runtime-acceptance-2026-06-11.json --json
	PATH="$(PWD):$$PATH" cornerstone release evidence collect --scope vs0-runtime-acceptance --json

verify-vs0-evux:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs0-evux --json --output reports/scenario/vs0-evux-2026-06-13.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs0-evux-2026-06-13.json --json
	PATH="$(PWD):$$PATH" cornerstone quickstart verify vs0-evux --json --output reports/quickstart/vs0-evux-quickstart.json
	PATH="$(PWD):$$PATH" cornerstone release evidence collect --scope vs0-evux --json

verify-vs0-operator-ui:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs0-operator-acceptance-ui --json --output reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json --json

verify-vs1-ontology:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs1-ontology-suggest-promote --json --output reports/scenario/vs1-ontology-suggest-promote-2026-06-15.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs1-ontology-suggest-promote-2026-06-15.json --json

verify-vs4-product-alpha-shell:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-012 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-REG-003 --scenario VS4-REG-006 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-brief-detail:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-UI-002 --scenario VS4-UI-003 --scenario VS4-UI-004 --scenario VS4-UI-005 --scenario VS4-UI-006 --scenario VS4-UI-007 --scenario VS4-UI-008 --scenario VS4-UI-009 --scenario VS4-UI-010 --scenario VS4-UI-011 --scenario VS4-REF-002 --scenario VS4-REG-004 --scenario VS4-REG-005 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-slice-003:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-UI-013 --scenario VS4-UI-014 --scenario VS4-STATE-001 --scenario VS4-REF-001 --scenario VS4-REG-001 --scenario VS4-REG-002 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-human-package:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json
	PATH="$(PWD):$$PATH" cornerstone human-gate package --scope vs4 --scenario-report reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --record-template-output reports/human-gates/vs4/record-templates/VS4-H01.review-record.template.json --json --output reports/human-gates/vs4/review-kit.json

verify-vs4-product-alpha-ux-polish-learn:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-responsive-mobile:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-keyboard-focus:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-ask-readability:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-decision-pages:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-ask-injection-boundary:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-007 --scenario VS4-UI-009 --scenario VS4-UI-011 --scenario VS4-UI-013 --scenario VS4-REG-004 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-ops-inbox-triage:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-012 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-001 --scenario VS4-REG-003 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-action-execution-boundary:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-007 --scenario VS4-UI-009 --scenario VS4-UI-010 --scenario VS4-UI-011 --scenario VS4-REF-002 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-desktop-overflow:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-004 --scenario VS4-UI-010 --scenario VS4-UI-011 --scenario VS4-REF-002 --scenario VS4-REG-003 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-human-review-handoff:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-012 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REG-003 --scenario VS4-REG-005 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-gate-integrity:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json

verify-vs4-product-alpha-evidence-audit-detail:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-005 --scenario VS4-UI-012 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REG-003 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-user-drop-ask-source:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-002 --scenario VS4-UI-003 --scenario VS4-UI-004 --scenario VS4-UI-013 --scenario VS4-UI-016 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json

verify-vs4-product-alpha-drop-ask-trust-boundary:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-002 --scenario VS4-UI-007 --scenario VS4-UI-009 --scenario VS4-UI-011 --scenario VS4-UI-013 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json

verify-vs4-product-alpha-interactive-ops-inbox:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-001 --scenario VS4-REG-003 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json

verify-vs4-product-alpha-runtime-backed-ops-inbox:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-008 --scenario VS4-UI-009 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-015 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-001 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json

verify-vs4-product-alpha-runtime-loop-coherence:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --scenario VS4-GATE-001 --scenario VS4-UI-001 --scenario VS4-UI-008 --scenario VS4-UI-009 --scenario VS4-UI-010 --scenario VS4-UI-011 --scenario VS4-UI-012 --scenario VS4-UI-013 --scenario VS4-UI-016 --scenario VS4-STATE-001 --scenario VS4-REF-002 --scenario VS4-REG-003 --scenario VS4-REG-004 --scenario VS4-REG-005 --scenario VS4-REG-006 --scenario VS4-REG-007 --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs4-product-alpha-ui-daily-loop-2026-07-03.json --json --output reports/scenario/vs4-product-alpha-ui-daily-loop-gate-2026-07-03.json

verify-vs2-security:
	mkdir -p reports/scenario reports/security
	PATH="$(PWD):$$PATH" cornerstone security sensitive-change-test --category vs2_policy_tenancy_egress --json > reports/scenario/vs2-sensitive-change-gate-2026-06-19.json
	PATH="$(PWD):$$PATH" cornerstone security vs2-h01-approval-package --json > reports/scenario/vs2-h01-approval-package-2026-06-19.json
	PATH="$(PWD):$$PATH" cornerstone security vs2-local-range --json > reports/security/vs2-local-range-command.json
	PATH="$(PWD):$$PATH" cornerstone security vs2-local-proof --reuse-local-range-report reports/security/vs2-local-range.json --json > reports/security/vs2-local-security-proof-command.json || true
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json --output reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json || true
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json --json

verify-vs2-local-range:
	mkdir -p reports/security
	PATH="$(PWD):$$PATH" cornerstone security vs2-local-range --json > reports/security/vs2-local-range-command.json

verify-vs2-production-like:
	mkdir -p reports/security
	PATH="$(PWD):$$PATH" cornerstone security vs2-production-like-integration --json > reports/security/vs2-production-like-integration-command.json

verify-connector-contract-adapter:
	PATH="$(PWD):$$PATH" cornerstone security vs2-local-proof --json > reports/security/vs2-local-security-proof-command.json
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json --output reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-001 --json --output reports/scenario/connector-contract-adapter-cs-ch-001-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-001-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-002 --json --output reports/scenario/connector-contract-adapter-cs-ch-002-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-002-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-003 --json --output reports/scenario/connector-contract-adapter-cs-ch-003-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-003-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-004 --json --output reports/scenario/connector-contract-adapter-cs-ch-004-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-004-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-005 --json --output reports/scenario/connector-contract-adapter-cs-ch-005-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-005-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-006 --json --output reports/scenario/connector-contract-adapter-cs-ch-006-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-006-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-007 --json --output reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-007-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-008 --json --output reports/scenario/connector-contract-adapter-cs-ch-008-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-008-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-009 --json --output reports/scenario/connector-contract-adapter-cs-ch-009-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-009-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-010 --json --output reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-010-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-011 --json --output reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-011-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-012 --json --output reports/scenario/connector-contract-adapter-cs-ch-012-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-012-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-013 --json --output reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-013-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-014 --json --output reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-014-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-015 --json --output reports/scenario/connector-contract-adapter-cs-ch-015-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-015-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-016 --json --output reports/scenario/connector-contract-adapter-cs-ch-016-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-016-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-017 --json --output reports/scenario/connector-contract-adapter-cs-ch-017-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-017-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-018 --json --output reports/scenario/connector-contract-adapter-cs-ch-018-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-018-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-019 --json --output reports/scenario/connector-contract-adapter-cs-ch-019-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-019-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-020 --json --output reports/scenario/connector-contract-adapter-cs-ch-020-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-020-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-021 --json --output reports/scenario/connector-contract-adapter-cs-ch-021-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-021-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-022 --json --output reports/scenario/connector-contract-adapter-cs-ch-022-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-022-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-023 --json --output reports/scenario/connector-contract-adapter-cs-ch-023-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-023-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-024 --json --output reports/scenario/connector-contract-adapter-cs-ch-024-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-024-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-025 --json --output reports/scenario/connector-contract-adapter-cs-ch-025-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-025-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-026 --json --output reports/scenario/connector-contract-adapter-cs-ch-026-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-026-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-027 --json --output reports/scenario/connector-contract-adapter-cs-ch-027-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-027-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-028 --json --output reports/scenario/connector-contract-adapter-cs-ch-028-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-028-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-029 --json --output reports/scenario/connector-contract-adapter-cs-ch-029-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-029-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-030 --json --output reports/scenario/connector-contract-adapter-cs-ch-030-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-030-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-031 --json --output reports/scenario/connector-contract-adapter-cs-ch-031-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-031-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-032 --json --output reports/scenario/connector-contract-adapter-cs-ch-032-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-032-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-033 --json --output reports/scenario/connector-contract-adapter-cs-ch-033-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-033-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-034 --json --output reports/scenario/connector-contract-adapter-cs-ch-034-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-034-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-035 --json --output reports/scenario/connector-contract-adapter-cs-ch-035-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-035-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json --output reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-037 --json --output reports/scenario/connector-contract-adapter-cs-ch-037-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-037-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-038 --json --output reports/scenario/connector-contract-adapter-cs-ch-038-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-038-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-039 --json --output reports/scenario/connector-contract-adapter-cs-ch-039-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-039-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --scenario CS-CH-040 --json --output reports/scenario/connector-contract-adapter-cs-ch-040-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-cs-ch-040-2026-06-23.json --json
	PATH="$(PWD):$$PATH" cornerstone scenario verify connector-contract-adapter --json --output reports/scenario/connector-contract-adapter-2026-06-23.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter-2026-06-23.json --json
	python3 scripts/compact_connectorhub_reports.py --delete-sources
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/connector-contract-adapter/aggregate-2026-06-23.json --json
	for report in reports/scenario/connector-contract-adapter/scenarios/CS-CH-*.json; do \
		PATH="$(PWD):$$PATH" cornerstone scenario gate "$$report" --json; \
	done
	python3 -m unittest tests.scenario.test_connectorhub_cli

verify-connectorhub-engineering-trail:
	python3 scripts/verify_connectorhub_engineering_trail.py

generate-connectorhub-human-gate-artifacts:
	mkdir -p reports/scenario
	for scenario in CS-CH-H01 CS-CH-H02 CS-CH-H03 CS-CH-H04 CS-CH-H05 CS-CH-H06 CS-CH-H07; do \
		lower=$$(printf "%s" "$$scenario" | tr '[:upper:]' '[:lower:]'); \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate package --scenario "$$scenario" --state-dir tmp/manual-connector-human-gates --json --output "reports/scenario/connectorhub-human-gate-package-$${lower}-2026-06-24.json" >/dev/null; \
	done
	PATH="$(PWD):$$PATH" cornerstone connector human-gate field-ref-contract --scenario CS-CH-H04 --state-dir tmp/manual-connector-human-gates --json --output reports/scenario/connectorhub-human-gate-field-ref-contract-cs-ch-h04-2026-06-24.json >/dev/null
	for scenario in CS-CH-H01 CS-CH-H02 CS-CH-H03 CS-CH-H04 CS-CH-H05 CS-CH-H06 CS-CH-H07; do \
		lower=$$(printf "%s" "$$scenario" | tr '[:upper:]' '[:lower:]'); \
		case "$$scenario" in \
			CS-CH-H01) packet_dir="<h01-github-readonly-packet-dir>" ;; \
			CS-CH-H02) packet_dir="<h02-macos-permission-packet-dir>" ;; \
			CS-CH-H03) packet_dir="<h03-chrome-privacy-packet-dir>" ;; \
			CS-CH-H04) packet_dir="<h04-acceptance-packet-dir>" ;; \
			CS-CH-H05) packet_dir="<h05-live-action-packet-dir>" ;; \
			CS-CH-H06) packet_dir="<h06-usability-trust-packet-dir>" ;; \
			CS-CH-H07) packet_dir="<h07-recovery-packet-dir>" ;; \
		esac; \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate evidence-packet-contract --scenario "$$scenario" --state-dir tmp/manual-connector-human-gates --json --output "reports/scenario/connectorhub-human-gate-evidence-packet-contract-$${lower}-2026-06-24.json" >/dev/null; \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate evidence-packet-file-contract --scenario "$$scenario" --state-dir tmp/manual-connector-human-gates --json --output "reports/scenario/connectorhub-human-gate-evidence-packet-file-contract-$${lower}-2026-06-24.json" >/dev/null; \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate evidence-packet-scaffold --scenario "$$scenario" --packet-dir "$$packet_dir" --state-dir tmp/manual-connector-human-gates --json --output "reports/scenario/connectorhub-human-gate-evidence-packet-scaffold-$${lower}-2026-06-24.json" >/dev/null; \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate evidence-packet-validate --scenario "$$scenario" --packet-dir "$$packet_dir" --state-dir tmp/manual-connector-human-gates --json --output "reports/scenario/connectorhub-human-gate-evidence-packet-validation-$${lower}-2026-06-24.json" >/dev/null; status=$$?; if test $$status -ne 1; then exit 1; fi; \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate evidence-packet-record-draft --scenario "$$scenario" --packet-dir "$$packet_dir" --state-dir tmp/manual-connector-human-gates --json --output "reports/scenario/connectorhub-human-gate-evidence-packet-record-draft-$${lower}-2026-06-24.json" >/dev/null; status=$$?; if test $$status -ne 1; then exit 1; fi; \
	done
	PATH="$(PWD):$$PATH" cornerstone connector human-gate preflight-bundle --scenario CS-CH-H04 --state-dir tmp/manual-connector-human-gates --json --output reports/scenario/connectorhub-human-gate-preflight-bundle-cs-ch-h04-2026-06-24.json >/dev/null
	PATH="$(PWD):$$PATH" cornerstone connector human-gate report --state-dir tmp/manual-connector-human-gates --json --output reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json >/dev/null
	PATH="$(PWD):$$PATH" cornerstone connector human-gate next --state-dir tmp/manual-connector-human-gates --json --output reports/scenario/connectorhub-human-gate-next-2026-06-24.json >/dev/null
	PATH="$(PWD):$$PATH" cornerstone connector human-gate validation-handoff --state-dir tmp/manual-connector-human-gates --json --output reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json >/dev/null
	for scenario in CS-CH-H01 CS-CH-H02 CS-CH-H03 CS-CH-H04 CS-CH-H05 CS-CH-H06 CS-CH-H07; do \
		lower=$$(printf "%s" "$$scenario" | tr '[:upper:]' '[:lower:]'); \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate package --scenario "$$scenario" --state-dir tmp/manual-connector-human-gate-template-checks --json --record-template-output "reports/scenario/connectorhub-human-gate-record-template-$${lower}-2026-06-24.json" >/dev/null; \
		PATH="$(PWD):$$PATH" cornerstone connector human-gate validate-record --scenario "$$scenario" --record-file "reports/scenario/connectorhub-human-gate-record-template-$${lower}-2026-06-24.json" --state-dir tmp/manual-connector-human-gate-template-checks --json --output "reports/scenario/connectorhub-human-gate-validation-blank-$${lower}-2026-06-24.json" >/dev/null; status=$$?; if test $$status -ne 1; then exit 1; fi; \
	done

generate-connectorhub-engineering-trail-manifest: generate-connectorhub-human-gate-artifacts
	python3 scripts/generate_connectorhub_engineering_trail_manifest.py

verify-local-fast: verify-docs verify-scenario-matrix verify-scaffold-cli
