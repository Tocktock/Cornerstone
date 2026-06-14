.PHONY: verify-docs verify-scenario-matrix verify-scaffold-cli verify-vs0-runtime verify-vs0-acceptance verify-vs0-evux verify-vs0-operator-ui verify-local-fast

verify-docs:
	scripts/verify_sot_docs.sh

verify-scenario-matrix:
	python3 scripts/generate_scenario_verification_matrix.py --check
	python3 scripts/verify_scenario_matrix.py

verify-scaffold-cli:
	scripts/verify_scaffold_cli.sh

verify-vs0-runtime:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs0-product-runtime --output reports/scenario/vs0-product-runtime-2026-06-11.json
	PATH="$(PWD):$$PATH" cornerstone scenario gate reports/scenario/vs0-product-runtime-2026-06-11.json --json
	python3 -m unittest tests.scenario.test_scaffold_cli

verify-vs0-acceptance:
	PATH="$(PWD):$$PATH" cornerstone scenario verify vs0-runtime-acceptance --output reports/scenario/vs0-runtime-acceptance-2026-06-11.json
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

verify-local-fast: verify-docs verify-scenario-matrix verify-scaffold-cli
