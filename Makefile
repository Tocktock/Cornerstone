.PHONY: verify-docs verify-scenario-matrix verify-scaffold-cli verify-vs0-runtime verify-local-fast

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

verify-local-fast: verify-docs verify-scenario-matrix verify-scaffold-cli
