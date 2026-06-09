.PHONY: verify-docs verify-scenario-matrix verify-scaffold-cli verify-local-fast

verify-docs:
	scripts/verify_sot_docs.sh

verify-scenario-matrix:
	python3 scripts/generate_scenario_verification_matrix.py --check
	python3 scripts/verify_scenario_matrix.py

verify-scaffold-cli:
	scripts/verify_scaffold_cli.sh

verify-local-fast: verify-docs verify-scenario-matrix verify-scaffold-cli
