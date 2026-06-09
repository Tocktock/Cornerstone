.PHONY: verify-docs verify-scaffold-cli verify-local-fast

verify-docs:
	scripts/verify_sot_docs.sh

verify-scaffold-cli:
	scripts/verify_scaffold_cli.sh

verify-local-fast: verify-docs verify-scaffold-cli
