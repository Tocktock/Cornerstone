TEST_DB_NAME ?= cornerstone_test
BROWSER_TEST_DB_NAME ?= cornerstone_browser_test
TEST_DB_PORT ?= 55433
TEST_DB_URL ?= postgresql+psycopg://cornerstone:cornerstone@localhost:$(TEST_DB_PORT)/$(TEST_DB_NAME)
BROWSER_TEST_DB_URL ?= postgresql+psycopg://cornerstone:cornerstone@localhost:$(TEST_DB_PORT)/$(BROWSER_TEST_DB_NAME)
COMPOSE_TEST := COMPOSE_PROJECT_NAME=cornerstone-test docker compose -f compose.test.yml

.PHONY: test-stack-up test-stack-down ensure-test-db ensure-browser-db lint typecheck backend-fast backend-integration symptoms corpus-smoke

test-stack-up:
	$(COMPOSE_TEST) up -d --wait db

test-stack-down:
	$(COMPOSE_TEST) down -v

ensure-test-db: test-stack-up
	$(COMPOSE_TEST) exec -T db sh -lc 'psql -U "$$POSTGRES_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '\''$(TEST_DB_NAME)'\''" | grep -q 1 || psql -U "$$POSTGRES_USER" -d postgres -c "CREATE DATABASE $(TEST_DB_NAME)"'

ensure-browser-db: test-stack-up
	$(COMPOSE_TEST) exec -T db sh -lc 'psql -U "$$POSTGRES_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '\''$(BROWSER_TEST_DB_NAME)'\''" | grep -q 1 || psql -U "$$POSTGRES_USER" -d postgres -c "CREATE DATABASE $(BROWSER_TEST_DB_NAME)"'

lint:
	. .venv/bin/activate && ruff check backend/src backend/tests

typecheck:
	cd frontend && npm run build

backend-fast:
	$(MAKE) ensure-test-db
	. .venv/bin/activate && CORNERSTONE_TEST_DATABASE_URL='$(TEST_DB_URL)' pytest backend/tests/domain -q

backend-integration:
	$(MAKE) ensure-test-db
	. .venv/bin/activate && CORNERSTONE_TEST_DATABASE_URL='$(TEST_DB_URL)' pytest backend/tests/integration backend/tests/contract backend/tests/smoke -q -m 'not corpus'

symptoms:
	$(MAKE) ensure-browser-db
	cd frontend && CORNERSTONE_BROWSER_TEST_DATABASE_URL='$(BROWSER_TEST_DB_URL)' npm run test:symptoms

corpus-smoke:
	$(MAKE) ensure-test-db
	. .venv/bin/activate && CORNERSTONE_TEST_DATABASE_URL='$(TEST_DB_URL)' CORNERSTONE_RUN_CORPUS_SMOKE=1 pytest backend/tests/smoke -q -m corpus
