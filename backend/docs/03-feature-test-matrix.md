# 03 — Feature Test Matrix

| Feature | Unit tests | Integration tests | Status |
| --- | --- | --- | --- |
| Honest production empty state | N/A | `test_empty_production_state_has_no_demo_context` | Implemented |
| Pending OAuth source not real connected source | N/A | `test_pending_auth_source_does_not_count_as_real_connected_source` | Implemented |
| Source registration | Source route integration | `test_legacy_source_sync_route_is_removed` | Implemented |
| OAuth completion | Source route integration | `test_oauth_completion_does_not_allow_manual_sync_for_notion` | Implemented |
| Sync to Artifact | Extraction/hash behavior | `test_sync_creates_artifact_and_evidence_with_provenance` | Implemented |
| Sync idempotency | Store/source route behavior | `test_sync_is_idempotent_for_same_source_external_id_and_hash` | Implemented |
| Sync rollback/degraded state | Store transaction behavior | `test_failed_sync_after_prior_success_marks_degraded_and_rolls_back` | Implemented |
| Required Evidence provenance | `test_extracted_evidence_contains_required_provenance` | Sync provenance assertions | Implemented |
| Freshness state | `test_freshness_policy_*` | Sync response assertions | Implemented |
| Evidence review | Review request validation | `test_authorized_reviewer_can_mark_evidence_reviewed` | Implemented |
| Unauthorized evidence review blocked | Reviewer auth service | `test_unauthorized_reviewer_cannot_review_evidence` | Implemented |
| Concept candidate creation | Concept route integration | `test_concept_list_and_read_return_created_candidate` | Implemented |
| Missing evidence blocks Concept creation | Store lookup behavior | `test_create_concept_with_missing_evidence_returns_not_found` | Implemented |
| Fake DecisionRecord blocks Concept creation | Store lookup behavior | `test_create_concept_with_fake_decision_record_id_returns_not_found` | Implemented |
| Unsupported officialization blocked | `test_officialization_rejects_unsupported_concept` | `test_officialization_without_support_returns_conflict` | Implemented |
| Unreviewed evidence officialization blocked | `test_officialization_rejects_unreviewed_evidence` | `test_officialization_with_unreviewed_evidence_returns_conflict` | Implemented |
| Non-production evidence officialization blocked | `test_officialization_rejects_non_production_source_evidence` | `test_non_production_source_evidence_cannot_officialize` | Implemented |
| Unauthorized officialization blocked | `test_officialization_rejects_unauthorized_reviewer` | `test_unauthorized_reviewer_cannot_officialize` | Implemented |
| Reviewed evidence can officialize | `test_officialization_accepts_reviewed_evidence_supported_concept` | `test_create_and_officialize_supported_concept` | Implemented |
| DecisionRecord requires reviewed evidence | Officialization service decision validation | `test_decision_record_creation_requires_reviewed_evidence` | Implemented |
| DecisionRecord can support officialization | Officialization service decision validation | `test_decision_record_can_support_concept_officialization` | Implemented |
| Grounded unsupported response | Grounded context unit tests | `test_unknown_query_returns_unsupported` | Implemented |
| Grounded official response | Grounded context unit tests | `test_official_concept_query_returns_citation` | Implemented |
| Unknown freshness not official | `test_official_concept_with_unknown_freshness_is_not_labeled_official` | Planned API fixture | Implemented unit |
| Production excludes demo evidence | `test_production_mode_excludes_non_production_source_evidence` | Covered by officialization failure path | Implemented |
| Structured JSON logging | `test_log_event_emits_parseable_structured_json` | source/officialization/context log tests | Implemented |
| Audit event API | Audit route tests | review, blocked officialization, officialization events | Implemented |
| ConceptRelation officialization | `test_relation_officialization_rejects_candidate_concepts`, `test_relation_officialization_accepts_reviewed_evidence_supported_relation` | `test_relation_creation_and_officialization_requires_reviewed_support`, `test_sqlalchemy_store_persists_concept_relation_officialization` | Implemented v0.10.0 |
| Real Notion connector | Notion adapter/gateway tests | Mocked OAuth/discovery/selection/sync tests | Partial |
| Unsupported provider object selection blocked | `test_notion_mock_discovery_returns_snapshots_for_selection` | `test_source_selection_rejects_unsupported_notion_database_object` | Implemented |
| `all_accessible` selects only syncable objects | Route/service behavior | `test_all_accessible_selection_expands_only_to_syncable_objects` | Implemented |
| Workspace-limited selection blocked until implemented | Request validation | `test_workspace_limited_selection_is_rejected_until_implemented` | Implemented |
| Production config fails closed | `test_production_mode_reports_all_unsafe_default_config_values`, `test_safe_production_runtime_config_has_no_issues` | `test_create_app_fails_closed_when_production_config_is_unsafe`, `test_worker_cli_fails_closed_when_production_env_is_unsafe` | Implemented |
| Atomic worker success writes | Service transaction boundary | `test_sync_success_writes_roll_back_when_cursor_advance_fails`, `test_persistent_sync_success_writes_roll_back_when_cursor_advance_fails` | Implemented |
| Worker lease claim primitives | `test_in_memory_store_claims_sync_job_once_with_worker_lease`, `test_sync_job_table_has_worker_lease_and_enqueue_primitives` | `test_worker_claim_event_records_worker_identity_and_clears_lease_on_success`, `test_already_claimed_job_is_skipped_by_second_worker` | Implemented |
| Scheduled enqueue idempotency primitive | `test_schedule_enqueue_key_is_stable_and_duplicate_jobs_are_rejected` | Scheduler due-job tests | Implemented |
| SQLAlchemy/PostgreSQL adapter | Model and migration tests plus SQLite-backed store tests | Live PostgreSQL tests are gated operator-run checks | Partial; live PostgreSQL execution pending |
| Evaluation framework | Planned | Eval task tests planned | Not started |

## P0 pass/fail checklist

- [x] No source connected → honest empty state.
- [x] OAuth-backed source can be `pending_auth`.
- [x] Pending auth does not count as connected production source.
- [x] Sync is blocked before OAuth completion.
- [x] Sync creates Artifacts.
- [x] Sync creates EvidenceFragments.
- [x] Evidence includes required provenance.
- [x] Freshness state is visible.
- [x] Sync is idempotent for unchanged source object hash.
- [x] Failed sync after previous success marks source degraded.
- [x] Failed sync rolls back partial Artifact/Evidence writes.
- [x] Evidence can be reviewed or rejected by authorized reviewers.
- [x] Concept can be created from evidence.
- [x] Concept cannot become official without support.
- [x] Concept cannot become official with unreviewed evidence.
- [x] Concept cannot become official using non-production evidence in production mode.
- [x] Fake DecisionRecord ID cannot be used.
- [x] DecisionRecord requires reviewed evidence.
- [x] Grounded query can return unsupported.
- [x] Official answer cannot use unknown freshness as official.
- [x] Manual source sync emits structured logs.
- [x] Review and officialization emit structured logs.
- [x] Context query emits trust-label logs.
- [x] Review/officialization audit events are visible through API.
- [x] Evidence review queue returns source/artifact context and suggested reviewer actions.
- [x] Reviewers can create Concept candidates from EvidenceFragments.
- [x] Reviewers can mark EvidenceFragments as conflicted.
- [x] ConceptRelations can be created, persisted, read, listed, and officialized.
- [x] Official ConceptRelations require official source/target Concepts plus reviewed evidence or a valid DecisionRecord.
- [x] Provider-backed manual sync bypass is removed.
- [x] Unsupported Notion database/data_source selection is blocked before sync.
- [x] `all_accessible` selection expands only to ingestion-supported provider objects.

- [x] Production mode refuses unsafe local/default runtime config.
- [x] API and worker process startup share the same fail-closed config guard.
- [x] Worker sync success writes roll back Artifact/Evidence/source changes if cursor/job success writes fail.
- [x] Sync workers claim jobs before provider work starts.
- [x] Claimed jobs expose worker lease metadata and clear it on terminal states.
- [x] Scheduled jobs have enqueue idempotency keys for v0.9 PostgreSQL concurrency verification.

| Worker active lease protection | `test_in_memory_store_rejects_active_running_job_claim_by_second_worker` | `test_already_claimed_job_is_skipped_by_second_worker` | Implemented v0.9.0 |
| Expired lease recovery | `test_in_memory_store_reclaims_expired_running_job_lease` | `test_expired_running_job_lease_can_be_reclaimed_by_second_worker` | Implemented v0.9.0 |
| Worker lease heartbeat | `test_in_memory_store_lease_heartbeat_requires_owner` | `test_sync_job_lease_heartbeat_extends_active_worker_lease` | Implemented v0.9.0 |
| Live PostgreSQL worker concurrency | `test_sync_job_table_has_live_postgres_claim_columns_and_indexes` | `tests/postgres/test_live_postgres_worker_concurrency.py` | Gated live test || Strict live PostgreSQL runner | `test_live_postgres_report_rejects_skipped_tests_when_required`, `test_live_postgres_environment_accepts_psycopg_url` | `scripts/run_live_postgres_tests.py` and GitHub Actions PostgreSQL service | Implemented v0.9.1 |
| Live PostgreSQL heartbeat ownership | Planned local unit check through lease heartbeat primitives | `test_live_postgres_heartbeat_requires_owner_and_extends_lease` | Gated live test v0.9.1 |
| Live PostgreSQL duplicate scheduled enqueue key | Existing duplicate-key primitive test | `test_live_postgres_duplicate_scheduled_enqueue_key_is_rejected` | Gated live test v0.9.1 |
| Stable mypy verification | N/A | `./scripts/run_tests.sh` writes deterministic `reports/mypy-report.txt` | Implemented v0.9.1 |
## v0.9.2 Live Notion E2E Matrix

| Feature | Test coverage | Status |
|---|---|---:|
| Notion page snapshot lookup | unit | done |
| E2E config fail-closed behavior | unit | done |
| Live page → Artifact → EvidenceFragment pilot | gated live_notion | ready, not run by default |



## v0.10.0 Evidence review and officialization matrix

| Feature | Unit coverage | Integration coverage | Status |
|---|---|---|---:|
| Evidence review queue | helper/action state tests through API | `test_review_queue_lists_unreviewed_evidence_with_source_context` | implemented |
| Evidence conflict state | API/state validation | `test_reviewer_can_mark_evidence_conflicted` | implemented |
| Concept candidate from evidence | API authorization checks | `test_reviewer_can_create_concept_candidate_from_evidence` | implemented |
| ConceptRelation officialization | `test_relation_officialization_*` | `test_concept_relations_api.py` | implemented |
| Relation persistence | SQLAlchemy store converters | `test_sqlalchemy_store_persists_concept_relation_officialization` | implemented |

## v0.11.0 Grounded serving contract matrix

| Feature | Unit coverage | Integration coverage | Status |
|---|---|---|---:|
| Evidence-only grounded response | `test_related_reviewed_evidence_without_concept_is_evidence_supported` | `test_reviewed_evidence_without_concept_returns_evidence_supported` | implemented |
| Rejected evidence excluded | `test_related_rejected_evidence_is_not_served_as_support` | covered through serving contract behavior | implemented |
| Conflicted evidence/Concept handling | `test_related_conflicted_evidence_returns_conflicted`, existing conflicted Concept test | context route logs trust label | implemented |
| Relation citations | `test_official_concept_response_includes_official_relations_and_relation_citations` | `test_grounded_response_includes_relation_and_support_metadata` | implemented |
| Decision citations | `test_official_concept_response_includes_decisions_and_decision_citations` | Concept/Decision route coverage plus context unit | implemented |
| OpenAPI grounded contract snapshot | snapshot helper | `test_grounded_serving_openapi_contract_matches_snapshot` | implemented |


## v0.12.0 Evaluation framework

| Feature | Tests | Status |
|---|---|---:|
| Eval task schema | unsupported/evidence validation | Covered |
| Eval result scoring | answer/evidence/provenance/trust/freshness/unsupported checks | Covered |
| Metric summary | grounded_context_task_success_rate and supporting rates | Covered |
| Eval API | create/list/run/batch/results/summary | Covered |
| Eval persistence | SQLAlchemy metadata + migration | Covered |


## v0.12.0 Evaluation framework matrix

| Feature | Unit coverage | Integration coverage | Status |
|---|---|---|---:|
| Evaluation task validation | `test_eval_task_rejects_unsupported_with_required_evidence` | `POST /v1/evaluations/tasks` 422 case | implemented |
| Unsupported answer evaluation | `test_unsupported_eval_task_succeeds_when_unsupported_is_expected` | `test_evaluation_api_detects_unsupported_official_claim` | implemented |
| Answer correctness check | `test_eval_task_requires_expected_answer_text` | official context evaluation API | implemented |
| Evidence/provenance/trust/freshness checks | evaluation service tests | run-one and run-many evaluation API | implemented |
| Metric summary | `test_eval_summary_computes_grounded_context_task_success_rate` | `GET /v1/evaluations/summary` | implemented |
| Evaluation persistence | `test_in_memory_store_persists_eval_tasks_and_results` | `test_persistent_evaluation_api_stores_tasks_and_results` | implemented |
| Evaluation migration | `test_evaluation_tables_are_persisted_and_indexed` | Alembic offline SQL through 0011 | implemented |


## v0.12.1 Product trust cleanup tests

- Direct provider-backed source creation is rejected.
- Fake source OAuth completion route is removed.
- Degraded-source serving keeps reviewed captured evidence available with explicit limitations.
- Officialization remains stricter than serving.
- Vague evaluation tasks are rejected.
- Evaluation OpenAPI surface is snapshot-protected.

## v0.13.0 release-candidate cleanup

| Feature / Gate | Test / Check | Expected |
|---|---|---|
| Release docs | `tests/unit/test_release_candidate_docs.py` | Required release docs exist and are actionable |
| Static release readiness | `python scripts/check_release_candidate.py` | Version/docs/API-freeze/hygiene checks pass |
| API freeze | `docs/release/api-freeze-review.md` | Unsafe removed routes documented |
| Known limitations | `docs/release/known-limitations.md` | Accepted MVP limitations explicit |
| Operator proof | `docs/release/backend-operator-runbook.md` | Live proof loop is repeatable |
