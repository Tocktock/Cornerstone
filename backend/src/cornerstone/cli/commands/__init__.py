from __future__ import annotations

from ..proof import command_proof_run
from .completion import command_completion
from .concept import command_concept_create_from_evidence, command_concept_list, command_concept_officialize, command_concept_show
from .config import command_config
from .context import command_context
from .evaluation import command_eval_create, command_eval_results, command_eval_run, command_eval_summary
from .evidence import command_evidence_conflict, command_evidence_queue, command_evidence_reject, command_evidence_review, command_evidence_show
from .runtime import command_api, command_db, command_doctor, command_env_init, command_live, command_local, command_setup, command_stack, command_status, command_version, command_worker
from .review import command_review_preview, command_review_queue
from .source import command_source_jobs, command_source_list, command_source_objects, command_source_show, command_source_sync

__all__ = [name for name in globals() if name.startswith("command_")]
