#!/usr/bin/env python3
"""Cron-friendly script to regenerate query hints for scheduled projects."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from cornerstone.app import create_app
from cornerstone.glossary import GlossaryEntry
from cornerstone.query_hints import merge_hint_sources


async def refresh_project(app, project_id: str) -> None:
    services = app.state.services
    project_store = services.project_store
    base_hints = services.query_hints
    generator = services.query_hint_generator
    scheduler = services.hint_scheduler

    project = project_store.get_project(project_id)
    if project is None:
        scheduler.update_job(project_id, None)
        return

    raw_entries = project_store.list_glossary_entries(project_id)
    entries: list[GlossaryEntry] = []
    for item in raw_entries:
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if not term or not definition:
            continue
        synonyms = item.get("synonyms") or []
        keywords = item.get("keywords") or []
        entries.append(
            GlossaryEntry(
                term=term,
                definition=definition,
                synonyms=[str(value).strip() for value in synonyms if str(value).strip()],
                keywords=[str(value).strip() for value in keywords if str(value).strip()],
            )
        )

    if not entries:
        return

    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(None, generator.generate, entries)
    project_hints = project_store.get_query_hints(project_id)
    merged_project = merge_hint_sources(project_hints, report.hints)
    metadata = project_store.get_query_hint_metadata(project_id) or {}
    metadata.update(
        {
            "backend": report.backend,
            "prompts": report.prompts_sent,
            "last_generated": datetime.now(timezone.utc).isoformat(),
        }
    )
    project_store.set_query_hints(project_id, merged_project, metadata=metadata)
    scheduler.update_job(project_id, metadata.get("schedule"), start=datetime.now(timezone.utc))


def seed_scheduler(app) -> None:
    services = app.state.services
    project_store = services.project_store
    scheduler = services.hint_scheduler
    for project in project_store.list_projects():
        metadata = project_store.get_query_hint_metadata(project.id)
        schedule = metadata.get("schedule") if isinstance(metadata, dict) else None
        last = metadata.get("last_generated") if isinstance(metadata, dict) else None
        start = None
        if last:
            try:
                start = datetime.fromisoformat(last)
            except ValueError:
                start = None
        scheduler.update_job(project.id, schedule, start=start)


async def main() -> int:
    app = create_app()
    seed_scheduler(app)
    services = app.state.services
    due = list(services.hint_scheduler.due_projects())
    if not due:
        print("No projects due for hint refresh")
        return 0
    for project_id in due:
        await refresh_project(app, project_id)
        print(f"Refreshed query hints for project {project_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
