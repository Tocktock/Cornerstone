#!/usr/bin/env python
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from cornerstone.config import Settings
from cornerstone.store import InMemoryStore
from cornerstone.verification.notion_e2e import (
    NotionE2EConfigError,
    NotionE2ERunError,
    notion_e2e_config_from_env,
    run_live_notion_page_e2e,
)


def _create_store(settings: Settings) -> Any:
    if settings.persistence_backend == "postgres":
        from cornerstone.persistence.database import create_persistent_store

        return create_persistent_store(settings)
    return InMemoryStore()


async def _main() -> int:
    settings = Settings.from_env()
    config = notion_e2e_config_from_env(os.environ)
    store = _create_store(settings)
    try:
        result = await run_live_notion_page_e2e(store=store, settings=settings, config=config)
    except NotionE2EConfigError as exc:
        print(
            json.dumps(
                {
                    "status": "configuration_failed",
                    "issues": [issue.__dict__ for issue in exc.issues],
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except NotionE2ERunError as exc:
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "result": result.to_jsonable()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
