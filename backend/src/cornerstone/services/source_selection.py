from __future__ import annotations

from typing import Any, cast

from cornerstone.schemas import SourceSelection
from cornerstone.store import NotFoundError


def get_source_selection_for_sync(source_id: str, store: Any) -> SourceSelection:
    """Return the persisted source selection or the default sync-all selection."""
    try:
        return cast(SourceSelection, store.get_source_selection(source_id))
    except NotFoundError:
        return SourceSelection(datasource_id=source_id)
