from __future__ import annotations

from fastapi import APIRouter, Depends

from cornerstone.api.dependencies import get_store
from cornerstone.schemas import AuditEvent
from cornerstone.store import InMemoryStore

router = APIRouter(prefix="/audit-events", tags=["audit"])


@router.get("", response_model=list[AuditEvent])
def list_audit_events(store: InMemoryStore = Depends(get_store)) -> list[AuditEvent]:
    return store.list_audit_events()
