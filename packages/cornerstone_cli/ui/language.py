from __future__ import annotations


RECORD_ACTIONS = {
    "Source": "Open source",
    "Brief": "Open Brief",
    "Claim": "Review decision",
    "Action": "Review action",
}


RECORD_ICONS = {
    "Source": "document",
    "Brief": "brief",
    "Claim": "shield-check",
    "Action": "action",
}


def record_action(kind: str) -> str:
    return RECORD_ACTIONS.get(kind, "Open record")


def record_icon(kind: str) -> str:
    return RECORD_ICONS.get(kind, "document")

