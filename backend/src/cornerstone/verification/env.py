from __future__ import annotations


def int_env(value: str | None, *, default: int) -> int:
    """Parse optional integer environment values used by live proof scripts."""
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default
