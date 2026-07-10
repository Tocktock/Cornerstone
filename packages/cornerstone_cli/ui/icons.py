from __future__ import annotations

from html import escape


ICON_NAMES = frozenset(
    {
        "action",
        "brief",
        "chat",
        "check",
        "chevron-right",
        "document",
        "external",
        "help",
        "history",
        "home",
        "review",
        "search",
        "shield-check",
        "upload",
        "warning",
    }
)


def icon(name: str, *, label: str | None = None, class_name: str = "cs-icon") -> str:
    """Render a checked-in Heroicons asset with safe accessible semantics."""

    icon_name = name if name in ICON_NAMES else "document"
    classes = escape(class_name, quote=True)
    src = f"/assets/icons/{icon_name}.svg"
    if label:
        return f'<img class="{classes}" src="{src}" width="20" height="20" alt="{escape(label, quote=True)}">'
    return f'<img class="{classes}" src="{src}" width="20" height="20" alt="" aria-hidden="true">'
