"""Utilities for loading and querying glossary definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - optional dependency guard
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:  # pragma: no cover
    YAML_IMPORT_ERROR = None


@dataclass(slots=True)
class GlossaryEntry:
    """Represents a single domain-specific definition."""

    term: str
    definition: str
    synonyms: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        if self.term.lower() in lowered:
            return True
        return any(syn.lower() in lowered for syn in self.synonyms)


class Glossary:
    """Container with lookup helpers for glossary entries."""

    def __init__(self, entries: Iterable[GlossaryEntry] | None = None) -> None:
        self._entries: list[GlossaryEntry] = list(entries or [])

    def __len__(self) -> int:
        return len(self._entries)

    def top_matches(self, text: str, limit: int) -> List[GlossaryEntry]:
        if not text:
            return []
        matches = [entry for entry in self._entries if entry.matches(text)]
        return matches[:limit]

    def to_prompt_section(self, text: str, limit: int) -> str:
        entries = self.top_matches(text, limit)
        if not entries:
            return ""
        lines = ["Key domain definitions:"]
        for entry in entries:
            synonym_suffix = f" (synonyms: {', '.join(entry.synonyms)})" if entry.synonyms else ""
            lines.append(f"- {entry.term}{synonym_suffix}: {entry.definition}")
        return "\n".join(lines)


class GlossaryLoadError(RuntimeError):
    """Raised when the glossary cannot be loaded."""


def load_glossary(path: str | Path) -> Glossary:
    """Load glossary entries from a YAML file; return empty glossary if missing."""

    glossary_path = Path(path)
    if not glossary_path.exists():
        return Glossary()

    if yaml is None:  # pragma: no cover - requires pyyaml
        raise GlossaryLoadError(
            "pyyaml is required to load glossary definitions"
        ) from YAML_IMPORT_ERROR

    data = yaml.safe_load(glossary_path.read_text(encoding="utf-8")) or []
    entries: list[GlossaryEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if not term or not definition:
            continue
        synonyms = item.get("synonyms") or []
        if not isinstance(synonyms, list):
            synonyms = [str(synonyms)]
        entries.append(
            GlossaryEntry(
                term=term,
                definition=definition,
                synonyms=[str(value).strip() for value in synonyms if str(value).strip()],
            )
        )
    return Glossary(entries)


__all__ = [
    "Glossary",
    "GlossaryEntry",
    "GlossaryLoadError",
    "load_glossary",
]
