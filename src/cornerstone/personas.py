"""Persona catalog models and persistence helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, List

import logging


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PersonaOverrides:
    """Per-project overrides that adjust a base persona."""

    name: str | None = None
    tone: str | None = None
    system_prompt: str | None = None
    avatar_url: str | None = None


@dataclass(slots=True)
class Persona:
    """Reusable persona profile shared across projects."""

    id: str
    name: str
    description: str | None
    tone: str | None
    system_prompt: str | None
    avatar_url: str | None
    tags: list[str]
    created_at: str


@dataclass(slots=True)
class PersonaSnapshot:
    """Resolved persona combining catalog entry and overrides."""

    id: str | None
    name: str | None
    tone: str | None
    system_prompt: str | None
    avatar_url: str | None
    base_persona: Persona | None = None
    overrides: PersonaOverrides = field(default_factory=PersonaOverrides)


class PersonaStore:
    """File-backed catalog of personas available to projects."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._personas_file = root / "personas.json"
        self._root.mkdir(parents=True, exist_ok=True)
        self._ensure_seed_personas()

    def list_personas(self) -> List[Persona]:
        data = self._read_personas()
        personas = [self._deserialize_persona(item) for item in data.get("personas", [])]
        return sorted(personas, key=lambda persona: persona.name.lower())

    def get_persona(self, persona_id: str | None) -> Persona | None:
        if not persona_id:
            return None
        for persona in self.list_personas():
            if persona.id == persona_id:
                return persona
        return None

    def create_persona(
        self,
        *,
        name: str,
        description: str | None = None,
        tone: str | None = None,
        system_prompt: str | None = None,
        avatar_url: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> Persona:
        payload = self._read_personas()
        existing_ids = {item["id"] for item in payload.get("personas", [])}
        persona_id = self._generate_id(name, existing_ids)
        persona = Persona(
            id=persona_id,
            name=name,
            description=_normalize(description),
            tone=_normalize(tone),
            system_prompt=_normalize(system_prompt),
            avatar_url=_normalize(avatar_url),
            tags=sorted({tag.strip() for tag in (tags or []) if tag.strip()}),
            created_at=self._now(),
        )
        payload.setdefault("personas", []).append(self._serialize_persona(persona))
        self._write_personas(payload)
        logger.info("persona.created id=%s name=%s", persona.id, persona.name)
        return persona

    def update_persona(
        self,
        persona_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tone: str | None = None,
        system_prompt: str | None = None,
        avatar_url: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> Persona:
        payload = self._read_personas()
        for item in payload.get("personas", []):
            if item.get("id") != persona_id:
                continue
            if name is not None:
                item["name"] = name.strip()
            if description is not None:
                item["description"] = _normalize(description)
            if tone is not None:
                item["tone"] = _normalize(tone)
            if system_prompt is not None:
                item["system_prompt"] = _normalize(system_prompt)
            if avatar_url is not None:
                item["avatar_url"] = _normalize(avatar_url)
            if tags is not None:
                item["tags"] = sorted({tag.strip() for tag in tags if tag.strip()})
            self._write_personas(payload)
            updated = self._deserialize_persona(item)
            logger.info("persona.updated id=%s name=%s", updated.id, updated.name)
            return updated
        raise ValueError(f"Persona {persona_id} not found")

    def delete_persona(self, persona_id: str) -> bool:
        payload = self._read_personas()
        personas = payload.get("personas", [])
        remaining = [item for item in personas if item.get("id") != persona_id]
        if len(remaining) == len(personas):
            return False
        payload["personas"] = remaining
        self._write_personas(payload)
        logger.info("persona.deleted id=%s", persona_id)
        return True

    def resolve_persona(
        self,
        persona_id: str | None,
        overrides: PersonaOverrides | None = None,
    ) -> PersonaSnapshot:
        base = self.get_persona(persona_id)
        overrides = overrides or PersonaOverrides()
        name = overrides.name or (base.name if base else None)
        tone = overrides.tone or (base.tone if base else None)
        system_prompt = overrides.system_prompt or (base.system_prompt if base else None)
        avatar_url = overrides.avatar_url or (base.avatar_url if base else None)
        return PersonaSnapshot(
            id=base.id if base else None,
            name=name,
            tone=tone,
            system_prompt=system_prompt,
            avatar_url=avatar_url,
            base_persona=base,
            overrides=overrides,
        )

    def _ensure_seed_personas(self) -> None:
        if self._personas_file.exists():
            return
        logger.info("persona.seed.start")
        seeds = [
            Persona(
                id="chatty-guide",
                name="Chatty Guide",
                description="High-energy helper focused on quick, upbeat answers.",
                tone="friendly and enthusiastic",
                system_prompt=(
                    "You are a cheerful assistant who keeps responses short, energetic, and motivating. "
                    "Always include a quick recap and a cheerful encouragement."
                ),
                avatar_url=None,
                tags=["support", "friendly"],
                created_at=self._now(),
            ),
            Persona(
                id="sleepy-consultant",
                name="Sleepy Consultant",
                description="Calm, methodical persona that slows the conversation down.",
                tone="slow-paced and reassuring",
                system_prompt=(
                    "You respond thoughtfully and with patience. Focus on clarity over speed and reassure the user "
                    "that you're carefully considering each step."
                ),
                avatar_url=None,
                tags=["support", "calm"],
                created_at=self._now(),
            ),
            Persona(
                id="analysis-agent",
                name="Analysis Agent",
                description="Analytical support persona that cites evidence and next steps.",
                tone="confident and data-driven",
                system_prompt=(
                    "Adopt an analytical voice, reference relevant documentation, and highlight risks or unknowns. "
                    "Close with a recommended action plan and escalation criteria."
                ),
                avatar_url=None,
                tags=["support", "analysis"],
                created_at=self._now(),
            ),
        ]
        payload = {"personas": [self._serialize_persona(persona) for persona in seeds]}
        self._write_personas(payload)
        logger.info("persona.seed.completed count=%s", len(seeds))

    def _read_personas(self) -> dict:
        if not self._personas_file.exists():
            return {"personas": []}
        with self._personas_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_personas(self, data: dict) -> None:
        with self._personas_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _deserialize_persona(self, payload: dict) -> Persona:
        return Persona(
            id=payload.get("id"),
            name=payload.get("name"),
            description=payload.get("description"),
            tone=payload.get("tone"),
            system_prompt=payload.get("system_prompt"),
            avatar_url=payload.get("avatar_url"),
            tags=list(payload.get("tags", [])),
            created_at=payload.get("created_at") or self._now(),
        )

    def _serialize_persona(self, persona: Persona) -> dict:
        payload = asdict(persona)
        payload["tags"] = list(persona.tags)
        return payload

    def _generate_id(self, name: str, existing_ids: Iterable[str]) -> str:
        base = name.strip().lower().replace(" ", "-")
        base = "".join(char for char in base if char.isalnum() or char == "-") or "persona"
        candidate = base
        counter = 1
        while candidate in existing_ids:
            counter += 1
            candidate = f"{base}-{counter}"
        return candidate

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


__all__ = [
    "Persona",
    "PersonaOverrides",
    "PersonaSnapshot",
    "PersonaStore",
]
