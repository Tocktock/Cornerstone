from __future__ import annotations

import hashlib
import re


def normalize_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return normalized


def split_paragraphs(body: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    return parts


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_key(value))
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "item"


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"
