from __future__ import annotations

import re


def normalize_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return normalized


def split_paragraphs(body: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    return parts
