from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cornerstone.services.normalization import split_paragraphs


@dataclass(slots=True)
class ParsedArtifact:
    external_id: str
    title: str
    artifact_type: str
    canonical_url: str
    content_text: str
    content_hash: str
    source_updated_at: datetime
    metadata: dict[str, Any]
    frontmatter: dict[str, Any]
    evidence_fragments: list[tuple[str, str]]


class FilesystemConnector:
    provider = "filesystem"

    def __init__(self, root_path: str):
        self.root = Path(root_path)

    def list_artifacts(self) -> list[ParsedArtifact]:
        if not self.root.exists():
            return []
        artifacts: list[ParsedArtifact] = []
        for path in sorted(self.root.rglob("*.md")):
            artifacts.append(self._parse_file(path))
        return artifacts

    def _parse_file(self, path: Path) -> ParsedArtifact:
        raw_text = path.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(raw_text)
        title = str(frontmatter.get("title") or frontmatter.get("canonical_name") or path.stem.replace("-", " ").title())
        evidence = [(f"paragraph:{idx}", paragraph) for idx, paragraph in enumerate(split_paragraphs(body), start=1)]
        stat = path.stat()
        return ParsedArtifact(
            external_id=str(path.relative_to(self.root)),
            title=title,
            artifact_type=str(frontmatter.get("kind", "document")),
            canonical_url=path.resolve().as_uri(),
            content_text=body,
            content_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            source_updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            metadata={"path": str(path), "file_name": path.name},
            frontmatter=frontmatter,
            evidence_fragments=evidence,
        )

    @staticmethod
    def _split_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
        if raw_text.startswith("---\n"):
            _, rest = raw_text.split("---\n", 1)
            frontmatter_text, body = rest.split("\n---\n", 1)
            return yaml.safe_load(frontmatter_text) or {}, body.strip()
        return {}, raw_text.strip()
