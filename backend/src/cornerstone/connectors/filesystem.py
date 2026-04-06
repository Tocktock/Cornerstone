from __future__ import annotations

import csv
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
    source_locator: str
    content_text: str
    content_hash: str
    source_updated_at: datetime
    metadata: dict[str, Any]
    support_fragments: list[tuple[str, str]]


class FilesystemConnector:
    provider = "filesystem"

    def __init__(self, root_path: str):
        self.root = Path(root_path)

    def list_artifacts(self) -> list[ParsedArtifact]:
        if not self.root.exists():
            return []
        artifacts: list[ParsedArtifact] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".md":
                artifacts.append(self._parse_markdown(path))
            elif path.suffix.lower() == ".csv":
                artifacts.append(self._parse_csv(path))
        return artifacts

    def _parse_markdown(self, path: Path) -> ParsedArtifact:
        raw_text = path.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(raw_text)
        title = str(frontmatter.get("title") or path.stem.replace("-", " ").title())
        support_fragments = [
            (f"paragraph:{idx}", paragraph)
            for idx, paragraph in enumerate(split_paragraphs(body), start=1)
        ]
        stat = path.stat()
        return ParsedArtifact(
            external_id=str(path.relative_to(self.root)),
            title=title,
            artifact_type=str(frontmatter.get("kind", "markdown_document")),
            source_locator=path.resolve().as_uri(),
            content_text=body,
            content_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            source_updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            metadata={
                "path": str(path.relative_to(self.root)),
                "file_name": path.name,
                "frontmatter": frontmatter,
            },
            support_fragments=support_fragments,
        )

    def _parse_csv(self, path: Path) -> ParsedArtifact:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        support_fragments: list[tuple[str, str]] = []
        rendered_rows: list[str] = []
        for idx, row in enumerate(rows, start=1):
            rendered = "; ".join(
                f"{key}={value}" for key, value in row.items() if value not in {None, ""}
            )
            if not rendered:
                continue
            rendered_rows.append(rendered)
            support_fragments.append((f"row:{idx}", rendered))
        content_text = "\n".join(rendered_rows)
        stat = path.stat()
        return ParsedArtifact(
            external_id=str(path.relative_to(self.root)),
            title=path.stem.replace("-", " ").title(),
            artifact_type="csv_snapshot",
            source_locator=path.resolve().as_uri(),
            content_text=content_text,
            content_hash=hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
            source_updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            metadata={
                "path": str(path.relative_to(self.root)),
                "file_name": path.name,
                "row_count": len(rows),
            },
            support_fragments=support_fragments,
        )

    @staticmethod
    def _split_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
        if raw_text.startswith("---\n"):
            try:
                _, rest = raw_text.split("---\n", 1)
                frontmatter_text, body = rest.split("\n---\n", 1)
            except ValueError:
                return {}, raw_text.strip()
            return yaml.safe_load(frontmatter_text) or {}, body.strip()
        return {}, raw_text.strip()
