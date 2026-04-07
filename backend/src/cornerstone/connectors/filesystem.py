from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cornerstone.connectors.base import (
    ParsedArtifact,
    PreparedConnection,
    PreviewArtifact,
    ProviderSyncResult,
)
from cornerstone.domain.enums import SyncMode, VisibilityClass
from cornerstone.services.normalization import split_paragraphs


class FilesystemConnector:
    provider = "filesystem"

    def __init__(self, root_path: str):
        self.root = Path(root_path)

    def prepare_connection(
        self,
        *,
        template_key: str,
        source_label: str,
        selected_scope_input: str,
        visibility_class: VisibilityClass,
        settings,
        provider_credential=None,
    ) -> PreparedConnection:
        artifacts = self.list_artifacts()
        preview_items = [
            PreviewArtifact(
                upstream_id=artifact.external_id,
                title=artifact.title,
                artifact_type=artifact.artifact_type,
                source_locator=artifact.source_locator,
                excerpt=artifact.support_fragments[0][1] if artifact.support_fragments else None,
                source_updated_at=artifact.source_updated_at,
            )
            for artifact in artifacts[:5]
        ]
        return PreparedConnection(
            provider=self.provider,
            template_key=template_key,
            source_boundary_locator=str(self.root),
            selected_scope={
                "scope_kind": "filesystem_root",
                "input": selected_scope_input,
                "resolved_root": str(self.root),
                "source_label": source_label,
            },
            visibility_class=visibility_class,
            sync_mode=SyncMode.POLLING,
            sync_interval_seconds=settings.default_sync_interval_seconds,
            effective_sync_policy={"mode": "filesystem", "template_key": template_key},
            preview_items=preview_items,
        )

    def sync(self, *, connection, settings, provider_credential=None) -> ProviderSyncResult:
        if not self.root.exists():
            raise FileNotFoundError(f"Source root does not exist: {self.root}")
        return ProviderSyncResult(
            parsed_artifacts=self.list_artifacts(),
            sync_checkpoint=connection.sync_checkpoint_json or {},
            effective_sync_policy=connection.effective_sync_policy,
        )

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
