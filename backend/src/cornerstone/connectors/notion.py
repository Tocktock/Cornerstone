from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from cornerstone.connectors.base import (
    ParsedArtifact,
    PreparedConnection,
    PreviewArtifact,
    ProviderSyncResult,
)
from cornerstone.domain.enums import SyncMode, VisibilityClass
from cornerstone.services.normalization import split_paragraphs

NOTION_ID_PATTERN = re.compile(
    r"(?P<id>[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12})"
)


@dataclass(slots=True)
class NotionScope:
    scope_kind: str
    object_id: str
    title: str
    url: str
    preview_items: list[PreviewArtifact]


class NotionConnector:
    provider = "notion"

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
        scope = self._resolve_scope(
            template_key=template_key,
            selected_scope_input=selected_scope_input,
            settings=settings,
            provider_credential=provider_credential,
        )
        source_boundary_locator = (
            f"notion://page-tree/{scope.object_id}"
            if scope.scope_kind == "page_tree"
            else f"notion://database/{scope.object_id}"
        )
        return PreparedConnection(
            provider=self.provider,
            template_key=template_key,
            source_boundary_locator=source_boundary_locator,
            selected_scope={
                "scope_kind": scope.scope_kind,
                "object_id": scope.object_id,
                "title": scope.title,
                "url": scope.url,
                "input": selected_scope_input,
                "source_label": source_label,
            },
            visibility_class=visibility_class,
            sync_mode=SyncMode.SCHEDULED_SYNC,
            sync_interval_seconds=900,
            effective_sync_policy={
                "mode": "scheduled_incremental_with_fallback",
                "provider": "notion",
                "template_key": template_key,
            },
            preview_items=scope.preview_items,
        )

    def sync(self, *, connection, settings, provider_credential=None) -> ProviderSyncResult:
        selected_scope = connection.selected_scope_json or {}
        scope_kind = str(selected_scope.get("scope_kind", ""))
        object_id = str(selected_scope.get("object_id", ""))
        if not object_id:
            raise ValueError("Notion connection is missing selected scope.")

        if self._is_demo_mode(settings, provider_credential):
            fixture = self._load_fixture_data(Path(settings.notion_fixture_root))
            if scope_kind == "page_tree":
                artifacts = self._sync_demo_page_tree(fixture, object_id)
            else:
                artifacts = self._sync_demo_database(fixture, object_id)
            return ProviderSyncResult(
                parsed_artifacts=artifacts,
                sync_checkpoint=self._build_checkpoint(artifacts, strategy="fallback_scan"),
                effective_sync_policy={
                    **(connection.effective_sync_policy or {}),
                    "strategy": "fallback_scan",
                    "provider": "notion",
                },
            )

        if provider_credential is None:
            raise ValueError("Notion sync requires a provider credential.")

        if scope_kind == "page_tree":
            artifacts = self._sync_live_page_tree(
                object_id=object_id,
                settings=settings,
                access_token=self._access_token(provider_credential),
            )
        else:
            checkpoint = connection.sync_checkpoint_json or {}
            artifacts = self._sync_live_database(
                object_id=object_id,
                checkpoint=checkpoint,
                settings=settings,
                access_token=self._access_token(provider_credential),
            )

        return ProviderSyncResult(
            parsed_artifacts=artifacts,
            sync_checkpoint=self._build_checkpoint(artifacts, strategy="incremental_or_fallback"),
            effective_sync_policy={
                **(connection.effective_sync_policy or {}),
                "strategy": "incremental_or_fallback",
                "provider": "notion",
            },
        )

    def build_demo_binding_payload(self, settings) -> dict[str, Any]:
        return {
            "mode": "demo_fixture",
            "fixture_root": settings.notion_fixture_root,
            "account_label": "Demo Notion workspace",
            "binding_code": secrets.token_hex(8),
        }

    def build_authorization_url(self, *, binding_state: str, settings) -> str:
        redirect_uri = settings.notion_oauth_redirect_uri or (
            f"{settings.frontend_base_url.rstrip('/')}/sources"
        )
        query = (
            f"client_id={quote(settings.notion_client_id or '')}"
            f"&response_type=code"
            f"&owner=user"
            f"&redirect_uri={quote(redirect_uri)}"
            f"&state={quote(binding_state)}"
        )
        return f"{settings.notion_oauth_authorize_url}?{query}"

    def exchange_code_for_token(self, *, code: str, settings) -> tuple[dict[str, Any], str | None]:
        if not settings.notion_client_id or not settings.notion_client_secret:
            raise ValueError("Notion OAuth credentials are not configured.")
        redirect_uri = settings.notion_oauth_redirect_uri or (
            f"{settings.frontend_base_url.rstrip('/')}/sources"
        )
        response = httpx.post(
            settings.notion_oauth_token_url,
            auth=(settings.notion_client_id, settings.notion_client_secret),
            headers={"Content-Type": "application/json"},
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        return payload, payload.get("workspace_name")

    def _resolve_scope(
        self,
        *,
        template_key: str,
        selected_scope_input: str,
        settings,
        provider_credential,
    ) -> NotionScope:
        object_id = self._parse_notion_id(selected_scope_input)
        scope_kind = "page_tree" if template_key == "notion_shared_page_tree" else "database"
        if self._is_demo_mode(settings, provider_credential):
            fixture = self._load_fixture_data(Path(settings.notion_fixture_root))
            if scope_kind == "page_tree":
                scope = fixture["page_trees"].get(object_id)
                if scope is None:
                    raise ValueError(f"Unknown demo Notion page tree: {selected_scope_input}")
                preview_items = [
                    PreviewArtifact(
                        upstream_id=page["id"],
                        title=page["title"],
                        artifact_type="notion_page",
                        source_locator=page.get("url"),
                        excerpt=page["blocks"][0] if page.get("blocks") else None,
                        source_updated_at=self._parse_datetime(page.get("last_edited_time")),
                    )
                    for page in scope.get("pages", [])[:5]
                ]
            else:
                scope = fixture["databases"].get(object_id)
                if scope is None:
                    raise ValueError(f"Unknown demo Notion database: {selected_scope_input}")
                preview_items = [
                    PreviewArtifact(
                        upstream_id=entry["id"],
                        title=entry["title"],
                        artifact_type="notion_database_entry",
                        source_locator=entry.get("url"),
                        excerpt=self._render_properties(entry.get("properties", {})),
                        source_updated_at=self._parse_datetime(entry.get("last_edited_time")),
                    )
                    for entry in scope.get("entries", [])[:5]
                ]
            return NotionScope(
                scope_kind=scope_kind,
                object_id=object_id,
                title=str(scope["title"]),
                url=str(scope["url"]),
                preview_items=preview_items,
            )

        if provider_credential is None:
            raise ValueError("Notion preview requires a provider credential.")
        access_token = self._access_token(provider_credential)
        if scope_kind == "page_tree":
            page = self._request_json(
                settings,
                access_token=access_token,
                method="GET",
                path=f"/pages/{object_id}",
            )
            blocks = self._collect_block_children(
                page_id=object_id,
                access_token=access_token,
                settings=settings,
                page_limit=5,
            )
            preview_items = [
                PreviewArtifact(
                    upstream_id=object_id,
                    title=self._page_title(page),
                    artifact_type="notion_page",
                    source_locator=page.get("url"),
                    excerpt=self._flatten_block_texts(blocks)[:400] or None,
                    source_updated_at=self._parse_datetime(page.get("last_edited_time")),
                )
            ]
            return NotionScope(
                scope_kind=scope_kind,
                object_id=object_id,
                title=self._page_title(page),
                url=str(page.get("url") or ""),
                preview_items=preview_items,
            )

        database = self._request_json(
            settings,
            access_token=access_token,
            method="GET",
            path=f"/databases/{object_id}",
        )
        entries = self._query_database_entries(
            database_id=object_id,
            settings=settings,
            access_token=access_token,
            page_size=5,
            checkpoint=None,
        )
        preview_items = [
            PreviewArtifact(
                upstream_id=entry["id"],
                title=self._page_title(entry),
                artifact_type="notion_database_entry",
                source_locator=entry.get("url"),
                excerpt=self._render_entry_summary(entry),
                source_updated_at=self._parse_datetime(entry.get("last_edited_time")),
            )
            for entry in entries[:5]
        ]
        return NotionScope(
            scope_kind=scope_kind,
            object_id=object_id,
            title=self._plain_text(database.get("title", [])) or "Notion database",
            url=str(database.get("url") or ""),
            preview_items=preview_items,
        )

    def _sync_demo_page_tree(self, fixture: dict[str, Any], object_id: str) -> list[ParsedArtifact]:
        scope = fixture["page_trees"][object_id]
        return [self._demo_page_to_artifact(page) for page in scope.get("pages", [])]

    def _sync_demo_database(self, fixture: dict[str, Any], object_id: str) -> list[ParsedArtifact]:
        scope = fixture["databases"][object_id]
        return [
            self._demo_entry_to_artifact(entry, scope_title=scope["title"])
            for entry in scope.get("entries", [])
        ]

    def _sync_live_page_tree(
        self, *, object_id: str, settings, access_token: str
    ) -> list[ParsedArtifact]:
        pages = []
        queue = [object_id]
        seen: set[str] = set()
        while queue:
            page_id = queue.pop(0)
            if page_id in seen:
                continue
            seen.add(page_id)
            page = self._request_json(
                settings,
                access_token=access_token,
                method="GET",
                path=f"/pages/{page_id}",
            )
            blocks = self._collect_block_children(
                page_id=page_id,
                access_token=access_token,
                settings=settings,
            )
            pages.append((page, blocks))
            queue.extend(self._child_page_ids(blocks))
        return [self._live_page_to_artifact(page, blocks) for page, blocks in pages]

    def _sync_live_database(
        self,
        *,
        object_id: str,
        checkpoint: dict[str, Any],
        settings,
        access_token: str,
    ) -> list[ParsedArtifact]:
        entries = self._query_database_entries(
            database_id=object_id,
            settings=settings,
            access_token=access_token,
            page_size=100,
            checkpoint=checkpoint,
        )
        if not entries:
            entries = self._query_database_entries(
                database_id=object_id,
                settings=settings,
                access_token=access_token,
                page_size=100,
                checkpoint=None,
            )
        return [self._live_entry_to_artifact(entry, object_id=object_id) for entry in entries]

    def _query_database_entries(
        self,
        *,
        database_id: str,
        settings,
        access_token: str,
        page_size: int,
        checkpoint: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        database = self._request_json(
            settings,
            access_token=access_token,
            method="GET",
            path=f"/databases/{database_id}",
        )
        query_path = self._database_query_path(database)
        payload: dict[str, Any] = {"page_size": page_size}
        last_synced_at = str((checkpoint or {}).get("last_synced_at") or "")
        if last_synced_at:
            payload["filter"] = {
                "timestamp": "last_edited_time",
                "last_edited_time": {"after": last_synced_at},
            }
        response = self._request_json(
            settings,
            access_token=access_token,
            method="POST",
            path=query_path,
            payload=payload,
        )
        return list(response.get("results", []))

    def _database_query_path(self, database: dict[str, Any]) -> str:
        data_sources = database.get("data_sources")
        if isinstance(data_sources, list) and data_sources:
            return f"/data_sources/{data_sources[0]['id']}/query"
        return f"/databases/{database['id']}/query"

    def _collect_block_children(
        self,
        *,
        page_id: str,
        access_token: str,
        settings,
        page_limit: int = 100,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        next_cursor: str | None = None
        while True:
            query = f"?page_size={page_limit}"
            if next_cursor:
                query += f"&start_cursor={quote(next_cursor)}"
            response = self._request_json(
                settings,
                access_token=access_token,
                method="GET",
                path=f"/blocks/{page_id}/children{query}",
            )
            blocks.extend(response.get("results", []))
            next_cursor = response.get("next_cursor")
            if not response.get("has_more"):
                break
        return blocks

    def _request_json(
        self,
        settings,
        *,
        access_token: str,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Notion-Version": settings.notion_version,
            "Content-Type": "application/json",
        }
        response = httpx.request(
            method,
            f"{settings.notion_api_base_url.rstrip('/')}{path}",
            headers=headers,
            json=payload,
            timeout=20.0,
        )
        response.raise_for_status()
        return response.json()

    def _demo_page_to_artifact(self, page: dict[str, Any]) -> ParsedArtifact:
        content_text = "\n\n".join(page.get("blocks", []))
        raw = json.dumps(page, sort_keys=True)
        return ParsedArtifact(
            external_id=page["id"],
            title=page["title"],
            artifact_type="notion_page",
            source_locator=page["url"],
            content_text=content_text,
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            source_updated_at=self._parse_datetime(page["last_edited_time"]) or datetime.now(UTC),
            metadata={
                "provider": "notion",
                "object_type": "page",
                "object_id": page["id"],
            },
            support_fragments=[
                (f"paragraph:{index}", paragraph)
                for index, paragraph in enumerate(split_paragraphs(content_text), start=1)
            ],
        )

    def _demo_entry_to_artifact(self, entry: dict[str, Any], *, scope_title: str) -> ParsedArtifact:
        content_text = self._render_properties(entry.get("properties", {}))
        raw = json.dumps(entry, sort_keys=True)
        return ParsedArtifact(
            external_id=entry["id"],
            title=entry["title"],
            artifact_type="notion_database_entry",
            source_locator=entry["url"],
            content_text=content_text,
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            source_updated_at=self._parse_datetime(entry["last_edited_time"]) or datetime.now(UTC),
            metadata={
                "provider": "notion",
                "object_type": "database_entry",
                "object_id": entry["id"],
                "database_title": scope_title,
                "properties": entry.get("properties", {}),
            },
            support_fragments=[
                (f"row:{index}", fragment)
                for index, fragment in enumerate(split_paragraphs(content_text), start=1)
            ],
        )

    def _live_page_to_artifact(
        self, page: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> ParsedArtifact:
        content_text = self._flatten_block_texts(blocks)
        raw = json.dumps({"page": page, "blocks": blocks}, sort_keys=True)
        return ParsedArtifact(
            external_id=page["id"],
            title=self._page_title(page),
            artifact_type="notion_page",
            source_locator=str(page.get("url") or ""),
            content_text=content_text,
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            source_updated_at=self._parse_datetime(page.get("last_edited_time"))
            or datetime.now(UTC),
            metadata={
                "provider": "notion",
                "object_type": "page",
                "object_id": page["id"],
            },
            support_fragments=[
                (f"paragraph:{index}", paragraph)
                for index, paragraph in enumerate(split_paragraphs(content_text), start=1)
            ],
        )

    def _live_entry_to_artifact(self, entry: dict[str, Any], *, object_id: str) -> ParsedArtifact:
        content_text = self._render_entry_summary(entry)
        raw = json.dumps(entry, sort_keys=True)
        return ParsedArtifact(
            external_id=entry["id"],
            title=self._page_title(entry),
            artifact_type="notion_database_entry",
            source_locator=str(entry.get("url") or ""),
            content_text=content_text,
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            source_updated_at=self._parse_datetime(entry.get("last_edited_time"))
            or datetime.now(UTC),
            metadata={
                "provider": "notion",
                "object_type": "database_entry",
                "object_id": entry["id"],
                "database_id": object_id,
                "properties": entry.get("properties", {}),
            },
            support_fragments=[
                (f"row:{index}", fragment)
                for index, fragment in enumerate(split_paragraphs(content_text), start=1)
            ],
        )

    def _render_entry_summary(self, entry: dict[str, Any]) -> str:
        return self._render_properties(self._flatten_properties(entry.get("properties", {})))

    def _flatten_properties(self, properties: dict[str, Any]) -> dict[str, str]:
        flattened: dict[str, str] = {}
        for key, value in properties.items():
            if not isinstance(value, dict):
                continue
            prop_type = value.get("type")
            if prop_type == "title":
                flattened[key] = self._plain_text(value.get("title", []))
            elif prop_type == "rich_text":
                flattened[key] = self._plain_text(value.get("rich_text", []))
            elif prop_type == "select":
                flattened[key] = str((value.get("select") or {}).get("name") or "")
            elif prop_type == "status":
                flattened[key] = str((value.get("status") or {}).get("name") or "")
            elif prop_type == "multi_select":
                flattened[key] = ", ".join(
                    item.get("name", "") for item in value.get("multi_select", [])
                )
            elif prop_type == "number":
                flattened[key] = str(value.get("number") or "")
            elif prop_type == "people":
                flattened[key] = ", ".join(
                    person.get("name", "") for person in value.get("people", [])
                )
            elif prop_type == "checkbox":
                flattened[key] = "true" if value.get("checkbox") else "false"
            elif prop_type == "url":
                flattened[key] = str(value.get("url") or "")
            elif prop_type == "email":
                flattened[key] = str(value.get("email") or "")
            elif prop_type == "phone_number":
                flattened[key] = str(value.get("phone_number") or "")
            elif prop_type == "date":
                date_value = value.get("date") or {}
                flattened[key] = str(date_value.get("start") or "")
            elif prop_type == "formula":
                flattened[key] = str((value.get("formula") or {}).get("string") or "")
            else:
                flattened[key] = ""
        return {key: value for key, value in flattened.items() if value}

    def _render_properties(self, properties: dict[str, Any]) -> str:
        return "\n".join(f"{key}: {value}" for key, value in properties.items() if value)

    def _flatten_block_texts(self, blocks: list[dict[str, Any]]) -> str:
        fragments: list[str] = []
        for block in blocks:
            block_type = block.get("type")
            value = block.get(block_type or "", {})
            if not isinstance(value, dict):
                continue
            rich_text = value.get("rich_text")
            if isinstance(rich_text, list):
                text = self._plain_text(rich_text)
                if text:
                    fragments.append(text)
            child_page = value.get("title")
            if isinstance(child_page, str) and child_page:
                fragments.append(child_page)
        return "\n\n".join(fragment for fragment in fragments if fragment)

    def _child_page_ids(self, blocks: list[dict[str, Any]]) -> list[str]:
        child_ids: list[str] = []
        for block in blocks:
            if block.get("type") == "child_page" and block.get("id"):
                child_ids.append(str(block["id"]))
        return child_ids

    def _page_title(self, page: dict[str, Any]) -> str:
        properties = page.get("properties")
        if isinstance(properties, dict):
            for value in properties.values():
                if isinstance(value, dict) and value.get("type") == "title":
                    title = self._plain_text(value.get("title", []))
                    if title:
                        return title
        title = self._plain_text(page.get("title", []))
        return title or str(page.get("id") or "Untitled Notion item")

    def _plain_text(self, rich_text_items: list[dict[str, Any]] | Any) -> str:
        if not isinstance(rich_text_items, list):
            return ""
        return "".join(str(item.get("plain_text") or "") for item in rich_text_items).strip()

    def _build_checkpoint(
        self, artifacts: list[ParsedArtifact], *, strategy: str
    ) -> dict[str, Any]:
        if not artifacts:
            return {"strategy": strategy}
        latest = max(artifact.source_updated_at for artifact in artifacts)
        return {"strategy": strategy, "last_synced_at": latest.isoformat()}

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _parse_notion_id(self, selected_scope_input: str) -> str:
        candidate = selected_scope_input.strip()
        if re.fullmatch(r"[0-9a-fA-F]{32}", candidate):
            return (
                f"{candidate[0:8]}-{candidate[8:12]}-{candidate[12:16]}-"
                f"{candidate[16:20]}-{candidate[20:32]}"
            ).lower()
        if re.fullmatch(r"[0-9a-fA-F-]{36}", candidate):
            return candidate.lower()
        match = NOTION_ID_PATTERN.search(candidate)
        if not match:
            raise ValueError("Expected a Notion page or database URL or UUID.")
        raw_id = match.group("id").replace("-", "")
        return (
            f"{raw_id[0:8]}-{raw_id[8:12]}-{raw_id[12:16]}-"
            f"{raw_id[16:20]}-{raw_id[20:32]}"
        ).lower()

    def _load_fixture_data(self, fixture_root: Path) -> dict[str, Any]:
        return json.loads((fixture_root / "scopes.json").read_text(encoding="utf-8"))

    def _access_token(self, provider_credential) -> str:
        access_token = str(provider_credential.auth_payload.get("access_token") or "")
        if not access_token:
            raise ValueError("Notion credential is missing an access token.")
        return access_token

    def _is_demo_mode(self, settings, provider_credential) -> bool:
        if provider_credential is None:
            return bool(settings.notion_demo_oauth_mode)
        return provider_credential.auth_payload.get("mode") == "demo_fixture"
