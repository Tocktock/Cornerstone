"""Full-text search index backed by SQLite FTS5."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence

logger = logging.getLogger(__name__)

_SAFE_FTS_QUERY_RE = re.compile(r'[0-9A-Za-z가-힣_]+', re.UNICODE)


class FTSIndex:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            needs_rebuild = False
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_search'"
            )
            if cursor.fetchone():
                pragma = conn.execute("PRAGMA table_info(chunk_search)").fetchall()
                column_names = {row[1] for row in pragma}
                if "metadata" not in column_names:
                    needs_rebuild = True
            if needs_rebuild:
                conn.execute("DROP TABLE IF EXISTS chunk_search")
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(
                    chunk_id UNINDEXED,
                    project_id UNINDEXED,
                    doc_id UNINDEXED,
                    title,
                    body,
                    metadata UNINDEXED,
                    tokenize = 'unicode61'
                )
                """
            )

    def upsert_chunks(
        self,
        *,
        project_id: str,
        doc_id: str,
        entries: Sequence[dict[str, str]],
    ) -> None:
        if not entries:
            return
        with self._connect() as conn:
            conn.execute("BEGIN")
            chunk_ids = [(entry["chunk_id"],) for entry in entries]
            conn.executemany("DELETE FROM chunk_search WHERE chunk_id = ?", chunk_ids)
            conn.executemany(
                """
                INSERT INTO chunk_search (chunk_id, project_id, doc_id, title, body, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        entry["chunk_id"],
                        project_id,
                        doc_id,
                        entry.get("title") or "",
                        entry.get("text") or "",
                        json.dumps(entry.get("metadata") or {}),
                    )
                    for entry in entries
                ],
            )
            conn.commit()

    def delete_document(self, project_id: str, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunk_search WHERE project_id = ? AND doc_id = ?",
                (project_id, doc_id),
            )

    def delete_project(self, project_id: str) -> None:
        """Remove all indexed chunks for the provided project."""

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunk_search WHERE project_id = ?",
                (project_id,),
            )

    def search(self, project_id: str, query: str, *, limit: int = 10) -> List[dict[str, str]]:
        query = (query or "").strip()
        if not query:
            return []
        sanitized_query = query
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT chunk_id, doc_id, title, body, metadata, bm25(chunk_search) as score
                    FROM chunk_search
                    WHERE project_id = ? AND chunk_search MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (project_id, sanitized_query, limit),
                )
            except sqlite3.OperationalError as exc:
                fallback_terms = []
                for match in _SAFE_FTS_QUERY_RE.findall(query):
                    if match and match not in fallback_terms:
                        fallback_terms.append(match)
                fallback = " OR ".join(fallback_terms)
                if not fallback:
                    logger.debug("fts.query.skip", exc_info=exc)
                    return []
                logger.debug(
                    "fts.query.retry",
                    extra={"project_id": project_id, "query": query},
                )
                cursor = conn.execute(
                    """
                    SELECT chunk_id, doc_id, title, body, metadata, bm25(chunk_search) as score
                    FROM chunk_search
                    WHERE project_id = ? AND chunk_search MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (project_id, fallback, limit),
                )
            rows = cursor.fetchall()
        results: List[dict[str, str]] = []
        for row in rows:
            metadata = {}
            if row[4]:
                try:
                    metadata = json.loads(row[4])
                except json.JSONDecodeError:
                    metadata = {}
            results.append(
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "title": row[2],
                    "text": row[3],
                    "metadata": metadata,
                    "score": float(row[5]),
                }
            )
        return results

    def close(self) -> None:
        """Close resources (included for API symmetry)."""

        # Connections are opened per-operation; nothing to close.
        return
