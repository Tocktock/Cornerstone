from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import InitializeResult


@asynccontextmanager
async def mcp_session(
    app: FastAPI,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[tuple[ClientSession, InitializeResult]]:
    base_url = "http://localhost:8000"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=base_url,
        headers=headers or {},
    ) as http_client, streamable_http_client(
        f"{base_url}/mcp",
        http_client=http_client,
    ) as streams:
        read_stream, write_stream, _ = streams
        async with ClientSession(read_stream, write_stream) as session:
            yield session, await session.initialize()
