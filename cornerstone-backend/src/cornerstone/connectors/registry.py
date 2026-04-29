from __future__ import annotations

from functools import lru_cache

from cornerstone.config import Settings
from cornerstone.connectors.base import Connector
from cornerstone.connectors.manual import ManualConnector
from cornerstone.connectors.notion import NotionConnector
from cornerstone.connectors.security import TokenCipher
from cornerstone.schemas import DataSourceType


@lru_cache(maxsize=16)
def _cached_cipher(secret: str) -> TokenCipher:
    return TokenCipher(secret)


def get_token_cipher(settings: Settings) -> TokenCipher:
    return _cached_cipher(settings.connector_encryption_secret)


def get_connector(provider: DataSourceType, settings: Settings) -> Connector:
    if provider == DataSourceType.NOTION:
        return NotionConnector(settings)
    if provider == DataSourceType.MANUAL:
        return ManualConnector(settings)
    raise KeyError(f"Unsupported connector provider: {provider}")
