from __future__ import annotations

from cornerstone.config import get_settings
from cornerstone.persistence.extensions import parse_required_extensions, verify_required_extensions
from cornerstone.persistence.store import create_sqlalchemy_engine

if __name__ == "__main__":
    settings = get_settings()
    engine = create_sqlalchemy_engine(settings.database_url, echo=settings.database_echo)
    installed = verify_required_extensions(
        engine,
        parse_required_extensions(settings.postgres_required_extensions_raw),
    )
    required = parse_required_extensions(settings.postgres_required_extensions_raw)
    print("required_extensions=" + ",".join(required))
    print("installed_extensions=" + ",".join(sorted(installed)))
