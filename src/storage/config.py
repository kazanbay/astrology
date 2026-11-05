"""Environment-backed configuration shared by the storage entry points.

No endpoint or credential is hardcoded: every value comes from the process
environment or the project ``.env`` file, so the same code runs against a local
PostgreSQL and against Neon.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_database_url(dsn: str | None = None) -> str:
    """Return a plain ``postgresql://`` DSN for asyncpg.

    An explicit ``dsn`` wins over the environment. The SQLAlchemy-style
    ``postgresql+asyncpg://`` prefix is accepted as well, because it is what
    hosting providers usually hand out.
    """
    if dsn is not None and str(dsn).strip():
        return _normalize_dsn(str(dsn).strip())

    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to the project's .env file "
            "or export it in the environment before using the database."
        )
    return _normalize_dsn(database_url)


def _normalize_dsn(value: str) -> str:
    if not value.startswith("postgres"):
        raise RuntimeError(f"DATABASE_URL must be a PostgreSQL DSN, got {value.split(':', 1)[0]!r}")
    if value.startswith("postgresql+asyncpg://"):
        return "postgresql://" + value.removeprefix("postgresql+asyncpg://")
    return value
