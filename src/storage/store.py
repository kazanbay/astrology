"""PostgreSQL persistence for users, cached charts and request history.

The module exposes one narrow interface, ``ChartStore``, and its PostgreSQL
implementation. The pipeline talks to the protocol only, so tests and future
backends (another database, a read-through cache) do not need PostgreSQL.

Schema (created by ``init_db.py``):

* ``users`` - one row per person, Telegram id for the MVP, email reserved for
  web/app clients;
* ``charts_cache`` - one row per unique birth input, shared by every user with
  the same chart (twins included);
* ``user_chart_requests`` - append-only history of "who asked for which chart
  under which profile name" plus the report it received;
* ``llm_generations`` - cached report per (chart, prompt version, model,
  temperature).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import asyncpg

from .config import resolve_database_url

PROFILE_NAME_LIMIT = 100
USER_NAME_LIMIT = 255
EMAIL_LIMIT = 255

UPSERT_USER_SQL = """
WITH inserted AS (
    INSERT INTO users (telegram_id, email, first_name, last_name, display_name)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT DO NOTHING
    RETURNING user_id
),
candidates AS (
    SELECT user_id, 0 AS priority FROM inserted
    UNION ALL
    SELECT user_id,
           CASE WHEN telegram_id = $1::bigint THEN 1 ELSE 2 END AS priority
      FROM users
     WHERE ($1::bigint IS NOT NULL AND telegram_id = $1::bigint)
        OR ($2::text IS NOT NULL AND email = $2::text)
)
SELECT user_id FROM candidates ORDER BY priority LIMIT 1
"""

SET_DISPLAY_NAME_SQL = """
UPDATE users SET display_name = $2 WHERE user_id = $1
"""

SELECT_USER_SQL = """
SELECT user_id
  FROM users
 WHERE ($1::bigint IS NOT NULL AND telegram_id = $1::bigint)
    OR ($2::text IS NOT NULL AND email = $2::text)
 ORDER BY (CASE WHEN telegram_id = $1::bigint THEN 0 ELSE 1 END)
 LIMIT 1
"""

SELECT_CHART_SQL = """
SELECT chart_hash, input_params, calculated_chart, retrieved_chunks, created_at
  FROM charts_cache
 WHERE chart_hash = $1
"""

INSERT_CHART_SQL = """
INSERT INTO charts_cache (chart_hash, input_params, calculated_chart, retrieved_chunks)
VALUES ($1, $2, $3, $4)
ON CONFLICT (chart_hash) DO NOTHING
RETURNING chart_hash
"""

INSERT_REQUEST_SQL = """
INSERT INTO user_chart_requests (user_id, chart_hash, profile_name)
VALUES ($1, $2, $3)
RETURNING request_id
"""

SELECT_REQUESTS_SQL = """
SELECT r.request_id, r.chart_hash, r.profile_name, r.requested_at, r.generation_id, c.input_params
  FROM user_chart_requests AS r
  JOIN charts_cache AS c ON c.chart_hash = r.chart_hash
 WHERE r.user_id = $1
 ORDER BY r.requested_at DESC, r.request_id DESC
 LIMIT $2
"""

SELECT_GENERATION_SQL = """
SELECT generated_text
  FROM llm_generations
 WHERE generation_id = $1
"""

INSERT_GENERATION_SQL = """
INSERT INTO llm_generations
       (generation_id, chart_hash, prompt_version, model_name, temperature, generated_text)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (generation_id) DO NOTHING
RETURNING generation_id
"""

ATTACH_GENERATION_SQL = """
UPDATE user_chart_requests
   SET generation_id = $2
 WHERE request_id = $1
"""


@dataclass(frozen=True)
class CachedChart:
    """One ``charts_cache`` row: birth inputs plus everything derived from them."""

    chart_hash: str
    input_params: dict[str, Any]
    calculated_chart: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    created_at: datetime | None = None


@dataclass(frozen=True)
class RequestRecord:
    """One history entry, enriched with the cached birth inputs for display."""

    request_id: int
    chart_hash: str
    profile_name: str
    requested_at: datetime
    input_params: dict[str, Any] = field(default_factory=dict)
    #: Report attached to this request; ``None`` when generation was skipped.
    generation_id: str | None = None


@dataclass(frozen=True)
class GenerationRecord:
    """One ``llm_generations`` row: a cached report and its identity."""

    generation_id: str
    chart_hash: str
    prompt_version: str
    model_name: str
    temperature: float
    generated_text: str
    created_at: datetime | None = None


@runtime_checkable
class ChartStore(Protocol):
    """Persistence contract used by ``ChartPipeline``."""

    async def upsert_user(
        self,
        *,
        telegram_id: int | None = None,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        display_name: str | None = None,
    ) -> int: ...

    async def find_user(
        self, *, telegram_id: int | None = None, email: str | None = None
    ) -> int | None: ...

    async def get_chart(self, chart_hash: str) -> CachedChart | None: ...

    async def save_chart(self, cached: CachedChart) -> bool: ...

    async def record_request(
        self, *, user_id: int, chart_hash: str, profile_name: str
    ) -> int: ...

    async def list_requests(self, user_id: int, *, limit: int = 10) -> list[RequestRecord]: ...

    async def get_generation(self, generation_id: str) -> str | None: ...

    async def save_generation(self, record: GenerationRecord) -> bool: ...

    async def attach_generation(self, request_id: int, generation_id: str) -> None: ...


async def _configure_connection(connection: asyncpg.Connection) -> None:
    """Decode JSONB into Python objects on every pooled connection."""
    for type_name in ("json", "jsonb"):
        await connection.set_type_codec(
            type_name,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


def _as_json(value: Any, default: Any) -> Any:
    """Accept both decoded values (pool codec) and raw JSON strings."""
    if value is None:
        return default
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except ValueError as exc:  # pragma: no cover - corrupt row, cannot recover
            raise ValueError("charts_cache contains a JSON column that is not valid JSON") from exc
    return value


def normalize_profile_name(value: str) -> str:
    """Validate and trim the profile label shown in the user's history."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("profile_name must be a non-empty string")
    name = value.strip()
    if len(name) > PROFILE_NAME_LIMIT:
        raise ValueError(f"profile_name must be at most {PROFILE_NAME_LIMIT} characters")
    return name


def _telegram_id(value: int | str | None) -> int | None:
    """Validate a Telegram id; ``0`` is reserved for the CLI/service profile."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("telegram_id must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"telegram_id must be an integer, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"telegram_id must not be negative, got {number}")
    return number


def _email(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    address = str(value).strip().lower()
    if len(address) > EMAIL_LIMIT or "@" not in address:
        raise ValueError(f"email looks invalid: {value!r}")
    return address


def _optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    return name[:USER_NAME_LIMIT] or None


def _display_name(value: str | None) -> str | None:
    """Name the user chose for themselves; used for address and mailings."""
    if value is None:
        return None
    name = str(value).strip()
    return name[:USER_NAME_LIMIT] or None


class PostgresChartStore:
    """asyncpg-backed ``ChartStore``."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(
        cls,
        dsn: str | None = None,
        *,
        min_size: int = 1,
        max_size: int = 5,
    ) -> "PostgresChartStore":
        """Open a connection pool; the DSN falls back to ``DATABASE_URL``."""
        if min_size < 1 or max_size < min_size:
            raise ValueError("pool sizes must satisfy 1 <= min_size <= max_size")
        pool = await asyncpg.create_pool(
            resolve_database_url(dsn),
            min_size=min_size,
            max_size=max_size,
            init=_configure_connection,
        )
        if pool is None:  # pragma: no cover - asyncpg never returns None
            raise RuntimeError("could not create a PostgreSQL connection pool")
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def upsert_user(
        self,
        *,
        telegram_id: int | str | None = None,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        display_name: str | None = None,
    ) -> int:
        """Return the internal ``user_id``, creating the account on first contact."""
        telegram = _telegram_id(telegram_id)
        address = _email(email)
        if telegram is None and address is None:
            raise ValueError("a user needs at least one of telegram_id or email")
        display = _display_name(display_name)
        user_id = await self.pool.fetchval(
            UPSERT_USER_SQL,
            telegram,
            address,
            _optional_name(first_name),
            _optional_name(last_name),
            display,
        )
        if user_id is None:  # pragma: no cover - would mean a lost row
            raise RuntimeError("could not resolve user_id after upsert")
        if display:
            await self.pool.execute(SET_DISPLAY_NAME_SQL, int(user_id), display)
        return int(user_id)

    async def find_user(
        self, *, telegram_id: int | str | None = None, email: str | None = None
    ) -> int | None:
        """Return the internal id of an existing user, or ``None``."""
        telegram = _telegram_id(telegram_id)
        address = _email(email)
        if telegram is None and address is None:
            raise ValueError("a user needs at least one of telegram_id or email")
        user_id = await self.pool.fetchval(SELECT_USER_SQL, telegram, address)
        return int(user_id) if user_id is not None else None

    async def get_chart(self, chart_hash: str) -> CachedChart | None:
        """Return the cached chart, or ``None`` on a cache miss."""
        if not isinstance(chart_hash, str) or len(chart_hash) != 64:
            raise ValueError("chart_hash must be a 64-character SHA-256 hex string")
        row = await self.pool.fetchrow(SELECT_CHART_SQL, chart_hash)
        if row is None:
            return None
        return CachedChart(
            chart_hash=str(row["chart_hash"]),
            input_params=_as_json(row["input_params"], {}),
            calculated_chart=_as_json(row["calculated_chart"], {}),
            retrieved_chunks=_as_json(row["retrieved_chunks"], []),
            created_at=row["created_at"],
        )

    async def save_chart(self, cached: CachedChart) -> bool:
        """Insert a cache row; ``False`` means another request stored it first.

        Concurrent misses compute the same deterministic payload, so losing the
        race is harmless and the loser does not overwrite the winner.
        """
        if not isinstance(cached, CachedChart):
            raise ValueError("save_chart expects a CachedChart")
        if len(cached.chart_hash) != 64:
            raise ValueError("chart_hash must be a 64-character SHA-256 hex string")
        stored = await self.pool.fetchval(
            INSERT_CHART_SQL,
            cached.chart_hash,
            cached.input_params,
            cached.calculated_chart,
            cached.retrieved_chunks,
        )
        return stored is not None

    async def record_request(
        self, *, user_id: int, chart_hash: str, profile_name: str
    ) -> int:
        """Append one history entry and return its ``request_id``."""
        name = normalize_profile_name(profile_name)
        request_id = await self.pool.fetchval(INSERT_REQUEST_SQL, int(user_id), chart_hash, name)
        if request_id is None:  # pragma: no cover - would mean a lost row
            raise RuntimeError("could not record the chart request")
        return int(request_id)

    async def list_requests(self, user_id: int, *, limit: int = 10) -> list[RequestRecord]:
        """Return the latest requests of one user, newest first."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        rows = await self.pool.fetch(SELECT_REQUESTS_SQL, int(user_id), int(limit))
        return [
            RequestRecord(
                request_id=int(row["request_id"]),
                chart_hash=str(row["chart_hash"]),
                profile_name=str(row["profile_name"] or ""),
                requested_at=row["requested_at"],
                input_params=_as_json(row["input_params"], {}),
                generation_id=row["generation_id"],
            )
            for row in rows
        ]

    async def get_generation(self, generation_id: str) -> str | None:
        """Return the cached report text, or ``None`` on a miss."""
        if not isinstance(generation_id, str) or len(generation_id) != 64:
            raise ValueError("generation_id must be a 64-character SHA-256 hex string")
        text = await self.pool.fetchval(SELECT_GENERATION_SQL, generation_id)
        return str(text) if text is not None else None

    async def save_generation(self, record: GenerationRecord) -> bool:
        """Insert a report; ``False`` means another request stored it first."""
        if not isinstance(record, GenerationRecord):
            raise ValueError("save_generation expects a GenerationRecord")
        if len(record.generation_id) != 64:
            raise ValueError("generation_id must be a 64-character SHA-256 hex string")
        stored = await self.pool.fetchval(
            INSERT_GENERATION_SQL,
            record.generation_id,
            record.chart_hash,
            record.prompt_version,
            record.model_name,
            record.temperature,
            record.generated_text,
        )
        return stored is not None

    async def attach_generation(self, request_id: int, generation_id: str) -> None:
        """Link a history entry to the report it received."""
        if len(generation_id) != 64:
            raise ValueError("generation_id must be a 64-character SHA-256 hex string")
        status = await self.pool.execute(ATTACH_GENERATION_SQL, int(request_id), generation_id)
        if status.endswith(" 0"):
            raise ValueError(f"request_id {request_id} does not exist")


