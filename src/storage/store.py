"""PostgreSQL persistence for users, cached charts and request history.

The module exposes one narrow interface, ``ChartStore``, and its PostgreSQL
implementation. The pipeline talks to the protocol only, so tests and future
backends (another database, a read-through cache) do not need PostgreSQL.

Schema (created by ``init_db.py``):

* ``users`` - one row per person, Telegram id for the MVP, email reserved for
  web/app clients; also the owner's natal profile (date, time, coordinates,
  zone, chart hash) and the premium flag used by the forecast layer;
* ``user_contacts`` - natal profiles of other people, owned by one user and
  always resolved through the pair (contact_id, user_id);
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
from datetime import date, datetime, time
from typing import Any, Protocol, runtime_checkable

import asyncpg

from .config import resolve_database_url

PROFILE_NAME_LIMIT = 100
USER_NAME_LIMIT = 255
EMAIL_LIMIT = 255
CONTACT_NAME_LIMIT = 255
CONTACT_RELATION_LIMIT = 100

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

SELECT_PROFILE_SQL = """
SELECT user_id, telegram_id, email, first_name, last_name, display_name, is_premium,
       birth_date, birth_time, birth_time_unknown, birth_lat, birth_lon,
       birth_timezone, birth_city, birth_chart_hash
  FROM users
 WHERE ($1::bigint IS NOT NULL AND telegram_id = $1::bigint)
    OR ($2::text IS NOT NULL AND email = $2::text)
 ORDER BY (CASE WHEN telegram_id = $1::bigint THEN 0 ELSE 1 END)
 LIMIT 1
"""

SAVE_NATAL_PROFILE_SQL = """
UPDATE users
   SET birth_date = $2, birth_time = $3, birth_time_unknown = $4,
       birth_lat = $5, birth_lon = $6, birth_timezone = $7,
       birth_city = $8, birth_chart_hash = $9
 WHERE user_id = $1
"""

SET_PREMIUM_SQL = "UPDATE users SET is_premium = $2 WHERE user_id = $1"

SELECT_CONTACTS_SQL = """
SELECT contact_id, user_id, name, relation, birth_date, birth_time, birth_time_unknown,
       birth_lat, birth_lon, birth_timezone, birth_city, created_at
  FROM user_contacts
 WHERE user_id = $1
 ORDER BY contact_id
"""

SELECT_CONTACT_SQL = """
SELECT contact_id, user_id, name, relation, birth_date, birth_time, birth_time_unknown,
       birth_lat, birth_lon, birth_timezone, birth_city, created_at
  FROM user_contacts
 WHERE contact_id = $1 AND user_id = $2
"""

INSERT_CONTACT_SQL = """
INSERT INTO user_contacts
       (user_id, name, relation, birth_date, birth_time, birth_time_unknown,
        birth_lat, birth_lon, birth_timezone, birth_city)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
RETURNING contact_id, user_id, name, relation, birth_date, birth_time, birth_time_unknown,
          birth_lat, birth_lon, birth_timezone, birth_city, created_at
"""

DELETE_CONTACT_SQL = "DELETE FROM user_contacts WHERE contact_id = $1 AND user_id = $2"


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


@dataclass(frozen=True)
class NatalProfile:
    """Natal data of one person as stored on ``users`` (the owner's profile)."""

    birth_date: date
    birth_time: time | None
    birth_time_unknown: bool
    birth_lat: float
    birth_lon: float
    birth_timezone: str
    birth_city: str | None
    birth_chart_hash: str


@dataclass(frozen=True)
class UserProfile:
    """One ``users`` row: identity, premium flag and the owner's natal profile."""

    user_id: int
    telegram_id: int | None = None
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    is_premium: bool = False
    birth_date: date | None = None
    birth_time: time | None = None
    birth_time_unknown: bool = False
    birth_lat: float | None = None
    birth_lon: float | None = None
    birth_timezone: str | None = None
    birth_city: str | None = None
    birth_chart_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for LangGraph tools and logs (dates as ISO strings)."""
        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "telegram_id": self.telegram_id,
            "email": self.email,
            "display_name": self.display_name,
            "is_premium": self.is_premium,
            "birth_time_unknown": self.birth_time_unknown,
            "birth_city": self.birth_city,
            "birth_timezone": self.birth_timezone,
            "birth_chart_hash": self.birth_chart_hash,
        }
        for key in ("birth_date", "birth_time"):
            value = getattr(self, key)
            payload[key] = value.isoformat() if value is not None else None
        for key in ("birth_lat", "birth_lon"):
            payload[key] = getattr(self, key)
        return payload


@dataclass(frozen=True)
class ContactInput:
    """Validated payload for one ``user_contacts`` row."""

    name: str
    relation: str | None = None
    birth_date: date | None = None
    birth_time: time | None = None
    birth_time_unknown: bool = False
    birth_lat: float | None = None
    birth_lon: float | None = None
    birth_timezone: str | None = None
    birth_city: str | None = None


@dataclass(frozen=True)
class ContactProfile:
    """One ``user_contacts`` row: a person the owner may calculate charts for."""

    contact_id: int
    user_id: int
    name: str
    relation: str | None = None
    birth_date: date | None = None
    birth_time: time | None = None
    birth_time_unknown: bool = False
    birth_lat: float | None = None
    birth_lon: float | None = None
    birth_timezone: str | None = None
    birth_city: str | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form; see ``UserProfile.to_dict``."""
        payload: dict[str, Any] = {
            "contact_id": self.contact_id,
            "user_id": self.user_id,
            "name": self.name,
            "relation": self.relation,
            "birth_time_unknown": self.birth_time_unknown,
            "birth_city": self.birth_city,
            "birth_timezone": self.birth_timezone,
        }
        for key in ("birth_date", "birth_time"):
            value = getattr(self, key)
            payload[key] = value.isoformat() if value is not None else None
        for key in ("birth_lat", "birth_lon"):
            payload[key] = getattr(self, key)
        return payload


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

    async def get_user_profile(
        self, *, telegram_id: int | str | None = None, email: str | None = None
    ) -> UserProfile | None: ...

    async def save_user_natal_profile(self, user_id: int, profile: NatalProfile) -> bool: ...

    async def set_user_premium(self, user_id: int, is_premium: bool) -> bool: ...

    async def list_contacts(self, user_id: int) -> list[ContactProfile]: ...

    async def get_contact(self, contact_id: int, user_id: int) -> ContactProfile | None: ...

    async def add_contact(self, user_id: int, contact: ContactInput) -> ContactProfile: ...

    async def delete_contact(self, contact_id: int, user_id: int) -> bool: ...


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


def _contact_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("contact name must be a non-empty string")
    name = value.strip()
    if len(name) > CONTACT_NAME_LIMIT:
        raise ValueError(f"contact name must be at most {CONTACT_NAME_LIMIT} characters")
    return name


def _contact_relation(value: str | None) -> str | None:
    if value is None:
        return None
    relation = str(value).strip()
    if not relation:
        return None
    if len(relation) > CONTACT_RELATION_LIMIT:
        raise ValueError(f"relation must be at most {CONTACT_RELATION_LIMIT} characters")
    return relation


def _contact_city(value: str | None) -> str | None:
    if value is None:
        return None
    city = str(value).strip()
    return city[:255] or None


def _contact_from_row(row: asyncpg.Record) -> ContactProfile:
    return ContactProfile(
        contact_id=int(row["contact_id"]),
        user_id=int(row["user_id"]),
        name=str(row["name"]),
        relation=row["relation"],
        birth_date=row["birth_date"],
        birth_time=row["birth_time"],
        birth_time_unknown=bool(row["birth_time_unknown"]),
        birth_lat=row["birth_lat"],
        birth_lon=row["birth_lon"],
        birth_timezone=row["birth_timezone"],
        birth_city=row["birth_city"],
        created_at=row["created_at"],
    )


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

    async def get_user_profile(
        self, *, telegram_id: int | str | None = None, email: str | None = None
    ) -> UserProfile | None:
        """Return identity, premium flag and natal data of one account."""
        telegram = _telegram_id(telegram_id)
        address = _email(email)
        if telegram is None and address is None:
            raise ValueError("a user needs at least one of telegram_id or email")
        row = await self.pool.fetchrow(SELECT_PROFILE_SQL, telegram, address)
        if row is None:
            return None
        return UserProfile(
            user_id=int(row["user_id"]),
            telegram_id=row["telegram_id"],
            email=row["email"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            display_name=row["display_name"],
            is_premium=bool(row["is_premium"]),
            birth_date=row["birth_date"],
            birth_time=row["birth_time"],
            birth_time_unknown=bool(row["birth_time_unknown"]),
            birth_lat=row["birth_lat"],
            birth_lon=row["birth_lon"],
            birth_timezone=row["birth_timezone"],
            birth_city=row["birth_city"],
            birth_chart_hash=row["birth_chart_hash"],
        )

    async def save_user_natal_profile(self, user_id: int, profile: NatalProfile) -> bool:
        """Write the owner's natal profile; ``False`` means the user does not exist."""
        if len(profile.birth_chart_hash) != 64:
            raise ValueError("birth_chart_hash must be a 64-character SHA-256 hex string")
        status = await self.pool.execute(
            SAVE_NATAL_PROFILE_SQL,
            int(user_id),
            profile.birth_date,
            profile.birth_time,
            bool(profile.birth_time_unknown),
            profile.birth_lat,
            profile.birth_lon,
            profile.birth_timezone,
            profile.birth_city,
            profile.birth_chart_hash,
        )
        return status.endswith(" 1")

    async def set_user_premium(self, user_id: int, is_premium: bool) -> bool:
        """Toggle the premium flag; ``False`` means the user does not exist."""
        status = await self.pool.execute(SET_PREMIUM_SQL, int(user_id), bool(is_premium))
        return status.endswith(" 1")

    async def list_contacts(self, user_id: int) -> list[ContactProfile]:
        """Return every contact of one user, oldest first."""
        rows = await self.pool.fetch(SELECT_CONTACTS_SQL, int(user_id))
        return [_contact_from_row(row) for row in rows]

    async def get_contact(self, contact_id: int, user_id: int) -> ContactProfile | None:
        """Resolve a contact only through its owner: foreign ids look absent."""
        row = await self.pool.fetchrow(SELECT_CONTACT_SQL, int(contact_id), int(user_id))
        return _contact_from_row(row) if row is not None else None

    async def add_contact(self, user_id: int, contact: ContactInput) -> ContactProfile:
        """Insert one contact row and return the stored profile."""
        name = _contact_name(contact.name)
        relation = _contact_relation(contact.relation)
        if contact.birth_date is None:
            raise ValueError("a contact needs a birth date")
        if contact.birth_lat is None or contact.birth_lon is None:
            raise ValueError("a contact needs birth coordinates")
        if not contact.birth_timezone:
            raise ValueError("a contact needs a birth time zone")
        row = await self.pool.fetchrow(
            INSERT_CONTACT_SQL,
            int(user_id),
            name,
            relation,
            contact.birth_date,
            contact.birth_time,
            bool(contact.birth_time_unknown),
            float(contact.birth_lat),
            float(contact.birth_lon),
            str(contact.birth_timezone),
            _contact_city(contact.birth_city),
        )
        if row is None:  # pragma: no cover - would mean a lost row
            raise RuntimeError("could not insert the contact row")
        return _contact_from_row(row)

    async def delete_contact(self, contact_id: int, user_id: int) -> bool:
        """Delete a contact of this user; ``False`` when there was nothing to delete."""
        status = await self.pool.execute(DELETE_CONTACT_SQL, int(contact_id), int(user_id))
        return status.endswith(" 1")


