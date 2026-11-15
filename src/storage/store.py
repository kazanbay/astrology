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

#: Allowed audit sources of a premium grant.
PREMIUM_SOURCES: tuple[str, ...] = ("manual", "referral", "telegram_stars", "other")

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

SELECT_FORECAST_REPORT_SQL = """
SELECT generated_text
  FROM forecast_reports
 WHERE report_id = $1
"""

INSERT_FORECAST_REPORT_SQL = """
INSERT INTO forecast_reports
       (report_id, chart_hash, interval, forecast_date, prompt_version, model_name,
        temperature, generated_text)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (report_id) DO NOTHING
RETURNING report_id
"""

SELECT_PROFILE_SQL = """
SELECT user_id, telegram_id, email, first_name, last_name, display_name, is_premium,
       birth_date, birth_time, birth_time_unknown, birth_lat, birth_lon,
       birth_timezone, birth_city, birth_chart_hash,
       profile_changes_count, profile_changes_period, premium_until, referral_code
  FROM users
 WHERE ($1::bigint IS NOT NULL AND telegram_id = $1::bigint)
    OR ($2::text IS NOT NULL AND email = $2::text)
 ORDER BY (CASE WHEN telegram_id = $1::bigint THEN 0 ELSE 1 END)
 LIMIT 1
"""

#: First day of the current calendar month (UTC): a new month restarts the
#: profile-change allowance.
_PROFILE_MONTH_START_SQL = "date_trunc('month', now() AT TIME ZONE 'UTC')::date"

SAVE_NATAL_PROFILE_SQL = f"""
UPDATE users
   SET birth_date = $2, birth_time = $3, birth_time_unknown = $4,
       birth_lat = $5, birth_lon = $6, birth_timezone = $7,
       birth_city = $8, birth_chart_hash = $9,
       profile_changes_count = CASE
           WHEN $10::boolean AND profile_changes_period = {_PROFILE_MONTH_START_SQL}
               THEN profile_changes_count + 1
           WHEN $10::boolean THEN 1
           ELSE profile_changes_count END,
       profile_changes_period = CASE
           WHEN $10::boolean THEN {_PROFILE_MONTH_START_SQL}
           ELSE profile_changes_period END
 WHERE user_id = $1
   AND (NOT $10::boolean
        OR $11::int IS NULL
        OR profile_changes_period IS DISTINCT FROM {_PROFILE_MONTH_START_SQL}
        OR profile_changes_count < $11::int)
"""

SET_PREMIUM_SQL = "UPDATE users SET is_premium = $2 WHERE user_id = $1"

GRANT_PREMIUM_SQL = """
WITH before AS (
    SELECT user_id, premium_until FROM users WHERE user_id = $1
), updated AS (
    UPDATE users
       SET premium_until = GREATEST(COALESCE(premium_until, now()), now())
                          + make_interval(months => $2::int)
     WHERE user_id = $1
    RETURNING user_id, premium_until
), logged AS (
    INSERT INTO premium_events
           (user_id, source, months, stars, plan_key, actor_user_id, reference, reason,
            premium_until_before, premium_until_after)
    SELECT updated.user_id, $3, $2::int, $4::int, $5, $6::int, $7, $8,
           before.premium_until, updated.premium_until
      FROM updated LEFT JOIN before ON before.user_id = updated.user_id
    RETURNING event_id
)
SELECT premium_until FROM updated
"""

ENSURE_REFERRAL_CODE_SQL = """
UPDATE users SET referral_code = $2
 WHERE user_id = $1 AND referral_code IS NULL
RETURNING referral_code
"""

GET_REFERRAL_CODE_SQL = "SELECT referral_code FROM users WHERE user_id = $1"

FIND_REFERRER_SQL = """
SELECT user_id, telegram_id FROM users WHERE referral_code = $1
"""

CREATE_REFERRAL_SQL = """
WITH inserted AS (
    INSERT INTO referrals (referrer_user_id, referee_user_id, bonus_months)
    VALUES ($1, $2, $3)
    ON CONFLICT (referee_user_id) DO NOTHING
    RETURNING referral_id
)
SELECT referral_id FROM inserted
"""

REFERRAL_STATS_SQL = """
SELECT count(*) AS invited, COALESCE(sum(bonus_months), 0) AS bonus_months
  FROM referrals
 WHERE referrer_user_id = $1
"""

COMPLETE_PAYMENT_SQL = """
WITH inserted AS (
    INSERT INTO payments (charge_id, user_id, plan_key, months, stars)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (charge_id) DO NOTHING
    RETURNING payment_id
), before AS (
    SELECT user_id, premium_until FROM users WHERE user_id = $2
), updated AS (
    UPDATE users
       SET premium_until = GREATEST(COALESCE(premium_until, now()), now())
                          + make_interval(months => $4::int)
     WHERE user_id = $2 AND EXISTS (SELECT 1 FROM inserted)
    RETURNING user_id, premium_until
), logged AS (
    INSERT INTO premium_events
           (user_id, source, months, stars, plan_key, actor_user_id, reference, reason,
            premium_until_before, premium_until_after)
    SELECT updated.user_id, 'telegram_stars', $4::int, $5::int, $3, NULL, $1, $6,
           before.premium_until, updated.premium_until
      FROM updated LEFT JOIN before ON before.user_id = updated.user_id
    RETURNING event_id
)
SELECT premium_until FROM updated
"""

SELECT_PREMIUM_EVENTS_SQL = """
SELECT event_id, user_id, source, months, stars, plan_key, actor_user_id,
       reference, reason, premium_until_before, premium_until_after, created_at
  FROM premium_events
 WHERE user_id = $1
 ORDER BY created_at DESC, event_id DESC
 LIMIT $2
"""

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

UPDATE_CONTACT_SQL = """
UPDATE user_contacts
   SET name = $3, relation = $4, birth_date = $5, birth_time = $6,
       birth_time_unknown = $7, birth_lat = $8, birth_lon = $9,
       birth_timezone = $10, birth_city = $11
 WHERE contact_id = $1 AND user_id = $2
"""

SELECT_SYNASTRY_REPORT_SQL = """
SELECT generated_text
  FROM synastry_reports
 WHERE report_id = $1
"""

INSERT_SYNASTRY_REPORT_SQL = """
INSERT INTO synastry_reports
       (report_id, chart_hash_a, chart_hash_b, prompt_version, model_name,
        temperature, generated_text)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (report_id) DO NOTHING
RETURNING report_id
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


@dataclass(frozen=True)
class ForecastReportRecord:
    """One ``forecast_reports`` row: a cached LLM interpretation of a forecast."""

    report_id: str
    chart_hash: str
    interval: str
    forecast_date: date
    prompt_version: str
    model_name: str
    temperature: float
    generated_text: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class SynastryReportRecord:
    """One ``synastry_reports`` row: a cached LLM reading of one pair."""

    report_id: str
    chart_hash_a: str
    chart_hash_b: str
    prompt_version: str
    model_name: str
    temperature: float
    generated_text: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class PremiumEvent:
    """One ``premium_events`` row: an audited premium grant."""

    event_id: int
    user_id: int
    source: str
    months: int
    stars: int | None = None
    plan_key: str | None = None
    actor_user_id: int | None = None
    reference: str | None = None
    reason: str | None = None
    premium_until_before: datetime | None = None
    premium_until_after: datetime | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for tools, logs and admin views."""
        return {
            "event_id": self.event_id,
            "source": self.source,
            "months": self.months,
            "stars": self.stars,
            "plan_key": self.plan_key,
            "actor_user_id": self.actor_user_id,
            "reference": self.reference,
            "reason": self.reason,
            "premium_until_before": (
                self.premium_until_before.isoformat()
                if self.premium_until_before is not None
                else None
            ),
            "premium_until_after": (
                self.premium_until_after.isoformat()
                if self.premium_until_after is not None
                else None
            ),
            "created_at": (
                self.created_at.isoformat() if self.created_at is not None else None
            ),
        }


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
    #: Natal-profile changes spent in ``profile_changes_period``.
    profile_changes_count: int = 0
    #: First day of the month the counter belongs to (UTC).
    profile_changes_period: date | None = None
    #: Paid premium period end (UTC); ``None`` for free accounts.
    premium_until: datetime | None = None
    #: Short public token used in the referral link.
    referral_code: str | None = None

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
            "profile_changes_count": self.profile_changes_count,
            "profile_changes_period": (
                self.profile_changes_period.isoformat()
                if self.profile_changes_period is not None
                else None
            ),
            "premium_until": (
                self.premium_until.isoformat() if self.premium_until is not None else None
            ),
            "referral_code": self.referral_code,
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

    async def get_forecast_report(self, report_id: str) -> str | None: ...

    async def save_forecast_report(self, record: ForecastReportRecord) -> bool: ...

    async def get_synastry_report(self, report_id: str) -> str | None: ...

    async def save_synastry_report(self, record: SynastryReportRecord) -> bool: ...

    async def get_user_profile(
        self, *, telegram_id: int | str | None = None, email: str | None = None
    ) -> UserProfile | None: ...

    async def save_user_natal_profile(
        self,
        user_id: int,
        profile: NatalProfile,
        *,
        count_change: bool = False,
        max_changes_per_month: int | None = None,
    ) -> bool: ...

    async def set_user_premium(self, user_id: int, is_premium: bool) -> bool: ...

    async def list_contacts(self, user_id: int) -> list[ContactProfile]: ...

    async def get_contact(self, contact_id: int, user_id: int) -> ContactProfile | None: ...

    async def add_contact(self, user_id: int, contact: ContactInput) -> ContactProfile: ...

    async def delete_contact(self, contact_id: int, user_id: int) -> bool: ...

    async def update_contact(
        self, contact_id: int, user_id: int, contact: ContactInput
    ) -> bool: ...

    async def grant_premium(
        self,
        user_id: int,
        months: int,
        *,
        source: str = "other",
        stars: int | None = None,
        plan_key: str | None = None,
        actor_user_id: int | None = None,
        reference: str | None = None,
        reason: str | None = None,
    ) -> datetime | None: ...

    async def ensure_referral_code(self, user_id: int, code: str) -> str | None: ...

    async def get_referral_code(self, user_id: int) -> str | None: ...

    async def find_user_by_referral_code(
        self, code: str
    ) -> tuple[int, int | None] | None: ...

    async def create_referral(
        self, *, referrer_user_id: int, referee_user_id: int, bonus_months: int
    ) -> bool: ...

    async def referral_stats(self, user_id: int) -> dict[str, int]: ...

    async def complete_payment(
        self,
        *,
        charge_id: str,
        user_id: int,
        plan_key: str,
        months: int,
        stars: int,
        reason: str | None = None,
    ) -> datetime | None: ...

    async def list_premium_events(
        self, user_id: int, *, limit: int = 20
    ) -> list[PremiumEvent]: ...


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

    async def get_forecast_report(self, report_id: str) -> str | None:
        """Return the cached forecast interpretation, or ``None`` on a miss."""
        if not isinstance(report_id, str) or len(report_id) != 64:
            raise ValueError("report_id must be a 64-character SHA-256 hex string")
        text = await self.pool.fetchval(SELECT_FORECAST_REPORT_SQL, report_id)
        return str(text) if text is not None else None

    async def save_forecast_report(self, record: ForecastReportRecord) -> bool:
        """Insert a forecast interpretation; ``False`` means another request won."""
        if not isinstance(record, ForecastReportRecord):
            raise ValueError("save_forecast_report expects a ForecastReportRecord")
        if len(record.report_id) != 64:
            raise ValueError("report_id must be a 64-character SHA-256 hex string")
        if len(record.chart_hash) != 64:
            raise ValueError("chart_hash must be a 64-character SHA-256 hex string")
        stored = await self.pool.fetchval(
            INSERT_FORECAST_REPORT_SQL,
            record.report_id,
            record.chart_hash,
            record.interval,
            record.forecast_date,
            record.prompt_version,
            record.model_name,
            record.temperature,
            record.generated_text,
        )
        return stored is not None

    async def get_synastry_report(self, report_id: str) -> str | None:
        """Return the cached synastry reading, or ``None`` on a miss."""
        if not isinstance(report_id, str) or len(report_id) != 64:
            raise ValueError("report_id must be a 64-character SHA-256 hex string")
        text = await self.pool.fetchval(SELECT_SYNASTRY_REPORT_SQL, report_id)
        return str(text) if text is not None else None

    async def save_synastry_report(self, record: SynastryReportRecord) -> bool:
        """Insert a synastry reading; ``False`` means another request won."""
        if not isinstance(record, SynastryReportRecord):
            raise ValueError("save_synastry_report expects a SynastryReportRecord")
        for value, label in (
            (record.report_id, "report_id"),
            (record.chart_hash_a, "chart_hash_a"),
            (record.chart_hash_b, "chart_hash_b"),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be a 64-character SHA-256 hex string")
        stored = await self.pool.fetchval(
            INSERT_SYNASTRY_REPORT_SQL,
            record.report_id,
            record.chart_hash_a,
            record.chart_hash_b,
            record.prompt_version,
            record.model_name,
            record.temperature,
            record.generated_text,
        )
        return stored is not None

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
            profile_changes_count=int(row["profile_changes_count"] or 0),
            profile_changes_period=row["profile_changes_period"],
            premium_until=row["premium_until"],
            referral_code=row["referral_code"],
        )

    async def save_user_natal_profile(
        self,
        user_id: int,
        profile: NatalProfile,
        *,
        count_change: bool = False,
        max_changes_per_month: int | None = None,
    ) -> bool:
        """Write the owner's natal profile.

        ``count_change`` marks the write as a real change: the monthly counter
        is incremented, or reset when the stored period is a past month. With
        ``max_changes_per_month`` the UPDATE is guarded by the limit, so a
        concurrent double-tap cannot exceed it. ``False`` means the guard
        rejected the write (limit reached) or the user does not exist.
        """
        if len(profile.birth_chart_hash) != 64:
            raise ValueError("birth_chart_hash must be a 64-character SHA-256 hex string")
        if max_changes_per_month is not None and max_changes_per_month < 1:
            raise ValueError("max_changes_per_month must be positive when provided")
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
            bool(count_change),
            max_changes_per_month,
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

    async def update_contact(
        self, contact_id: int, user_id: int, contact: ContactInput
    ) -> bool:
        """Rewrite a contact of this user; ``False`` when it does not exist."""
        name = _contact_name(contact.name)
        relation = _contact_relation(contact.relation)
        if contact.birth_date is None:
            raise ValueError("a contact needs a birth date")
        if contact.birth_lat is None or contact.birth_lon is None:
            raise ValueError("a contact needs birth coordinates")
        if not contact.birth_timezone:
            raise ValueError("a contact needs a birth time zone")
        status = await self.pool.execute(
            UPDATE_CONTACT_SQL,
            int(contact_id),
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
        return status.endswith(" 1")

    async def grant_premium(
        self,
        user_id: int,
        months: int,
        *,
        source: str = "other",
        stars: int | None = None,
        plan_key: str | None = None,
        actor_user_id: int | None = None,
        reference: str | None = None,
        reason: str | None = None,
    ) -> datetime | None:
        """Extend the paid period by whole months and write the audit event.

        ``None`` means no such user. ``source`` names the mechanism (manual /
        referral / telegram_stars / other); ``actor_user_id`` is the account
        that caused the grant (the referee for a referral bonus), ``reference``
        holds the charge id or referral code, ``reason`` is free text.
        """
        if int(months) < 0:
            raise ValueError("months must not be negative")
        if source not in PREMIUM_SOURCES:
            raise ValueError(f"source must be one of {', '.join(PREMIUM_SOURCES)}")
        until = await self.pool.fetchval(
            GRANT_PREMIUM_SQL,
            int(user_id),
            int(months),
            source,
            stars,
            plan_key,
            actor_user_id,
            reference,
            reason,
        )
        return until

    async def ensure_referral_code(self, user_id: int, code: str) -> str | None:
        """Set the referral code once; ``None`` when the account already has one."""
        value = str(code).strip()
        if not value or len(value) > 16:
            raise ValueError("referral code must be 1-16 characters")
        created = await self.pool.fetchval(ENSURE_REFERRAL_CODE_SQL, int(user_id), value)
        return str(created) if created is not None else None

    async def get_referral_code(self, user_id: int) -> str | None:
        """Return the account's referral code, or ``None``."""
        code = await self.pool.fetchval(GET_REFERRAL_CODE_SQL, int(user_id))
        return str(code) if code is not None else None

    async def find_user_by_referral_code(
        self, code: str
    ) -> tuple[int, int | None] | None:
        """Resolve a referral code to ``(user_id, telegram_id)``."""
        value = str(code).strip()
        if not value:
            return None
        row = await self.pool.fetchrow(FIND_REFERRER_SQL, value)
        if row is None:
            return None
        return int(row["user_id"]), row["telegram_id"]

    async def create_referral(
        self, *, referrer_user_id: int, referee_user_id: int, bonus_months: int
    ) -> bool:
        """Record one attribution; ``False`` means the referee was already invited."""
        if int(bonus_months) < 0:
            raise ValueError("bonus_months must not be negative")
        created = await self.pool.fetchval(
            CREATE_REFERRAL_SQL, int(referrer_user_id), int(referee_user_id), int(bonus_months)
        )
        return created is not None

    async def referral_stats(self, user_id: int) -> dict[str, int]:
        """Invited users and the bonus months they brought."""
        row = await self.pool.fetchrow(REFERRAL_STATS_SQL, int(user_id))
        if row is None:  # pragma: no cover - aggregate always returns a row
            return {"invited": 0, "bonus_months": 0}
        return {
            "invited": int(row["invited"] or 0),
            "bonus_months": int(row["bonus_months"] or 0),
        }

    async def complete_payment(
        self,
        *,
        charge_id: str,
        user_id: int,
        plan_key: str,
        months: int,
        stars: int,
        reason: str | None = None,
    ) -> datetime | None:
        """Record a Stars payment and extend premium in one transaction.

        ``charge_id`` is the idempotency key: a repeated notification inserts
        nothing and returns ``None`` instead of granting the term twice. Every
        successful payment leaves a ``telegram_stars`` row in premium_events.
        """
        charge = str(charge_id).strip()
        if not charge:
            raise ValueError("charge_id must be a non-empty string")
        if int(months) < 1 or int(stars) < 1:
            raise ValueError("months and stars must be positive")
        until = await self.pool.fetchval(
            COMPLETE_PAYMENT_SQL,
            charge,
            int(user_id),
            str(plan_key),
            int(months),
            int(stars),
            reason,
        )
        return until

    async def list_premium_events(
        self, user_id: int, *, limit: int = 20
    ) -> list[PremiumEvent]:
        """Return the latest premium grants of one account, newest first."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        rows = await self.pool.fetch(SELECT_PREMIUM_EVENTS_SQL, int(user_id), int(limit))
        return [
            PremiumEvent(
                event_id=int(row["event_id"]),
                user_id=int(row["user_id"]),
                source=str(row["source"]),
                months=int(row["months"]),
                stars=row["stars"],
                plan_key=row["plan_key"],
                actor_user_id=row["actor_user_id"],
                reference=row["reference"],
                reason=row["reason"],
                premium_until_before=row["premium_until_before"],
                premium_until_after=row["premium_until_after"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


