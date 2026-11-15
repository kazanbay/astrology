"""Account-level natal profiles: owner profile, premium contacts, permissions.

The forecast and synastry tools are the security boundary required by the TZ:
they never accept birth data from a request. Every chart is rebuilt from the
account's stored natal profile (owner) or from a ``user_contacts`` row owned by
that account, so arbitrary coordinates or dates cannot enter the calculation
through the bot or a LangGraph tool call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Protocol

from storage.birth import (
    BirthData,
    BirthInputError,
    local_to_utc,
    parse_birth_date,
    parse_birth_time,
)
from storage.hashing import generate_chart_hash
from storage.store import ChartStore, ContactInput, ContactProfile, NatalProfile, UserProfile

#: The pipeline defaults the bot uses; forecasts must produce charts that
#: match the owner's cached chart identity.
HOUSE_SYSTEM = "P"
LANGUAGE = "ru"

#: When the birth time is unknown the chart is built for 12:00 local time,
#: exactly like the onboarding wizard does.
DEFAULT_TIME = time(12, 0)

CONTACT_COMMAND_USAGE = "/contact Мария | супруга | 15.03.1992 14:30 | Казань"

_PREMIUM_ENV = "PREMIUM_TELEGRAM_IDS"

PREMIUM_REQUIRED_MESSAGE = (
    "👑 Прогнозы для других людей и проверка совместимости — премиум-функция.\n"
    "Свой собственный прогноз доступен всегда: «📅 Прогноз» → «👤 Мой прогноз»."
)

CONTACT_PERMISSION_MESSAGE = "Контакт не найден или принадлежит другому аккаунту."

PROFILE_MISSING_MESSAGE = (
    "Профиль не найден. Сначала рассчитайте свою карту через «🔮 Рассчитать карту»."
)

PROFILE_NATAL_MISSING_MESSAGE = (
    "В профиле нет данных рождения. Рассчитайте карту через «🔮 Рассчитать карту», "
    "и я сохраню их для прогнозов."
)

#: Env knob: natal-profile changes allowed per calendar month. ``0`` or a
#: negative value disables the limit; invalid values fall back to the default.
PROFILE_CHANGE_LIMIT_ENV = "PROFILE_CHANGES_PER_MONTH"
DEFAULT_PROFILE_CHANGE_LIMIT = 3

PROFILE_CHANGE_LIMIT_MESSAGE = (
    "Данные рождения можно менять не чаще {limit} раз в месяц. "
    "Лимит на этот месяц исчерпан — следующая смена станет доступна с 1-го числа."
)

logger = logging.getLogger(__name__)


class ProfileError(RuntimeError):
    """The account has no usable natal profile (or no account at all)."""


class ContactPermissionError(RuntimeError):
    """A contact id was passed that does not belong to this account."""


class PremiumRequiredError(RuntimeError):
    """The feature is premium and the account has no premium access."""


class ProfileChangeLimitError(RuntimeError):
    """The monthly natal-profile change limit is exhausted."""


class NatalRecord(Protocol):
    """Structural contract shared by ``UserProfile`` and ``ContactProfile``."""

    birth_date: date | None
    birth_time: time | None
    birth_time_unknown: bool
    birth_lat: float | None
    birth_lon: float | None
    birth_timezone: str | None
    birth_city: str | None


def premium_telegram_ids() -> frozenset[int]:
    """Telegram ids granted premium through ``PREMIUM_TELEGRAM_IDS``.

    The database flag (``users.is_premium``) is the source of truth; this env
    override exists so hosting environments can grant premium without touching
    the database. Nothing is hardcoded: the value comes from the environment.
    """
    raw = os.getenv(_PREMIUM_ENV, "")
    ids: set[int] = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            ids.add(int(token))
    return frozenset(ids)


def has_premium(profile: UserProfile, telegram_id: int | str | None = None) -> bool:
    """Premium flag of the account, including the environment override."""
    if profile.is_premium:
        return True
    identity = telegram_id if telegram_id is not None else profile.telegram_id
    try:
        number = int(identity)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number in premium_telegram_ids()


async def resolve_owner_profile(
    store: ChartStore,
    *,
    telegram_id: int | str | None = None,
    email: str | None = None,
) -> UserProfile:
    """Load the full account profile or raise ``ProfileError``."""
    if telegram_id is None and email is None:
        raise ProfileError(PROFILE_MISSING_MESSAGE)
    profile = await store.get_user_profile(telegram_id=telegram_id, email=email)
    if profile is None:
        raise ProfileError(PROFILE_MISSING_MESSAGE)
    return profile


def natal_data_present(record: NatalRecord) -> bool:
    """True when the stored profile carries every field the calculator needs."""
    return (
        record.birth_date is not None
        and record.birth_lat is not None
        and record.birth_lon is not None
        and bool(record.birth_timezone)
    )


def birth_data_from_profile(record: NatalRecord) -> BirthData:
    """Rebuild the canonical ``BirthData`` of a stored natal profile.

    This is the only path from a stored profile to a chart: no caller may
    construct birth data for the forecast tools from request input. Unknown
    birth time is handled like the wizard does - 12:00 local time.
    """
    if not natal_data_present(record):
        raise ProfileError(PROFILE_NATAL_MISSING_MESSAGE)
    local_time = record.birth_time or DEFAULT_TIME
    moment_local = datetime.combine(record.birth_date, local_time)
    timezone_name = str(record.birth_timezone)
    utc, _offset, _notes = local_to_utc(moment_local, timezone_name)
    return BirthData(
        dt_utc=utc,
        latitude=float(record.birth_lat),
        longitude=float(record.birth_lon),
        house_system=HOUSE_SYSTEM,
        language=LANGUAGE,
        city=record.birth_city,
        timezone=timezone_name,
        datetime_local=moment_local.isoformat(),
    )


async def save_owner_profile(
    store: ChartStore,
    user_id: int,
    birth: BirthData,
    *,
    time_unknown: bool = False,
    profile: UserProfile | None = None,
) -> str:
    """Persist the owner's natal profile after a successful chart request.

    Returns ``"unchanged"`` when the stored data already matches (no write and
    no spent change), ``"saved"`` otherwise. Pass the already loaded
    ``profile`` to avoid an extra lookup. Raises ``ProfileChangeLimitError``
    when the monthly change limit is exhausted.
    """
    natal = natal_profile_from_birth(birth, time_unknown=time_unknown)
    if profile is not None and natal_data_present(profile) and profile_matches(profile, natal):
        return "unchanged"
    is_initial = profile is None or not natal_data_present(profile)
    limit = profile_change_limit()
    saved = await store.save_user_natal_profile(
        user_id,
        natal,
        count_change=not is_initial,
        max_changes_per_month=None if is_initial or limit <= 0 else limit,
    )
    if not saved:
        if is_initial:
            raise RuntimeError("could not store the natal profile: the user does not exist")
        raise ProfileChangeLimitError(PROFILE_CHANGE_LIMIT_MESSAGE.format(limit=limit))
    return "saved"


def natal_profile_from_birth(birth: BirthData, *, time_unknown: bool) -> NatalProfile:
    """Build the stored profile payload of a birth input (canonical hash included)."""
    if birth.datetime_local:
        moment_local = datetime.fromisoformat(birth.datetime_local)
        birth_date = moment_local.date()
        local_time = moment_local.time()
    else:
        # Direct UTC input (CLI/service calls): the local date is approximated
        # by the UTC date and the exact local time stays unknown.
        birth_date = birth.dt_utc.date()
        local_time = None
    chart_hash = generate_chart_hash(
        birth.dt_utc,
        birth.latitude,
        birth.longitude,
        birth.house_system,
        birth.language,
    )
    return NatalProfile(
        birth_date=birth_date,
        birth_time=local_time,
        birth_time_unknown=bool(time_unknown),
        birth_lat=birth.latitude,
        birth_lon=birth.longitude,
        birth_timezone=str(birth.timezone or "UTC"),
        birth_city=birth.city,
        birth_chart_hash=chart_hash,
    )


def profile_matches(record: NatalRecord, natal: NatalProfile) -> bool:
    """True when the stored natal data already equals the proposed one."""
    if record.birth_date != natal.birth_date:
        return False
    if record.birth_time != natal.birth_time:
        return False
    if bool(record.birth_time_unknown) != bool(natal.birth_time_unknown):
        return False
    if record.birth_lat is None or abs(float(record.birth_lat) - float(natal.birth_lat)) > 1e-6:
        return False
    if record.birth_lon is None or abs(float(record.birth_lon) - float(natal.birth_lon)) > 1e-6:
        return False
    if str(record.birth_timezone or "") != natal.birth_timezone:
        return False
    if (record.birth_city or None) != (natal.birth_city or None):
        return False
    return True


def profile_change_limit() -> int:
    """Monthly natal-profile change limit from the environment (default 3).

    ``0`` or a negative value disables the limit; a non-integer value falls
    back to the default so a typo in ``.env`` cannot lock users out.
    """
    raw = os.getenv(PROFILE_CHANGE_LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_PROFILE_CHANGE_LIMIT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using %d",
            PROFILE_CHANGE_LIMIT_ENV,
            raw,
            DEFAULT_PROFILE_CHANGE_LIMIT,
        )
        return DEFAULT_PROFILE_CHANGE_LIMIT
    return value if value > 0 else 0


def changes_used_this_month(profile: UserProfile) -> int:
    """Natal-profile changes already spent in the current calendar month."""
    period = profile.profile_changes_period
    if period is None:
        return 0
    today = datetime.now(timezone.utc).date()
    if (period.year, period.month) != (today.year, today.month):
        return 0
    return max(0, int(profile.profile_changes_count))


def profile_changes_note(profile: UserProfile) -> str | None:
    """One-line hint for the confirmation card; ``None`` when unlimited."""
    limit = profile_change_limit()
    if limit <= 0:
        return None
    used = changes_used_this_month(profile)
    return f"🔄 Изменений данных рождения в этом месяце: {used} из {limit}"


def ensure_profile_change_allowed(profile: UserProfile, natal: NatalProfile) -> None:
    """Raise when the proposed profile exceeds the monthly change limit.

    Called before the expensive calculation, so an exhausted limit fails fast.
    An unchanged profile and the very first save never spend the allowance.
    """
    if not natal_data_present(profile) or profile_matches(profile, natal):
        return
    limit = profile_change_limit()
    if limit <= 0:
        return
    if changes_used_this_month(profile) >= limit:
        raise ProfileChangeLimitError(PROFILE_CHANGE_LIMIT_MESSAGE.format(limit=limit))


def ensure_birth_change_allowed(
    profile: UserProfile, birth: BirthData, *, time_unknown: bool
) -> None:
    """Convenience wrapper around ``ensure_profile_change_allowed``."""
    ensure_profile_change_allowed(
        profile, natal_profile_from_birth(birth, time_unknown=time_unknown)
    )


@dataclass(frozen=True)
class ContactCommandInput:
    """Parsed ``/contact`` payload before geocoding."""

    name: str
    relation: str | None
    birth_date: date
    birth_time: time | None
    time_unknown: bool
    city: str


_UNKNOWN_TIME_MARKERS = frozenset({"не знаю", "неизвестно", "неизвестен", "-", "н/д"})


def parse_contact_command(raw: str | None) -> ContactCommandInput:
    """Parse ``/contact Имя | [отношение] | дата [время] | Город``.

    The relation part is optional and the birth time may be omitted or marked
    unknown («не знаю») - such contacts are still usable for synastry, which
    works planet-to-planet and does not need houses.
    """
    text = (raw or "").strip()
    if not text:
        raise BirthInputError(
            "Укажите данные контакта. Формат: " + CONTACT_COMMAND_USAGE
        )
    parts = [part.strip() for part in text.split("|")]
    if len(parts) == 3:
        relation = None
        when_text = parts[1]
        city = parts[2]
    elif len(parts) == 4:
        relation = parts[1] or None
        when_text = parts[2]
        city = parts[3]
    else:
        raise BirthInputError(f"Не удалось разобрать команду. Формат: {CONTACT_COMMAND_USAGE}")

    name = parts[0]
    if not name:
        raise BirthInputError("Имя контакта не может быть пустым.")

    tokens = when_text.split()
    if not tokens:
        raise BirthInputError("Не хватает даты рождения контакта.")
    birth_date = parse_birth_date(tokens[0])
    time_text = " ".join(tokens[1:]).strip().lower()
    if not time_text or time_text in _UNKNOWN_TIME_MARKERS:
        birth_time = None
        time_unknown = True
    else:
        birth_time = parse_birth_time(time_text)
        time_unknown = False

    if not city:
        raise BirthInputError("Не хватает города рождения контакта.")
    return ContactCommandInput(
        name=name,
        relation=relation,
        birth_date=birth_date,
        birth_time=birth_time,
        time_unknown=time_unknown,
        city=city,
    )


async def create_contact_from_command(
    store: ChartStore,
    user_id: int,
    parsed: ContactCommandInput,
    geo: dict[str, Any],
    timezone_name: str,
) -> ContactProfile:
    """Store one contact from a parsed ``/contact`` command (geo + zone resolved)."""
    moment_local = datetime.combine(parsed.birth_date, parsed.birth_time or DEFAULT_TIME)
    utc, _offset, _notes = local_to_utc(moment_local, timezone_name)
    birth = BirthData(
        dt_utc=utc,
        latitude=geo["lat"],
        longitude=geo["lon"],
        house_system=HOUSE_SYSTEM,
        language=LANGUAGE,
        city=str(geo.get("display_name") or parsed.city),
        timezone=timezone_name,
        datetime_local=moment_local.isoformat(),
    )
    return await store.add_contact(
        user_id,
        ContactInput(
            name=parsed.name,
            relation=parsed.relation,
            birth_date=parsed.birth_date,
            birth_time=parsed.birth_time,
            birth_time_unknown=parsed.time_unknown,
            birth_lat=birth.latitude,
            birth_lon=birth.longitude,
            birth_timezone=timezone_name,
            birth_city=birth.city,
        ),
    )
