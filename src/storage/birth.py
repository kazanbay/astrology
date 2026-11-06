"""Birth input intake: validation, local time -> UTC, geocoding, command parsing.

The astronomy calculator accepts UTC only and never consults the local clock or
``zoneinfo``. This module is the backend layer above it:

* ``BirthData`` is the validated, canonical form of a birth input that both the
  hash and the calculator consume;
* ``local_to_utc`` applies historical time zone rules (``tzdata``) to a naive
  local instant;
* ``geocode_city`` / ``detect_timezone`` turn a city name into coordinates and
  an IANA zone through Nominatim and ``timezonefinder``;
* ``parse_birth_command`` turns a bot command payload into a datetime and city.

Network calls live inside the functions, so importing the module stays cheap
and offline-safe.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "astrology-rag/0.1 (natal chart structural lookup; educational project)"

BIRTH_COMMAND_USAGE = "/chart 1990-08-15 12:00 Москва"

_HOUSE_SYSTEM_PATTERN = re.compile(r"[A-Z]")
_LANGUAGE_PATTERN = re.compile(r"[a-z0-9-]{2,16}")


class BirthInputError(RuntimeError):
    """Birth input cannot be resolved into a UTC chart request."""


def normalize_utc(value: datetime | str) -> datetime:
    """Return a timezone-aware UTC datetime from a datetime or an ISO string.

    Naive values are rejected: silently guessing the zone is the classic source
    of wrong natal charts.
    """
    moment = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise BirthInputError("Пустое значение даты и времени")
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise BirthInputError(
                f"Не удалось разобрать дату и время {value!r}; ожидается ISO 8601"
            ) from exc
    if not isinstance(moment, datetime):
        raise BirthInputError(f"Ожидалась дата и время, получено {type(value).__name__}")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise BirthInputError("Дата и время должны содержать смещение таймзоны (UTC)")
    return moment.astimezone(timezone.utc)


def canonical_utc_string(value: datetime | str) -> str:
    """Return one stable textual form of an instant: ``...+00:00``.

    The form is used as the hash input, so ``1990-08-15T08:00:00Z``,
    ``1990-08-15T12:00:00+04:00`` and the equivalent datetime must produce the
    same string. Microseconds appear only when they are non-zero.
    """
    moment = normalize_utc(value)
    base = moment.strftime("%Y-%m-%dT%H:%M:%S")
    if moment.microsecond:
        base = f"{base}.{moment.microsecond:06d}"
    return base + "+00:00"


def house_system_code(value: str) -> str:
    """Normalize a single-letter Swiss Ephemeris house-system code."""
    if not isinstance(value, str):
        raise BirthInputError("Система домов задаётся однобуквенным кодом, например 'P'")
    code = value.strip().upper()
    if not _HOUSE_SYSTEM_PATTERN.fullmatch(code):
        raise BirthInputError(f"Некорректный код системы домов: {value!r}")
    return code


def language_code(value: str) -> str:
    """Normalize the chunk language used for RAG retrieval."""
    if not isinstance(value, str) or not value.strip():
        raise BirthInputError("Язык базы знаний должен быть непустой строкой, например 'ru'")
    code = value.strip().lower()
    if not _LANGUAGE_PATTERN.fullmatch(code):
        raise BirthInputError(f"Некорректный код языка: {value!r}")
    return code


def _coordinate(value: Any, name: str, limit: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BirthInputError(f"{name} должен быть числом, получено {value!r}") from exc
    if not math.isfinite(number):
        raise BirthInputError(f"{name} должен быть конечным числом")
    if not -limit <= number <= limit:
        raise BirthInputError(f"{name} должен быть в диапазоне [-{limit:g}, {limit:g}], получено {number}")
    return number


@dataclass(frozen=True)
class BirthData:
    """Canonical birth input of one chart request."""

    dt_utc: datetime
    latitude: float
    longitude: float
    house_system: str = "P"
    language: str = "ru"
    #: Display name of the place, as returned by the geocoder.
    city: str | None = None
    #: IANA zone the local instant came from; ``None`` when the caller passed UTC.
    timezone: str | None = None
    #: The original local instant, kept for the prompt and for the user display.
    datetime_local: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt_utc", normalize_utc(self.dt_utc))
        object.__setattr__(self, "latitude", _coordinate(self.latitude, "Широта", 90.0))
        object.__setattr__(self, "longitude", _coordinate(self.longitude, "Долгота", 180.0))
        object.__setattr__(self, "house_system", house_system_code(self.house_system))
        object.__setattr__(self, "language", language_code(self.language))

    def input_params(self) -> dict[str, Any]:
        """Return the JSON-ready ``charts_cache.input_params`` payload."""
        params: dict[str, Any] = {
            "dt_utc": canonical_utc_string(self.dt_utc),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "house_system": self.house_system,
            "language": self.language,
        }
        for key in ("city", "timezone", "datetime_local"):
            value = getattr(self, key)
            if value:
                params[key] = value
        return params


_DATE_FORMATS = ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d")
_TIME_PATTERN = re.compile(r"(\d{1,2})(?:[:.](\d{1,2}))?")


def parse_birth_date(value: str) -> date:
    """Parse a birth date typed by a user (``15.08.1990`` and similar)."""
    text = str(value).strip()
    parsed: date | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        raise BirthInputError(
            f"Не удалось разобрать дату {value!r}; ожидается 15.08.1990"
        )
    if parsed > date.today():
        raise BirthInputError("Дата рождения не может быть в будущем")
    if parsed.year < 1900:
        raise BirthInputError("Поддерживаются даты рождения с 1900 года")
    return parsed


def parse_birth_time(value: str) -> time:
    """Parse a local birth time (``12:30``, ``12.30`` or ``12``)."""
    text = str(value).strip()
    match = _TIME_PATTERN.fullmatch(text)
    if not match:
        raise BirthInputError(f"Не удалось разобрать время {value!r}; ожидается 12:30")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise BirthInputError(f"Время вне диапазона: {value!r}")
    return time(hour, minute)


def parse_local_datetime(value: str) -> datetime:
    """Parse a naive local datetime; an explicit offset is a user error."""
    text = str(value).strip()
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise BirthInputError(
            f"Не удалось разобрать дату и время {value!r}; ожидается формат 1990-08-15 12:00"
        ) from exc
    if moment.tzinfo is not None and moment.utcoffset() is not None:
        raise BirthInputError(
            "Указывайте локальное время без смещения: таймзона определяется по городу"
        )
    return moment


def parse_birth_command(raw: str | None) -> tuple[datetime, str]:
    """Split a ``/chart`` payload into a local datetime and a city.

    Accepted shapes: ``1990-08-15 12:00 Москва`` and
    ``1990-08-15T12:00:00 Москва``. The city may contain spaces.
    """
    text = (raw or "").strip()
    if not text:
        raise BirthInputError(f"Укажите дату, время и город. Формат: {BIRTH_COMMAND_USAGE}")

    head, separator, tail = text.partition(" ")
    tail = tail.strip()
    if not separator or not tail:
        raise BirthInputError(f"Не хватает города. Формат: {BIRTH_COMMAND_USAGE}")

    if "T" in head:
        value = head
        city = tail
    else:
        time_part, separator, city = tail.partition(" ")
        city = city.strip()
        if not separator or not city:
            raise BirthInputError(f"Не хватает города. Формат: {BIRTH_COMMAND_USAGE}")
        value = f"{head}T{time_part.strip()}"

    return parse_local_datetime(value), city


def local_to_utc(
    moment_local: datetime,
    tz_name: str,
) -> tuple[datetime, timedelta, list[str]]:
    """Convert a naive local datetime to UTC using historical time zone rules."""
    try:
        tzinfo = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise BirthInputError(
            f"Таймзона {tz_name!r} недоступна; установите пакет tzdata (uv pip install tzdata)"
        ) from exc

    localized = moment_local.replace(tzinfo=tzinfo)
    utc = localized.astimezone(timezone.utc)
    notes: list[str] = []
    if utc.astimezone(tzinfo).replace(tzinfo=None) != moment_local:
        notes.append(
            "Локальное время не существует из-за перехода на летнее время; "
            "использована ближайшая допустимая интерпретация"
        )
    elif moment_local.replace(tzinfo=tzinfo, fold=1).utcoffset() != localized.utcoffset():
        notes.append("Локальное время неоднозначно из-за перехода часов; использован fold=0")

    offset = localized.utcoffset() or timedelta(0)
    return utc, offset, notes


def geocode_city(city: str, *, timeout: float = 15.0) -> dict[str, Any]:
    """Resolve a city name through Nominatim; return coordinates and a display name."""
    import requests

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"q": city, "format": "jsonv2", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru"},
            timeout=timeout,
        )
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as exc:
        raise BirthInputError(f"Nominatim недоступен: {exc}") from exc
    except ValueError as exc:
        raise BirthInputError("Nominatim вернул ответ, который не является JSON") from exc

    if not results:
        raise BirthInputError(f"Город {city!r} не найден через Nominatim")
    best = results[0]
    try:
        lat = float(best["lat"])
        lon = float(best["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BirthInputError("Nominatim вернул результат без координат") from exc
    if not -90.0 <= lat <= 90.0:
        raise BirthInputError(f"Nominatim вернул некорректную широту: {lat}")
    if not -180.0 <= lon <= 180.0:
        raise BirthInputError(f"Nominatim вернул некорректную долготу: {lon}")
    return {
        "lat": lat,
        "lon": lon,
        "display_name": str(best.get("display_name") or city),
        "place_type": str(best.get("type") or "—"),
    }


def detect_timezone(lat: float, lon: float) -> str:
    """Return the IANA time zone name for the coordinates."""
    from timezonefinder import TimezoneFinder

    name = TimezoneFinder().timezone_at(lat=lat, lng=lon)
    if not name:
        raise BirthInputError("Не удалось определить IANA-таймзону по координатам (точка в океане?)")
    return name
