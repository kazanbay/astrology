"""Telegram-sized Russian rendering of the forecast and synastry tool results.

Plain text only, deterministic: the numbers come straight from the calculation
and the labels from the shared dictionaries, so the report can never invent an
interpretation. LLM-based interpretation is a separate layer (LangGraph), not
this renderer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from storage.presentation import PLANET_LABELS_RU, SIGN_LABELS_RU

ASPECT_LABELS_RU: dict[str, str] = {
    "conjunction": "соединение",
    "sextile": "секстиль",
    "square": "квадрат",
    "trine": "трин",
    "opposition": "оппозиция",
}

INTERVAL_TITLES_RU: dict[str, str] = {
    "day": "на день",
    "week": "на неделю",
    "month": "на месяц",
    "year": "на год",
}

#: Natal points that are not planets (event lines and summaries).
POINT_LABELS_RU: dict[str, str] = {
    "Ascendant": "Асцендент",
    "Midheaven": "MC",
}

MAX_TEXT_CHARS = 3800

UNKNOWN_TIME_NOTE = (
    "⚠️ Время рождения неизвестно — взято 12:00, дома и углы карты приблизительны."
)


def _planet(name: Any) -> str:
    return PLANET_LABELS_RU.get(str(name or ""), str(name or "—"))


def _point(name: Any) -> str:
    """Label of a transited natal point (planet or angle)."""
    key = str(name or "")
    return POINT_LABELS_RU.get(key) or _planet(key)


def _sign(name: Any) -> str:
    text = str(name or "")
    return SIGN_LABELS_RU.get(text, text or "—")


def _aspect(name: Any) -> str:
    return ASPECT_LABELS_RU.get(str(name or ""), str(name or "—"))


def _local(value: Any, tz_name: Any) -> datetime | None:
    """Parse an ISO instant and shift it into the profile's zone for display."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    target: Any = timezone.utc
    if tz_name:
        try:
            target = ZoneInfo(str(tz_name))
        except Exception:  # pragma: no cover - stored zones come from tzfinder
            target = timezone.utc
    return moment.astimezone(target)


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _person_line(person: Mapping[str, Any]) -> str:
    name = str(person.get("name") or "")
    if person.get("kind") == "owner":
        return "👤 Мой прогноз" + (f" ({name})" if name else "")
    label = f"👥 {name}" if name else "👥 Другой человек"
    relation = person.get("relation")
    if relation:
        label += f" ({relation})"
    return label


def _event_period(event: Mapping[str, Any], tz_name: Any) -> str:
    first = _local(event.get("first_sample"), tz_name)
    last = _local(event.get("last_sample"), tz_name)
    if first is None:
        return ""
    if last is None or first.date() == last.date():
        return f" ({first.strftime('%d.%m %H:%M')})"
    return f" ({first.strftime('%d.%m')}–{last.strftime('%d.%m')})"


def format_forecast_text(result: Mapping[str, Any], *, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Render a forecast tool result as a Telegram-sized Russian message."""
    if max_chars < 200:
        raise ValueError("max_chars is too small to render a forecast")
    person = result.get("person") or {}
    birth = result.get("birth") or {}
    events = result.get("events") or []
    interval = str(result.get("interval") or "")
    title = INTERVAL_TITLES_RU.get(interval, interval)
    tz_name = birth.get("timezone")

    lines = [f"📅 Прогноз {title}", _person_line(person)]
    city = birth.get("city")
    if city:
        lines.append(f"📍 Место рождения: {city}")
    if birth.get("time_unknown"):
        lines.append(UNKNOWN_TIME_NOTE)
    lines.append("")

    snapshot = result.get("now_transits") or []
    if snapshot:
        parts = []
        for entry in snapshot:
            part = f"{_planet(entry.get('name'))} {_sign(entry.get('sign'))}"
            house = entry.get("house")
            if house:
                part += f" ({house} д.)"
            parts.append(part)
        lines.append("Небо сейчас: " + " · ".join(parts))
        lines.append("")

    if not events:
        lines.append("Транзитов с орбом до 3° к вашей карте в этом периоде нет.")
    else:
        lines.append("Ключевые транзиты к карте:")
        for index, event in enumerate(events, start=1):
            line = (
                f"{index}. {_planet(event.get('transit'))} {_aspect(event.get('aspect'))} "
                f"{_point(event.get('natal'))} — орб {event.get('orb')}°"
            )
            period = _event_period(event, tz_name)
            if period:
                line += period
            lines.append(line)

    return _trim("\n".join(lines), max_chars)


def forecast_query(result: Mapping[str, Any], *, limit: int = 5) -> str:
    """Knowledge-base search query built from the strongest transits."""
    events = result.get("events") or []
    parts: list[str] = []
    for event in events[:limit]:
        if not isinstance(event, Mapping):
            continue
        parts.append(
            f"{_planet(event.get('transit'))} {_aspect(event.get('aspect'))} "
            f"{_point(event.get('natal'))}"
        )
    interval = INTERVAL_TITLES_RU.get(str(result.get("interval") or ""), "")
    head = f"Транзиты {interval}" if interval else "Транзиты"
    body = "; ".join(parts) if parts else "спокойный период без точных транзитов"
    return f"{head}: {body}. Классическая астрология, толкование транзитов"


def _highlight(positions: Any) -> str:
    """The Sun/Moon/Venus/Mars placements of one person as a compact line."""
    if not isinstance(positions, Mapping):
        return "—"
    parts = []
    for name in ("Sun", "Moon", "Venus", "Mars"):
        placement = positions.get(name)
        if isinstance(placement, Mapping) and placement.get("sign"):
            parts.append(f"{_planet(name)} {_sign(placement.get('sign'))}")
    return ", ".join(parts) if parts else "—"


def format_synastry_text(result: Mapping[str, Any], *, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Render a synastry tool result as a Telegram-sized Russian message."""
    if max_chars < 200:
        raise ValueError("max_chars is too small to render a synastry report")
    person_a = result.get("person_a") or {}
    person_b = result.get("person_b") or {}
    events = result.get("events") or []

    name_a = str(person_a.get("name") or "Вы")
    name_b = str(person_b.get("name") or "")
    relation = person_b.get("relation")
    if relation:
        name_b += f" ({relation})"

    lines = [f"💞 Совместимость: {name_a} + {name_b}", ""]
    lines.append("Ключевые точки карт:")
    lines.append(f"• {name_a}: {_highlight(person_a.get('positions'))}")
    lines.append(f"• {name_b}: {_highlight(person_b.get('positions'))}")
    lines.append("")

    if not events:
        lines.append("Мажорных аспектов между картами в пределах орбов не найдено.")
    else:
        lines.append("Главные аспекты между картами:")
        for index, event in enumerate(events, start=1):
            lines.append(
                f"{index}. {_planet(event.get('body_a'))} {_aspect(event.get('aspect'))} "
                f"{_planet(event.get('body_b'))} — орб {event.get('orb')}°"
            )

    return _trim("\n".join(lines), max_chars)


def format_contacts_list(contacts: Iterable[Mapping[str, Any]], *, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Render the contact list; includes the add-command hint when empty."""
    if max_chars < 200:
        raise ValueError("max_chars is too small to render a contact list")
    rows = list(contacts)
    if not rows:
        return (
            "Список контактов пуст. Отправьте /contact — мастер спросит имя, дату, "
            "время и место рождения.\n\n"
            "Можно и одной строкой:\n"
            "/contact Мария | супруга | 15.03.1992 14:30 | Казань\n\n"
            "Время можно опустить: /contact Мария | 15.03.1992 | Казань"
        )
    lines = ["Ваши контакты:"]
    for index, contact in enumerate(rows, start=1):
        label = str(contact.get("name") or "—")
        relation = contact.get("relation")
        if relation:
            label += f" ({relation})"
        birth_date = contact.get("birth_date") or "—"
        suffix = " (время неизвестно)" if contact.get("birth_time_unknown") else ""
        lines.append(f"{index}. {label} — {birth_date}{suffix}")
    return _trim("\n".join(lines), max_chars)
