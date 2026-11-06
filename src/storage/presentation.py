"""Human-readable Russian rendering of pipeline results.

Telegram messages are plain text with a hard length limit, so the formatter
builds a compact summary first and appends section titles only while they fit.
The same renderer serves the CLI script, which keeps bot and script output in
sync.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from .chunks import element_stats

PLANET_LABELS_RU: dict[str, str] = {
    "Sun": "Солнце",
    "Moon": "Луна",
    "Mercury": "Меркурий",
    "Venus": "Венера",
    "Mars": "Марс",
    "Jupiter": "Юпитер",
    "Saturn": "Сатурн",
    "Uranus": "Уран",
    "Neptune": "Нептун",
    "Pluto": "Плутон",
    "NorthNode": "Северный узел",
    "SouthNode": "Южный узел",
}

SIGN_LABELS_RU: dict[str, str] = {
    "Aries": "Овен",
    "Taurus": "Телец",
    "Gemini": "Близнецы",
    "Cancer": "Рак",
    "Leo": "Лев",
    "Virgo": "Дева",
    "Libra": "Весы",
    "Scorpio": "Скорпион",
    "Sagittarius": "Стрелец",
    "Capricorn": "Козерог",
    "Aquarius": "Водолей",
    "Pisces": "Рыбы",
}

TELEGRAM_MESSAGE_LIMIT = 4096
DEFAULT_MAX_CHARS = 3800
#: Report parts are split below the hard limit because HTML tags add length.
REPORT_PART_LIMIT = 3000

_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)
#: Markdown list markers Telegram renders as literal stars; replace with a bullet.
_BULLET_RE = re.compile(r"^([ \t]*)[*+-][ \t]+", re.MULTILINE)
#: A report heading is one short line wrapped in bold, optionally with an emoji.
_REPORT_HEADING_RE = re.compile(r"^(?:[^\w\s]{1,4}\s*)?\*\*[^*\n]{1,130}\*\*$")
#: Roughly four lines of Russian text that must follow a heading in one message.
MIN_TEXT_AFTER_HEADING = 200

#: Display order of bodies. JSONB does not preserve key order, so the cache
#: round trip reorders ``chart['planets']``; rendering restores the canonical
#: order of the calculator.
PLANET_ORDER: tuple[str, ...] = (
    "Sun",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
    "NorthNode",
    "SouthNode",
)
_PLANET_RANK = {name: index for index, name in enumerate(PLANET_ORDER)}


def _planet_label(name: str) -> str:
    return PLANET_LABELS_RU.get(name, name)


def _sign_label(name: Any) -> str:
    text = str(name or "")
    return SIGN_LABELS_RU.get(text, text or "—")


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_place(input_params: Mapping[str, Any]) -> str | None:
    city = input_params.get("city")
    return str(city) if city else None


def _format_header(input_params: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    local = input_params.get("datetime_local")
    timezone_name = input_params.get("timezone")
    moment_utc = _parse_utc(input_params.get("dt_utc"))

    if local or moment_utc:
        local_text = str(local).replace("T", " ") if local else "—"
        zone_text = f" ({timezone_name})" if timezone_name else ""
        utc_text = moment_utc.strftime("%H:%M UTC") if moment_utc else "—"
        lines.append(f"Карта: {local_text}{zone_text} → {utc_text}")

    place = _format_place(input_params)
    if place:
        lines.append(f"Место: {place}")
    return lines


def _format_angles(chart: Mapping[str, Any]) -> str | None:
    asc = chart.get("asc")
    mc = chart.get("mc")
    if not isinstance(asc, Mapping) or not isinstance(mc, Mapping):
        return None
    return (
        f"ASC: {_sign_label(asc.get('sign'))} {float(asc.get('longitude') or 0):.1f}° · "
        f"MC: {_sign_label(mc.get('sign'))} {float(mc.get('longitude') or 0):.1f}°"
    )


def _format_planets(chart: Mapping[str, Any]) -> list[str]:
    planets = chart.get("planets")
    if not isinstance(planets, Mapping) or not planets:
        return []
    parts: list[str] = []
    ordered = sorted(
        planets.items(),
        key=lambda item: (_PLANET_RANK.get(str(item[0]), len(PLANET_ORDER)), str(item[0])),
    )
    for name, position in ordered:
        if not isinstance(position, Mapping):
            continue
        house = position.get("house")
        house_text = f" — {house} дом" if isinstance(house, int) else ""
        parts.append(
            f"{_planet_label(str(name))} {_sign_label(position.get('sign'))} "
            f"{float(position.get('longitude') or 0):.1f}°{house_text}"
        )
    return parts


def _element_total(chart: Mapping[str, Any]) -> int:
    """Number of discrete elements the structural router queries for this chart."""
    planets = chart.get("planets")
    total = 0
    if isinstance(planets, Mapping):
        for position in planets.values():
            if not isinstance(position, Mapping):
                continue
            total += 1 if position.get("sign") else 0
            total += 1 if position.get("house") is not None else 0
    aspects = chart.get("aspects")
    if isinstance(aspects, list):
        total += len(aspects)
    return total


def format_chart_reply(
    result: Any,
    *,
    max_sections: int = 8,
    max_chars: int = DEFAULT_MAX_CHARS,
    technical: bool = True,
) -> str:
    """Render a ``ChartResult`` as a Telegram-sized Russian message.

    ``technical=True`` (CLI) appends cache state, section counters and titles;
    the bot passes ``False`` and shows a clean chart card followed by the report.
    """
    if max_chars < 200:
        raise ValueError("max_chars is too small to render a chart summary")
    if max_sections < 0:
        raise ValueError("max_sections must not be negative")

    input_params = getattr(result, "input_params", {}) or {}
    chart = getattr(result, "chart", {}) or {}
    chunks = getattr(result, "chunks", []) or []

    lines = _format_header(input_params)
    angles = _format_angles(chart)
    if angles:
        lines.append(angles)
    planets = _format_planets(chart)
    if planets:
        lines.append("")
        lines.append(" · ".join(planets))

    total_elements = _element_total(chart)
    stats = element_stats(chunks)
    if technical:
        lines.append("")
        lines.append(
            f"Разделы базы знаний: {stats['elements_with_chunks']} из {total_elements} элементов, "
            f"{stats['chunks']} уникальных разделов"
        )

        cache_state = (
            "попадание (карта не пересчитывалась)" if getattr(result, "cache_hit", False) else "промах"
        )
        if getattr(result, "cache_written", False):
            cache_state += ", результат сохранён"
        lines.append(f"Кэш: {cache_state}; хэш {str(result.chart_hash)[:12]}")

    titles: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        title = str(chunk.get("section_title") or chunk.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        if len(titles) >= max_sections:
            break
        titles.append(title)

    if titles and technical:
        lines.append("")
        lines.append("Найденные разделы:")
        for index, title in enumerate(titles, start=1):
            lines.append(f"{index}. {title}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def format_section_previews(
    chunks: Iterable[Mapping[str, Any]],
    *,
    limit: int = 2,
    max_chars: int = 1500,
) -> list[str]:
    """Render the first chunks as short standalone messages.

    The bot cannot send thirty-five full sections, so the summary lists the
    titles and these previews show the actual source text of the best matches.
    """
    if limit < 0:
        raise ValueError("limit must not be negative")
    if max_chars < 300:
        raise ValueError("max_chars is too small to render a preview")

    previews: list[str] = []
    for chunk in chunks:
        if len(previews) >= limit:
            break
        title = str(chunk.get("section_title") or chunk.get("title") or "Раздел").strip()
        content = " ".join(str(chunk.get("content") or "").split())
        if not content:
            continue
        header = f"«{title}»\n\n"
        budget = max_chars - len(header) - 1
        if budget < 100:
            break
        body = content if len(content) <= budget else content[:budget].rstrip() + "…"
        previews.append(header + body)
    return previews


def _is_report_heading(paragraph: str) -> bool:
    """True for a short single-line heading such as ``💼 **2. Реализация**``."""
    line = paragraph.strip()
    if not line or "\n" in line or len(line) > 140:
        return False
    return bool(_REPORT_HEADING_RE.fullmatch(line))


def _keep_headings_with_text(parts: list[str], limit: int) -> list[str]:
    """Never end a message with a heading and never send a heading alone.

    A heading that lands at the tail of a part is moved to the next part. A
    heading followed by fewer than ~four lines (200 characters) of text is moved
    as well, and a part that is just a heading is merged into the next one, so
    the reader always sees the heading together with its text.
    """
    index = 0
    while index < len(parts) - 1:
        paragraphs = parts[index].split("\n\n")
        heading_index = next(
            (position for position in range(len(paragraphs) - 1, -1, -1)
             if _is_report_heading(paragraphs[position])),
            None,
        )
        if heading_index is None:
            index += 1
            continue

        tail = paragraphs[heading_index:]
        text_after = sum(len(paragraph.strip()) for paragraph in tail[1:])
        if text_after >= MIN_TEXT_AFTER_HEADING:
            index += 1
            continue

        if heading_index > 0:
            moved = "\n\n".join(tail + [parts[index + 1]]).strip()
            if len(moved) <= limit:
                parts[index] = "\n\n".join(paragraphs[:heading_index]).strip()
                parts[index + 1] = moved
        else:
            # The part starts with a heading and has almost no text: glue it to
            # the next part instead of sending a lone heading.
            merged = f"{parts[index]}\n\n{parts[index + 1]}".strip()
            if len(merged) <= TELEGRAM_MESSAGE_LIMIT:
                parts[index] = merged
                parts.pop(index + 1)
                continue
        index += 1
    return parts


def split_message(text: str, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split a long report into Telegram-sized messages without cutting words.

    Prefers paragraph breaks, then line breaks, then spaces; hard-splits only
    when a single token is longer than the limit.
    """
    if limit < 200:
        raise ValueError("limit is too small to split a report")
    remaining = (text or "").strip()
    if not remaining:
        return []

    parts: list[str] = []
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n ")
    if remaining:
        parts.append(remaining)
    return _keep_headings_with_text(parts, limit)


def to_telegram_html(text: str) -> str:
    """Turn the model's lightweight Markdown into Telegram-safe HTML.

    The synthesis prompt asks for ``**bold**`` and forbids ``###`` headings
    (Telegram renders neither), so the bot converts both into ``<b>`` tags and
    escapes the rest before sending with ``parse_mode=HTML``.
    """
    if not text:
        return ""
    escaped = html.escape(text, quote=False)
    escaped = _HEADING_RE.sub(lambda match: f"<b>{match.group(1)}</b>", escaped)
    escaped = _BULLET_RE.sub(lambda match: f"{match.group(1)}• ", escaped)
    return _BOLD_RE.sub(r"<b>\1</b>", escaped)


def format_history(records: Iterable[Any], *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Render the user's request history as a compact list."""
    records = list(records)
    if not records:
        return "История запросов пуста. Отправьте /chart 1990-08-15 12:00 Москва."

    lines = ["Последние запросы:"]
    for index, record in enumerate(records, start=1):
        params = getattr(record, "input_params", {}) or {}
        place = _format_place(params) or "—"
        moment = _parse_utc(params.get("dt_utc"))
        moment_text = moment.strftime("%d.%m.%Y %H:%M UTC") if moment else "—"
        requested = getattr(record, "requested_at", None)
        requested_text = requested.strftime("%d.%m.%Y %H:%M") if isinstance(requested, datetime) else "—"
        lines.append(
            f"{index}. {requested_text} · {getattr(record, 'profile_name', '—') or '—'} · "
            f"{place} · {moment_text} · хэш {str(getattr(record, 'chart_hash', ''))[:8]}"
        )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
