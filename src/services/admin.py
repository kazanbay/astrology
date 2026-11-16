"""Admin user directory: search results, the user card and its activity.

Pure formatting over ``UserProfile`` and the counters from the store; the only
personal data shown is what an administrator needs to help or moderate: the
Telegram identity, the account state and usage counters. Messages are HTML.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .premium import SOURCE_LABELS_RU

MSK = ZoneInfo("Europe/Moscow")

EVENT_LABELS_RU: dict[str, str] = {
    "chart": "карта",
    "forecast": "прогноз",
    "synastry": "синастрия",
    "other": "другое",
}


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _moment(value: datetime | None) -> str:
    if value is None:
        return "—"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(MSK).strftime("%d.%m.%Y %H:%M")


def _short_date(value: datetime | None) -> str:
    if value is None:
        return "—"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(MSK).strftime("%d.%m.%Y")


def display_name(profile: Any) -> str:
    """Human label: display name, then first/last, then @username."""
    parts = [profile.display_name, profile.first_name, profile.last_name]
    label = " ".join(str(part).strip() for part in parts if part)
    if label:
        return label
    if profile.username:
        return f"@{profile.username}"
    return str(profile.telegram_id or profile.user_id)


def format_user_list(users: list[Any], *, query: str | None = None) -> str:
    """Compact search result list under a /users command."""
    if not users:
        return "Никого не нашёл."
    header = f"🔎 Поиск: {_esc(query)}" if query else "👥 Последние пользователи"
    lines = [f"<b>{header}</b>", ""]
    for profile in users:
        flags = []
        if getattr(profile, "is_premium", False) or _is_premium_active(profile):
            flags.append("👑")
        if getattr(profile, "is_blocked", False):
            flags.append("⛔")
        marks = " " + " ".join(flags) if flags else ""
        username = f"@{_esc(profile.username)} · " if profile.username else ""
        lines.append(
            f"• <b>{_esc(display_name(profile))}</b>{marks}\n"
            f"  {username}id <code>{_esc(profile.telegram_id)}</code>"
        )
    return "\n".join(lines)


def _is_premium_active(profile: Any) -> bool:
    until = getattr(profile, "premium_until", None)
    if until is None:
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return until > datetime.now(timezone.utc)


def format_user_card(
    profile: Any,
    stats: dict[str, int],
    *,
    events: list[dict[str, Any]] | None = None,
    premium_events: list[Any] | None = None,
) -> str:
    """Full admin card: identity, access state, usage and premium history."""
    username = f"@{_esc(profile.username)}" if profile.username else "без @username"
    premium_state = "нет"
    if getattr(profile, "is_premium", False):
        premium_state = "бессрочно (флаг)"
    elif _is_premium_active(profile):
        premium_state = f"до {_short_date(profile.premium_until)}"
    block_state = "⛔ заблокирован" if getattr(profile, "is_blocked", False) else "✅ активен"

    lines = [
        f"👤 <b>{_esc(display_name(profile))}</b>",
        f"{username} · id <code>{_esc(profile.telegram_id)}</code> · db #{_esc(profile.user_id)}",
        f"Статус: {block_state} · Премиум: {premium_state}",
        f"Создан: {_moment(getattr(profile, 'created_at', None))}",
        "",
        "<b>Активность бота</b>",
        (
            f"🔮 карты {stats.get('charts', 0)} · 📅 прогнозы {stats.get('forecasts', 0)} · "
            f"💞 синастрия {stats.get('synastry', 0)}"
        ),
        (
            f"👥 контакты {stats.get('contacts', 0)} · 💬 обращения {stats.get('feedback', 0)} · "
            f"запросы карт {stats.get('requests', 0)}"
        ),
        (
            f"⭐ покупок {stats.get('payments', 0)} на {stats.get('stars', 0)} ⭐ · "
            f"начислений премиума {stats.get('grants', 0)}"
        ),
    ]
    if events:
        labels = ", ".join(
            f"{EVENT_LABELS_RU.get(str(item['event_type']), 'другое')} "
            f"{_moment(item.get('created_at'))}"
            for item in events[:5]
        )
        lines.append(f"🕒 Последнее: {labels}")
    if premium_events:
        lines.append("")
        lines.append("<b>Премиум: последние начисления</b>")
        for event in premium_events[:3]:
            source = SOURCE_LABELS_RU.get(str(event.source), str(event.source))
            lines.append(
                f"• {_short_date(event.created_at)} — {source}: +{event.months} мес."
                + (f", {event.stars} ⭐" if event.stars else "")
                + (f" → до {_short_date(event.premium_until_after)}"
                   if event.premium_until_after else "")
            )
    return "\n".join(lines)


def format_premium_history(events: list[Any], *, limit: int = 10) -> str:
    """Extended premium audit list for the admin card button."""
    if not events:
        return "📜 Начислений премиума нет."
    lines = ["📜 <b>История премиума</b>"]
    for event in events[:limit]:
        source = SOURCE_LABELS_RU.get(str(event.source), str(event.source))
        line = (
            f"• {_moment(event.created_at)} — {source}: +{event.months} мес."
            + (f", {event.stars} ⭐" if event.stars else "")
        )
        if event.premium_until_after:
            line += f" → до {_short_date(event.premium_until_after)}"
        if event.reason:
            line += f"\n  {_esc(event.reason)}"
        lines.append(line)
    return "\n".join(lines)


def format_ticket_list(tickets: list[Any]) -> str:
    """Unanswered support tickets for the admin chat."""
    if not tickets:
        return "✅ Неотвеченных обращений нет."
    lines = [f"🎫 <b>Неотвеченные обращения: {len(tickets)}</b>", ""]
    for ticket in tickets:
        who = f"@{_esc(ticket.username)}" if ticket.username else "без username"
        snippet = " ".join(str(ticket.message).split())
        if len(snippet) > 60:
            snippet = snippet[:59].rstrip() + "…"
        lines.append(
            f"<b>#{ticket.id}</b> · {_moment(ticket.created_at)} · {who} · "
            f"id <code>{_esc(ticket.user_id)}</code>\n{snippet}"
        )
    return "\n".join(lines)


def format_ticket(ticket: Any) -> str:
    """One ticket in full."""
    who = f"@{_esc(ticket.username)}" if ticket.username else "без username"
    status = "новое" if ticket.status == "new" else "обработано"
    return (
        f"🎫 <b>Обращение #{ticket.id}</b> ({status})\n"
        f"👤 {who} · id <code>{_esc(ticket.user_id)}</code>\n"
        f"🕒 {_moment(ticket.created_at)}\n\n"
        f"{_esc(ticket.message)}"
    )
