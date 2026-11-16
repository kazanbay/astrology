"""Unit tests of the admin user directory and ticket formatting."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from services.admin import (
    format_premium_history,
    format_ticket,
    format_ticket_list,
    format_user_card,
    format_user_list,
)

NOW = datetime.now(timezone.utc)

PROFILE = SimpleNamespace(
    user_id=5,
    telegram_id=380771563,
    username="sergey",
    first_name="Сергей",
    last_name=None,
    display_name="Сергей",
    is_premium=False,
    is_blocked=False,
    premium_until=NOW + timedelta(days=20),
    created_at=NOW - timedelta(days=3),
)

STATS = {
    "charts": 3,
    "forecasts": 7,
    "synastry": 1,
    "contacts": 2,
    "feedback": 1,
    "requests": 4,
    "payments": 1,
    "stars": 250,
    "grants": 2,
}


def test_user_card_shows_identity_premium_and_activity():
    text = format_user_card(PROFILE, STATS)

    assert "Сергей" in text and "@sergey" in text and "380771563" in text
    assert "Премиум: до" in text
    assert "карты 3" in text and "прогнозы 7" in text and "синастрия 1" in text
    assert "250 ⭐" in text


def test_blocked_state_is_visible():
    blocked = SimpleNamespace(**{**PROFILE.__dict__, "is_blocked": True})

    assert "заблокирован" in format_user_card(blocked, STATS)


def test_user_list_marks_premium_and_ids():
    text = format_user_list([PROFILE], query="sergey")

    assert "Поиск: sergey" in text
    assert "👑" in text
    assert "id <code>380771563</code>" in text


def test_ticket_formatting():
    ticket = SimpleNamespace(
        id=7,
        user_id=123456789,
        username="user",
        message="Не приходит прогноз",
        status="new",
        created_at=NOW,
    )

    card = format_ticket(ticket)
    assert "#7" in card and "Не приходит прогноз" in card

    listing = format_ticket_list([ticket])
    assert "Неотвеченные обращения: 1" in listing and "#7" in listing
    assert "Неотвеченных обращений нет" in format_ticket_list([])


def test_premium_history_lists_events():
    events = [
        SimpleNamespace(
            source="manual",
            months=1,
            stars=None,
            reason="Выдано администратором",
            created_at=NOW,
            premium_until_after=NOW + timedelta(days=30),
        )
    ]

    text = format_premium_history(events)

    assert "вручную" in text and "+1 мес." in text
    assert "Выдано администратором" in text
