"""Unit tests of the feedback formatting and parsing."""

from services.feedback import format_admin_reply, format_ticket_notice, parse_user_id


def test_notice_and_parse_roundtrip():
    notice = format_ticket_notice(
        ticket_id=7,
        username="sergey",
        user_id=123456789,
        message="Не приходит ответ",
        created_at="15.11.2026 19:30 UTC",
    )

    assert "Обращение #7" in notice
    assert "@sergey" in notice
    assert "user_id: 123456789" in notice
    assert "Не приходит ответ" in notice
    assert parse_user_id(notice) == 123456789


def test_notice_escapes_html_and_marks_missing_username():
    notice = format_ticket_notice(
        ticket_id=1,
        username=None,
        user_id=42,
        message="<b>жирный</b> & текст",
        created_at="x",
    )

    assert "без username" in notice
    assert "&lt;b&gt;" in notice
    assert "<b>жирный</b>" not in notice


def test_parse_user_id_edge_cases():
    assert parse_user_id(None) is None
    assert parse_user_id("нет id") is None


def test_admin_reply_text():
    assert format_admin_reply("  Готово  ") == "📩 Ответ от поддержки:\n\nГотово"
