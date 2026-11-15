"""Unit tests of the product-metrics digest formatting."""

from datetime import datetime, timezone

from services.analytics import format_daily_digest, format_stats_now, format_weekly_digest

METRICS = {
    "total_users": 12,
    "new_users": 3,
    "dau": 4,
    "wau": 7,
    "mau": 10,
    "active_premium": 2,
    "new_payments": 1,
    "revenue_stars": 250,
    "charts": 5,
    "forecasts": 9,
    "synastry": 2,
    "feedback_new": 2,
    "feedback_open": 1,
}

NOW = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)


def test_daily_digest_has_all_sections():
    text = format_daily_digest(METRICS, now=NOW)

    assert "Дайджест дня" in text
    assert "15.11.2026" in text
    assert "всего 12 (+3 за сутки)" in text
    assert "DAU 4" in text and "WAU 7" in text and "MAU 10" in text
    assert "⭐ 250" in text
    assert "карты 5" in text and "прогнозы 9" in text and "синастрия 2" in text
    assert "новых 2" in text and "в работе 1" in text


def test_weekly_digest_has_stickiness_and_top_features():
    text = format_weekly_digest(METRICS, now=NOW)

    assert "Итоги недели" in text
    assert "Stickiness 40%" in text
    assert "Топ функций: 📅 Прогнозы (9)" in text


def test_weekly_zero_mau_is_safe():
    zeros = {key: 0 for key in METRICS}

    text = format_weekly_digest(zeros, now=NOW)

    assert "Stickiness —" in text


def test_stats_now_is_marked_on_demand():
    text = format_stats_now(METRICS, now=NOW)

    assert text.startswith("📊 <b>Сводка на сейчас</b>")
