"""Unit tests of the Telegram rendering of the forecast and synastry tools."""

from services.presentation import (
    format_contacts_list,
    format_forecast_text,
    format_synastry_text,
)

FORECAST_RESULT = {
    "tool": "forecast",
    "interval": "day",
    "person": {"kind": "owner", "name": "Анна", "relation": None},
    "birth": {
        "city": "Москва",
        "timezone": "Europe/Moscow",
        "datetime_local": "1990-08-15T12:00:00",
        "time_unknown": False,
    },
    "now_transits": [
        {"name": "Sun", "longitude": 280.0, "sign": "Capricorn", "house": 10},
        {"name": "Moon", "longitude": 100.0, "sign": "Cancer", "house": 4},
    ],
    "events": [
        {
            "transit": "Saturn",
            "natal": "Sun",
            "aspect": "square",
            "orb": 0.8,
            "hits": 4,
            "first_sample": "2026-01-01T00:00:00+00:00",
            "last_sample": "2026-01-20T00:00:00+00:00",
        },
        {
            "transit": "Moon",
            "natal": "Moon",
            "aspect": "conjunction",
            "orb": 1.0,
            "hits": 1,
            "first_sample": "2026-01-01T03:00:00+00:00",
            "last_sample": "2026-01-01T03:00:00+00:00",
        },
    ],
    "event_count": 2,
}


def test_forecast_text_header_and_events():
    text = format_forecast_text(FORECAST_RESULT)

    assert text.startswith("📅 Прогноз на день")
    assert "👤 Мой прогноз" in text
    assert "📍 Место рождения: Москва" in text
    assert "Небо сейчас: Солнце Козерог (10 д.) · Луна Рак (4 д.)" in text
    assert "1. Сатурн квадрат Солнце — орб 0.8° (01.01–20.01)" in text
    # 03:00 UTC is 06:00 in Europe/Moscow.
    assert "2. Луна соединение Луна — орб 1.0° (01.01 06:00)" in text


def test_forecast_text_marks_unknown_time_and_contacts():
    result = {
        **FORECAST_RESULT,
        "person": {"kind": "contact", "name": "Мария", "relation": "супруга"},
        "birth": {**FORECAST_RESULT["birth"], "time_unknown": True},
    }

    text = format_forecast_text(result)

    assert "👥 Мария (супруга)" in text
    assert "Время рождения неизвестно" in text


def test_forecast_without_events():
    result = {**FORECAST_RESULT, "events": [], "event_count": 0, "now_transits": []}

    text = format_forecast_text(result)

    assert "Транзитов с орбом до 3°" in text
    assert "Небо сейчас" not in text


def test_forecast_titles_for_every_interval():
    for interval, title in (
        ("week", "на неделю"),
        ("month", "на месяц"),
        ("year", "на год"),
    ):
        text = format_forecast_text({**FORECAST_RESULT, "interval": interval, "events": []})
        assert text.startswith(f"📅 Прогноз {title}")


def test_synastry_text_renders_people_and_aspects():
    result = {
        "tool": "synastry",
        "person_a": {
            "kind": "owner",
            "name": "Анна",
            "positions": {
                "Sun": {"sign": "Leo", "longitude": 142.5},
                "Moon": {"sign": "Virgo", "longitude": 165.0},
            },
        },
        "person_b": {
            "kind": "contact",
            "name": "Мария",
            "relation": "супруга",
            "positions": {"Sun": {"sign": "Taurus", "longitude": 40.0}},
        },
        "events": [
            {"body_a": "Moon", "body_b": "Sun", "aspect": "sextile", "orb": 0.0},
            {"body_a": "Sun", "body_b": "Venus", "aspect": "conjunction", "orb": 5.0},
        ],
        "event_count": 2,
    }

    text = format_synastry_text(result)

    assert text.startswith("💞 Совместимость: Анна + Мария (супруга)")
    assert "• Анна: Солнце Лев, Луна Дева" in text
    assert "1. Луна секстиль Солнце — орб 0.0°" in text
    assert "2. Солнце соединение Венера — орб 5.0°" in text


def test_synastry_without_events():
    result = {
        "tool": "synastry",
        "person_a": {"kind": "owner", "name": "Анна", "positions": {}},
        "person_b": {"kind": "contact", "name": "Мария", "relation": None, "positions": {}},
        "events": [],
        "event_count": 0,
    }

    text = format_synastry_text(result)

    assert "Мажорных аспектов" in text
    assert "• Мария: —" in text


def test_contacts_list_empty_hint_and_rows():
    empty = format_contacts_list([])
    assert "/contact Мария | супруга | 15.03.1992 14:30 | Казань" in empty

    rows = format_contacts_list(
        [
            {"name": "Мария", "relation": "супруга", "birth_date": "15.03.1992",
             "birth_time_unknown": False},
            {"name": "Пётр", "relation": None, "birth_date": "02.11.1985",
             "birth_time_unknown": True},
        ]
    )
    assert "Ваши контакты:" in rows
    assert "1. Мария (супруга) — 15.03.1992" in rows
    assert "2. Пётр — 02.11.1985 (время неизвестно)" in rows


def test_long_text_is_trimmed():
    result = {
        **FORECAST_RESULT,
        "events": [
            {
                "transit": f"Planet{index}",
                "natal": "Sun",
                "aspect": "square",
                "orb": index / 10,
                "hits": 1,
                "first_sample": "2026-01-01T00:00:00+00:00",
                "last_sample": "2026-01-01T00:00:00+00:00",
            }
            for index in range(120)
        ],
    }

    text = format_forecast_text(result, max_chars=200)

    assert len(text) <= 200
    assert text.endswith("…")
