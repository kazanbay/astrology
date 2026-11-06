from datetime import datetime

from storage.pipeline import ChartResult
from storage.presentation import (
    TELEGRAM_MESSAGE_LIMIT,
    format_chart_reply,
    format_history,
    format_section_previews,
    split_message,
    to_telegram_html,
)
from storage.store import RequestRecord

CHART = {
    "asc": {"longitude": 200.68, "sign": "Libra", "house": 1},
    "mc": {"longitude": 118.94, "sign": "Cancer", "house": 10},
    "planets": {
        "Sun": {"longitude": 142.5, "sign": "Leo", "house": 10},
        "Moon": {"longitude": 63.4, "sign": "Gemini", "house": 8},
    },
    "aspects": [{"body_a": "Sun", "body_b": "Moon", "aspect": "sextile"}],
}

CHUNKS = [
    {
        "chunk_id": f"chunk-{index}",
        "section_title": f"2.2.{index}. Солнце в знаке Льва",
        "title": "Классическая астрология",
        "elements": [{"key": f"Sun_in_Leo_{index}", "rank": 0}],
    }
    for index in range(1, 13)
]


def build_result(**overrides):
    payload = {
        "user_id": 1,
        "request_id": 7,
        "chart_hash": "a" * 64,
        "cache_hit": False,
        "cache_written": True,
        "profile_name": "Я",
        "chart": CHART,
        "chunks": CHUNKS,
        "input_params": {
            "dt_utc": "1990-08-15T08:00:00+00:00",
            "datetime_local": "1990-08-15T12:00:00",
            "timezone": "Europe/Moscow",
            "city": "Москва, Россия",
            "latitude": 55.7558,
            "longitude": 37.6173,
            "house_system": "P",
            "language": "ru",
        },
    }
    payload.update(overrides)
    return ChartResult(**payload)


def test_reply_contains_header_angles_and_cache_state():
    text = format_chart_reply(build_result())

    assert "Карта: 1990-08-15 12:00:00 (Europe/Moscow) → 08:00 UTC" in text
    assert "Место: Москва, Россия" in text
    assert "ASC: Весы 200.7° · MC: Рак 118.9°" in text
    assert "Солнце Лев 142.5° — 10 дом" in text
    assert "Разделы базы знаний: 12 из 5 элементов" in text
    assert "Кэш: промах, результат сохранён" in text
    assert "хэш aaaaaaaaaaaa" in text


def test_reply_marks_cache_hit():
    text = format_chart_reply(build_result(cache_hit=True, cache_written=False))

    assert "Кэш: попадание (карта не пересчитывалась)" in text
    assert "сохранён" not in text


def test_planets_keep_canonical_order_after_cache_roundtrip():
    # JSONB returns object keys in its own order; the reply must not depend on it.
    reordered = {
        "planets": {
            "Moon": CHART["planets"]["Moon"],
            "Sun": CHART["planets"]["Sun"],
        },
        "asc": CHART["asc"],
        "mc": CHART["mc"],
        "aspects": CHART["aspects"],
    }

    text = format_chart_reply(build_result(chart=reordered))
    planets_line = next(line for line in text.splitlines() if line.startswith("Солнце"))

    assert planets_line.index("Солнце") < planets_line.index("Луна")


def test_reply_limits_the_number_of_sections():
    text = format_chart_reply(build_result(), max_sections=3)

    assert "1. 2.2.1. Солнце в знаке Льва" in text
    assert "3. 2.2.3. Солнце в знаке Льва" in text
    assert "4. 2.2.4" not in text


def test_reply_fits_the_message_budget():
    text = format_chart_reply(build_result(), max_chars=220)

    assert len(text) <= 220
    assert text.endswith("…")


def test_reply_without_technical_details_is_a_clean_chart_card():
    text = format_chart_reply(build_result(), technical=False)

    assert "ASC:" in text
    assert "Кэш:" not in text
    assert "Разделы базы знаний:" not in text
    assert "Найденные разделы:" not in text


def test_reply_without_chunks_reports_zero():
    text = format_chart_reply(build_result(chunks=[]))

    assert "Разделы базы знаний: 0 из 5 элементов, 0 уникальных разделов" in text
    assert "Найденные разделы:" not in text


def test_reply_stays_below_telegram_limit_by_default():
    huge_chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "section_title": "Очень длинное название раздела " * 20,
            "title": "Классическая астрология",
            "elements": [{"key": f"element-{index}", "rank": 0}],
        }
        for index in range(1, 60)
    ]
    text = format_chart_reply(build_result(chunks=huge_chunks))

    assert len(text) <= TELEGRAM_MESSAGE_LIMIT - 200


def test_section_previews_render_source_text():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "section_title": "2.2.5. Солнце в знаке Льва",
            "content": "Солнце во Льве находится на пике своего величия.",
        }
    ]

    previews = format_section_previews(chunks)

    assert len(previews) == 1
    assert previews[0].startswith("«2.2.5. Солнце в знаке Льва»")
    assert "на пике своего величия" in previews[0]


def test_section_previews_respect_limit_and_budget():
    chunks = [
        {"chunk_id": f"chunk-{index}", "section_title": f"Раздел {index}", "content": "слово " * 500}
        for index in range(1, 5)
    ]

    previews = format_section_previews(chunks, limit=2, max_chars=300)

    assert len(previews) == 2
    assert all(len(preview) <= 300 for preview in previews)
    assert all(preview.endswith("…") for preview in previews)


def test_section_previews_skip_empty_content():
    chunks = [{"chunk_id": "empty", "section_title": "Пусто", "content": "   "}]

    assert format_section_previews(chunks) == []
    assert format_section_previews([]) == []


def test_history_is_empty_without_requests():
    assert "История запросов пуста" in format_history([])


def test_split_message_keeps_short_text_intact():
    assert split_message("Короткий ответ") == ["Короткий ответ"]
    assert split_message("   ") == []


def test_split_message_respects_limit_and_paragraphs():
    paragraphs = ["Абзац номер {}. ".format(index) * 40 for index in range(5)]
    text = "\n\n".join(paragraphs)

    parts = split_message(text, limit=1000)

    assert len(parts) > 1
    assert all(len(part) <= 1000 for part in parts)
    # Only whitespace may be dropped at the split points.
    assert " ".join(parts).split() == text.split()


def test_split_message_hard_splits_a_long_token():
    parts = split_message("я" * 2500, limit=1000)

    assert [len(part) for part in parts] == [1000, 1000, 500]


def test_split_message_rejects_tiny_limit():
    import pytest

    with pytest.raises(ValueError):
        split_message("текст", limit=100)


def test_to_telegram_html_converts_bold_and_headings():
    text = "### 1. Введение\n\n**Солнце во Льве** — сила.\n\n<опасно> & 🚀"

    result = to_telegram_html(text)

    assert "<b>1. Введение</b>" in result
    assert "<b>Солнце во Льве</b>" in result
    assert "**" not in result
    assert "&lt;опасно&gt; &amp; 🚀" in result


def test_to_telegram_html_replaces_markdown_bullets():
    text = (
        "Карта насыщена аспектами:\n"
        "* Соединение Венеры и Марса: чувственность.\n"
        "  * Вложенный пункт\n"
        "- Второй пункт\n"
        "+ Третий пункт"
    )

    result = to_telegram_html(text)

    assert "* Соединение" not in result
    assert "• Соединение Венеры и Марса" in result
    assert "  • Вложенный пункт" in result
    assert "• Второй пункт" in result
    assert "• Третий пункт" in result


def test_to_telegram_html_keeps_plain_text_and_single_asterisks():
    assert to_telegram_html("2 * 3 = 6") == "2 * 3 = 6"
    assert to_telegram_html("") == ""


def test_split_message_moves_trailing_heading_to_next_part():
    intro = "А" * 200
    heading = "💼 **2. Социальная реализация и вектор судьбы**"
    body = "Б" * 200

    parts = split_message(f"{intro}\n\n{heading}\n\n{body}", limit=260)

    assert parts[0] == intro
    assert parts[1].startswith(heading)
    assert "**" not in parts[0]


def test_split_message_never_starts_with_a_lone_heading():
    text = "**1. Раздел**\n\n" + "текст " * 200

    parts = split_message(text, limit=300)

    assert parts[0].startswith("**1. Раздел**")
    assert len(parts[0]) > len("**1. Раздел**") + 50


def test_split_message_keeps_heading_when_four_lines_follow():
    intro = "А" * 50
    heading = "**2. Раздел**"
    body = "\n".join(["Строка текста номер " + "а" * 45 for _ in range(4)])
    tail = "В" * 200

    parts = split_message(f"{intro}\n\n{heading}\n\n{body}\n\n{tail}", limit=340)

    assert heading in parts[0]
    assert body in parts[0]


def test_history_lists_records():
    records = [
        RequestRecord(
            request_id=2,
            chart_hash="b" * 64,
            profile_name="Мама",
            requested_at=datetime(2026, 11, 6, 12, 30),
            input_params={
                "dt_utc": "1970-01-01T08:00:00+00:00",
                "city": "Казань, Россия",
            },
        )
    ]

    text = format_history(records)

    assert "1. 06.11.2026 12:30 · Мама · Казань, Россия · 01.01.1970 08:00 UTC · хэш bbbbbbbb" in text
