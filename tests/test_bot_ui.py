"""Tests for the onboarding texts and keyboards (no Telegram calls)."""

from bot_ui import (
    ASK_DATE,
    ASK_NAME,
    ASK_PLACE,
    ASK_TIME,
    BUTTON_CHART,
    BUTTON_HELP,
    BUTTON_HISTORY,
    CONFIRM_CALLBACK,
    EDIT_CALLBACK,
    GREETING,
    HELP,
    SKIP_NAME_CALLBACK,
    UNKNOWN_TIME_CALLBACK,
    build_birth_data,
    confirm_keyboard,
    format_confirmation,
    main_menu,
    skip_name_keyboard,
    unknown_time_keyboard,
)


def button_texts(markup):
    return [button.text for row in markup.keyboard for button in row]


def callback_data(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


def test_main_menu_has_chart_history_and_help():
    menu = main_menu()

    assert button_texts(menu) == [BUTTON_CHART, BUTTON_HISTORY, BUTTON_HELP]
    assert menu.is_persistent is True
    assert menu.resize_keyboard is True


def test_wizard_keyboards_expose_expected_callbacks():
    assert callback_data(skip_name_keyboard()) == [SKIP_NAME_CALLBACK]
    assert callback_data(unknown_time_keyboard()) == [UNKNOWN_TIME_CALLBACK]
    assert callback_data(confirm_keyboard()) == [CONFIRM_CALLBACK, EDIT_CALLBACK]


def test_confirmation_lists_every_field():
    text = format_confirmation(
        name="Анна",
        birth_date="1990-08-15",
        birth_time="12:30",
        timezone_name="Europe/Moscow",
        place={"display_name": "Москва, Россия"},
    )

    assert "Анна" in text
    assert "15.08.1990" in text
    assert "12:30 (Europe/Moscow)" in text
    assert "Москва, Россия" in text
    assert "неточными" not in text


def test_confirmation_marks_unknown_time_and_missing_name():
    text = format_confirmation(
        name=None,
        birth_date="1990-08-15",
        birth_time="12:00",
        timezone_name="Europe/Moscow",
        place={"display_name": "Казань"},
        time_unknown=True,
    )

    assert "не указано" in text
    assert "неизвестно, возьму 12:00" in text
    assert "неточными" in text


def test_build_birth_data_converts_local_time_to_utc():
    birth = build_birth_data(
        {
            "birth_date": "1990-08-15",
            "birth_time": "12:30",
            "timezone": "Europe/Moscow",
            "place": {"lat": 55.75, "lon": 37.61, "display_name": "Москва"},
        }
    )

    assert birth.dt_utc.isoformat() == "1990-08-15T08:30:00+00:00"
    assert birth.city == "Москва"
    assert birth.timezone == "Europe/Moscow"
    assert birth.datetime_local == "1990-08-15T12:30:00"


def test_build_birth_data_requires_every_field():
    import pytest

    with pytest.raises(KeyError):
        build_birth_data({"birth_date": "1990-08-15", "birth_time": "12:30"})


def test_onboarding_texts_mention_the_steps():
    assert "Шаг 1 из 4" in ASK_NAME
    assert "Шаг 2 из 4" in ASK_DATE
    assert "Шаг 3 из 4" in ASK_TIME
    assert "Шаг 4 из 4" in ASK_PLACE


def test_greeting_and_help_point_to_the_menu_and_command():
    assert "Рассчитать карту" in GREETING
    assert "/chart 1990-08-15 12:00 Москва" in GREETING
    assert "История" in HELP
    assert "/chart 1990-08-15 12:00 Москва" in HELP
