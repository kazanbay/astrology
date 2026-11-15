"""Tests for the onboarding texts and keyboards (no Telegram calls)."""

from datetime import date, time

from bot_ui import (
    ASK_DATE,
    ASK_NAME,
    ASK_PLACE,
    ASK_TIME,
    BUTTON_CHART,
    BUTTON_FORECAST,
    BUTTON_HELP,
    BUTTON_HISTORY,
    BUTTON_SYNASTRY,
    CONFIRM_CALLBACK,
    CONTACTS_DELETE_PREFIX,
    EDIT_CALLBACK,
    FORECAST_CONTACTS_CALLBACK,
    FORECAST_PICK_PREFIX,
    FORECAST_PROFILE_CALLBACK,
    FORECAST_RUN_PREFIX,
    FORECAST_SELF_CALLBACK,
    GREETING,
    HELP,
    PROFILE_CONFIRM_CALLBACK,
    PROFILE_EDIT_CALLBACK,
    SKIP_NAME_CALLBACK,
    SYNASTRY_PREFIX,
    UNKNOWN_TIME_CALLBACK,
    build_birth_data,
    confirm_keyboard,
    contact_pick_keyboard,
    contacts_delete_keyboard,
    forecast_interval_keyboard,
    forecast_profile_keyboard,
    format_confirmation,
    format_stored_profile,
    main_menu,
    profile_confirm_keyboard,
    skip_name_keyboard,
    unknown_time_keyboard,
)
from storage.store import ContactProfile, UserProfile


def button_texts(markup):
    return [button.text for row in markup.keyboard for button in row]


def callback_data(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


def contact(name, relation, contact_id):
    return ContactProfile(contact_id=contact_id, user_id=1, name=name, relation=relation)


def test_main_menu_has_chart_forecast_synastry_history_and_help():
    menu = main_menu()

    assert button_texts(menu) == [
        BUTTON_CHART,
        BUTTON_FORECAST,
        BUTTON_SYNASTRY,
        BUTTON_HISTORY,
        BUTTON_HELP,
    ]
    assert menu.is_persistent is True
    assert menu.resize_keyboard is True


def test_wizard_keyboards_expose_expected_callbacks():
    assert callback_data(skip_name_keyboard()) == [SKIP_NAME_CALLBACK]
    assert callback_data(unknown_time_keyboard()) == [UNKNOWN_TIME_CALLBACK]
    assert callback_data(confirm_keyboard()) == [CONFIRM_CALLBACK, EDIT_CALLBACK]


def test_profile_confirm_keyboard_reuses_the_card_layout():
    markup = profile_confirm_keyboard()

    assert [button.text for row in markup.inline_keyboard for button in row] == [
        "✅ Рассчитать",
        "✏️ Изменить данные",
    ]
    assert callback_data(markup) == [PROFILE_CONFIRM_CALLBACK, PROFILE_EDIT_CALLBACK]


def test_format_stored_profile_renders_saved_data_and_note():
    profile = UserProfile(
        user_id=1,
        telegram_id=42,
        display_name="Сергей",
        birth_date=date(1985, 1, 29),
        birth_time=time(2, 0),
        birth_time_unknown=False,
        birth_lat=56.14,
        birth_lon=47.25,
        birth_timezone="Europe/Moscow",
        birth_city="Чебоксары, Россия",
    )

    text = format_stored_profile(profile, note="🔄 Изменений: 1 из 3")

    assert text.startswith("Проверьте данные перед расчётом:")
    assert "👤 Имя: Сергей" in text
    assert "📅 Дата: 29.01.1985" in text
    assert "🕐 Время: 02:00 (Europe/Moscow)" in text
    assert "📍 Место: Чебоксары, Россия" in text
    assert "Изменений: 1 из 3" in text


def test_forecast_profile_keyboard_offers_self_and_contacts():
    markup = forecast_profile_keyboard()

    assert [button.text for row in markup.inline_keyboard for button in row] == [
        "👤 Мой прогноз",
        "👥 Другие люди (Премиум)",
    ]
    assert callback_data(markup) == [FORECAST_SELF_CALLBACK, FORECAST_CONTACTS_CALLBACK]


def test_forecast_interval_keyboard_encodes_the_target():
    assert callback_data(forecast_interval_keyboard(None)) == [
        f"{FORECAST_RUN_PREFIX}day:self",
        f"{FORECAST_RUN_PREFIX}week:self",
        f"{FORECAST_RUN_PREFIX}month:self",
        f"{FORECAST_RUN_PREFIX}year:self",
    ]
    assert callback_data(forecast_interval_keyboard(7))[0] == f"{FORECAST_RUN_PREFIX}day:c7"


def test_contact_pick_keyboard_lists_names_and_cancel():
    contacts = [contact("Мария", "супруга", 7), contact("Олег", None, 8)]

    markup = contact_pick_keyboard(
        contacts, FORECAST_PICK_PREFIX, cancel_callback=FORECAST_PROFILE_CALLBACK
    )

    assert [button.text for row in markup.inline_keyboard for button in row] == [
        "Мария (супруга)",
        "Олег",
        "⬅️ Назад",
    ]
    assert callback_data(markup) == [
        f"{FORECAST_PICK_PREFIX}7",
        f"{FORECAST_PICK_PREFIX}8",
        FORECAST_PROFILE_CALLBACK,
    ]


def test_contact_pick_keyboard_without_cancel():
    markup = contact_pick_keyboard([contact("Мария", None, 7)], SYNASTRY_PREFIX)

    assert callback_data(markup) == [f"{SYNASTRY_PREFIX}7"]


def test_contacts_delete_keyboard():
    assert contacts_delete_keyboard([]) is None

    markup = contacts_delete_keyboard([contact("Мария", None, 7)])
    assert callback_data(markup) == [f"{CONTACTS_DELETE_PREFIX}7"]


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
    assert "📅 Прогноз" in GREETING
    assert "История" in HELP
    assert "«Прогноз»" in HELP
    assert "«Совместимость»" in HELP
    assert "/contact Мария | супруга | 15.03.1992 14:30 | Казань" in HELP
    assert "/chart 1990-08-15 12:00 Москва" in HELP
