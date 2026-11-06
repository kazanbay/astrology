"""Texts and keyboards of the Telegram onboarding wizard.

Kept separate from the handlers so the wording and the button layout can be
tested and edited without touching the routing code.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time
from typing import Any

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from storage.birth import BirthData, local_to_utc

BUTTON_CHART = "🔮 Рассчитать карту"
BUTTON_HISTORY = "🕘 История"
BUTTON_HELP = "❓ Помощь"

SKIP_NAME_CALLBACK = "wizard:skip_name"
UNKNOWN_TIME_CALLBACK = "wizard:unknown_time"
CONFIRM_CALLBACK = "wizard:confirm"
EDIT_CALLBACK = "wizard:edit"

GREETING = (
    "Здравствуйте! Я — астрологический помощник на классических первоисточниках.\n\n"
    "Я рассчитываю натальную карту по эфемеридам, подбираю трактовки из книг "
    "С. А. Вронского и готовлю персональный разбор.\n\n"
    "Чтобы начать, нажмите «🔮 Рассчитать карту» — я по шагам спрошу имя "
    "(его можно пропустить), дату, время и место рождения.\n\n"
    "Можно и одной командой: /chart 1990-08-15 12:00 Москва"
)

HELP = (
    "Что я умею:\n\n"
    "🔮 «Рассчитать карту» — пошаговый ввод данных рождения и разбор карты.\n"
    "🕘 «История» — ваши прошлые запросы.\n"
    "💬 Любой вопрос текстом — поиск ответа по книгам.\n\n"
    "Быстрый расчёт одной командой:\n"
    "/chart 1990-08-15 12:00 Москва\n\n"
    "Важно: время рождения нужно указывать местное, по часам места рождения. "
    "Без точного времени дома и ASC определяются приблизительно."
)

ASK_NAME = (
    "Шаг 1 из 4. Как вас зовут?\n\n"
    "Имя нужно только для обращения — например, чтобы поздороваться по имени. "
    "Если не хотите, нажмите «Пропустить»."
)

ASK_DATE = "Шаг 2 из 4. Введите дату рождения.\n\nНапример: 15.08.1990"

ASK_TIME = (
    "Шаг 3 из 4. Введите время рождения.\n\n"
    "Например: 12:30. Время местное, по часам места рождения.\n"
    "Если время неизвестно — нажмите кнопку ниже."
)

ASK_PLACE = (
    "Шаг 4 из 4. Введите город рождения.\n\n"
    "Например: Москва. Можно уточнить: Москва, Россия"
)

CONFIRM_PROMPT = "Проверьте данные перед расчётом:"


def main_menu() -> ReplyKeyboardMarkup:
    """Persistent bottom menu shown after /start."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BUTTON_CHART)],
            [KeyboardButton(text=BUTTON_HISTORY), KeyboardButton(text=BUTTON_HELP)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def skip_name_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Пропустить", callback_data=SKIP_NAME_CALLBACK)]]
    )


def unknown_time_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Не знаю время", callback_data=UNKNOWN_TIME_CALLBACK)]
        ]
    )


def confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Рассчитать", callback_data=CONFIRM_CALLBACK)],
            [InlineKeyboardButton(text="✏️ Изменить данные", callback_data=EDIT_CALLBACK)],
        ]
    )


def format_confirmation(
    *,
    name: str | None,
    birth_date: str,
    birth_time: str,
    timezone_name: str | None,
    place: dict[str, Any],
    time_unknown: bool = False,
) -> str:
    """Render the "check your data" message before the expensive calculation."""
    moment = date.fromisoformat(birth_date)
    time_text = "неизвестно, возьму 12:00" if time_unknown else birth_time
    zone_text = f" ({timezone_name})" if timezone_name else ""
    place_name = str(place.get("display_name") or "—")

    lines = [
        CONFIRM_PROMPT,
        "",
        f"👤 Имя: {name or 'не указано'}",
        f"📅 Дата: {moment.strftime('%d.%m.%Y')}",
        f"🕐 Время: {time_text}{zone_text}",
        f"📍 Место: {place_name}",
    ]
    if time_unknown:
        lines += ["", "⚠️ Без точного времени дома и ASC могут быть неточными."]
    return "\n".join(lines)


def build_birth_data(profile: Mapping[str, Any], *, house_system: str = "P") -> BirthData:
    """Turn the wizard's collected profile into the canonical ``BirthData``."""
    moment_local = datetime.combine(
        date.fromisoformat(str(profile["birth_date"])),
        time.fromisoformat(str(profile["birth_time"])),
    )
    place = profile["place"]
    timezone_name = str(profile["timezone"])
    utc, _offset, _notes = local_to_utc(moment_local, timezone_name)
    return BirthData(
        dt_utc=utc,
        latitude=place["lat"],
        longitude=place["lon"],
        house_system=house_system,
        city=str(place.get("display_name") or "") or None,
        timezone=timezone_name,
        datetime_local=moment_local.isoformat(),
    )
