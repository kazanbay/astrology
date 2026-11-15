"""Texts and keyboards of the Telegram onboarding wizard.

Kept separate from the handlers so the wording and the button layout can be
tested and edited without touching the routing code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
BUTTON_FORECAST = "📅 Прогноз"
BUTTON_SYNASTRY = "💞 Совместимость"
BUTTON_PREMIUM = "👑 Премиум"
BUTTON_HISTORY = "🕘 История"
BUTTON_HELP = "❓ Помощь"

SKIP_NAME_CALLBACK = "wizard:skip_name"
UNKNOWN_TIME_CALLBACK = "wizard:unknown_time"
CONFIRM_CALLBACK = "wizard:confirm"
EDIT_CALLBACK = "wizard:edit"

PROFILE_CONFIRM_CALLBACK = "profile:confirm"
PROFILE_EDIT_CALLBACK = "profile:edit"

CONTACT_SKIP_RELATION_CALLBACK = "contact:skip_relation"
CONTACT_UNKNOWN_TIME_CALLBACK = "contact:unknown_time"

FORECAST_PROFILE_CALLBACK = "forecast:profile"
PREMIUM_MENU_CALLBACK = "premium:menu"
PREMIUM_BUY_PREFIX = "premium:buy:"
PREMIUM_HISTORY_CALLBACK = "premium:history"
INVITE_CALLBACK = "invite:show"
FORECAST_SELF_CALLBACK = "forecast:profile:self"
FORECAST_CONTACTS_CALLBACK = "forecast:profile:contacts"
FORECAST_PICK_PREFIX = "forecast:pick:"
FORECAST_RUN_PREFIX = "forecast:run:"
SYNASTRY_PREFIX = "synastry:run:"
CONTACTS_DELETE_PREFIX = "contacts:del:"

#: Forecast intervals of the TZ: day, week, month, year.
INTERVAL_BUTTONS: tuple[tuple[str, str], ...] = (
    ("📅 День", "day"),
    ("📆 Неделя", "week"),
    ("🗓 Месяц", "month"),
    ("📈 Год", "year"),
)

GREETING = (
    "Здравствуйте! Я — астрологический помощник на классических первоисточниках.\n\n"
    "Я рассчитываю натальную карту по эфемеридам, подбираю трактовки из книг "
    "С. А. Вронского и готовлю персональный разбор.\n\n"
    "Чтобы начать, нажмите «🔮 Рассчитать карту» — я по шагам спрошу имя "
    "(его можно пропустить), дату, время и место рождения.\n\n"
    "Можно и одной командой: /chart 1990-08-15 12:00 Москва\n"
    "После расчёта карты доступны «📅 Прогноз» и «💞 Совместимость»."
)

HELP = (
    "Что я умею:\n\n"
    "🔮 «Рассчитать карту» — пошаговый ввод данных рождения и разбор карты.\n"
    "📅 «Прогноз» — транзиты на день, неделю, месяц или год по вашему профилю.\n"
    "💞 «Совместимость» — синастрия с человеком из вашего списка (премиум).\n"
    "👑 «Премиум» — подписка за звёзды Telegram: 1, 3, 6, 12 или 24 месяца.\n"
    "🕘 «История» — ваши прошлые запросы.\n"
    "💬 Любой вопрос текстом — поиск ответа по книгам.\n\n"
    "Контакты для прогнозов и совместимости:\n"
    "/contact — мастер добавления (или одной строкой:\n"
    "/contact Мария | супруга | 15.03.1992 14:30 | Казань)\n"
    "/contacts — список и удаление.\n"
    "Приглашение друга: /invite — за каждого нового пользователя +1 мес. премиума.\n\n"
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

CONTACT_WIZARD_ASK_NAME = "Шаг 1 из 5. Как зовут человека?\n\nНапример: Анна"

CONTACT_WIZARD_ASK_RELATION = (
    "Шаг 2 из 5. Кто это для вас?\n\n"
    "Например: супруга, друг, коллега. Можно пропустить."
)

CONTACT_WIZARD_ASK_DATE = "Шаг 3 из 5. Дата рождения?\n\nНапример: 05.06.1990"

CONTACT_WIZARD_ASK_TIME = (
    "Шаг 4 из 5. Время рождения?\n\n"
    "Например: 14:30. Время местное, по часам места рождения.\n"
    "Если неизвестно — нажмите кнопку ниже."
)

CONTACT_WIZARD_ASK_PLACE = "Шаг 5 из 5. Город рождения?\n\nНапример: Тула"


def main_menu() -> ReplyKeyboardMarkup:
    """Persistent bottom menu shown after /start."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BUTTON_CHART), KeyboardButton(text=BUTTON_FORECAST)],
            [KeyboardButton(text=BUTTON_SYNASTRY), KeyboardButton(text=BUTTON_HISTORY)],
            [KeyboardButton(text=BUTTON_PREMIUM), KeyboardButton(text=BUTTON_HELP)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def premium_keyboard(catalog: Any) -> InlineKeyboardMarkup:
    """One button per premium term plus the referral shortcut."""
    rows = [
        [
            InlineKeyboardButton(
                text=f"{plan.title} — {plan.stars} ⭐",
                callback_data=f"{PREMIUM_BUY_PREFIX}{plan.key}",
            )
        ]
        for plan in catalog.ordered()
    ]
    rows.append(
        [InlineKeyboardButton(text="📜 История начислений", callback_data=PREMIUM_HISTORY_CALLBACK)]
    )
    rows.append([InlineKeyboardButton(text="🔗 Пригласить друга", callback_data=INVITE_CALLBACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


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


def profile_confirm_keyboard() -> InlineKeyboardMarkup:
    """Card of the stored profile: recalculate or open the wizard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Рассчитать", callback_data=PROFILE_CONFIRM_CALLBACK)],
            [InlineKeyboardButton(text="✏️ Изменить данные", callback_data=PROFILE_EDIT_CALLBACK)],
        ]
    )


def forecast_profile_keyboard() -> InlineKeyboardMarkup:
    """Who the forecast is built for: the owner (free) or a contact (premium)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="👤 Мой прогноз", callback_data=FORECAST_SELF_CALLBACK)],
            [
                InlineKeyboardButton(
                    text="👥 Другие люди (Премиум)", callback_data=FORECAST_CONTACTS_CALLBACK
                )
            ],
        ]
    )


def forecast_interval_keyboard(contact_id: int | None) -> InlineKeyboardMarkup:
    """Forecast intervals; the selected person is encoded in the callback."""
    target = "self" if contact_id is None else f"c{contact_id}"
    buttons = [
        InlineKeyboardButton(text=label, callback_data=f"{FORECAST_RUN_PREFIX}{key}:{target}")
        for label, key in INTERVAL_BUTTONS
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[:2], buttons[2:]])


def contact_pick_keyboard(
    contacts: Iterable[Any],
    callback_prefix: str,
    *,
    cancel_callback: str | None = None,
) -> InlineKeyboardMarkup:
    """One button per contact, namespaced by ``callback_prefix``."""
    rows = [
        [
            InlineKeyboardButton(
                text=contact.name + (f" ({contact.relation})" if contact.relation else ""),
                callback_data=f"{callback_prefix}{contact.contact_id}",
            )
        ]
        for contact in contacts
    ]
    if cancel_callback:
        rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data=cancel_callback)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def contacts_delete_keyboard(contacts: Iterable[Any]) -> InlineKeyboardMarkup | None:
    """One delete button per contact; ``None`` when the list is empty."""
    rows = [
        [
            InlineKeyboardButton(
                text=f"🗑 {contact.name}",
                callback_data=f"{CONTACTS_DELETE_PREFIX}{contact.contact_id}",
            )
        ]
        for contact in contacts
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def contact_relation_keyboard() -> InlineKeyboardMarkup:
    """Skip the optional relation step of the contact wizard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Пропустить", callback_data=CONTACT_SKIP_RELATION_CALLBACK)]
        ]
    )


def contact_time_keyboard() -> InlineKeyboardMarkup:
    """Mark the birth time of a contact as unknown."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Не знаю время", callback_data=CONTACT_UNKNOWN_TIME_CALLBACK)]
        ]
    )


def format_contact_saved(contact: Any) -> str:
    """Confirmation of a stored contact; shared by the wizard and /contact."""
    relation = f" ({contact.relation})" if contact.relation else ""
    time_text = "" if contact.birth_time is None else f" {contact.birth_time.strftime('%H:%M')}"
    zone = f" ({contact.birth_timezone})" if contact.birth_timezone else ""
    return (
        f"✅ Контакт сохранён: {contact.name}{relation}\n"
        f"📅 {contact.birth_date.strftime('%d.%m.%Y')}{time_text}{zone}\n"
        f"📍 {contact.birth_city or '—'}\n\n"
        "Прогноз по этому человеку и совместимость — в меню «📅 Прогноз» и «💞 Совместимость»."
    )


def format_confirmation(
    *,
    name: str | None,
    birth_date: str,
    birth_time: str,
    timezone_name: str | None,
    place: dict[str, Any],
    time_unknown: bool = False,
    note: str | None = None,
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
    if note:
        lines += ["", note]
    return "\n".join(lines)


def format_stored_profile(profile: Any, *, note: str | None = None) -> str:
    """Confirmation card built from the stored account profile.

    Used when the user re-opens «🔮 Рассчитать карту»: the wizard is skipped and
    the previously saved data is shown for confirmation.
    """
    birth_time = profile.birth_time or time(12, 0)
    return format_confirmation(
        name=profile.display_name,
        birth_date=profile.birth_date.isoformat(),
        birth_time=birth_time.strftime("%H:%M"),
        timezone_name=profile.birth_timezone,
        place={"display_name": profile.birth_city},
        time_unknown=bool(profile.birth_time_unknown),
        note=note,
    )


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
