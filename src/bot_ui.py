"""Texts and keyboards of the Telegram onboarding wizard.

Kept separate from the handlers so the wording and the button layout can be
tested and edited without touching the routing code.
"""

from __future__ import annotations

import html
import re
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
BUTTON_CONTACTS = "👥 Мои контакты"
BUTTON_FEEDBACK = "💬 Обратная связь"
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
GLOSSARY_CALLBACK = "glossary:show"
HELP_CALLBACK = "help:show"
FORECAST_SELF_CALLBACK = "forecast:profile:self"
FORECAST_CONTACTS_CALLBACK = "forecast:profile:contacts"
FORECAST_PICK_PREFIX = "forecast:pick:"
FORECAST_RUN_PREFIX = "forecast:run:"
SYNASTRY_PREFIX = "synastry:run:"
CONTACTS_DELETE_PREFIX = "contacts:del:"
CONTACTS_OPEN_PREFIX = "contacts:open:"
CONTACTS_EDIT_PREFIX = "contacts:edit:"
CONTACTS_ADD_CALLBACK = "contacts:add"
CONTACTS_ADD_SYNASTRY_CALLBACK = "contacts:add:synastry"
CONTACTS_ADD_FORECAST_CALLBACK = "contacts:add:forecast"
CONTACTS_ADD_CHART_CALLBACK = "contacts:add:chart"
CONTACTS_BACK_CALLBACK = "contacts:list"
CHART_CONTACT_PICK_CALLBACK = "chart:contact:pick"
CHART_CONTACT_RUN_PREFIX = "chart:contact:run:"
CHART_SELF_CARD_CALLBACK = "chart:self"

#: Forecast intervals of the TZ: day, week, month, year.
INTERVAL_BUTTONS: tuple[tuple[str, str], ...] = (
    ("📅 День", "day"),
    ("📆 Неделя", "week"),
    ("🗓 Месяц", "month"),
    ("📈 Год", "year"),
)

GREETING = (
    "🔮 *Добро пожаловать в классическую астрологию нового поколения!*\n\n"
    "Здесь нет случайных гороскопов из интернета. Мы рассчитываем вашу натальную карту по точным "
    "эфемеридам Швейцарского банка (`pyswisseph`), с учетом реального времени и исторических часовых поясов, "
    "а трактовки берем из академических трудов С.А. Вронского.\n\n"
    "Чтобы начать, нажмите *«🔮 Рассчитать карту»* ниже — я бережно проведу вас по шагам: спрошу имя, дату, точное время и город рождения.\n\n"
    "⚡️ *Лайфхак:* можно запустить расчет одной быстрой командой:\n"
    "`/chart 1990-08-15 12:00 Москва`\n\n"
    "📖 Незнакомы термины — загляните в словарь: /glossary.\n\n"
    "После расчёта вам станут доступны персональные *«📅 Прогноз»* и *«💞 Совместимость»*."
)

HELP = (
    "🔮 *Добро пожаловать в классическую астрологию нового поколения!*\n\n"
    "Здесь нет шаблонных гороскопов из интернета. Мы используем академический подход С.А. Вронского, "
    "точные эфемериды Швейцарского банка (`pyswisseph`) и учет исторических часовых поясов. "
    "Бот вычисляет реальные координаты планет и домов на момент вашего рождения.\n\n"
    "--- \n\n"
    "🛠 *Что я умею:*\n\n"
    "🔮 «Рассчитать карту» — разбор себе или контакту (премиум).\n"
    "📅 «Прогноз» — транзиты на день, неделю, месяц или год по вашему профилю.\n"
    "💞 «Совместимость» — синастрия с человеком из вашего списка (премиум).\n"
    "👥 «Мои контакты» — карточки людей: редактирование и удаление.\n"
    "💬 «Обратная связь» — написать разработчику; ответ придёт сюда же.\n"
    "👑 «Премиум» — подписка за звёзды Telegram: 1, 3, 6, 12 или 24 месяца.\n"
    "🕘 «История» — ваши прошлые запросы.\n"
    "💬 Любой вопрос текстом — поиск ответа по книгам.\n\n"
    "Контакты для прогнозов и совместимости:\n"
    "/contact — мастер добавления (или одной строкой:\n"
    "/contact Мария | супруга | 15.03.1992 14:30 | Казань)\n"
    "/contacts — список и удаление.\n"
    "Приглашение друга: /invite — премиум получают и вы, и приглашённый.\n\n"
    "📖 Незнакомы термины — словарь: /glossary (кнопка ниже).\n\n"
    "Быстрый расчёт одной командой:\n"
    "/chart 1990-08-15 12:00 Москва\n\n"
    "⚠️ *Важно:* время рождения нужно указывать местное, по часам места рождения. "
    "Без точного времени дома и ASC определяются приблизительно."
)

GLOSSARY = (
    "📖 *Словарь новичка*\n\n"
    "Главные термины, которые встречаются в разборах бота.\n\n"
    "--- \n\n"
    "*1. Основы и структура карты:*\n"
    "• *Натальная карта* — персональный снимок звездного неба в секунду вашего рождения. Показывает характер, таланты и жизненные сценарии.\n"
    "• *Зодиак и Знаки* — 12 секторов неба (от Овна до Рыб), каждый со своей стихией: Огонь, Земля, Воздух, Вода.\n"
    "• *Стихии* — Огонь (действие), Земля (практика), Воздух (общение), Вода (чувства): темперамент знака.\n"
    "• *Кресты (качества)* — кардинальный (начинает), фиксированный (держит), мутабельный (меняет): как знак действует.\n"
    "• *Дома (12 домов)* — сферы вашей жизни: от личности (1-й дом) и денег (2-й) до отношений (7-й) и карьеры (10-й).\n"
    "• *Куспид* — граница (начало) любого дома в карте.\n\n"
    "*2. Главные точки (углы карты):*\n"
    "• *Асцендент (Asc)* — знак на восточном горизонте в момент рождения. Ваша внешняя «маска», первое впечатление и стиль поведения.\n"
    "• *Десцендент (Dsc)* — точка напротив Асцендента. Отвечает за партнерство, брак и то, каких людей вы притягиваете.\n"
    "• *Середина Неба (MC)* — высшая точка карты. Отвечает за карьеру, признание, статус и главные жизненные цели.\n"
    "• *Имум Коэли (IC)* — точка напротив MC. Символизирует корни, семью, дом и ваш внутренний тыл.\n\n"
    "*3. Планеты (действующие лица):*\n"
    "• *Светила* — Солнце (ядро личности, воля) и Луна (эмоции, привычки, внутренняя опора).\n"
    "• *Личные планеты* (Меркурий, Венера, Марс) — мышление и речь, любовь и ценности, активность и воля.\n"
    "• *Социальные планеты* (Юпитер, Сатурн) — успех в обществе, амбиции, правила и ответственность.\n"
    "• *Высшие планеты* (Уран, Нептун, Плутон) — планеты поколений, глубоких трансформаций, кризисов и перемен.\n"
    "• *Управитель (диспозитор)* — планета, управляющая знаком или домом: «хозяин» этой темы в карте.\n"
    "• *Достоинства* — обитель и экзальтация (планета сильна), изгнание и падение (планета ослаблена).\n"
    "• *Лунные узлы* — Северный (Раху) и Южный (Кету): ось между привычным и тем, куда расти.\n"
    "• *Ретроградность* — период, когда кажется, что планета движется назад. Время замедления и работы над ошибками.\n\n"
    "*4. Аспекты (углы и диалог между планетами):*\n"
    "• *Соединение (0°)* — планеты стоят рядом и сливают свои энергии в одну мощную силу.\n"
    "• *Секстиль (60°)* — мягкий гармоничный аспект. Дает шансы и возможности.\n"
    "• *Квадратура (90°)* — напряженный аспект, точка трения и конфликта, заставляющая расти и действовать.\n"
    "• *Трин / Тригон (120°)* — самый гармоничный аспект. Приносит удачу, талант и поддержку без усилий.\n"
    "• *Оппозиция (180°)* — противостояние двух планет. Требует поиска баланса и компромисса.\n"
    "• *Орб* — допустимое отклонение от точного угла: чем меньше орб, тем сильнее и заметнее аспект.\n"
    "• *Стеллиум* — скопление трёх и более планет в одном знаке или доме: усиленная тема.\n"
    "• *Большой трин* — три планеты в тринах: устойчивый талант и «запас прочности».\n"
    "• *Тау-квадрат* — две квадратуры и оппозиция: зона напряжения, которая даёт мотивацию.\n\n"
    "*5. Дома и режимы работы бота:*\n"
    "• *Угловые дома* (1, 4, 7, 10) — самые заметные сферы: «я», дом, партнёр, карьера.\n"
    "• *Последующие и падающие* (2, 5, 8, 11 и 3, 6, 9, 12) — ресурсы, творчество и обучение.\n"
    "• *Транзиты* — «текущая погода на небе»: движение планет прямо сейчас и их влияние на вашу карту.\n"
    "• *Синастрия* — наложение двух карт друг на друга для анализа совместимости в паре.\n"
    "• *Эфемериды* — точные астрономические таблицы координат планет."
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

#: Light Markdown used in the authored UI texts: *bold*, `code`, --- rule.
_UI_BOLD_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_UI_CODE_RE = re.compile(r"`([^`\n]+)`")
_UI_RULE_RE = re.compile(r"^[ \t]*---[ \t]*$", re.MULTILINE)
_UI_TAG_RE = re.compile(r"<[^>]+>")


def render_ui_html(text: str) -> str:
    """Render authored UI texts as Telegram-safe HTML.

    Only the light markup our own texts use is converted: ``*bold*``,
    ```code``` and a ``---`` separator. Everything else is escaped, so a stray
    ``<`` or ``&`` in the copy can never break the message.
    """
    escaped = html.escape(text, quote=False)
    escaped = _UI_RULE_RE.sub("──────────", escaped)
    escaped = _UI_CODE_RE.sub(r"<code>\1</code>", escaped)
    return _UI_BOLD_RE.sub(r"<b>\1</b>", escaped)


def strip_ui_markdown(text: str) -> str:
    """Plain-text fallback: drop tags and the markup characters."""
    without_tags = _UI_TAG_RE.sub("", text)
    return (
        without_tags.replace("*", "")
        .replace("`", "")
        .replace("──────────", "")
    )

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

FEEDBACK_PROMPT = (
    "💬 Напишите сообщение для разработчика.\n\n"
    "Оно уйдёт в поддержку вместе с вашим @username, чтобы мы могли ответить вам здесь же.\n"
    "Отмена — /cancel."
)

FEEDBACK_SENT = "Спасибо! Ваше сообщение отправлено разработчику. Ожидайте ответа."

FEEDBACK_EMPTY = "Пока нечего отправлять: напишите текст сообщения. Отмена — /cancel."

FEEDBACK_UNAVAILABLE = "Обратная связь временно недоступна. Попробуйте позже."


def main_menu() -> ReplyKeyboardMarkup:
    """Persistent bottom menu shown after /start."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BUTTON_CHART), KeyboardButton(text=BUTTON_FORECAST)],
            [KeyboardButton(text=BUTTON_CONTACTS), KeyboardButton(text=BUTTON_SYNASTRY)],
            [KeyboardButton(text=BUTTON_HISTORY), KeyboardButton(text=BUTTON_PREMIUM)],
            [KeyboardButton(text=BUTTON_FEEDBACK), KeyboardButton(text=BUTTON_HELP)],
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
    """Card of the stored profile: recalculate, edit or calculate for a contact."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Рассчитать", callback_data=PROFILE_CONFIRM_CALLBACK)],
            [InlineKeyboardButton(text="✏️ Изменить данные", callback_data=PROFILE_EDIT_CALLBACK)],
            [
                InlineKeyboardButton(
                    text="👥 Рассчитать контакту", callback_data=CHART_CONTACT_PICK_CALLBACK
                )
            ],
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
    add_callback: str | None = None,
) -> InlineKeyboardMarkup:
    """One button per contact, optional add and back actions."""
    rows = [
        [
            InlineKeyboardButton(
                text=contact.name + (f" ({contact.relation})" if contact.relation else ""),
                callback_data=f"{callback_prefix}{contact.contact_id}",
            )
        ]
        for contact in contacts
    ]
    if add_callback:
        rows.append(
            [InlineKeyboardButton(text="➕ Добавить контакт", callback_data=add_callback)]
        )
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


def help_keyboard() -> InlineKeyboardMarkup:
    """Short help screen: the glossary lives behind one button."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📖 Словарь терминов", callback_data=GLOSSARY_CALLBACK)]
        ]
    )


def glossary_keyboard() -> InlineKeyboardMarkup:
    """Glossary screen: back to the short help."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ К помощи", callback_data=HELP_CALLBACK)]
        ]
    )


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


def contact_card_keyboard(contact_id: int) -> InlineKeyboardMarkup:
    """Actions of one contact card."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✏️ Изменить", callback_data=f"{CONTACTS_EDIT_PREFIX}{contact_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Удалить", callback_data=f"{CONTACTS_DELETE_PREFIX}{contact_id}"
                )
            ],
            [InlineKeyboardButton(text="⬅️ К списку", callback_data=CONTACTS_BACK_CALLBACK)],
        ]
    )


def contacts_list_keyboard(contacts: Iterable[Any]) -> InlineKeyboardMarkup:
    """One button per contact plus the add action."""
    rows = [
        [
            InlineKeyboardButton(
                text=contact.name + (f" ({contact.relation})" if contact.relation else ""),
                callback_data=f"{CONTACTS_OPEN_PREFIX}{contact.contact_id}",
            )
        ]
        for contact in contacts
    ]
    rows.append(
        [InlineKeyboardButton(text="➕ Добавить контакт", callback_data=CONTACTS_ADD_CALLBACK)]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def premium_upsell_keyboard() -> InlineKeyboardMarkup:
    """Both premium paths: buy with Stars or invite a friend for a free month."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="👑 Оформить премиум", callback_data=PREMIUM_MENU_CALLBACK)],
            [InlineKeyboardButton(text="🔗 Пригласить друга", callback_data=INVITE_CALLBACK)],
        ]
    )


def format_contact_card(contact: Any) -> str:
    """Detail card of one stored contact."""
    relation = f" ({contact.relation})" if contact.relation else ""
    time_text = (
        "неизвестно" if contact.birth_time is None else contact.birth_time.strftime("%H:%M")
    )
    zone = f" ({contact.birth_timezone})" if contact.birth_timezone else ""
    return (
        f"👤 {contact.name}{relation}\n"
        f"📅 {contact.birth_date.strftime('%d.%m.%Y')} · 🕐 {time_text}{zone}\n"
        f"📍 {contact.birth_city or '—'}"
    )


def format_contacts_screen(contacts: Iterable[Any]) -> str:
    """Header of the contacts screen; the contacts themselves are buttons."""
    if not list(contacts):
        return (
            "👥 Контактов пока нет.\n\n"
            "Добавьте человека: /contact — мастер спросит имя, дату, время и место рождения."
        )
    return (
        "👥 Ваши контакты\n\n"
        "Нажмите на контакт, чтобы открыть карточку: изменить данные или удалить."
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
