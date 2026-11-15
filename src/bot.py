"""Telegram entry point for the astrology knowledge base assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from pathlib import Path
from typing import Any

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, LabeledPrice, Message, PreCheckoutQuery, ReactionTypeEmoji, User
from dotenv import load_dotenv

from astronomy.calculator import AstronomyError
from bot_ui import (
    ASK_DATE,
    ASK_NAME,
    ASK_PLACE,
    ASK_TIME,
    BUTTON_CHART,
    BUTTON_CONTACTS,
    BUTTON_FEEDBACK,
    BUTTON_FORECAST,
    BUTTON_HELP,
    BUTTON_HISTORY,
    BUTTON_PREMIUM,
    BUTTON_SYNASTRY,
    CHART_CONTACT_PICK_CALLBACK,
    CHART_CONTACT_RUN_PREFIX,
    CHART_SELF_CARD_CALLBACK,
    CONTACTS_ADD_CALLBACK,
    CONTACTS_ADD_CHART_CALLBACK,
    CONTACTS_ADD_FORECAST_CALLBACK,
    CONTACTS_ADD_SYNASTRY_CALLBACK,
    CONTACTS_BACK_CALLBACK,
    CONTACTS_DELETE_PREFIX,
    CONTACTS_EDIT_PREFIX,
    CONTACTS_OPEN_PREFIX,
    CONTACT_SKIP_RELATION_CALLBACK,
    CONTACT_UNKNOWN_TIME_CALLBACK,
    CONTACT_WIZARD_ASK_DATE,
    CONTACT_WIZARD_ASK_NAME,
    CONTACT_WIZARD_ASK_PLACE,
    CONTACT_WIZARD_ASK_RELATION,
    CONTACT_WIZARD_ASK_TIME,
    CONFIRM_CALLBACK,
    EDIT_CALLBACK,
    FORECAST_CONTACTS_CALLBACK,
    FORECAST_PICK_PREFIX,
    FORECAST_PROFILE_CALLBACK,
    FORECAST_RUN_PREFIX,
    FORECAST_SELF_CALLBACK,
    GREETING,
    HELP,
    FEEDBACK_EMPTY,
    FEEDBACK_PROMPT,
    FEEDBACK_SENT,
    FEEDBACK_UNAVAILABLE,
    INVITE_CALLBACK,
    PREMIUM_BUY_PREFIX,
    PREMIUM_HISTORY_CALLBACK,
    PREMIUM_MENU_CALLBACK,
    PROFILE_CONFIRM_CALLBACK,
    PROFILE_EDIT_CALLBACK,
    SKIP_NAME_CALLBACK,
    SYNASTRY_PREFIX,
    UNKNOWN_TIME_CALLBACK,
    build_birth_data,
    confirm_keyboard,
    contact_card_keyboard,
    contact_pick_keyboard,
    contact_relation_keyboard,
    contact_time_keyboard,
    contacts_delete_keyboard,
    contacts_list_keyboard,
    forecast_interval_keyboard,
    forecast_profile_keyboard,
    format_confirmation,
    format_contact_card,
    format_contact_saved,
    format_contacts_screen,
    format_stored_profile,
    main_menu,
    premium_keyboard,
    premium_upsell_keyboard,
    profile_confirm_keyboard,
    render_ui_html,
    skip_name_keyboard,
    strip_ui_markdown,
    unknown_time_keyboard,
)
from generator.report import ChartReportService
from rag.generator import RAGGenerator
from rag.retriever import QdrantRetriever
from services.analytics import format_stats_now
from services.feedback import (
    format_admin_reply,
    format_ticket_notice,
    parse_user_id,
)
from services.forecast import ForecastService
from services.digest import run_digest_scheduler
from services.forecast_report import ForecastReportService
from services.premium import (
    PremiumCatalog,
    PremiumConfigError,
    format_invite_text,
    format_premium_history,
    format_premium_status,
    load_catalog,
)
from services.presentation import (
    format_forecast_text,
    format_synastry_text,
)
from services.profiles import (
    CONTACT_COMMAND_USAGE,
    PREMIUM_REQUIRED_MESSAGE,
    ContactCommandInput,
    ContactPermissionError,
    PremiumRequiredError,
    ProfileChangeLimitError,
    ProfileError,
    birth_data_from_profile,
    contact_input_from_command,
    create_contact_from_command,
    ensure_birth_change_allowed,
    has_premium,
    natal_data_present,
    parse_contact_command,
    profile_changes_note,
    resolve_owner_profile,
    save_owner_profile,
)
from services.referrals import (
    apply_referral,
    get_or_create_referral_code,
    parse_referral_payload,
    referral_link,
)
from services.synastry import SynastryService
from services.synastry_report import SynastryReportService
from storage.birth import (
    BIRTH_COMMAND_USAGE,
    BirthData,
    BirthInputError,
    detect_timezone,
    geocode_city,
    local_to_utc,
    parse_birth_command,
    parse_birth_date,
    parse_birth_time,
)
from storage.pipeline import ChartPipeline
from storage.presentation import (
    REPORT_PART_LIMIT,
    format_chart_reply,
    format_history,
    format_section_previews,
    split_message,
    to_telegram_html,
)
from storage.store import PostgresChartStore
from logging_utils import PrivacyFilter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_TIMEOUT = 15.0
HOUSE_SYSTEM = "P"
HISTORY_LIMIT = 10


class ChartForm(StatesGroup):
    """Steps of the birth-data wizard."""

    name = State()
    birth_date = State()
    birth_time = State()
    birth_place = State()
    confirm = State()


class ContactForm(StatesGroup):
    """Steps of the contact wizard (premium feature)."""

    name = State()
    relation = State()
    birth_date = State()
    birth_time = State()
    birth_place = State()


class FeedbackForm(StatesGroup):
    """Waiting for the user's feedback text."""

    message = State()


async def _send_previews(message: Message, chunks: list[dict[str, Any]]) -> None:
    """Send the best sections as short follow-up messages."""
    for preview in format_section_previews(chunks):
        await message.answer(preview, disable_web_page_preview=True)


async def _drop_markup(message: Message | None) -> None:
    """Remove inline buttons from the previous wizard message."""
    if message is None:
        return
    try:
        await message.edit_reply_markup(reply_markup=None)
    except TelegramBadRequest:
        pass


class LastActiveMiddleware(BaseMiddleware):
    """Touch ``users.last_active_at`` at most once a minute per account."""

    WINDOW_SECONDS = 60.0

    def __init__(self, store: Any) -> None:
        self.store = store
        self._seen: dict[int, float] = {}

    async def __call__(self, handler: Any, event: Any, data: dict[str, Any]) -> Any:
        user = getattr(event, "from_user", None)
        if user is not None and not user.is_bot:
            now = monotonic()
            if now - self._seen.get(user.id, 0.0) >= self.WINDOW_SECONDS:
                self._seen[user.id] = now
                try:
                    await self.store.touch_user(user.id)
                except Exception:
                    logger.debug("last_active_at update failed", exc_info=True)
        return await handler(event, data)


def create_dispatcher(
    retriever: QdrantRetriever,
    generator: RAGGenerator,
    chart_pipeline: ChartPipeline | None = None,
    report_service: ChartReportService | None = None,
    forecast_service: ForecastService | None = None,
    forecast_report_service: ForecastReportService | None = None,
    synastry_service: SynastryService | None = None,
    synastry_report_service: SynastryReportService | None = None,
    premium_catalog: PremiumCatalog | None = None,
    bot_username: str | None = None,
    admin_chat_id: int | None = None,
    admin_user_ids: set[int] | None = None,
) -> Dispatcher:
    """Create handlers bound to the single startup instances of the services."""
    dispatcher = Dispatcher()
    if chart_pipeline is not None:
        dispatcher.update.outer_middleware(LastActiveMiddleware(chart_pipeline.store))
    router = Router()

    def services_ready() -> bool:
        """True when the PostgreSQL-backed features (history, forecasts) are on."""
        return (
            chart_pipeline is not None
            and forecast_service is not None
            and synastry_service is not None
        )

    async def _show_inline(callback: CallbackQuery, text: str, markup: Any) -> None:
        """Replace the message text and keyboard; fall back to a new message."""
        if callback.message is None:
            return
        try:
            await callback.message.edit_text(text, reply_markup=markup)
        except TelegramBadRequest:
            await callback.message.answer(text, reply_markup=markup)

    async def _send_ui(
        message: Message, text: str, *, reply_markup: Any = None
    ) -> None:
        """Send an authored UI text with explicit HTML parsing.

        The texts keep a light Markdown for editing convenience; sending always
        declares the parse mode, and a rejected markup never loses the message:
        the same text is resent as plain text without markers. The main menu is
        attached by default so a long greeting never leaves the user without
        buttons.
        """
        markup = reply_markup if reply_markup is not None else main_menu()
        for part in split_message(render_ui_html(text), limit=REPORT_PART_LIMIT):
            try:
                await message.answer(
                    part,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )
            except TelegramBadRequest:
                await message.answer(
                    strip_ui_markdown(part),
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )

    async def _pick_contacts(
        telegram_id: int, *, require_premium: bool
    ) -> tuple[list[Any], str | None, bool]:
        """Load contacts; the third value tells whether the premium gate blocked it."""
        if chart_pipeline is None:
            return [], "Контакты недоступны: не настроено подключение к PostgreSQL.", False
        try:
            profile = await resolve_owner_profile(chart_pipeline.store, telegram_id=telegram_id)
        except ProfileError as exc:
            return [], str(exc), False
        if require_premium and not has_premium(profile, telegram_id):
            return [], PREMIUM_REQUIRED_MESSAGE, True
        contacts = await chart_pipeline.store.list_contacts(profile.user_id)
        return contacts, None, False

    async def _report_is_cached(service: ChartReportService, chart_hash: str) -> bool:
        """True when the report for this chart is stored already (no LLM needed)."""
        try:
            generation_id = service.generation_id(chart_hash)
            return await service.store.get_generation(generation_id) is not None
        except Exception:
            logger.exception("report cache probe failed")
            return False

    async def start_wizard(message: Message, state: FSMContext) -> None:
        await state.clear()
        await state.set_state(ChartForm.name)
        await message.answer(ASK_NAME, reply_markup=skip_name_keyboard())

    async def start_chart_flow(message: Message, state: FSMContext) -> None:
        """Open the chart card: reuse the stored profile when it exists.

        A returning user lands on the confirmation card with the saved data
        instead of retyping name, date, time and place; the wizard opens only
        for a first-time profile or via «✏️ Изменить данные».
        """
        await state.clear()
        profile = None
        if chart_pipeline is not None and message.from_user is not None:
            try:
                profile = await chart_pipeline.store.get_user_profile(
                    telegram_id=message.from_user.id
                )
            except Exception:
                logger.exception("profile lookup failed")
        if profile is not None and natal_data_present(profile):
            await message.answer(
                format_stored_profile(profile, note=profile_changes_note(profile)),
                reply_markup=profile_confirm_keyboard(),
                disable_web_page_preview=True,
            )
            return
        await start_wizard(message, state)

    async def send_history(message: Message) -> None:
        if chart_pipeline is None:
            await message.answer("История недоступна: не настроено подключение к PostgreSQL.")
            return
        user = message.from_user
        if user is None:
            await message.answer("Не удалось определить пользователя Telegram.")
            return
        try:
            user_id = await chart_pipeline.resolve_user(telegram_id=user.id)
            records = (
                await chart_pipeline.history(user_id, limit=HISTORY_LIMIT) if user_id else []
            )
        except Exception:
            logger.exception("history request failed")
            await message.answer("Не удалось загрузить историю. Попробуйте позже.")
            return

        await message.answer(
            format_history(records), reply_markup=main_menu(), disable_web_page_preview=True
        )

    async def run_chart(
        message: Message,
        user: User,
        birth: BirthData,
        *,
        display_name: str | None = None,
        time_unknown: bool = False,
        profile_name: str = "Я",
        save_profile: bool = True,
    ) -> bool:
        """Run the chart pipeline, send the summary and the LLM report.

        ``save_profile=False`` computes a contact's chart: the owner's stored
        natal profile and the monthly change limit stay untouched, and the
        history row is labelled with the contact's name.
        """
        if chart_pipeline is None:
            await message.answer("Расчёт карты недоступен: не настроено подключение к PostgreSQL.")
            return False

        profile = None
        if save_profile:
            try:
                profile = await chart_pipeline.store.get_user_profile(telegram_id=user.id)
            except Exception:
                logger.exception("profile lookup failed")
            if profile is not None:
                # Fail fast before the expensive calculation: the monthly change
                # limit protects the profile (and the LLM budget) from churn.
                try:
                    ensure_birth_change_allowed(profile, birth, time_unknown=time_unknown)
                except ProfileChangeLimitError as exc:
                    await message.answer(str(exc), reply_markup=main_menu())
                    return False

        await message.answer(
            "Считаю карту и подбираю разделы из книг…\n"
            "Первый разбор занимает около минуты."
        )
        try:
            result = await chart_pipeline.load(
                birth,
                telegram_id=user.id,
                first_name=user.first_name,
                last_name=user.last_name,
                display_name=display_name,
                profile_name=profile_name,
            )
        except BirthInputError as exc:
            await message.answer(f"Не удалось рассчитать карту: {exc}")
            return False
        except AstronomyError as exc:
            await message.answer(f"Астрономический расчёт не удался: {exc}")
            return False
        except Exception:
            logger.exception("chart request failed")
            await message.answer("Не удалось рассчитать карту. Попробуйте позже.")
            return False

        await _record_event(user.id, "chart")

        # The forecast tools read birth data from the stored profile only, so
        # every successful chart request refreshes the owner's natal profile.
        if save_profile:
            try:
                await save_owner_profile(
                    chart_pipeline.store,
                    result.user_id,
                    birth,
                    time_unknown=time_unknown,
                    profile=profile,
                )
            except ProfileChangeLimitError as exc:
                await message.answer(str(exc))
            except Exception:
                logger.exception("owner natal profile save failed")

        summary = format_chart_reply(result, technical=False)
        if time_unknown:
            summary += (
                "\n\n⚠️ Время рождения неизвестно: взято 12:00, "
                "дома и ASC могут быть неточными."
            )
        await message.answer(summary, reply_markup=main_menu(), disable_web_page_preview=True)

        if report_service is None:
            await _send_previews(message, result.chunks)
            return True

        # Unchanged data must not trigger an LLM call: probe the report cache
        # first and skip the "generating" notice when the text is already there.
        if not await _report_is_cached(report_service, result.chart_hash):
            await message.answer("Готовлю разбор карты…")
        try:
            report = await report_service.get_or_generate(result)
        except Exception:
            logger.exception("chart report generation failed")
            await message.answer("Текстовый разбор сейчас недоступен — показываю найденные разделы.")
            await _send_previews(message, result.chunks)
            return True

        for part in split_message(report.text, limit=REPORT_PART_LIMIT):
            try:
                await message.answer(
                    to_telegram_html(part),
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except TelegramBadRequest:
                # Never lose the report because of one unparsable character.
                await message.answer(part, disable_web_page_preview=True)
        return True

    @router.message(CommandStart(deep_link=True))
    async def start_referral(
        message: Message, command: CommandObject, state: FSMContext, bot: Bot
    ) -> None:
        """First contact via an invite link: attribute the account, reward the inviter."""
        await state.clear()
        code = parse_referral_payload(command.args)
        user = message.from_user
        if code is None or user is None or chart_pipeline is None or premium_catalog is None:
            await _send_ui(message, GREETING)
            return
        outcome = None
        try:
            existing = await chart_pipeline.store.find_user(telegram_id=user.id)
            if existing is None:
                user_id = await chart_pipeline.store.upsert_user(
                    telegram_id=user.id, first_name=user.first_name, last_name=user.last_name
                )
                outcome = await apply_referral(
                    chart_pipeline.store,
                    referee_user_id=user_id,
                    code=code,
                    bonus_months=premium_catalog.referral_bonus_months,
                )
        except Exception:
            logger.exception("referral registration failed")
        await _send_ui(message, GREETING)
        if outcome is None or outcome.status != "granted" or not outcome.referrer_telegram_id:
            return
        try:
            await bot.send_message(
                outcome.referrer_telegram_id,
                "🎉 По вашей ссылке пришёл новый пользователь!\n"
                f"Начислено {outcome.bonus_months} мес. премиума.",
            )
        except Exception:
            logger.exception("referral notification failed")

    @router.message(CommandStart())
    async def start_handler(message: Message, state: FSMContext) -> None:
        await state.clear()
        await _send_ui(message, GREETING)

    @router.message(Command("help"))
    async def help_handler(message: Message, state: FSMContext) -> None:
        await state.clear()
        await _send_ui(message, HELP)

    @router.message(F.text == BUTTON_HELP)
    async def help_button(message: Message, state: FSMContext) -> None:
        await state.clear()
        await _send_ui(message, HELP)

    @router.message(F.text == BUTTON_CHART)
    async def chart_button(message: Message, state: FSMContext) -> None:
        await start_chart_flow(message, state)

    @router.message(Command("history"))
    @router.message(F.text == BUTTON_HISTORY)
    async def history_handler(message: Message, state: FSMContext) -> None:
        await state.clear()
        await send_history(message)

    @router.message(Command("chart"))
    async def chart_handler(
        message: Message, command: CommandObject, state: FSMContext
    ) -> None:
        if chart_pipeline is None:
            await message.answer("Расчёт карты недоступен: не настроено подключение к PostgreSQL.")
            return
        if not command.args:
            await start_chart_flow(message, state)
            return

        await state.clear()
        try:
            moment_local, city = parse_birth_command(command.args)
        except BirthInputError as exc:
            await message.answer(f"{exc}\nФормат: {BIRTH_COMMAND_USAGE}")
            return

        try:
            geo = await asyncio.to_thread(geocode_city, city, timeout=GEO_TIMEOUT)
            tz_name = await asyncio.to_thread(detect_timezone, geo["lat"], geo["lon"])
            utc, _offset, _notes = local_to_utc(moment_local, tz_name)
            birth = BirthData(
                dt_utc=utc,
                latitude=geo["lat"],
                longitude=geo["lon"],
                house_system=HOUSE_SYSTEM,
                city=str(geo["display_name"]),
                timezone=tz_name,
                datetime_local=moment_local.isoformat(),
            )
        except BirthInputError as exc:
            await message.answer(f"Не удалось подготовить данные: {exc}")
            return
        except Exception:
            logger.exception("quick chart geocoding failed")
            await message.answer("Не удалось определить место рождения. Попробуйте ещё раз.")
            return

        await run_chart(message, message.from_user, birth)

    @router.message(ChartForm.name)
    async def name_step(message: Message, state: FSMContext) -> None:
        if not message.text:
            await message.answer(ASK_NAME, reply_markup=skip_name_keyboard())
            return
        name = message.text.strip()[:255] or None
        await state.update_data(name=name)
        await state.set_state(ChartForm.birth_date)
        await message.answer(ASK_DATE)

    @router.message(ChartForm.birth_date)
    async def date_step(message: Message, state: FSMContext) -> None:
        try:
            parsed = parse_birth_date(message.text or "")
        except BirthInputError as exc:
            await message.answer(f"{exc}\n\n{ASK_DATE}")
            return
        await state.update_data(birth_date=parsed.isoformat())
        await state.set_state(ChartForm.birth_time)
        await message.answer(ASK_TIME, reply_markup=unknown_time_keyboard())

    @router.message(ChartForm.birth_time)
    async def time_step(message: Message, state: FSMContext) -> None:
        try:
            parsed = parse_birth_time(message.text or "")
        except BirthInputError as exc:
            await message.answer(f"{exc}\n\n{ASK_TIME}", reply_markup=unknown_time_keyboard())
            return
        await state.update_data(birth_time=parsed.strftime("%H:%M"), time_unknown=False)
        await state.set_state(ChartForm.birth_place)
        await message.answer(ASK_PLACE)

    @router.message(ChartForm.birth_place)
    async def place_step(message: Message, state: FSMContext) -> None:
        city = (message.text or "").strip()
        if not city:
            await message.answer(ASK_PLACE)
            return
        try:
            geo = await asyncio.to_thread(geocode_city, city, timeout=GEO_TIMEOUT)
            tz_name = await asyncio.to_thread(detect_timezone, geo["lat"], geo["lon"])
        except BirthInputError as exc:
            await message.answer(f"{exc}\n\nПопробуйте иначе: «Москва» или «Москва, Россия».")
            return
        except Exception:
            logger.exception("wizard geocoding failed")
            await message.answer("Не удалось определить место. Попробуйте позже.")
            return

        await state.update_data(place=geo, timezone=tz_name)
        data = await state.get_data()
        await state.set_state(ChartForm.confirm)
        await message.answer(
            format_confirmation(
                name=data.get("name"),
                birth_date=str(data["birth_date"]),
                birth_time=str(data["birth_time"]),
                timezone_name=tz_name,
                place=geo,
                time_unknown=bool(data.get("time_unknown")),
            ),
            reply_markup=confirm_keyboard(),
        )

    @router.message(ChartForm.confirm)
    async def confirm_text(message: Message) -> None:
        await message.answer(
            "Нажмите «✅ Рассчитать» или «✏️ Изменить данные».",
            reply_markup=confirm_keyboard(),
        )

    @router.callback_query(ChartForm.name, F.data == SKIP_NAME_CALLBACK)
    async def skip_name(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await state.update_data(name=None)
        await state.set_state(ChartForm.birth_date)
        await _drop_markup(callback.message)
        if callback.message is not None:
            await callback.message.answer(ASK_DATE)

    @router.callback_query(ChartForm.birth_time, F.data == UNKNOWN_TIME_CALLBACK)
    async def unknown_time(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await state.update_data(birth_time="12:00", time_unknown=True)
        await state.set_state(ChartForm.birth_place)
        await _drop_markup(callback.message)
        if callback.message is not None:
            await callback.message.answer(ASK_PLACE)

    @router.callback_query(ChartForm.confirm, F.data == EDIT_CALLBACK)
    async def edit_data(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await state.set_state(ChartForm.birth_date)
        await _drop_markup(callback.message)
        if callback.message is not None:
            await callback.message.answer(f"Хорошо, введём заново.\n\n{ASK_DATE}")

    @router.callback_query(ChartForm.confirm, F.data == CONFIRM_CALLBACK)
    async def confirm_data(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        data = await state.get_data()
        await _drop_markup(callback.message)
        if callback.message is None:
            return

        required = ("birth_date", "birth_time", "place", "timezone")
        if any(key not in data for key in required):
            await state.set_state(ChartForm.birth_date)
            await callback.message.answer(f"Данные потерялись, начнём заново.\n\n{ASK_DATE}")
            return

        try:
            birth = build_birth_data(data, house_system=HOUSE_SYSTEM)
        except (BirthInputError, KeyError, TypeError, ValueError) as exc:
            await state.set_state(ChartForm.birth_date)
            await callback.message.answer(f"Не удалось собрать данные: {exc}\n\n{ASK_DATE}")
            return

        await state.clear()
        await run_chart(
            callback.message,
            callback.from_user,
            birth,
            display_name=data.get("name"),
            time_unknown=bool(data.get("time_unknown")),
        )

    @router.callback_query(F.data == PROFILE_CONFIRM_CALLBACK)
    async def profile_confirm(callback: CallbackQuery, state: FSMContext) -> None:
        """Recalculate from the stored profile: no wizard and no re-geocoding."""
        await callback.answer()
        await state.clear()
        if chart_pipeline is None or callback.message is None:
            return
        try:
            profile = await chart_pipeline.store.get_user_profile(
                telegram_id=callback.from_user.id
            )
        except Exception:
            logger.exception("profile lookup failed")
            profile = None
        if profile is None or not natal_data_present(profile):
            await _show_inline(
                callback, "Профиль не найден — заполните данные рождения заново.", None
            )
            await start_wizard(callback.message, state)
            return
        try:
            birth = birth_data_from_profile(profile)
        except (ProfileError, BirthInputError) as exc:
            await _show_inline(callback, f"Данные профиля неполны: {exc}", None)
            return
        await _drop_markup(callback.message)
        await run_chart(
            callback.message,
            callback.from_user,
            birth,
            display_name=profile.display_name,
            time_unknown=profile.birth_time_unknown,
        )

    @router.callback_query(F.data == PROFILE_EDIT_CALLBACK)
    async def profile_edit(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await _drop_markup(callback.message)
        if callback.message is not None:
            await start_wizard(callback.message, state)

    def _premium_view(profile: Any) -> tuple[str, Any]:
        text = "\n\n".join(
            [
                format_premium_status(profile),
                "Премиум даёт: прогнозы для контактов, синастрию и персональные разборы.",
                "🎁 Пригласите друга — и получите месяц премиума бесплатно (/invite).",
                "Выберите срок — оплата звёздами Telegram:",
            ]
        )
        return text, premium_keyboard(premium_catalog)

    @router.message(Command("premium"))
    @router.message(F.text == BUTTON_PREMIUM)
    async def premium_entry(message: Message, state: FSMContext) -> None:
        await state.clear()
        if chart_pipeline is None or premium_catalog is None:
            await message.answer("Магазин премиума временно недоступен.")
            return
        user = message.from_user
        if user is None:
            return
        try:
            profile = await resolve_owner_profile(chart_pipeline.store, telegram_id=user.id)
        except ProfileError as exc:
            await message.answer(str(exc), reply_markup=main_menu())
            return
        text, markup = _premium_view(profile)
        await message.answer(text, reply_markup=markup)

    @router.callback_query(F.data == PREMIUM_MENU_CALLBACK)
    async def premium_menu(callback: CallbackQuery) -> None:
        await callback.answer()
        if chart_pipeline is None or premium_catalog is None:
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await _show_inline(callback, str(exc), None)
            return
        text, markup = _premium_view(profile)
        await _show_inline(callback, text, markup)

    @router.callback_query(F.data == PREMIUM_HISTORY_CALLBACK)
    async def premium_history(callback: CallbackQuery) -> None:
        await callback.answer()
        if chart_pipeline is None:
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await _show_inline(callback, str(exc), None)
            return
        events = await chart_pipeline.store.list_premium_events(profile.user_id, limit=10)
        markup = premium_keyboard(premium_catalog) if premium_catalog is not None else None
        await _show_inline(callback, format_premium_history(events, profile), markup)

    @router.callback_query(F.data.startswith(PREMIUM_BUY_PREFIX))
    async def premium_buy(callback: CallbackQuery) -> None:
        key = callback.data[len(PREMIUM_BUY_PREFIX):]
        plan = premium_catalog.get(key) if premium_catalog is not None else None
        if plan is None:
            await callback.answer(
                "Тариф недоступен — откройте «👑 Премиум» заново.", show_alert=True
            )
            return
        await callback.answer()
        if callback.message is None:
            return
        try:
            await callback.message.answer_invoice(
                title=f"Премиум: {plan.title}",
                description=(
                    f"Доступ на {plan.months} мес.: прогнозы для контактов, синастрия "
                    "и персональные разборы."
                ),
                payload=f"premium:{plan.key}:{callback.from_user.id}",
                currency="XTR",
                prices=[LabeledPrice(label=plan.title, amount=plan.stars)],
            )
        except TelegramBadRequest:
            logger.exception("invoice creation failed")
            await callback.message.answer("Не удалось выставить счёт. Попробуйте позже.")

    @router.pre_checkout_query()
    async def pre_checkout(query: PreCheckoutQuery) -> None:
        parts = (query.invoice_payload or "").split(":")
        valid = (
            len(parts) == 3
            and parts[0] == "premium"
            and premium_catalog is not None
            and premium_catalog.get(parts[1]) is not None
        )
        await query.answer(
            ok=bool(valid),
            error_message=None if valid else "Тариф устарел — выберите заново в «👑 Премиум».",
        )

    @router.message(F.successful_payment)
    async def premium_paid(message: Message) -> None:
        payment = message.successful_payment
        user = message.from_user
        if payment is None or user is None or chart_pipeline is None:
            return
        parts = (payment.invoice_payload or "").split(":")
        plan = (
            premium_catalog.get(parts[1])
            if premium_catalog is not None and len(parts) == 3 and parts[0] == "premium"
            else None
        )
        if plan is None:
            await message.answer(
                "Оплата получена, но тариф не распознан — напишите администратору."
            )
            return
        try:
            user_id = await chart_pipeline.store.upsert_user(
                telegram_id=user.id, first_name=user.first_name, last_name=user.last_name
            )
            until = await chart_pipeline.store.complete_payment(
                charge_id=payment.telegram_payment_charge_id,
                user_id=user_id,
                plan_key=plan.key,
                months=plan.months,
                stars=int(payment.total_amount),
                reason=f"Тариф «{plan.title}» ({plan.key})",
            )
        except Exception:
            logger.exception("payment processing failed")
            await message.answer(
                "Оплата прошла, но начисление не удалось — напишите администратору."
            )
            return
        if until is None:
            await message.answer("Этот платёж уже обработан.", reply_markup=main_menu())
            return
        await message.answer(
            f"✅ Оплата получена: {plan.title}.\n"
            f"Премиум активен до {until.strftime('%d.%m.%Y')}.",
            reply_markup=main_menu(),
        )

    async def _invite_text(telegram_id: int) -> str | None:
        if chart_pipeline is None or premium_catalog is None or not bot_username:
            return None
        try:
            profile = await resolve_owner_profile(chart_pipeline.store, telegram_id=telegram_id)
        except ProfileError as exc:
            return str(exc)
        code = await get_or_create_referral_code(chart_pipeline.store, profile.user_id)
        stats = await chart_pipeline.store.referral_stats(profile.user_id)
        return format_invite_text(
            link=referral_link(bot_username, code),
            invited=stats["invited"],
            bonus_months=stats["bonus_months"],
            reward_months=premium_catalog.referral_bonus_months,
        )

    @router.message(Command("invite"))
    async def invite_command(message: Message, state: FSMContext) -> None:
        await state.clear()
        user = message.from_user
        if user is None:
            return
        text = await _invite_text(user.id)
        await message.answer(
            text or "Реферальная программа временно недоступна.", reply_markup=main_menu()
        )

    @router.callback_query(F.data == INVITE_CALLBACK)
    async def invite_callback(callback: CallbackQuery) -> None:
        await callback.answer()
        text = await _invite_text(callback.from_user.id)
        await _show_inline(
            callback, text or "Реферальная программа временно недоступна.", None
        )

    @router.message(Command("forecast"))
    @router.message(F.text == BUTTON_FORECAST)
    async def forecast_entry(message: Message, state: FSMContext) -> None:
        await state.clear()
        if not services_ready():
            await message.answer("Прогнозы недоступны: не настроено подключение к PostgreSQL.")
            return
        await message.answer(
            "Выберите, для кого построить прогноз:", reply_markup=forecast_profile_keyboard()
        )

    @router.callback_query(F.data == FORECAST_PROFILE_CALLBACK)
    async def forecast_back_to_profile(callback: CallbackQuery) -> None:
        await callback.answer()
        await _show_inline(
            callback, "Выберите, для кого построить прогноз:", forecast_profile_keyboard()
        )

    @router.callback_query(F.data == FORECAST_SELF_CALLBACK)
    async def forecast_self(callback: CallbackQuery) -> None:
        await callback.answer()
        await _show_inline(callback, "На какой период?", forecast_interval_keyboard(None))

    @router.callback_query(F.data == FORECAST_CONTACTS_CALLBACK)
    async def forecast_contacts(callback: CallbackQuery) -> None:
        await callback.answer()
        contacts, error_text, premium_denied = await _pick_contacts(
            callback.from_user.id, require_premium=True
        )
        if error_text:
            await _show_inline(
                callback, error_text, premium_upsell_keyboard() if premium_denied else None
            )
            return
        if not contacts:
            await _show_inline(
                callback,
                "Список контактов пуст — добавьте человека, чтобы построить прогноз.",
                contact_pick_keyboard(
                    [],
                    FORECAST_PICK_PREFIX,
                    add_callback=CONTACTS_ADD_FORECAST_CALLBACK,
                    cancel_callback=FORECAST_PROFILE_CALLBACK,
                ),
            )
            return
        await _show_inline(
            callback,
            "Выберите человека:",
            contact_pick_keyboard(
                contacts,
                FORECAST_PICK_PREFIX,
                add_callback=CONTACTS_ADD_FORECAST_CALLBACK,
                cancel_callback=FORECAST_PROFILE_CALLBACK,
            ),
        )

    @router.callback_query(F.data.startswith(FORECAST_PICK_PREFIX))
    async def forecast_pick(callback: CallbackQuery) -> None:
        await callback.answer()
        raw = callback.data[len(FORECAST_PICK_PREFIX):]
        try:
            contact_id = int(raw)
        except ValueError:
            await _show_inline(
                callback, "Не удалось определить контакт. Откройте прогноз заново.", None
            )
            return
        await _show_inline(callback, "На какой период?", forecast_interval_keyboard(contact_id))

    @router.callback_query(F.data.startswith(FORECAST_RUN_PREFIX))
    async def forecast_run(callback: CallbackQuery) -> None:
        payload = callback.data[len(FORECAST_RUN_PREFIX):]
        interval, separator, target = payload.partition(":")
        contact_id: int | None = None
        if target.startswith("c"):
            try:
                contact_id = int(target[1:])
            except ValueError:
                contact_id = None
        if not separator or (target != "self" and contact_id is None):
            await callback.answer(
                "Не удалось разобрать команду. Откройте прогноз заново.", show_alert=True
            )
            return
        await callback.answer()
        if callback.message is None:
            return
        await run_forecast(callback.message, callback.from_user, interval, contact_id)

    async def run_forecast(
        message: Message, user: User, interval: str, contact_id: int | None
    ) -> None:
        """Run the forecast tool and send the rendered report."""
        if forecast_service is None:
            await message.answer("Прогнозы недоступны: не настроено подключение к PostgreSQL.")
            return
        await message.answer("Считаю транзиты и готовлю разбор…")
        try:
            result = await forecast_service.get_forecast_for_interval(
                user.id, interval, contact_id=contact_id
            )
        except ProfileError as exc:
            await message.answer(str(exc))
            return
        except PremiumRequiredError as exc:
            await message.answer(str(exc), reply_markup=premium_upsell_keyboard())
            return
        except ContactPermissionError as exc:
            await message.answer(str(exc))
            return
        except BirthInputError as exc:
            await message.answer(f"Данные профиля неполны: {exc}")
            return
        except AstronomyError as exc:
            await message.answer(f"Астрономический расчёт не удался: {exc}")
            return
        except Exception:
            logger.exception("forecast request failed")
            await message.answer("Не удалось построить прогноз. Попробуйте позже.")
            return

        await _record_event(user.id, "forecast")

        # The interpretation is the product answer; the deterministic table is
        # only a fallback when the LLM or the knowledge base is unavailable.
        if forecast_report_service is not None:
            try:
                interpretation = await forecast_report_service.get_or_generate(
                    result, retriever=retriever
                )
            except Exception:
                # No service messages in chat: fall back to the deterministic grid.
                logger.exception("forecast interpretation failed")
            else:
                for part in split_message(interpretation["text"], limit=REPORT_PART_LIMIT):
                    try:
                        await message.answer(
                            to_telegram_html(part),
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                    except TelegramBadRequest:
                        # Never lose the report because of one unparsable character.
                        await message.answer(part, disable_web_page_preview=True)
                return
        for part in split_message(format_forecast_text(result)):
            await message.answer(part, disable_web_page_preview=True)

    @router.message(Command("synastry"))
    @router.message(F.text == BUTTON_SYNASTRY)
    async def synastry_entry(message: Message, state: FSMContext) -> None:
        await state.clear()
        if not services_ready():
            await message.answer("Совместимость недоступна: не настроено подключение к PostgreSQL.")
            return
        user = message.from_user
        if user is None:
            await message.answer("Не удалось определить пользователя Telegram.")
            return
        contacts, error_text, premium_denied = await _pick_contacts(
            user.id, require_premium=True
        )
        if error_text:
            await message.answer(
                error_text,
                reply_markup=premium_upsell_keyboard() if premium_denied else main_menu(),
            )
            return
        if not contacts:
            await message.answer(
                "Список контактов пуст — добавьте человека, чтобы проверить совместимость.",
                reply_markup=contact_pick_keyboard(
                    [],
                    SYNASTRY_PREFIX,
                    add_callback=CONTACTS_ADD_SYNASTRY_CALLBACK,
                ),
            )
            return
        await message.answer(
            "Выберите человека для проверки совместимости:",
            reply_markup=contact_pick_keyboard(
                contacts,
                SYNASTRY_PREFIX,
                add_callback=CONTACTS_ADD_SYNASTRY_CALLBACK,
            ),
        )

    @router.callback_query(F.data.startswith(SYNASTRY_PREFIX))
    async def synastry_run(callback: CallbackQuery) -> None:
        raw = callback.data[len(SYNASTRY_PREFIX):]
        await callback.answer()
        if callback.message is None:
            return
        try:
            contact_id = int(raw)
        except ValueError:
            await callback.message.answer(
                "Не удалось определить контакт. Откройте совместимость заново."
            )
            return
        if synastry_service is None:
            await callback.message.answer(
                "Совместимость недоступна: не настроено подключение к PostgreSQL."
            )
            return
        await callback.message.answer("Считаю совместимость и готовлю разбор…")
        try:
            result = await synastry_service.check_synastry(callback.from_user.id, contact_id)
        except ProfileError as exc:
            await callback.message.answer(str(exc))
            return
        except PremiumRequiredError as exc:
            await callback.message.answer(str(exc), reply_markup=premium_upsell_keyboard())
            return
        except ContactPermissionError as exc:
            await callback.message.answer(str(exc))
            return
        except BirthInputError as exc:
            await callback.message.answer(f"Данные профиля неполны: {exc}")
            return
        except AstronomyError as exc:
            await callback.message.answer(f"Астрономический расчёт не удался: {exc}")
            return
        except Exception:
            logger.exception("synastry request failed")
            await callback.message.answer("Не удалось проверить совместимость. Попробуйте позже.")
            return

        await _record_event(callback.from_user.id, "synastry")

        # The reading is the product answer; the deterministic grid is only a
        # fallback, and the user never sees a service apology in chat.
        if synastry_report_service is not None:
            try:
                interpretation = await synastry_report_service.get_or_generate(
                    result, retriever=retriever
                )
            except Exception:
                logger.exception("synastry interpretation failed")
            else:
                for part in split_message(interpretation["text"], limit=REPORT_PART_LIMIT):
                    try:
                        await callback.message.answer(
                            to_telegram_html(part),
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                    except TelegramBadRequest:
                        await callback.message.answer(part, disable_web_page_preview=True)
                return
        for part in split_message(format_synastry_text(result)):
            await callback.message.answer(part, disable_web_page_preview=True)

    @router.callback_query(F.data == CHART_CONTACT_PICK_CALLBACK)
    async def chart_contact_pick(callback: CallbackQuery) -> None:
        """Choose whose chart to compute: the owner's or a contact's."""
        await callback.answer()
        if chart_pipeline is None or premium_catalog is None:
            await _show_inline(
                callback, "Расчёт карты недоступен: не настроено подключение к PostgreSQL.", None
            )
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await _show_inline(callback, str(exc), None)
            return
        if not has_premium(profile, callback.from_user.id):
            # Two ways to unlock: buy a term with Stars or invite a friend.
            await _show_inline(callback, PREMIUM_REQUIRED_MESSAGE, premium_upsell_keyboard())
            return
        contacts = await chart_pipeline.store.list_contacts(profile.user_id)
        if not contacts:
            await _show_inline(
                callback,
                "Список контактов пуст — добавьте человека, чтобы рассчитать его карту.",
                contact_pick_keyboard(
                    [],
                    CHART_CONTACT_RUN_PREFIX,
                    add_callback=CONTACTS_ADD_CHART_CALLBACK,
                    cancel_callback=CHART_SELF_CARD_CALLBACK,
                ),
            )
            return
        await _show_inline(
            callback,
            "Кому рассчитать карту?",
            contact_pick_keyboard(
                contacts,
                CHART_CONTACT_RUN_PREFIX,
                add_callback=CONTACTS_ADD_CHART_CALLBACK,
                cancel_callback=CHART_SELF_CARD_CALLBACK,
            ),
        )

    @router.callback_query(F.data.startswith(CHART_CONTACT_RUN_PREFIX))
    async def chart_contact_run(callback: CallbackQuery) -> None:
        raw = callback.data[len(CHART_CONTACT_RUN_PREFIX):]
        await callback.answer()
        if chart_pipeline is None or callback.message is None:
            return
        try:
            contact_id = int(raw)
        except ValueError:
            await callback.message.answer("Не удалось определить контакт.")
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await callback.message.answer(str(exc))
            return
        if not has_premium(profile, callback.from_user.id):
            await callback.message.answer(
                PREMIUM_REQUIRED_MESSAGE, reply_markup=premium_upsell_keyboard()
            )
            return
        contact = await chart_pipeline.store.get_contact(contact_id, profile.user_id)
        if contact is None:
            await callback.message.answer("Контакт не найден.")
            return
        try:
            birth = birth_data_from_profile(contact)
        except (ProfileError, BirthInputError) as exc:
            await callback.message.answer(f"Данные контакта неполны: {exc}")
            return
        await _drop_markup(callback.message)
        await run_chart(
            callback.message,
            callback.from_user,
            birth,
            time_unknown=contact.birth_time_unknown,
            profile_name=contact.name,
            save_profile=False,
        )

    @router.callback_query(F.data == CHART_SELF_CARD_CALLBACK)
    async def chart_self_card(callback: CallbackQuery) -> None:
        await callback.answer()
        if chart_pipeline is None:
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await _show_inline(callback, str(exc), None)
            return
        await _show_inline(
            callback,
            format_stored_profile(profile, note=profile_changes_note(profile)),
            profile_confirm_keyboard(),
        )

    async def show_contacts(message: Message) -> None:
        if chart_pipeline is None:
            await message.answer("Контакты недоступны: не настроено подключение к PostgreSQL.")
            return
        user = message.from_user
        if user is None:
            return
        contacts, error_text, _premium_denied = await _pick_contacts(
            user.id, require_premium=False
        )
        if error_text:
            await message.answer(error_text)
            return
        await message.answer(
            format_contacts_screen(contacts), reply_markup=contacts_list_keyboard(contacts)
        )

    @router.message(Command("contacts"))
    @router.message(F.text == BUTTON_CONTACTS)
    async def contacts_command(message: Message, state: FSMContext) -> None:
        await state.clear()
        await show_contacts(message)

    @router.callback_query(F.data == CONTACTS_BACK_CALLBACK)
    async def contacts_back(callback: CallbackQuery) -> None:
        await callback.answer()
        if chart_pipeline is None or callback.message is None:
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await _show_inline(callback, str(exc), None)
            return
        contacts = await chart_pipeline.store.list_contacts(profile.user_id)
        await _show_inline(
            callback, format_contacts_screen(contacts), contacts_list_keyboard(contacts)
        )

    @router.callback_query(F.data.startswith(CONTACTS_OPEN_PREFIX))
    async def contacts_open(callback: CallbackQuery) -> None:
        await callback.answer()
        if chart_pipeline is None or callback.message is None:
            return
        try:
            contact_id = int(callback.data[len(CONTACTS_OPEN_PREFIX):])
        except ValueError:
            await _show_inline(callback, "Не удалось определить контакт.", None)
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await _show_inline(callback, str(exc), None)
            return
        contact = await chart_pipeline.store.get_contact(contact_id, profile.user_id)
        if contact is None:
            await _show_inline(callback, "Контакт не найден.", contacts_list_keyboard([]))
            return
        await _show_inline(
            callback, format_contact_card(contact), contact_card_keyboard(contact.contact_id)
        )

    @router.callback_query(F.data == CONTACTS_ADD_CALLBACK)
    async def contacts_add(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if callback.message is not None:
            await start_contact_wizard(callback.message, state)

    @router.callback_query(F.data == CONTACTS_ADD_SYNASTRY_CALLBACK)
    async def contacts_add_synastry(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if callback.message is not None:
            await start_contact_wizard(callback.message, state, return_to="synastry")

    @router.callback_query(F.data == CONTACTS_ADD_FORECAST_CALLBACK)
    async def contacts_add_forecast(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if callback.message is not None:
            await start_contact_wizard(callback.message, state, return_to="forecast")

    @router.callback_query(F.data == CONTACTS_ADD_CHART_CALLBACK)
    async def contacts_add_chart(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if callback.message is not None:
            await start_contact_wizard(callback.message, state, return_to="chart")

    @router.callback_query(F.data.startswith(CONTACTS_EDIT_PREFIX))
    async def contacts_edit(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        if chart_pipeline is None or callback.message is None:
            return
        try:
            contact_id = int(callback.data[len(CONTACTS_EDIT_PREFIX):])
        except ValueError:
            await callback.message.answer("Не удалось определить контакт.")
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await callback.message.answer(str(exc))
            return
        contact = await chart_pipeline.store.get_contact(contact_id, profile.user_id)
        if contact is None:
            await callback.message.answer("Контакт не найден.")
            return
        await _drop_markup(callback.message)
        await start_contact_wizard(callback.message, state, contact=contact)

    @router.callback_query(F.data.startswith(CONTACTS_DELETE_PREFIX))
    async def contacts_delete(callback: CallbackQuery) -> None:
        await callback.answer()
        raw = callback.data[len(CONTACTS_DELETE_PREFIX):]
        if chart_pipeline is None or callback.message is None:
            return
        try:
            contact_id = int(raw)
        except ValueError:
            await callback.message.answer("Не удалось определить контакт.")
            return
        try:
            profile = await resolve_owner_profile(
                chart_pipeline.store, telegram_id=callback.from_user.id
            )
        except ProfileError as exc:
            await callback.message.answer(str(exc))
            return
        deleted = await chart_pipeline.store.delete_contact(contact_id, profile.user_id)
        contacts = await chart_pipeline.store.list_contacts(profile.user_id)
        text = format_contacts_screen(contacts)
        if not deleted:
            text = "Контакт не найден.\n\n" + text
        await _show_inline(callback, text, contacts_list_keyboard(contacts))

    async def start_contact_wizard(
        message: Message,
        state: FSMContext,
        *,
        contact: Any = None,
        return_to: str | None = None,
    ) -> None:
        """Step-by-step contact entry; ``contact`` switches to edit mode."""
        await state.clear()
        if chart_pipeline is None or premium_catalog is None:
            await message.answer("Контакты недоступны: не настроено подключение к PostgreSQL.")
            return
        user = message.from_user
        if user is None:
            return
        try:
            profile = await resolve_owner_profile(chart_pipeline.store, telegram_id=user.id)
        except ProfileError as exc:
            await message.answer(str(exc), reply_markup=main_menu())
            return
        if not has_premium(profile, user.id):
            await message.answer(PREMIUM_REQUIRED_MESSAGE, reply_markup=premium_upsell_keyboard())
            return
        await state.update_data(contact_owner_id=profile.user_id)
        if contact is not None:
            await state.update_data(contact_edit_id=contact.contact_id)
        if return_to:
            await state.update_data(contact_return_to=return_to)
        await state.set_state(ContactForm.name)
        if contact is not None:
            await message.answer(
                f"✏️ Изменяем контакт «{contact.name}». Заполните данные заново.\n\n"
                f"{CONTACT_WIZARD_ASK_NAME}"
            )
        else:
            await message.answer(CONTACT_WIZARD_ASK_NAME)

    @router.message(ContactForm.name)
    async def contact_name_step(message: Message, state: FSMContext) -> None:
        name = (message.text or "").strip()
        if not name:
            await message.answer(CONTACT_WIZARD_ASK_NAME)
            return
        await state.update_data(contact_name=name[:255])
        await state.set_state(ContactForm.relation)
        await message.answer(CONTACT_WIZARD_ASK_RELATION, reply_markup=contact_relation_keyboard())

    @router.message(ContactForm.relation)
    async def contact_relation_step(message: Message, state: FSMContext) -> None:
        text = (message.text or "").strip()
        await state.update_data(contact_relation=text[:100] if text else None)
        await state.set_state(ContactForm.birth_date)
        await message.answer(CONTACT_WIZARD_ASK_DATE)

    @router.callback_query(ContactForm.relation, F.data == CONTACT_SKIP_RELATION_CALLBACK)
    async def contact_relation_skip(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await state.update_data(contact_relation=None)
        await state.set_state(ContactForm.birth_date)
        await _drop_markup(callback.message)
        if callback.message is not None:
            await callback.message.answer(CONTACT_WIZARD_ASK_DATE)

    @router.message(ContactForm.birth_date)
    async def contact_date_step(message: Message, state: FSMContext) -> None:
        try:
            parsed = parse_birth_date(message.text or "")
        except BirthInputError as exc:
            await message.answer(f"{exc}\n\n{CONTACT_WIZARD_ASK_DATE}")
            return
        await state.update_data(contact_birth_date=parsed.isoformat())
        await state.set_state(ContactForm.birth_time)
        await message.answer(CONTACT_WIZARD_ASK_TIME, reply_markup=contact_time_keyboard())

    @router.message(ContactForm.birth_time)
    async def contact_time_step(message: Message, state: FSMContext) -> None:
        try:
            parsed = parse_birth_time(message.text or "")
        except BirthInputError as exc:
            await message.answer(
                f"{exc}\n\n{CONTACT_WIZARD_ASK_TIME}", reply_markup=contact_time_keyboard()
            )
            return
        await state.update_data(
            contact_birth_time=parsed.strftime("%H:%M"), contact_time_unknown=False
        )
        await state.set_state(ContactForm.birth_place)
        await message.answer(CONTACT_WIZARD_ASK_PLACE)

    @router.callback_query(ContactForm.birth_time, F.data == CONTACT_UNKNOWN_TIME_CALLBACK)
    async def contact_time_unknown(callback: CallbackQuery, state: FSMContext) -> None:
        await callback.answer()
        await state.update_data(contact_birth_time=None, contact_time_unknown=True)
        await state.set_state(ContactForm.birth_place)
        await _drop_markup(callback.message)
        if callback.message is not None:
            await callback.message.answer(CONTACT_WIZARD_ASK_PLACE)

    @router.message(ContactForm.birth_place)
    async def contact_place_step(message: Message, state: FSMContext) -> None:
        city = (message.text or "").strip()
        if not city:
            await message.answer(CONTACT_WIZARD_ASK_PLACE)
            return
        if chart_pipeline is None:
            await message.answer("Контакты недоступны: не настроено подключение к PostgreSQL.")
            return
        data = await state.get_data()
        try:
            parsed = ContactCommandInput(
                name=str(data.get("contact_name") or "Без имени"),
                relation=data.get("contact_relation"),
                birth_date=date.fromisoformat(str(data["contact_birth_date"])),
                birth_time=(
                    time.fromisoformat(str(data["contact_birth_time"]))
                    if data.get("contact_birth_time")
                    else None
                ),
                time_unknown=bool(data.get("contact_time_unknown")),
                city=city,
            )
            owner_id = int(data["contact_owner_id"])
        except (KeyError, TypeError, ValueError) as exc:
            await state.clear()
            await message.answer(f"Данные мастера потерялись ({exc}). Начните заново: /contact")
            return
        await message.answer(f"Ищу город «{city}»…")
        try:
            geo = await asyncio.to_thread(geocode_city, city, timeout=GEO_TIMEOUT)
            tz_name = await asyncio.to_thread(detect_timezone, geo["lat"], geo["lon"])
            contact_input = contact_input_from_command(parsed, geo, tz_name)
            edit_id = data.get("contact_edit_id")
            if edit_id:
                updated = await chart_pipeline.store.update_contact(
                    int(edit_id), owner_id, contact_input
                )
                contact = None
                if updated:
                    contact = await chart_pipeline.store.get_contact(
                        int(edit_id), owner_id
                    )
                if contact is None:
                    await state.clear()
                    await message.answer(
                        "Контакт не найден — возможно, он удалён. Начните заново: /contact"
                    )
                    return
                await state.clear()
                await message.answer(
                    "✏️ Контакт обновлён.\n\n" + format_contact_saved(contact),
                    reply_markup=main_menu(),
                )
                return
            contact = await chart_pipeline.store.add_contact(owner_id, contact_input)
        except (BirthInputError, ValueError) as exc:
            await message.answer(f"Не удалось сохранить контакт: {exc}")
            return
        except Exception:
            logger.exception("contact creation failed")
            await message.answer("Не удалось сохранить контакт. Попробуйте позже.")
            return

        return_to = data.get("contact_return_to")
        await state.clear()
        if return_to == "synastry":
            contacts = await chart_pipeline.store.list_contacts(owner_id)
            await message.answer(
                "✅ Контакт сохранён.\n\nВыберите человека для проверки совместимости:",
                reply_markup=contact_pick_keyboard(
                    contacts,
                    SYNASTRY_PREFIX,
                    add_callback=CONTACTS_ADD_SYNASTRY_CALLBACK,
                ),
            )
            return
        if return_to == "forecast":
            contacts = await chart_pipeline.store.list_contacts(owner_id)
            await message.answer(
                "✅ Контакт сохранён.\n\nВыберите человека:",
                reply_markup=contact_pick_keyboard(
                    contacts,
                    FORECAST_PICK_PREFIX,
                    add_callback=CONTACTS_ADD_FORECAST_CALLBACK,
                    cancel_callback=FORECAST_PROFILE_CALLBACK,
                ),
            )
            return
        if return_to == "chart":
            contacts = await chart_pipeline.store.list_contacts(owner_id)
            await message.answer(
                "✅ Контакт сохранён.\n\nКому рассчитать карту?",
                reply_markup=contact_pick_keyboard(
                    contacts,
                    CHART_CONTACT_RUN_PREFIX,
                    add_callback=CONTACTS_ADD_CHART_CALLBACK,
                    cancel_callback=CHART_SELF_CARD_CALLBACK,
                ),
            )
            return
        await message.answer(format_contact_saved(contact), reply_markup=main_menu())

    @router.message(Command("contact"))
    async def contact_add_command(
        message: Message, command: CommandObject, state: FSMContext
    ) -> None:
        if not (command.args or "").strip():
            await start_contact_wizard(message, state)
            return
        if not services_ready():
            await message.answer("Контакты недоступны: не настроено подключение к PostgreSQL.")
            return
        user = message.from_user
        if user is None:
            await message.answer("Не удалось определить пользователя Telegram.")
            return
        try:
            profile = await resolve_owner_profile(chart_pipeline.store, telegram_id=user.id)
        except ProfileError as exc:
            await message.answer(str(exc))
            return
        if not has_premium(profile, user.id):
            await message.answer(PREMIUM_REQUIRED_MESSAGE, reply_markup=premium_upsell_keyboard())
            return
        try:
            parsed = parse_contact_command(command.args)
        except BirthInputError as exc:
            await message.answer(f"{exc}\nФормат: {CONTACT_COMMAND_USAGE}")
            return
        await message.answer(f"Ищу город «{parsed.city}»…")
        try:
            geo = await asyncio.to_thread(geocode_city, parsed.city, timeout=GEO_TIMEOUT)
            tz_name = await asyncio.to_thread(detect_timezone, geo["lat"], geo["lon"])
            contact = await create_contact_from_command(
                chart_pipeline.store, profile.user_id, parsed, geo, tz_name
            )
        except BirthInputError as exc:
            await message.answer(f"Не удалось сохранить контакт: {exc}")
            return
        except Exception:
            logger.exception("contact creation failed")
            await message.answer("Не удалось сохранить контакт. Попробуйте позже.")
            return
        await message.answer(
            format_contact_saved(contact),
            reply_markup=main_menu(),
        )

    async def _react(bot: Bot, chat_id: int, message_id: int, emoji: str) -> None:
        """React to an admin message; a missing reaction right is not fatal."""
        try:
            await bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception:
            logger.warning("could not set a reaction on the admin message", exc_info=True)

    async def _record_event(telegram_id: int, event_type: str) -> None:
        """Append one product-analytics event; never breaks the user flow."""
        if chart_pipeline is None:
            return
        try:
            await chart_pipeline.store.record_event(telegram_id, event_type)
        except Exception:
            logger.exception("analytics event failed")

    @router.message(Command("stats"))
    async def stats_command(message: Message) -> None:
        """On-demand product snapshot: admin group or admin ids only."""
        user = message.from_user
        allowed = (admin_chat_id is not None and message.chat.id == admin_chat_id) or (
            user is not None and bool(admin_user_ids) and user.id in admin_user_ids
        )
        if not allowed or chart_pipeline is None:
            return
        now = datetime.now(timezone.utc)
        try:
            metrics = await chart_pipeline.store.collect_analytics(
                since=now - timedelta(days=7)
            )
        except Exception:
            logger.exception("stats collection failed")
            await message.answer("Не удалось собрать статистику.")
            return
        await message.answer(
            format_stats_now(metrics, now=now),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    @router.message(Command("id"))
    async def id_handler(message: Message) -> None:
        """Show chat and user ids; used to fill ADMIN_CHAT_ID in .env."""
        chat = message.chat
        user = message.from_user
        lines = [
            "🆔 Информация о чате:",
            f"Тип чата: {chat.type}",
            f"ID этого чата: {chat.id}",
        ]
        if user is not None:
            lines.append(f"Ваш личный ID: {user.id}")
        await message.answer("\n".join(lines))

    @router.message(Command("feedback"))
    @router.message(F.text == BUTTON_FEEDBACK)
    async def feedback_entry(message: Message, state: FSMContext) -> None:
        await state.set_state(FeedbackForm.message)
        await message.answer(FEEDBACK_PROMPT)

    @router.message(Command("cancel"))
    async def cancel_feedback(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer("Отменено.", reply_markup=main_menu())

    @router.message(FeedbackForm.message)
    async def feedback_message(message: Message, state: FSMContext, bot: Bot) -> None:
        user = message.from_user
        text = (message.text or "").strip()
        if user is None:
            return
        if text.startswith("/"):
            await message.answer(FEEDBACK_EMPTY)
            return
        if not text:
            await message.answer(FEEDBACK_EMPTY)
            return
        if chart_pipeline is None:
            await state.clear()
            await message.answer(FEEDBACK_UNAVAILABLE, reply_markup=main_menu())
            return
        try:
            ticket = await chart_pipeline.store.add_feedback(user.id, user.username, text)
        except Exception:
            logger.exception("feedback storage failed")
            await message.answer(FEEDBACK_UNAVAILABLE, reply_markup=main_menu())
            return
        await state.clear()
        if admin_chat_id:
            try:
                created = (
                    ticket.created_at.strftime("%d.%m.%Y %H:%M UTC")
                    if ticket.created_at is not None
                    else "—"
                )
                sent = await bot.send_message(
                    admin_chat_id,
                    format_ticket_notice(
                        ticket_id=ticket.id,
                        username=ticket.username,
                        user_id=ticket.user_id,
                        message=ticket.message,
                        created_at=created,
                    ),
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                await chart_pipeline.store.attach_feedback_message(
                    ticket.id, admin_chat_id, sent.message_id
                )
            except Exception:
                logger.exception("feedback delivery to the admin chat failed")
        else:
            logger.warning(
                "ADMIN_CHAT_ID is not configured; feedback #%s is stored without forwarding",
                ticket.id,
            )
        await message.answer(FEEDBACK_SENT, reply_markup=main_menu())

    @router.message(F.chat.id == admin_chat_id, F.reply_to_message)
    async def admin_feedback_reply(message: Message, bot: Bot) -> None:
        """Deliver an admin reply from the group to the user in private."""
        if chart_pipeline is None or admin_chat_id is None:
            return
        sender = message.from_user
        if sender is None or sender.is_bot:
            return
        reply = message.reply_to_message
        if reply is None:
            return
        text = (message.text or message.caption or "").strip()
        if not text:
            return
        ticket = await chart_pipeline.store.find_feedback_by_admin_message(
            admin_chat_id, reply.message_id
        )
        if ticket is None:
            user_id = parse_user_id(reply.text or reply.caption)
            if user_id is not None:
                ticket = await chart_pipeline.store.find_open_feedback(user_id)
        if ticket is None:
            return
        try:
            await bot.send_message(ticket.user_id, format_admin_reply(text))
        except Exception:
            logger.exception("feedback reply delivery failed")
            await _react(bot, admin_chat_id, message.message_id, "👎")
            return
        await chart_pipeline.store.mark_feedback_answered(ticket.id)
        await _react(bot, admin_chat_id, message.message_id, "👍")

    @router.message()
    async def text_handler(message: Message, bot: Bot) -> None:
        user_query = message.text
        if not user_query:
            return

        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        try:
            chunks = await asyncio.to_thread(retriever.search, user_query, top_k=5)
            answer = await asyncio.to_thread(generator.generate_answer, user_query, chunks)
        except Exception:
            logger.exception("knowledge-base answer failed")
            await message.answer("Поиск по книгам сейчас недоступен. Попробуйте позже.")
            return

        try:
            await message.answer(
                answer,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except TelegramBadRequest:
            # Если вдруг HTML упадет, отправляем как чистый текст
            await message.answer(answer, disable_web_page_preview=True)

    dispatcher.include_router(router)
    return dispatcher


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    handler = logging.StreamHandler()
    handler.addFilter(PrivacyFilter())
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[handler],
    )
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from the environment or .env")

    # Build the potentially heavyweight clients once, before polling starts.
    retriever = QdrantRetriever()
    try:
        generator = RAGGenerator()
    except Exception:
        retriever.close()
        raise

    bot = Bot(token=token)
    premium_catalog = None
    try:
        premium_catalog = load_catalog()
    except PremiumConfigError:
        logger.exception("premium plans are unavailable; the shop is disabled")
    admin_chat_id: int | None = None
    raw_admin_chat = os.getenv("ADMIN_CHAT_ID", "").strip()
    if raw_admin_chat:
        try:
            admin_chat_id = int(raw_admin_chat)
        except ValueError:
            logger.error(
                "ADMIN_CHAT_ID must be an integer, got %r; feedback forwarding is disabled",
                raw_admin_chat,
            )
    bot_username = None
    try:
        bot_username = (await bot.get_me()).username
    except Exception:
        logger.exception("could not resolve the bot username; referral links are disabled")

    admin_user_ids: set[int] = set()
    for token in os.getenv("ADMIN_USER_IDS", "").replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            admin_user_ids.add(int(token))

    store: PostgresChartStore | None = None
    chart_pipeline: ChartPipeline | None = None
    report_service: ChartReportService | None = None
    forecast_service: ForecastService | None = None
    forecast_report_service: ForecastReportService | None = None
    synastry_service: SynastryService | None = None
    synastry_report_service: SynastryReportService | None = None
    if os.getenv("DATABASE_URL", "").strip():
        try:
            store = await PostgresChartStore.connect()
            chart_pipeline = ChartPipeline(store)
            report_service = ChartReportService(store)
            forecast_service = ForecastService(store)
            forecast_report_service = ForecastReportService(store)
            synastry_service = SynastryService(store)
            synastry_report_service = SynastryReportService(store)
        except Exception:
            logger.exception("chart storage is unavailable; /chart and /history are disabled")
            store = None
            chart_pipeline = None
            report_service = None
            forecast_service = None
            forecast_report_service = None
            synastry_service = None
            synastry_report_service = None

    dispatcher = create_dispatcher(
        retriever,
        generator,
        chart_pipeline,
        report_service,
        forecast_service,
        forecast_report_service,
        synastry_service,
        synastry_report_service,
        premium_catalog,
        bot_username,
        admin_chat_id,
        admin_user_ids,
    )
    digest_task: asyncio.Task | None = None
    if admin_chat_id and store is not None:
        digest_task = asyncio.create_task(run_digest_scheduler(bot, store, admin_chat_id))
    try:
        await dispatcher.start_polling(bot)
    finally:
        if digest_task is not None:
            digest_task.cancel()
        await bot.session.close()
        retriever.close()
        generator.close()
        if store is not None:
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())
