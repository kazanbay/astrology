"""Telegram entry point for the astrology knowledge base assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message, User
from dotenv import load_dotenv

from astronomy.calculator import AstronomyError
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
    confirm_keyboard,
    format_confirmation,
    main_menu,
    build_birth_data,
    skip_name_keyboard,
    unknown_time_keyboard,
)
from generator.report import ChartReportService
from rag.generator import RAGGenerator
from rag.retriever import QdrantRetriever
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


def create_dispatcher(
    retriever: QdrantRetriever,
    generator: RAGGenerator,
    chart_pipeline: ChartPipeline | None = None,
    report_service: ChartReportService | None = None,
) -> Dispatcher:
    """Create handlers bound to the single startup instances of the services."""
    dispatcher = Dispatcher()
    router = Router()

    async def start_wizard(message: Message, state: FSMContext) -> None:
        await state.clear()
        await state.set_state(ChartForm.name)
        await message.answer(ASK_NAME, reply_markup=skip_name_keyboard())

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
    ) -> bool:
        """Run the chart pipeline, send the summary and the LLM report."""
        if chart_pipeline is None:
            await message.answer("Расчёт карты недоступен: не настроено подключение к PostgreSQL.")
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

    @router.message(CommandStart())
    async def start_handler(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(GREETING, reply_markup=main_menu())

    @router.message(Command("help"))
    async def help_handler(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(HELP, reply_markup=main_menu())

    @router.message(F.text == BUTTON_HELP)
    async def help_button(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(HELP, reply_markup=main_menu())

    @router.message(F.text == BUTTON_CHART)
    async def chart_button(message: Message, state: FSMContext) -> None:
        await start_wizard(message, state)

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
            await start_wizard(message, state)
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
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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
    store: PostgresChartStore | None = None
    chart_pipeline: ChartPipeline | None = None
    report_service: ChartReportService | None = None
    if os.getenv("DATABASE_URL", "").strip():
        try:
            store = await PostgresChartStore.connect()
            chart_pipeline = ChartPipeline(store)
            report_service = ChartReportService(store)
        except Exception:
            logger.exception("chart storage is unavailable; /chart and /history are disabled")
            store = None
            chart_pipeline = None
            report_service = None

    dispatcher = create_dispatcher(retriever, generator, chart_pipeline, report_service)
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()
        retriever.close()
        generator.close()
        if store is not None:
            await store.close()


if __name__ == "__main__":
    asyncio.run(main())
