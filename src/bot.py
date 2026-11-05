"""Telegram entry point for the astrology knowledge base assistant."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from astronomy.calculator import AstronomyError
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
)
from storage.pipeline import ChartPipeline
from storage.presentation import format_chart_reply, format_history, format_section_previews
from storage.store import PostgresChartStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_TIMEOUT = 15.0
HOUSE_SYSTEM = "P"
HISTORY_LIMIT = 10
GREETING = (
    "Здравствуйте! Я помощник базы знаний по астрологии. "
    "Задайте вопрос, и я поищу ответ в материалах книг.\n\n"
    "Команды: /chart 1990-08-15 12:00 Москва — натальная карта и разделы книг, "
    "/history — история ваших запросов."
)


def create_dispatcher(
    retriever: QdrantRetriever,
    generator: RAGGenerator,
    chart_pipeline: ChartPipeline | None = None,
) -> Dispatcher:
    """Create handlers bound to the single startup instances of the RAG services."""
    dispatcher = Dispatcher()
    router = Router()

    @router.message(CommandStart())
    async def start_handler(message: Message) -> None:
        await message.answer(GREETING)

    @router.message(Command("chart"))
    async def chart_handler(message: Message, command: CommandObject) -> None:
        if chart_pipeline is None:
            await message.answer("Расчёт карты недоступен: не настроено подключение к PostgreSQL.")
            return
        user = message.from_user
        if user is None:
            await message.answer("Не удалось определить пользователя Telegram.")
            return

        try:
            moment_local, city = parse_birth_command(command.args)
        except BirthInputError as exc:
            await message.answer(f"{exc}\nФормат: {BIRTH_COMMAND_USAGE}")
            return

        await message.answer("Считаю карту и подбираю разделы из книг…")
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
            result = await chart_pipeline.load(
                birth,
                telegram_id=user.id,
                first_name=user.first_name,
                last_name=user.last_name,
            )
        except BirthInputError as exc:
            await message.answer(f"Не удалось рассчитать карту: {exc}")
            return
        except AstronomyError as exc:
            await message.answer(f"Астрономический расчёт не удался: {exc}")
            return
        except Exception:
            logger.exception("chart request failed")
            await message.answer("Не удалось рассчитать карту. Попробуйте позже.")
            return

        await message.answer(format_chart_reply(result), disable_web_page_preview=True)
        for preview in format_section_previews(result.chunks):
            await message.answer(preview, disable_web_page_preview=True)

    @router.message(Command("history"))
    async def history_handler(message: Message) -> None:
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

        await message.answer(format_history(records), disable_web_page_preview=True)

    @router.message()
    async def text_handler(message: Message, bot: Bot) -> None:
        user_query = message.text
        if not user_query:
            return

        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        chunks = await asyncio.to_thread(retriever.search, user_query, top_k=5)
        answer = await asyncio.to_thread(generator.generate_answer, user_query, chunks)

        try:
            await message.answer(
                answer, 
                parse_mode=ParseMode.HTML, 
                disable_web_page_preview=True
            )
        except TelegramBadRequest:
            # Если вдруг HTML упадет, отправляем как чистый текст
            await message.answer(answer, disable_web_page_preview=True)

    dispatcher.include_router(router)
    return dispatcher


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
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
    if os.getenv("DATABASE_URL", "").strip():
        try:
            store = await PostgresChartStore.connect()
            chart_pipeline = ChartPipeline(store)
        except Exception:
            logger.exception("chart storage is unavailable; /chart and /history are disabled")
            store = None
            chart_pipeline = None

    dispatcher = create_dispatcher(retriever, generator, chart_pipeline)
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
